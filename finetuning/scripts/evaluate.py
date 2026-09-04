"""Score model output against the held-out set.

The checks are the ones validate_dataset.py applies to the corpus, pointed at
what the model generated instead. Each eval example is replayed up to its final
assistant turn; the model produces that turn; we read it.

Everything scored here is prose, because prose is what fine-tuning owns. Which
tool to call is read off the <tools> block in the prompt and executed by code
that validates the call -- scoring the model on it would be scoring the prompt,
and would go stale the day the shop adds a tool. A tool call in the output is
therefore stripped before scoring rather than punished: the served prompt does
advertise tools, so a reply may legitimately carry one.

Metrics
-------
grounded      names no product that is absent from that example's menu
in_stock      never offers something the menu marks OUT OF STOCK
one_question  asks at most one thing in a turn
alternative   when the ask cannot be met, names a real product instead
"""

import json
import pathlib
import re
import sys

# Same reason as validate_dataset.py: running this file directly puts its own
# directory on sys.path, not the repo root. Spelled out here rather than left to
# validate_dataset's import doing it first, which would make the order of these
# two lines load-bearing.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from shared.tool_call import strip_tool_calls  # noqa: E402
from shared.tools import tool_block  # noqa: E402
from validate_dataset import (METRICS, mentions, offers,  # noqa: E402
                              parse_menu, score_reply, wants_alternative)

SHARED = pathlib.Path(__file__).resolve().parent.parent.parent / "shared"
NEUTRAL_PROMPT_PATH = SHARED / "system_prompt.txt"

BRAND_LINE = re.compile(r"You are the ordering assistant for (.+?)\.")
MENU_BLOCK = re.compile(r"<menu>\n(.*?)\n</menu>", re.DOTALL)


def neutral_system(record):
    """The record's system prompt with every instruction about tone removed.

    Each voice was trained with its own rules -- "Be warm and concise" against
    "Be blunt. No pleasantries." Scoring each adapter under its own prompt
    cannot separate tone carried in the weights from tone read off the prompt,
    since a base model would follow those lines too. This swaps in one prompt
    that says nothing about how to sound, keeping the brand and the menu, so
    any remaining difference has to come from the adapter.
    """
    original = record["messages"][0]["content"]
    brand = BRAND_LINE.search(original)
    menu = MENU_BLOCK.search(original)
    if not (brand and menu):
        raise ValueError("system prompt is not in the expected brand/menu shape")
    template = NEUTRAL_PROMPT_PATH.read_text(encoding="utf-8")
    # The tools come from shared/tools.py for the same reason the prompt comes
    # from shared/: this has to be the prompt the Space serves, or the scores
    # describe something nobody is running. No adapter was trained on a <tools>
    # block, which is the point -- the tool list is prompt, not weights.
    return template.format(brand=brand.group(1), menu=menu.group(1),
                           tools=tool_block())


def prompt_messages(record, neutral=False):
    """Everything up to (not including) the final assistant turn.

    neutral=True replaces the voice's system prompt with the shared one, which
    is the ablation that tells tone in the weights from tone in the prompt.
    """
    messages = record["messages"][:-1]
    if not neutral:
        return messages
    return [dict(messages[0], content=neutral_system(record))] + messages[1:]


def score_one(record, output):
    """Return a verdict per metric for one generated turn."""
    menu = parse_menu(record["messages"][0]["content"])
    asked, offered = set(), False
    for message in record["messages"][1:-1]:
        if message["role"] == "user":
            asked |= mentions(message["content"], menu["brand"])
        else:
            offered = offered or bool(offers(message["content"], menu, asked))

    needs = wants_alternative(record["meta"]["category"], menu, asked, offered)
    verdicts, _ = score_reply(strip_tool_calls(output), menu, asked, needs)
    return verdicts


def _rate(bucket, metric):
    """Share of passes among examples where the metric applied, or None."""
    hits, applicable = bucket[metric]
    return hits / applicable if applicable else None


def summarise(records, outputs):
    """Aggregate scores overall and per category, skipping N/A metrics."""
    def empty():
        return {m: [0, 0] for m in METRICS}

    totals, per_category = empty(), {}

    for record, output in zip(records, outputs):
        scores = score_one(record, output)
        bucket = per_category.setdefault(record["meta"]["category"], empty())
        bucket["n"] = bucket.get("n", 0) + 1
        for metric in METRICS:
            if scores[metric] is None:
                continue
            for target in (totals, bucket):
                target[metric][0] += scores[metric]
                target[metric][1] += 1

    return {
        "n": len(records),
        "overall": {m: _rate(totals, m) for m in METRICS},
        "per_category": per_category,
    }


def _cell(value):
    return "  n/a" if value is None else f"{value:>5.0%}"


def format_table(summary, label="model"):
    lines = [f"{label}  (n={summary['n']})", ""]
    for metric in METRICS:
        value = summary["overall"][metric]
        lines.append(f"  {metric:12} {'n/a' if value is None else f'{value:6.1%}'}")
    lines.append("")
    lines.append(f"  {'category':26} {'n':>4}  "
                 f"{'grnd':>5} {'stock':>5} {'1 q':>5} {'alt':>5}")
    for category, bucket in sorted(summary["per_category"].items()):
        lines.append(
            f"  {category:26} {bucket['n']:>4}  "
            + " ".join(_cell(_rate(bucket, m)) for m in METRICS))
    return "\n".join(lines)


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]
