"""Score model output against the held-out set.

The checks are the ones validate_dataset.py already applies to the corpus, now
pointed at what the model generated instead. Each eval example is replayed up to
its final assistant turn; the model produces that turn; we compare.

Metrics
-------
format_ok    output parses -- a tool call is well-formed JSON when one is due
restraint    no order emitted for questions and off-menu requests
grounded     ordered products exist on that example's menu
valid_slots  size and milk are offered, nothing out of stock
exact_match  order items identical to the reference (the strict one)
"""

import json
import pathlib
import re

from validate_dataset import extract_tool_calls, parse_menu

NO_ORDER = {"off_menu_request", "price_question_no_order"}


SHARED = pathlib.Path(__file__).resolve().parent.parent.parent / "shared"
SCHEMA_PATH = SHARED / "order_schema.json"
NEUTRAL_PROMPT_PATH = SHARED / "system_prompt.txt"

# The prompt text below is hand-formatted for the model to read, while
# shared/order_schema.json is the machine-readable contract the serving layer
# and frontend use. Rewording the prompt changes what the base model is scored
# against, so the two are kept as separate renderings of one contract and
# checked for drift at import rather than generated from each other.
TOOL_SCHEMA = """

You have one tool:

<tools>
{"name": "create_order", "description": "Place the customer's order.",
 "parameters": {"type": "object", "properties": {"items": {"type": "array", "items":
 {"type": "object", "properties": {
   "name": {"type": "string", "description": "exact product name from the menu"},
   "size": {"type": "string", "enum": ["S", "M", "L"], "description": "null for food"},
   "milk": {"type": "string", "description": "null unless the customer asked"},
   "extras": {"type": "array", "items": {"type": "string"}},
   "quantity": {"type": "integer"}},
  "required": ["name", "size", "milk", "extras", "quantity"]}}},
 "required": ["items"]}}
</tools>

Call it by writing exactly:
<tool_call>
{"name": "create_order", "arguments": {"items": [...]}}
</tool_call>"""


def _check_schema_drift():
    """Fail loudly if the prompt text and the shared contract disagree."""
    if not SCHEMA_PATH.exists():
        return
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fields = set(schema["parameters"]["properties"]["items"]["items"]["properties"])
    missing = [f for f in fields if f'"{f}"' not in TOOL_SCHEMA]
    if missing or schema["name"] not in TOOL_SCHEMA:
        raise AssertionError(
            f"TOOL_SCHEMA has drifted from {SCHEMA_PATH.name}: missing {missing}")


_check_schema_drift()


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
    return template.format(brand=brand.group(1), menu=menu.group(1))


def prompt_messages(record, tool_schema=False, neutral=False):
    """Everything up to (not including) the final assistant turn.

    The training prompt names create_order but never defines it -- the tuned
    model learns the schema from weights. A base model cannot, so scoring it on
    that prompt measures the omission rather than the model. Pass
    tool_schema=True to give it the definition and get a fair baseline.

    neutral=True replaces the voice's system prompt with the shared one, which
    is the ablation that tells tone in the weights from tone in the prompt.
    """
    messages = record["messages"][:-1]
    head = messages[0]
    if neutral:
        head = dict(head, content=neutral_system(record))
    if tool_schema:
        head = dict(head, content=head["content"] + TOOL_SCHEMA)
    if head is messages[0]:
        return messages
    return [head] + messages[1:]


def reference_items(record):
    calls, _ = extract_tool_calls(record["messages"][-1]["content"])
    return [i for c in calls for i in c.get("arguments", {}).get("items", [])]


def expects_order(record):
    """Whether a correct answer places an order.

    The hard set states this per record, since cases like an ambiguous product
    name or an unavailable size call for a question rather than an order and
    cannot be read off the category alone.
    """
    meta = record["meta"]
    if "expects_order" in meta:
        return meta["expects_order"]
    return meta["category"] not in NO_ORDER


def score_one(record, output):
    """Return a dict of pass/fail flags for one generated turn."""
    menu = parse_menu(record["messages"][0]["content"])
    calls, malformed = extract_tool_calls(output)
    items = [i for c in calls if c.get("name") == "create_order"
             for i in c.get("arguments", {}).get("items", [])]

    should_order = expects_order(record)
    # grounded and valid_slots are None when no order was emitted: a model that
    # never orders anything would otherwise score 100% on both.
    result = {
        "format_ok": malformed == 0 and (bool(items) if should_order else True),
        "restraint": not items if not should_order else True,
        "grounded": True if items else None,
        "valid_slots": True if items else None,
        "exact_match": False,
    }

    for entry in items:
        product = menu["products"].get(str(entry.get("name", "")).lower())
        if product is None:
            result["grounded"] = False
            continue
        if not product["in_stock"]:
            result["valid_slots"] = False
        size = entry.get("size")
        if product["sizes"] and size not in product["sizes"]:
            result["valid_slots"] = False
        milk = entry.get("milk")
        if milk is not None and milk not in menu["milks"]:
            result["valid_slots"] = False

    def key(entries):
        return sorted(
            (str(e.get("name", "")).lower(), e.get("size"), e.get("milk"),
             tuple(sorted(e.get("extras") or [])), e.get("quantity", 1))
            for e in entries
        )

    result["exact_match"] = key(items) == key(reference_items(record))
    return result


METRICS = ["format_ok", "restraint", "grounded", "valid_slots", "exact_match"]


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
    lines.append(f"  {'category':26} {'n':>4}  {'fmt':>5} {'grnd':>5} {'slot':>5} {'exact':>5}")
    for category, bucket in sorted(summary["per_category"].items()):
        lines.append(
            f"  {category:26} {bucket['n']:>4}  "
            + " ".join(_cell(_rate(bucket, m))
                       for m in ("format_ok", "grounded", "valid_slots", "exact_match")))
    return "\n".join(lines)


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]
