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

from validate_dataset import extract_tool_calls, parse_menu

NO_ORDER = {"off_menu_request", "price_question_no_order"}


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


def prompt_messages(record, tool_schema=False):
    """Everything up to (not including) the final assistant turn.

    The training prompt names create_order but never defines it -- the tuned
    model learns the schema from weights. A base model cannot, so scoring it on
    that prompt measures the omission rather than the model. Pass
    tool_schema=True to give it the definition and get a fair baseline.
    """
    messages = record["messages"][:-1]
    if not tool_schema:
        return messages
    head = dict(messages[0], content=messages[0]["content"] + TOOL_SCHEMA)
    return [head] + messages[1:]


def reference_items(record):
    calls, _ = extract_tool_calls(record["messages"][-1]["content"])
    return [i for c in calls for i in c.get("arguments", {}).get("items", [])]


def score_one(record, output):
    """Return a dict of pass/fail flags for one generated turn."""
    category = record["meta"]["category"]
    menu = parse_menu(record["messages"][0]["content"])
    calls, malformed = extract_tool_calls(output)
    items = [i for c in calls if c.get("name") == "create_order"
             for i in c.get("arguments", {}).get("items", [])]

    should_order = category not in NO_ORDER
    # grounded and valid_slots are None when no order was emitted: a model that
    # never orders anything would otherwise score 100% on both.
    result = {
        "format_ok": malformed == 0 and (bool(items) if should_order else True),
        "restraint": not items if category in NO_ORDER else True,
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
