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


def prompt_messages(record):
    """Everything up to (not including) the final assistant turn."""
    return record["messages"][:-1]


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
    result = {
        "format_ok": malformed == 0 and (bool(items) if should_order else True),
        "restraint": not items if category in NO_ORDER else True,
        "grounded": True,
        "valid_slots": True,
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


def summarise(records, outputs):
    """Aggregate scores overall and per category."""
    metrics = ["format_ok", "restraint", "grounded", "valid_slots", "exact_match"]
    totals = {m: 0 for m in metrics}
    per_category = {}

    for record, output in zip(records, outputs):
        scores = score_one(record, output)
        category = record["meta"]["category"]
        bucket = per_category.setdefault(category, {m: 0 for m in metrics} | {"n": 0})
        bucket["n"] += 1
        for metric in metrics:
            totals[metric] += scores[metric]
            bucket[metric] += scores[metric]

    n = len(records) or 1
    return {
        "n": len(records),
        "overall": {m: totals[m] / n for m in metrics},
        "per_category": per_category,
    }


def format_table(summary, label="model"):
    lines = [f"{label}  (n={summary['n']})", ""]
    for metric, value in summary["overall"].items():
        lines.append(f"  {metric:12} {value:6.1%}")
    lines.append("")
    lines.append(f"  {'category':26} {'n':>4}  {'fmt':>5} {'grnd':>5} {'slot':>5} {'exact':>5}")
    for category, bucket in sorted(summary["per_category"].items()):
        count = bucket["n"]
        lines.append(
            f"  {category:26} {count:>4}  "
            f"{bucket['format_ok']/count:>5.0%} {bucket['grounded']/count:>5.0%} "
            f"{bucket['valid_slots']/count:>5.0%} {bucket['exact_match']/count:>5.0%}")
    return "\n".join(lines)


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]
