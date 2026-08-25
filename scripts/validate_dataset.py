"""Validate a coffee-ordering dataset file against its own <menu> context.

Every check here is a property the trained model will later be scored on, so
this module is deliberately reusable: eval runs parse model output with the
same helpers and apply the same rules.

Checks
------
1. structural      -- system first, roles alternate, assistant speaks last
2. parseable       -- every <tool_call> block is valid JSON with the right schema
3. grounded        -- ordered product names exist in that example's <menu>
4. valid size      -- ordered size is offered for that product
5. valid milk      -- ordered milk is offered by that brand
6. in stock        -- nothing OUT OF STOCK is ever ordered
7. restraint       -- off-menu requests and price questions emit no tool call

Usage
-----
    python scripts/validate_dataset.py data/train.jsonl
"""

import json
import pathlib
import re
import sys

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
MENU_RE = re.compile(r"<menu>\n(.*?)\n</menu>", re.DOTALL)

# Categories where a correct assistant produces no order at all.
NO_ORDER_CATEGORIES = {"off_menu_request", "price_question_no_order"}


def extract_tool_calls(text):
    """Return (calls, malformed_count) for one assistant message."""
    calls, malformed = [], 0
    for raw in TOOL_CALL_RE.findall(text):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            malformed += 1
    # A bare <tool_call> with no JSON body is a malformed call too.
    opens = text.count("<tool_call>")
    if opens > len(calls) + malformed:
        malformed += opens - len(calls) - malformed
    return calls, malformed


def parse_menu(system_prompt):
    """Parse the <menu> block into a lookup the checks can query."""
    match = MENU_RE.search(system_prompt)
    if not match:
        return None

    products, milks = {}, set()
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Milk options:"):
            body = line.split(":", 1)[1].split("(")[0]
            milks = {m.strip() for m in body.split(",") if m.strip()}
            continue
        if line.startswith("Extras:"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        name, sizes, prices, stock = parts
        products[name.lower()] = {
            "sizes": [] if sizes == "-" else sizes.split(","),
            "prices": [float(p) for p in prices.split("/")],
            "in_stock": stock == "in stock",
        }
    return {"products": products, "milks": milks}


def check_record(record, index):
    """Return a list of human-readable problems with this record."""
    problems = []
    messages = record.get("messages", [])
    category = record.get("meta", {}).get("category", "?")

    # 1. structural
    if not messages or messages[0]["role"] != "system":
        problems.append("no system message")
        return problems
    if messages[-1]["role"] != "assistant":
        problems.append("does not end on an assistant turn")
    expected = "user"
    for message in messages[1:]:
        if message["role"] != expected:
            problems.append(f"role order broken at '{message['role']}'")
            break
        expected = "assistant" if expected == "user" else "user"

    menu = parse_menu(messages[0]["content"])
    if menu is None:
        problems.append("system prompt has no <menu> block")
        return problems

    ordered_any = False
    for message in messages:
        if message["role"] != "assistant":
            continue
        calls, malformed = extract_tool_calls(message["content"])
        if malformed:
            problems.append(f"{malformed} malformed tool_call block(s)")

        for call in calls:
            ordered_any = True
            # 2. schema
            if call.get("name") != "create_order":
                problems.append(f"unexpected tool name {call.get('name')!r}")
                continue
            items = call.get("arguments", {}).get("items")
            if not isinstance(items, list) or not items:
                problems.append("create_order with no items")
                continue

            for entry in items:
                name = str(entry.get("name", "")).lower()
                product = menu["products"].get(name)

                # 3. grounded
                if product is None:
                    problems.append(f"ordered off-menu product {entry.get('name')!r}")
                    continue
                # 6. in stock
                if not product["in_stock"]:
                    problems.append(f"ordered out-of-stock product {entry.get('name')!r}")
                # 4. size
                size = entry.get("size")
                if product["sizes"]:
                    if size not in product["sizes"]:
                        problems.append(
                            f"{entry.get('name')!r} size {size!r} not in {product['sizes']}")
                elif size is not None:
                    problems.append(f"{entry.get('name')!r} is sizeless but got size {size!r}")
                # 5. milk
                milk = entry.get("milk")
                if milk is not None and milk not in menu["milks"]:
                    problems.append(
                        f"{entry.get('name')!r} milk {milk!r} not offered "
                        f"(offered: {sorted(menu['milks'])})")
                # quantity sanity
                quantity = entry.get("quantity")
                if not isinstance(quantity, int) or quantity < 1:
                    problems.append(f"{entry.get('name')!r} bad quantity {quantity!r}")

    # 7. restraint -- the hard set states this per record, since an ambiguous
    # product name or an unavailable size calls for a question, not an order.
    meta = record.get("meta", {})
    should_order = meta.get("expects_order", category not in NO_ORDER_CATEGORIES)
    if not should_order and ordered_any:
        problems.append(f"'{category}' should emit no tool call")
    if should_order and not ordered_any:
        problems.append(f"'{category}' emitted no tool call")

    return problems


def main():
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/train.jsonl")
    records = [json.loads(line) for line in path.open(encoding="utf-8")]

    failures, total_problems = 0, 0
    for index, record in enumerate(records):
        problems = check_record(record, index)
        if problems:
            failures += 1
            total_problems += len(problems)
            if failures <= 10:
                print(f"record {index} [{record.get('meta', {}).get('category')}]")
                for problem in problems:
                    print(f"    - {problem}")

    print(f"\n{len(records)} records checked")
    print(f"{failures} records with problems ({total_problems} problems total)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
