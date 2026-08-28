"""Turning a generated tool call into a cart line, or refusing to.

This is the layer the architecture calls "code owns actually placing the
order", and the whole point of it is that the model's output is *not* trusted.
The adapters score 100% on format and ~92% on exact match, which is another way
of saying they are wrong about one order in twelve. A cart built by believing
them would be wrong that often too.

So every item is checked against the menu that was actually retrieved for that
turn -- the same rules validate_dataset.py applies to the training data -- and
anything that fails is dropped with a reason rather than silently corrected.
Prices are computed here from the catalog, never read out of the model's prose.
"""

from shared.tool_call import extract_tool_calls, strip_tool_calls

# The menu line writes one surcharge for the whole milk list, and the corpus
# always puts the house milk first, so anything else is the one that costs.
DEFAULT_MILK = "whole"


def _price(product, size):
    """What one of these costs before milk and extras."""
    prices = [float(p) for p in product.prices.split("/")]
    if product.sizes == "-":
        return prices[0]
    return prices[product.sizes.split(",").index(size)]


def _check(entry, products, shop):
    """Return (line, None) if the item is orderable, else (None, reason)."""
    name = str(entry.get("name", ""))
    product = products.get(name.lower())
    if product is None:
        return None, f"{name!r} is not on the menu that was retrieved"
    if not product.in_stock:
        return None, f"{product.name} is out of stock"

    size = entry.get("size")
    sizes = [] if product.sizes == "-" else product.sizes.split(",")
    if sizes and size not in sizes:
        return None, f"{product.name} does not come in size {size!r}"
    if not sizes and size is not None:
        return None, f"{product.name} has no sizes but was ordered as {size!r}"

    milk = entry.get("milk")
    if milk is not None and milk not in shop.milks:
        return None, f"{product.name}: {milk!r} milk is not offered"

    quantity = entry.get("quantity")
    if not isinstance(quantity, int) or quantity < 1:
        return None, f"{product.name}: {quantity!r} is not a quantity"

    extras = [e for e in entry.get("extras") or [] if isinstance(e, str)]
    surcharges = dict(shop.extras)
    unknown = [e for e in extras if e not in surcharges]
    if unknown:
        return None, f"{product.name}: {', '.join(unknown)} not offered"

    unit = _price(product, size)
    if milk is not None and milk != DEFAULT_MILK:
        unit += shop.milk_surcharge
    unit += sum(surcharges[e] for e in extras)

    return {
        "name": product.name,
        "size": size,
        "milk": milk,
        "extras": extras,
        "quantity": quantity,
        "unit_price": round(unit, 2),
        "line_total": round(unit * quantity, 2),
    }, None


def parse(reply, products, shop):
    """Read one assistant turn into a reply, a cart and a list of refusals.

    `products` is what retrieval put in front of the model this turn. Checking
    against anything wider would let the model order from a menu it was never
    shown, which is the failure this layer exists to catch.

    The cart is whatever the last valid create_order held, not an accumulation:
    the corpus teaches the model to emit the complete order at confirmation, so
    adding to a running total would double every item.
    """
    calls, malformed = extract_tool_calls(reply)
    by_name = {p.name.lower(): p for p in products}

    items, rejections = [], []
    if malformed:
        rejections.append(f"{malformed} tool call(s) were not valid JSON")

    for call in calls:
        if call.get("name") != "create_order":
            rejections.append(f"unexpected tool {call.get('name')!r}")
            continue
        entries = call.get("arguments", {}).get("items")
        if not isinstance(entries, list) or not entries:
            rejections.append("create_order arrived with no items")
            continue
        for entry in entries:
            line, reason = _check(entry, by_name, shop)
            (items if line else rejections).append(line or reason)

    return {
        "text": strip_tool_calls(reply),
        "items": items,
        "total": round(sum(i["line_total"] for i in items), 2),
        "rejections": rejections,
        "ordered": bool(calls),
    }
