"""Turning a proposed item into a cart line, or refusing to.

This is the layer the architecture calls "code owns actually placing the
order", and the whole point of it is that the model's output is *not* trusted.
The adapters score 100% on format and ~92% on exact match, which is another way
of saying they are wrong about one order in twelve. A cart built by believing
them would be wrong that often too.

So every item is checked against the menu that was actually retrieved -- the
same rules validate_dataset.py applies to the training data -- and anything
that fails is refused with a reason rather than silently corrected. Prices are
computed here from the catalog, never read out of the model's prose.

The agent calls this once per add_item. Nothing here knows about tool calls or
about conversations; tools.py owns that, and this owns what a valid line is.
"""

# The menu line writes one surcharge for the whole milk list, and the corpus
# always puts the house milk first, so anything else is the one that costs.
DEFAULT_MILK = "whole"


def price(product, size):
    """What one of these costs before milk and extras."""
    prices = [float(p) for p in product.prices.split("/")]
    if product.sizes == "-":
        return prices[0]
    return prices[product.sizes.split(",").index(size)]


def check_item(entry, products, shop):
    """Return (line, None) if the item is orderable, else (None, reason).

    `products` is what retrieval has put in front of the model. Checking
    against anything wider would let the model order from a menu it was never
    shown, which is the failure this layer exists to catch.
    """
    by_name = {p.name.lower(): p for p in products}

    name = str(entry.get("name", ""))
    product = by_name.get(name.lower())
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

    unit = price(product, size)
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


def total(lines):
    return round(sum(line["line_total"] for line in lines), 2)
