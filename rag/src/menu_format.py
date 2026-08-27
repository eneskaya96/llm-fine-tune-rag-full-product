"""The one place that knows what a menu line looks like.

The adapters were fine-tuned to read exactly this shape, so it is not ours to
improve. `finetuning/scripts/validate_dataset.py:parse_menu` is the reference
parser and the round-trip test drives output back through it.

Drinks come before food, matching the order the training corpus generated
(generate_dataset.py:make_menu appends food after the shuffled drinks).
"""


def product_line(product):
    stock = "in stock" if product.in_stock else "OUT OF STOCK"
    return f"{product.name} | {product.sizes} | {product.prices} | {stock}"


def menu_block(products, shop):
    """The full <menu> body: product lines, then extras, then milks."""
    drinks = [p for p in products if p.kind == "drink"]
    food = [p for p in products if p.kind == "food"]

    lines = [product_line(p) for p in drinks + food]
    lines.append("Extras: " + ", ".join(
        f"{name} (+{price:.2f})" for name, price in shop.extras))
    lines.append(
        f"Milk options: {', '.join(shop.milks)} (+{shop.milk_surcharge:.2f})")
    return "\n".join(lines)
