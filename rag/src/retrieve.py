"""Catalog -> the <menu> block for one turn of one conversation.

The whole reason this layer exists: the adapters were trained on menus of four
to nine drinks and one to three food items, and never saw a longer one. A shop
with 28 products cannot put its catalog in the prompt, so something has to
choose. Choosing badly is worse than not retrieving at all -- the model
correctly refuses a product the shop actually sells, and the refusal looks
grounded.

Four rules do the choosing. Three of them are about what semantic search alone
gets wrong.
"""

import argparse

from . import catalog, menu_format

MAX_DRINKS = 9      # the top of the range the adapters were trained on
MAX_FOOD = 3


def _all_products(col):
    rows = col.get(include=["metadatas"])
    return {i: catalog.Product.from_metadata(i, m)
            for i, m in zip(rows["ids"], rows["metadatas"])}


def _mentioned(products, history):
    """Products named anywhere in the conversation so far.

    Rule 2. Without this the menu is rebuilt from scratch every turn, and a
    latte ordered on turn one is gone by turn three -- the model then denies
    an order it just took. The model cannot fix this from its side; it reads
    the menu it is given and has no other record of the product.

    Longest name first so "Iced Latte" claims the match before "Latte" can.
    """
    text = " ".join(history).lower()
    found, claimed = [], []
    for product in sorted(products, key=lambda p: -len(p.name)):
        lowered = product.name.lower()
        if lowered in text and not any(lowered in c for c in claimed):
            found.append(product)
            claimed.append(lowered)
    return found


def _nearest(col, query, kind, wanted, exclude):
    """Rule 1, run once per kind.

    Drinks and food are queried separately because one ranked list lets a
    coffee query take every slot and leave the pastries invisible.
    """
    if wanted <= 0:
        return []
    result = col.query(
        query_texts=[query],
        n_results=min(wanted + len(exclude), col.count()),
        where={"kind": kind},
    )
    picked = []
    for pid, meta, distance in zip(result["ids"][0], result["metadatas"][0],
                                   result["distances"][0]):
        if pid in exclude:
            continue
        picked.append((catalog.Product.from_metadata(pid, meta), distance))
        if len(picked) == wanted:
            break
    return picked


def select(slug, query, history=(), n_drinks=6, n_food=2):
    """The products this turn's menu should carry, and why each is there.

    Returns (shop, products, reasons) so a caller can show its working. The
    Space does: retrieval that nobody can see reads as magic, and the choosing
    is the part worth looking at.
    """
    shop, col = catalog.collection(slug)
    pinned = _mentioned(_all_products(col).values(), history)
    pinned_ids = {p.id for p in pinned}
    reasons = {p.id: "mentioned earlier in the conversation" for p in pinned}

    chosen = list(pinned)
    for kind, wanted, cap in (("drink", n_drinks, MAX_DRINKS),
                              ("food", n_food, MAX_FOOD)):
        room = cap - sum(1 for p in pinned if p.kind == kind)
        for product, distance in _nearest(col, query, kind,
                                          min(wanted, room), pinned_ids):
            chosen.append(product)
            reasons[product.id] = f"closest to the query (distance {distance:.2f})"

    # Rule 4 is the absence of a filter here: a sold-out product stays on the
    # menu as OUT OF STOCK. Dropping it would have the model say the shop does
    # not sell it, when it does and has simply run out -- and the adapters were
    # trained to offer the nearest listed alternative instead, which needs the
    # product to be listed.
    return shop, chosen, reasons


def retrieve(slug, query, history=(), **kwargs):
    """The <menu> body, ready to drop into the system prompt."""
    shop, products, _ = select(slug, query, history, **kwargs)
    # Rule 3: milks and extras go in whole, every time. They are two short
    # lines and every order may need them, so there is nothing to retrieve.
    return menu_format.menu_block(products, shop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shop", default="ember_and_oak")
    parser.add_argument("--query", required=True)
    parser.add_argument("--history", nargs="*", default=[])
    args = parser.parse_args()

    shop, products, reasons = select(args.shop, args.query, args.history)
    print(menu_format.menu_block(products, shop))
    print("\n-- why --")
    for product in products:
        print(f"{product.name:24} {reasons[product.id]}")


if __name__ == "__main__":
    main()
