"""What retrieval must not get wrong.

No benchmark here. Whether vector search beats substring matching over 28
products is not a question worth a report -- the answer is known and the
catalog is far too small for the comparison to mean anything. These are
regression tests: each one pins a rule that, if it broke, would produce a
plausible-looking menu with something wrong in it.
"""

import pytest
import validate_dataset as vd

from rag.src import menu_format
from rag.src.retrieve import MAX_DRINKS, MAX_FOOD, retrieve, select

SHOP = "ember_and_oak"


@pytest.fixture(scope="module")
def catalog_products():
    """Every product in the shop, keyed by name. Built once per session."""
    from rag.src.retrieve import _all_products
    from rag.src import catalog
    _, col = catalog.collection(SHOP)
    return {p.name: p for p in _all_products(col).values()}


def parse(query, history=()):
    """Retrieval output, read back through the trainer's own parser."""
    menu = retrieve(SHOP, query, history)
    parsed = vd.parse_menu(f"<menu>\n{menu}\n</menu>")
    assert parsed is not None, "parse_menu could not read the block"
    return parsed


def test_round_trips_through_the_reference_parser(catalog_products):
    """Every field survives the trip catalog -> Chroma -> menu line -> parser.

    This is the test that locks the format. The adapters read this shape and
    nothing else, so a drifted separator or a stray decimal is a silent
    regression rather than an error.
    """
    shop, products, _ = select(SHOP, "something cold and refreshing")
    parsed = parse("something cold and refreshing")

    assert len(parsed["products"]) == len(products)
    for product in products:
        got = parsed["products"][product.name.lower()]
        expected_sizes = [] if product.sizes == "-" else product.sizes.split(",")
        assert got["sizes"] == expected_sizes
        assert got["prices"] == [float(p) for p in product.prices.split("/")]
        assert got["in_stock"] == product.in_stock
    assert parsed["milks"] == set(shop.milks)


@pytest.mark.parametrize("query", [
    "a latte please", "something to eat", "what's cheap",
    "large flat white with oat", "anything without milk",
])
def test_menu_stays_inside_the_trained_length(query):
    """The adapters never saw a menu longer than this.

    9 drinks and 3 food is the top of the range generate_dataset.py produced.
    Beyond it the model is reading a prompt shape it has no training for.
    """
    _, products, _ = select(SHOP, query)
    assert sum(1 for p in products if p.kind == "drink") <= MAX_DRINKS
    assert sum(1 for p in products if p.kind == "food") <= MAX_FOOD


def test_a_product_ordered_earlier_survives_an_unrelated_query():
    """Rule 2, the one the model cannot compensate for.

    Turn three asks about cookies. If the latte from turn one falls off the
    menu, the model has no record of it anywhere and will deny an order it
    took two turns ago.
    """
    history = ["a latte please", "One latte. Anything else?",
               "actually add a cookie too"]
    parsed = parse("actually add a cookie too", history)
    assert "latte" in parsed["products"]
    assert "chocolate chip cookie" in parsed["products"]


def test_longer_product_name_wins_the_mention():
    """"Iced Latte" must not be read as a mention of "Latte"."""
    parsed = parse("something hot instead", ["can I get an iced latte"])
    assert "iced latte" in parsed["products"]


def test_sold_out_products_stay_on_the_menu(catalog_products):
    """Rule 4, and the reason it is a rule.

    Filtering the sold-out product out would have the model say the shop does
    not sell it. It does; it has run out. The adapters were trained to name the
    closest listed alternative, which they can only do if it is listed.
    """
    assert catalog_products["Iced Latte"].in_stock is False
    parsed = parse("can I get an iced latte", ["can I get an iced latte"])
    assert parsed["products"]["iced latte"]["in_stock"] is False


def test_extras_and_milks_go_in_whole(catalog_products):
    """Rule 3. Two short lines, needed by any order, so nothing to retrieve."""
    shop, _, _ = select(SHOP, "espresso")
    menu = retrieve(SHOP, "espresso")
    assert f"Milk options: {', '.join(shop.milks)}" in menu
    for name, _price in shop.extras:
        assert name in menu


@pytest.mark.parametrize("query,expected", [
    ("something cold and refreshing", "cold brew"),
    ("do you have tea", "earl grey"),
    ("something sweet to eat", "chocolate chip cookie"),
    ("just a plain black coffee", "americano"),
])
def test_the_obvious_semantic_cases(query, expected):
    """None of these queries contain the product's name.

    Not a benchmark -- a floor. If one of these stops matching, something
    about the descriptions or the embedding has changed.
    """
    assert expected in parse(query)["products"]


def test_food_is_retrieved_separately_from_drinks():
    """One ranked list would let a coffee query take every slot."""
    _, products, _ = select(SHOP, "a strong espresso, no milk")
    assert any(p.kind == "food" for p in products)


def test_reasons_cover_every_product_on_the_menu():
    """The Space shows its working; every line needs an explanation."""
    _, products, reasons = select(SHOP, "iced coffee", ["one blueberry muffin"])
    assert {p.id for p in products} == set(reasons)
    assert reasons["blueberry-muffin"] == "mentioned earlier in the conversation"
