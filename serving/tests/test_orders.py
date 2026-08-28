"""What the order layer must refuse.

The adapters are wrong about one order in twelve. Each test here is one of the
ways that goes wrong, and the assertion is always the same shape: the bad item
does not reach the cart, and the reason is visible rather than swallowed.
"""

import json

import orders
import pytest

from rag.src.retrieve import select

SHOP = "ember_and_oak"


@pytest.fixture(scope="module")
def context():
    """A real retrieved menu, since that is what parse() checks against."""
    shop, products, _ = select(SHOP, "a latte and something to eat")
    return shop, products


@pytest.fixture(scope="module")
def catalog_names():
    """Every product the shop sells, retrieved or not."""
    from rag.src import catalog
    _, products, _ = catalog.load(SHOP)
    return {p.name for p in products}


def call(items):
    return ("Sure.\n<tool_call>\n"
            + json.dumps({"name": "create_order", "arguments": {"items": items}})
            + "\n</tool_call>")


def item(name, size="M", milk=None, extras=None, quantity=1):
    return {"name": name, "size": size, "milk": milk,
            "extras": extras or [], "quantity": quantity}


def test_a_valid_order_reaches_the_cart(context):
    shop, products = context
    result = orders.parse(call([item("Latte", "L")]), products, shop)
    assert result["rejections"] == []
    assert result["items"][0]["name"] == "Latte"
    assert result["total"] == 4.40      # the L price from the catalog
    assert result["ordered"] is True


def test_the_tool_call_never_reaches_the_customer(context):
    shop, products = context
    result = orders.parse(call([item("Latte", "L")]), products, shop)
    assert "<tool_call>" not in result["text"]
    assert result["text"] == "Sure."


def test_prices_come_from_the_catalog_not_the_prose(context):
    """The model can say any number; the cart uses the one on the menu."""
    shop, products = context
    reply = "That's £99.00.\n" + call([item("Latte", "S")])
    assert orders.parse(reply, products, shop)["total"] == 3.30


def test_milk_and_extras_are_priced_in(context):
    shop, products = context
    result = orders.parse(
        call([item("Latte", "M", milk="oat", extras=["extra shot"])]),
        products, shop)
    # 3.85 base + 0.50 oat + 0.80 shot
    assert result["items"][0]["unit_price"] == 5.15


def test_the_house_milk_costs_nothing(context):
    shop, products = context
    result = orders.parse(call([item("Latte", "M", milk="whole")]), products, shop)
    assert result["items"][0]["unit_price"] == 3.85


def test_quantity_multiplies_the_line(context):
    shop, products = context
    result = orders.parse(call([item("Latte", "S", quantity=3)]), products, shop)
    assert result["items"][0]["line_total"] == 9.90


def test_a_product_outside_the_retrieved_menu_is_refused(context, catalog_names):
    """The check is against what the model was shown, not the whole catalog.

    The product is picked at run time rather than named: which of the 28 a
    query leaves out depends on the embedding, and hard-coding one makes the
    test a hostage to that.
    """
    shop, products = context
    absent = sorted(catalog_names - {p.name for p in products})[0]
    result = orders.parse(call([item(absent, None)]), products, shop)
    assert result["items"] == []
    assert "not on the menu" in result["rejections"][0]


def test_an_invented_product_is_refused(context):
    shop, products = context
    result = orders.parse(call([item("Unicorn Frappuccino")]), products, shop)
    assert result["items"] == []
    assert result["rejections"]


def test_a_size_the_product_does_not_offer_is_refused(context):
    """Espresso is S only. Ordering a large one is the model's error, not ours."""
    shop, products = context
    _, espresso_menu, _ = select(SHOP, "a strong espresso")
    result = orders.parse(call([item("Espresso", "L")]), espresso_menu, shop)
    assert result["items"] == []
    assert "does not come in size" in result["rejections"][0]


def test_an_out_of_stock_product_is_refused():
    """Retrieval deliberately leaves it listed; the cart is where it stops."""
    shop, products, _ = select(SHOP, "can I get an iced latte")
    assert any(p.name == "Iced Latte" and not p.in_stock for p in products)
    result = orders.parse(call([item("Iced Latte", "L")]), products, shop)
    assert result["items"] == []
    assert "out of stock" in result["rejections"][0]


def test_a_milk_the_shop_does_not_offer_is_refused(context):
    shop, products = context
    result = orders.parse(call([item("Latte", "M", milk="camel")]), products, shop)
    assert result["items"] == []


def test_an_unlisted_extra_is_refused(context):
    """Otherwise a free extra would be charged at zero and quietly served."""
    shop, products = context
    result = orders.parse(
        call([item("Latte", "M", extras=["gold leaf"])]), products, shop)
    assert result["items"] == []


@pytest.mark.parametrize("quantity", [0, -1, "two", None, 1.5])
def test_a_bad_quantity_is_refused(context, quantity):
    shop, products = context
    result = orders.parse(call([item("Latte", "M", quantity=quantity)]),
                          products, shop)
    assert result["items"] == []


def test_one_bad_item_does_not_lose_the_good_ones(context):
    shop, products = context
    result = orders.parse(
        call([item("Latte", "L"), item("Unicorn Frappuccino")]), products, shop)
    assert [i["name"] for i in result["items"]] == ["Latte"]
    assert len(result["rejections"]) == 1


def test_malformed_json_is_reported_not_swallowed(context):
    shop, products = context
    result = orders.parse("Sure.\n<tool_call>\n{not json\n</tool_call>",
                          products, shop)
    assert result["items"] == []
    assert "not valid JSON" in result["rejections"][0]


def test_a_plain_question_produces_no_order(context):
    """Restraint is a trained behaviour; this checks we do not invent one."""
    shop, products = context
    result = orders.parse("What size would you like?", products, shop)
    assert result["items"] == []
    assert result["rejections"] == []
    assert result["ordered"] is False
    assert result["text"] == "What size would you like?"
