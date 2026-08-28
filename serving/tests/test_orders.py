"""What a cart line must satisfy before it is one.

The adapters are wrong about one order in twelve. Each test here is one of the
ways that goes wrong, and the assertion is always the same shape: the bad item
does not become a line, and the reason is a sentence rather than a silence.

Nothing here knows about tool calls or conversations. check_item is the rule;
test_tools.py covers running it from a call, test_agent.py the loop around it.
"""

import orders
import pytest

from rag.src.retrieve import select

SHOP = "ember_and_oak"


@pytest.fixture(scope="module")
def context():
    """A real retrieved menu, since that is what check_item checks against."""
    shop, products, _ = select(SHOP, "a latte and something to eat")
    return shop, products


@pytest.fixture(scope="module")
def catalog_names():
    """Every product the shop sells, retrieved or not."""
    from rag.src import catalog
    _, products, _ = catalog.load(SHOP)
    return {p.name for p in products}


def item(name, size="M", milk=None, extras=None, quantity=1):
    return {"name": name, "size": size, "milk": milk,
            "extras": extras or [], "quantity": quantity}


def test_a_valid_item_becomes_a_line(context):
    shop, products = context
    line, reason = orders.check_item(item("Latte", "L"), products, shop)
    assert reason is None
    assert line["name"] == "Latte"
    assert line["line_total"] == 4.40      # the L price from the catalog


def test_prices_come_from_the_catalog_not_the_prose(context):
    """The model can say any number; the cart uses the one on the menu."""
    shop, products = context
    line, _ = orders.check_item(item("Latte", "S"), products, shop)
    assert line["unit_price"] == 3.30


def test_milk_and_extras_are_priced_in(context):
    shop, products = context
    line, _ = orders.check_item(
        item("Latte", "M", milk="oat", extras=["extra shot"]), products, shop)
    # 3.85 base + 0.50 oat + 0.80 shot
    assert line["unit_price"] == 5.15


def test_the_house_milk_costs_nothing(context):
    shop, products = context
    line, _ = orders.check_item(item("Latte", "M", milk="whole"), products, shop)
    assert line["unit_price"] == 3.85


def test_quantity_multiplies_the_line(context):
    shop, products = context
    line, _ = orders.check_item(item("Latte", "S", quantity=3), products, shop)
    assert line["line_total"] == 9.90


def test_a_product_outside_the_retrieved_menu_is_refused(context, catalog_names):
    """The check is against what the model was shown, not the whole catalog.

    The product is picked at run time rather than named: which of the 28 a
    query leaves out depends on the embedding, and hard-coding one makes the
    test a hostage to that.
    """
    shop, products = context
    absent = sorted(catalog_names - {p.name for p in products})[0]
    line, reason = orders.check_item(item(absent, None), products, shop)
    assert line is None
    assert "not on the menu" in reason


def test_an_invented_product_is_refused(context):
    shop, products = context
    line, reason = orders.check_item(item("Unicorn Frappuccino"), products, shop)
    assert line is None and reason


def test_a_size_the_product_does_not_offer_is_refused(context):
    """Espresso is S only. Ordering a large one is the model's error, not ours."""
    shop, _ = context
    _, espresso_menu, _ = select(SHOP, "a strong espresso")
    line, reason = orders.check_item(item("Espresso", "L"), espresso_menu, shop)
    assert line is None
    assert "does not come in size" in reason


def test_an_out_of_stock_product_is_refused():
    """Retrieval deliberately leaves it listed; the cart is where it stops."""
    shop, products, _ = select(SHOP, "can I get an iced latte")
    assert any(p.name == "Iced Latte" and not p.in_stock for p in products)
    line, reason = orders.check_item(item("Iced Latte", "L"), products, shop)
    assert line is None
    assert "out of stock" in reason


def test_a_milk_the_shop_does_not_offer_is_refused(context):
    shop, products = context
    line, _ = orders.check_item(item("Latte", "M", milk="camel"), products, shop)
    assert line is None


def test_an_unlisted_extra_is_refused(context):
    """Otherwise a free extra would be charged at zero and quietly served."""
    shop, products = context
    line, _ = orders.check_item(
        item("Latte", "M", extras=["gold leaf"]), products, shop)
    assert line is None


@pytest.mark.parametrize("quantity", [0, -1, "two", None, 1.5])
def test_a_bad_quantity_is_refused(context, quantity):
    shop, products = context
    line, _ = orders.check_item(
        item("Latte", "M", quantity=quantity), products, shop)
    assert line is None


def test_total_adds_the_lines_up(context):
    shop, products = context
    lines = [orders.check_item(item("Latte", size), products, shop)[0]
             for size in ("S", "L")]
    assert orders.total(lines) == 7.70
