"""Running one tool call against a cart.

The tools are where the agent's decisions meet the shop's rules, and the thing
that matters most is what a refusal does: it comes back as a sentence, in the
return value, because that sentence is the next thing the model reads. A tool
that swallowed the reason would leave the model believing the item went in.
"""

import pytest
import tools

from rag.src.retrieve import MAX_DRINKS, MAX_FOOD, select

SHOP = "ember_and_oak"


@pytest.fixture
def session():
    shop, products, _ = select(SHOP, "a latte and something to eat")
    made = tools.Session(slug=SHOP, shop=shop)
    made.refresh(products)
    return made


def run(session, tool, **arguments):
    # `tool` rather than `name`, which is add_item's own first argument.
    return tools.run({"name": tool, "arguments": arguments}, session)


def test_add_item_fills_the_cart_and_says_the_total(session):
    result = run(session, "add_item", name="Latte", size="L", milk=None,
                 extras=[], quantity=1)
    assert session.cart[0]["name"] == "Latte"
    assert session.total == 4.40
    assert "4.40" in result


def test_a_refusal_is_returned_to_the_model_not_only_recorded(session):
    """The whole point of the loop: the model has to be able to read this."""
    result = run(session, "add_item", name="Unicorn Frappuccino", size="M",
                 milk=None, extras=[], quantity=1)
    assert session.cart == []
    assert result.startswith("refused:")
    assert session.rejections == ["'Unicorn Frappuccino' is not on the menu "
                                  "that was retrieved"]


def test_remove_item_takes_it_back_out(session):
    run(session, "add_item", name="Latte", size="L", quantity=1)
    result = run(session, "remove_item", name="Latte")
    assert session.cart == []
    assert "removed Latte" in result


def test_removing_something_that_is_not_there_is_refused(session):
    assert run(session, "remove_item", name="Latte").startswith("refused:")


def test_confirming_an_empty_cart_is_refused(session):
    assert run(session, "confirm_order").startswith("refused:")
    assert session.confirmed is False


def test_confirm_places_what_is_in_the_cart(session):
    run(session, "add_item", name="Latte", size="M", quantity=2)
    result = run(session, "confirm_order")
    assert session.confirmed is True
    assert "7.70" in result


def test_search_widens_the_menu_and_names_what_it_added(session):
    before = {p.id for p in session.products}
    result = run(session, "search_menu", query="something sweet to eat")
    assert {p.id for p in session.products} >= before or session.products
    assert result.startswith(("added to the menu:", "nothing new"))


def test_search_never_grows_the_menu_past_what_the_model_was_trained_on(session):
    """The adapters saw menus of 4-9 drinks and 1-3 food. An agent that keeps
    searching must not hand them something longer than anything they know."""
    for query in ("iced drinks", "something sweet", "strong coffee", "tea",
                  "a sandwich", "hot chocolate"):
        run(session, "search_menu", query=query)
    kinds = [p.kind for p in session.products]
    assert kinds.count("drink") <= MAX_DRINKS
    assert kinds.count("food") <= MAX_FOOD


def test_a_product_in_the_cart_is_never_dropped_from_the_menu(session):
    """Otherwise check_item would start refusing the cart's own contents."""
    run(session, "add_item", name="Latte", size="M", quantity=1)
    for query in ("iced drinks", "something sweet", "strong coffee", "tea",
                  "a sandwich", "hot chocolate", "juice"):
        run(session, "search_menu", query=query)
    assert "Latte" in {p.name for p in session.products}


def test_an_unknown_tool_is_refused(session):
    assert run(session, "delete_shop").startswith("refused:")
    assert session.rejections


def test_a_missing_required_argument_is_refused(session):
    result = run(session, "add_item", size="L")
    assert "name, quantity missing" in result


def test_arguments_that_are_not_an_object_are_refused(session):
    result = tools.run({"name": "add_item", "arguments": "Latte"}, session)
    assert result.startswith("refused:")


def test_create_order_replaces_the_cart_rather_than_adding_to_it(session):
    """The trained call carries the whole order, and the model repeats it on
    the confirmation turn. Accumulating would double every line."""
    items = [{"name": "Latte", "size": "M", "milk": None, "extras": [],
              "quantity": 1}]
    run(session, "create_order", items=items)
    run(session, "create_order", items=items)
    assert len(session.cart) == 1
    assert session.confirmed is True


def test_create_order_keeps_the_good_items_when_one_is_bad(session):
    result = run(session, "create_order", items=[
        {"name": "Latte", "size": "L", "milk": None, "extras": [], "quantity": 1},
        {"name": "Unicorn Frappuccino", "size": "M", "milk": None, "extras": [],
         "quantity": 1},
    ])
    assert [line["name"] for line in session.cart] == ["Latte"]
    assert "refused" in result
    assert len(session.rejections) == 1


def test_create_order_with_no_items_is_refused(session):
    assert run(session, "create_order", items=[]).startswith("refused:")
