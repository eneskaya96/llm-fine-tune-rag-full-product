"""The loop, driven by a scripted model.

`generate` is injected into the graph, so these run the real nodes, the real
edges and the real tools with a list of replies standing in for the weights.
That is the only way to test a loop whose other half needs a GPU -- and it is
also the only way to write down "the model tried X, the shop said no, the model
tried Y" as a test rather than as a hope.
"""

import json

import agent
import pytest
import tools

from rag.src.retrieve import select

SHOP = "ember_and_oak"


@pytest.fixture
def session():
    shop, products, _ = select(SHOP, "a latte and something to eat")
    made = tools.Session(slug=SHOP, shop=shop)
    made.refresh(products)
    return made


def call(tool, **arguments):
    return ("<tool_call>"
            + json.dumps({"name": tool, "arguments": arguments})
            + "</tool_call>")


class Model:
    """Says the next scripted line, and keeps what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, menu, brand, turns):
        self.seen.append(list(turns))
        # Repeat the last line forever, so a test of the step cap does not have
        # to guess how many times the graph will come round.
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def turn(session, model, message="a latte please"):
    graph = agent.build(model)
    return agent.run_turn(graph, session, [{"role": "user", "content": message}])


def test_an_answer_with_no_tool_call_ends_the_turn(session):
    result = turn(session, Model("What size would you like?"))
    assert result["steps"] == 1
    assert result["text"] == "What size would you like?"
    assert result["items"] == []
    assert result["rejections"] == []


def test_the_tool_call_never_reaches_the_customer(session):
    result = turn(session, Model("Sure. " + call("confirm_order")))
    assert "<tool_call>" not in result["text"]
    assert result["text"] == "Sure."


def test_a_turn_that_is_only_a_tool_call_keeps_the_last_words(session):
    """Otherwise the customer is shown a blank reply because the model's final
    act was a call rather than a sentence."""
    model = Model(
        "Let me look that up. " + call("search_menu", query="something sweet"),
        call("confirm_order"),
    )
    result = turn(session, model)
    assert result["text"] == "Let me look that up."


def test_a_refused_item_comes_back_and_the_model_answers_it():
    """The difference between this and a single reply, as one test.

    Iced Latte is out of stock in the catalog. The single-shot layer put that
    on a card for the customer to read; here the model is told, and gets a turn
    in which to offer something else.
    """
    shop, products, _ = select(SHOP, "an iced latte")
    session = tools.Session(slug=SHOP, shop=shop)
    session.refresh(products)

    # Whichever drink retrieval offered alongside it. Named at run time because
    # which of the 28 a query returns depends on the embedding, and writing one
    # in makes the test a hostage to that.
    alternative = next(p.name for p in session.products
                       if p.kind == "drink" and p.in_stock)

    model = Model(
        "One iced latte. " + call("add_item", name="Iced Latte", size="M",
                                  milk=None, extras=[], quantity=1),
        f"Sorry, that one's off today -- a {alternative} instead? "
        + call("add_item", name=alternative, size="M", milk=None,
               extras=[], quantity=1),
        "Added. Anything else?",
    )
    result = turn(session, model, "an iced latte please")

    told = model.seen[1][-1]["content"]
    assert "out of stock" in told, "the model was not told why it was refused"
    assert [line["name"] for line in result["items"]] == [alternative]
    assert result["rejections"] == ["Iced Latte is out of stock"]
    assert result["text"] == "Added. Anything else?"


def test_a_terminal_tool_ends_the_turn_without_another_generation(session):
    """confirm_order answers the customer by itself. Going round again would
    buy a second goodbye at the price of a whole generation."""
    model = Model(
        "Adding it. " + call("add_item", name="Latte", size="M", quantity=1),
        "That's everything. " + call("confirm_order"),
        "and another thing",
    )
    result = turn(session, model)
    assert result["steps"] == 2
    assert result["ordered"] is True


def test_the_loop_stops_at_the_step_cap(session):
    """A model that keeps calling tools must not keep a GPU slot forever."""
    model = Model("Still looking. " + call("search_menu", query="coffee"))
    result = turn(session, model)
    assert result["steps"] == agent.MAX_STEPS


def test_malformed_json_is_reported_not_swallowed(session):
    result = turn(session, Model("Sure.\n<tool_call>\n{not json\n</tool_call>"))
    assert result["items"] == []
    assert "not valid JSON" in result["rejections"][0]


def test_the_trace_records_what_was_called_and_what_came_back(session):
    model = Model(
        "Looking. " + call("search_menu", query="flat white"),
        "Adding. " + call("add_item", name="Flat White", size="L",
                          milk="oat", extras=[], quantity=1),
        "Done. " + call("confirm_order"),
    )
    result = turn(session, model)
    assert [step["tool"] for step in result["trace"]] == [
        "search_menu", "add_item", "confirm_order"]
    assert result["total"] == 5.00      # 4.50 large flat white + 0.50 oat


def test_the_menu_the_agent_ended_with_is_reported_back(session):
    """The caller may not share memory with the graph: on ZeroGPU the turn runs
    behind @spaces.GPU and what comes back is a return value."""
    model = Model("Looking. " + call("search_menu", query="avocado toast"),
                  "Here you go.")
    result = turn(session, model)
    assert set(result["menu_ids"]) == {p.id for p in session.products}


def test_the_call_the_shipped_adapters_emit_still_works(session):
    """create_order is not advertised any more, but the weights in production
    were trained to emit it. Dropping it would break the live Space."""
    reply = "Sure. " + call("create_order", items=[
        {"name": "Latte", "size": "L", "milk": None, "extras": [], "quantity": 1}])
    result = turn(session, Model(reply))
    assert result["ordered"] is True
    assert result["total"] == 4.40
    assert result["steps"] == 1


def test_a_turn_reports_only_its_own_refusals(session):
    """The cart survives the turn; the reasons it refused something do not.

    Otherwise the screen would keep showing a customer the item they were told
    about three turns ago, every turn, forever.
    """
    graph = agent.build(Model(
        "One iced latte. " + call("add_item", name="Iced Latte", size="M",
                                  milk=None, extras=[], quantity=1),
        "Sorry, that one's off.",
    ))
    first = agent.run_turn(graph, session,
                           [{"role": "user", "content": "an iced latte"}])
    assert first["rejections"]

    graph = agent.build(Model("What else can I get you?"))
    second = agent.run_turn(graph, session,
                            [{"role": "user", "content": "nothing else"}])
    assert second["rejections"] == []
