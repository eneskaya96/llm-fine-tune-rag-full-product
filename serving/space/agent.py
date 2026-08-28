"""The loop that makes this an agent rather than a single reply.

Before this, the model spoke once, code ran whatever tool call the reply
carried, and the customer read the result. The model never saw what its own
call did -- so an item the shop refused became a card on the screen saying no,
rather than the assistant noticing and offering something else.

Here the tool result goes back in as a turn and the model gets to answer it.
That is the whole difference; everything below is bookkeeping around it.

Written as a LangGraph StateGraph. The loop is small enough to be a while loop;
what the graph buys is that the stopping rules are edges you can read off the
page rather than break statements scattered through a body, and somewhere to
hang a checkpointer when the cart moves off the browser.

`generate` is injected rather than imported. app.py passes the real one, which
needs a GPU and eight gigabytes of weights; the tests pass a scripted one and
run the same graph.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

import tools
from rag.src import menu_format
from shared import tools as schema
from shared.tool_call import extract_tool_calls, strip_tool_calls

# How many times the model may speak in one turn. Every step is another
# generation, and on ZeroGPU a whole day's allowance is measured in seconds.
# Three covers search -> add -> confirm, the longest useful sequence this shop
# has.
MAX_STEPS = 3

# Tools that end the turn. The customer was already answered by the message
# that carried them, so going round again would buy a second confirmation at
# the price of another generation.
TERMINAL = frozenset({"confirm_order", "create_order"})


class State(TypedDict):
    turns: list
    session: object
    steps: int
    text: str
    done: bool


def build(generate):
    """Compile the graph. `generate(menu, brand, turns) -> str`."""

    def speak(state):
        session = state["session"]
        menu = menu_format.menu_block(session.products, session.shop)
        reply = generate(menu, session.shop.brand, state["turns"])
        spoken = strip_tool_calls(reply)
        return {
            "steps": state["steps"] + 1,
            "turns": state["turns"] + [{"role": "assistant", "content": reply}],
            # An assistant turn can be nothing but a tool call. Keeping the last
            # text that had words in it means the customer is never shown a
            # blank reply because the model's final act was a call.
            "text": spoken or state["text"],
        }

    def act(state):
        session = state["session"]
        calls, malformed = extract_tool_calls(state["turns"][-1]["content"])
        if malformed:
            session.rejections.append(f"{malformed} tool call(s) were not valid JSON")

        turns, done = list(state["turns"]), False
        for call in calls:
            name = call.get("name")
            result = tools.run(call, session)
            session.trace.append({"tool": name,
                                  "arguments": call.get("arguments"),
                                  "result": result})
            turns.append(schema.result_message(name, result))
            done = done or name in TERMINAL

        return {"turns": turns, "done": done}

    def after_speak(state):
        # `malformed` counts too. A reply whose call is broken JSON parses to no
        # calls at all, and routing that straight to END would drop the turn
        # without anyone recording that the model tried and failed to act.
        calls, malformed = extract_tool_calls(state["turns"][-1]["content"])
        return "act" if calls or malformed else END

    def after_act(state):
        return END if state["done"] or state["steps"] >= MAX_STEPS else "speak"

    graph = StateGraph(State)
    graph.add_node("speak", speak)
    graph.add_node("act", act)
    graph.set_entry_point("speak")
    graph.add_conditional_edges("speak", after_speak, {"act": "act", END: END})
    graph.add_conditional_edges("act", after_act, {"speak": "speak", END: END})
    return graph.compile()


def run_turn(graph, session, turns):
    """One customer turn, from their message to what they are told back.

    `session` rides in the state and is mutated by the tools rather than
    threaded through as a value. The cart is what the conversation is about,
    and copying it at every node would mean the tools' return values were no
    longer the record of what happened.
    """
    # Both are what happened *this* turn. Left to accumulate, the screen would
    # keep showing a customer the item they were refused three turns ago.
    session.trace = []
    session.rejections = []
    final = graph.invoke({"turns": list(turns), "session": session,
                          "steps": 0, "text": "", "done": False})
    return {
        "text": final["text"],
        "items": session.cart,
        "total": session.total,
        "rejections": session.rejections,
        "ordered": session.confirmed,
        "steps": final["steps"],
        "trace": session.trace,
        "turns": final["turns"],
        # search_menu can widen the menu mid-turn. Reported as ids because the
        # caller may not share memory with the graph -- on ZeroGPU the whole
        # turn runs behind @spaces.GPU, and what comes back is a return value,
        # not a mutated object.
        "menu_ids": [p.id for p in session.products],
    }
