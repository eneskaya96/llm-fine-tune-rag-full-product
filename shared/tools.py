"""What the agent is allowed to do, described once.

`tool_call.py` says how a call is written; this says which calls exist and what
they carry. The list reaches the model as prompt, never as weights: the corpus
teaches no tool calls at all, so adding a tool here is the whole change, with
no retraining behind it. That is the point of keeping it in one file -- a
second list anywhere is a second definition that drifts.

`create_order` is not here. It is the single call carrying a finished order
that the adapters in production were trained on, back when the corpus taught
tool calls; tools.run still accepts it so the live Space keeps working. It goes
when those adapters are replaced. The compatibility path is in the executor,
not in the contract.
"""

TOOLS = {
    "search_menu": {
        "summary": "look up products by what the customer described",
        "arguments": {
            "query": "what the customer asked for, in their own words",
        },
        "required": ["query"],
    },
    "add_item": {
        "summary": "put one product in the cart",
        "arguments": {
            "name": "the product name exactly as the menu writes it",
            "size": "S, M or L -- null for a product with no sizes",
            "milk": "a milk from the shop's list -- null for the house milk",
            "extras": "a list of extras, [] for none",
            "quantity": "how many, at least 1",
        },
        "required": ["name", "quantity"],
    },
    "remove_item": {
        "summary": "take a product back out of the cart",
        "arguments": {
            "name": "the product name to remove",
        },
        "required": ["name"],
    },
    "confirm_order": {
        "summary": "place the cart as an order once the customer has agreed",
        "arguments": {},
        "required": [],
    },
}


def tool_block():
    """The tools as the system prompt states them.

    One line each. The models being served are 4B and the menu already takes a
    large share of the prompt, so a full JSON schema here would cost context
    without telling the model anything the line does not.
    """
    lines = []
    for name, tool in TOOLS.items():
        signature = ", ".join(tool["arguments"])
        lines.append(f"- {name}({signature}) — {tool['summary']}")
        for argument, meaning in tool["arguments"].items():
            optional = "" if argument in tool["required"] else " (optional)"
            lines.append(f"    {argument}{optional}: {meaning}")
    return "\n".join(lines)


def result_message(name, result):
    """A tool's result on its way back to the model.

    Written as a user turn rather than a `tool` role: the chat template's
    handling of that role is one more thing that could only be discovered on the
    Space, and neither adapter has seen either shape. A user turn works under
    every template, so the risk here is the model's, not the runtime's -- and
    when the corpus is regenerated it will teach this exact line.
    """
    return {"role": "user", "content": f"[tool_result {name}] {result}"}
