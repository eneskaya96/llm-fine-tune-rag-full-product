"""How a create_order call is written into an assistant turn, and read back out.

`order_schema.json` says what an order contains; this says how it is wrapped.
Both halves of the contract live here because three layers depend on them: the
training data emits this shape, the evaluator scores against it, and the
serving layer parses it before touching a cart. A second copy in any of those
places would be a second definition of the format.

`<tool_call>` is ordinary text in Qwen3's vocabulary, not a special token, so
it survives `skip_special_tokens=True` and this is a plain text parse.
"""

import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text):
    """Return (calls, malformed_count) for one assistant message."""
    calls, malformed = [], 0
    for raw in TOOL_CALL_RE.findall(text):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            malformed += 1
    # A bare <tool_call> with no JSON body is a malformed call too.
    opens = text.count("<tool_call>")
    if opens > len(calls) + malformed:
        malformed += opens - len(calls) - malformed
    return calls, malformed


def strip_tool_calls(text):
    """The same message with the call blocks removed -- what a customer reads."""
    return TOOL_CALL_RE.sub("", text).replace("<tool_call>", "").strip()
