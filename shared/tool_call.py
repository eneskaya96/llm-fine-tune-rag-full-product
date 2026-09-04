"""How a tool call is written into an assistant turn, and read back out.

`tools.py` says which calls exist; this says how one is wrapped. Both halves
live in shared/ because the serving layer writes the list into the prompt and
parses what comes back, and the frontend renders the result. A second copy in
either place would be a second definition of the format.

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
