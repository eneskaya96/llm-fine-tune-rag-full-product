"""The API contract, checked without starting anything.

Two startup failures got to the Space before these existed, and both were the
same shape: something that only matters where the pieces meet, which is
production. The first was a file the sync workflow did not copy. The second was
gradio refusing a gr.api function with no type hints -- a rule discovered in a
throwaway script and then applied only there.

app.py cannot be imported here; it pulls in torch and the spaces runtime and
downloads eight gigabytes of weights. So this reads it as source instead: the
gr.api registrations are found in the syntax tree, and the frontend's calls are
found in client.ts. Neither check needs either side to run.
"""

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
APP = ROOT / "serving" / "space" / "app.py"
CLIENT = ROOT / "frontend" / "src" / "api" / "client.ts"

# `call<SomeType>("endpoint", [...])` in the frontend's only backend file.
FRONTEND_CALL = re.compile(r'call<[^>]+>\(\s*"([a-z_]+)"\s*,\s*\[')


def tree():
    return ast.parse(APP.read_text(encoding="utf-8"))


def registrations():
    """Every gr.api(fn, api_name=...) in app.py, as {api_name: function name}."""
    found = {}
    for node in ast.walk(tree()):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if not (isinstance(target, ast.Attribute) and target.attr == "api"):
            continue
        function = node.args[0]
        name = next((k.value.value for k in node.keywords if k.arg == "api_name"), None)
        found[name] = function.id
    return found


def definitions():
    return {n.name: n for n in ast.walk(tree()) if isinstance(n, ast.FunctionDef)}


def test_the_three_endpoints_are_registered():
    """A guard on the parser: if this stops finding them, the rest passes
    vacuously and the checks below would be worthless."""
    assert set(registrations()) == {"chat", "set_voice", "get_state"}


def test_every_endpoint_parameter_is_annotated():
    """The failure that took the Space down the second time.

    gradio builds an endpoint's schema from the annotations and raises at
    import time without them, so on a Space the container dies on startup
    rather than returning an error for one request.
    """
    defined = definitions()
    for api_name, function in registrations().items():
        node = defined[function]
        arguments = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
        missing = [a.arg for a in arguments if a.annotation is None]
        assert not missing, f"{api_name}: {', '.join(missing)} has no type hint"
        assert node.returns is not None, f"{api_name}: no return annotation"


def test_the_frontend_calls_the_endpoints_that_exist():
    """client.ts and app.py agree on names, or every request 404s."""
    called = set(FRONTEND_CALL.findall(CLIENT.read_text(encoding="utf-8")))
    assert called == set(registrations())


def test_the_frontend_passes_the_right_number_of_arguments():
    """Gradio matches the posted array to the signature by position, so adding
    a parameter on one side and not the other breaks every call at runtime."""
    source = CLIENT.read_text(encoding="utf-8")
    defined = definitions()

    for match in FRONTEND_CALL.finditer(source):
        endpoint = match.group(1)
        sent = count_arguments(source, match.end() - 1)
        node = defined[registrations()[endpoint]]
        accepted = len(node.args.args + node.args.posonlyargs)
        assert sent == accepted, (
            f"{endpoint}: client.ts sends {sent}, app.py takes {accepted}")


def count_arguments(source, opening):
    """Top-level commas in the array literal starting at `opening`, plus one.

    Counted rather than parsed because the entries nest -- `voice ?? null`,
    object literals -- and splitting on every comma would miscount them.
    """
    depth, items, seen_content = 0, 1, False
    for index in range(opening, len(source)):
        char = source[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return items if seen_content else 0
        elif char == "," and depth == 1:
            items += 1
        elif depth == 1 and not char.isspace():
            seen_content = True
    raise AssertionError("unterminated argument array in client.ts")


# ZeroGPU pads the declared duration and then compares it to what the caller
# has left, before running anything. An anonymous visitor gets roughly 180
# seconds a day, so a generous declaration does not slow the demo down -- it
# refuses the first message of every visit.
LONGEST_GPU_CALL = 60


def gpu_durations():
    """Every @spaces.GPU(duration=...) in app.py, as {function: seconds}."""
    found = {}
    for node in ast.walk(tree()):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if getattr(decorator.func, "attr", None) != "GPU":
                continue
            duration = next((k.value.value for k in decorator.keywords
                             if k.arg == "duration"), None)
            found[node.name] = duration
    return found


def test_no_gpu_function_declares_more_time_than_the_quota_allows():
    """The third failure that only showed up in production.

    Nothing here can measure how long a generation takes -- there is no GPU and
    no model. What it can do is stop the number from drifting back up, which is
    the part that took the demo down.
    """
    durations = gpu_durations()
    assert set(durations) == {"generate_both", "generate_one"}, durations
    for function, seconds in durations.items():
        assert seconds is not None, f"{function}: no duration, defaults to 60"
        assert seconds <= LONGEST_GPU_CALL, (
            f"{function}: declares {seconds}s, over the {LONGEST_GPU_CALL}s ceiling")
