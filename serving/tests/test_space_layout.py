"""Does the Space get every file it imports?

This exists because it did not, once. `shared/tool_call.py` was added and
imported by orders.py, the sync workflow still listed the two shared files it
knew about by name, and the Space died on startup with ModuleNotFoundError.
Nothing in the other tests could have caught it: conftest.py puts the repo root
on sys.path, which makes the repo's nested layout work and hides the flat one
the container actually runs.

So this test builds the Space's layout from the workflow's own rsync lines --
not from a list repeated here, which would be the same drift again -- and
imports into it from a fresh interpreter.
"""

import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "sync-space.yml"

# `rsync <flags> <src>/ <dest>/`, after shell line continuations are joined.
RSYNC = re.compile(r"^\s*rsync\s+(?P<flags>.*?)\s+(?P<src>\S+)\s+(?P<dest>\S+)\s*$")
EXCLUDE = re.compile(r"--exclude\s+'([^']+)'")


def copies():
    """Every (source, destination) the workflow rsyncs, and what it skips."""
    joined = WORKFLOW.read_text(encoding="utf-8").replace("\\\n", " ")
    found = []
    for line in joined.splitlines():
        match = RSYNC.match(line)
        if match:
            found.append((match["src"].rstrip("/"),
                          match["dest"].rstrip("/"),
                          EXCLUDE.findall(match["flags"])))
    return found


def build(destination):
    """Reproduce the Space's file tree on disk."""
    for source, target, excludes in copies():
        # The workflow's destinations are all under the Space clone, called
        # `space`; here that root is the temporary directory.
        relative = target[len("space"):].lstrip("/")
        shutil.copytree(
            ROOT / source,
            destination / relative if relative else destination,
            ignore=shutil.ignore_patterns(*excludes, ".git", "__pycache__"),
            dirs_exist_ok=True,
        )


def test_the_workflow_still_names_three_copies():
    """A guard on the parser, not on the workflow: if the rsync lines stop
    matching, every other assertion here would pass vacuously."""
    assert len(copies()) == 3


def test_every_import_app_makes_resolves_in_the_space_layout(tmp_path):
    """The failure that took the Space down, as a test.

    Run in a subprocess with the fake Space as the working directory, because
    this process has already imported these modules through the repo layout and
    would not notice a missing file.
    """
    build(tmp_path)

    check = """
import orders
from shared.tool_call import extract_tool_calls, strip_tool_calls
from rag.src import menu_format
from rag.src.retrieve import select

shop, products, _ = select("ember_and_oak", "a latte please")
assert products, "retrieval returned nothing"
call = ('<tool_call>{"name": "create_order", "arguments": {"items": ['
        '{"name": "Latte", "size": "L", "milk": null, "extras": [], '
        '"quantity": 1}]}}</tool_call>')
result = orders.parse("Sure. " + call, products, shop)
assert result["items"], result["rejections"]
print("ok")
"""
    done = subprocess.run([sys.executable, "-c", check], cwd=tmp_path,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_app_py_is_at_the_space_root(tmp_path):
    """A Gradio Space runs app.py from the root; the repo keeps it a directory
    down, which is the whole reason the sync is a copy rather than a mirror."""
    build(tmp_path)
    assert (tmp_path / "app.py").is_file()
    assert (tmp_path / "requirements.txt").is_file()


def test_the_tests_do_not_ship(tmp_path):
    """rag/tests has no business in the container, and pytest is not installed
    there to run it."""
    build(tmp_path)
    assert not (tmp_path / "rag" / "tests").exists()
