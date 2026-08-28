import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# On the Space, app.py and shared/ and rag/ all sit at the root together. In
# the repo they are three directories apart, so the paths are rebuilt here.
sys.path[:0] = [str(ROOT), str(ROOT / "serving" / "space")]
