import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# evaluate.py imports validate_dataset as a sibling and shared/ from the root,
# which is how both run as scripts.
sys.path[:0] = [str(ROOT), str(ROOT / "finetuning" / "scripts")]
