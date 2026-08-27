import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# validate_dataset is the reference menu parser; the round-trip test drives
# retrieval output back through the same code the training data was checked by.
sys.path[:0] = [str(ROOT), str(ROOT / "finetuning" / "scripts")]
