"""Compile data/eval_hard.yaml into the same JSONL shape as the other sets.

The hard set is written by hand in YAML because it is read and edited by hand;
this turns it into records the scorer already understands.

    python finetuning/scripts/build_eval_hard.py
"""

import json
import pathlib

import yaml

SYSTEM = """You are the ordering assistant for {brand}.

Rules:
- Be warm and concise. Two sentences at most before acting.
- Only ever reference items present in <menu>. Never invent products, prices, or sizes.
- If a requested item is unavailable, say so plainly and offer the closest listed alternative.
- Ask at most one clarifying question at a time, and only for details you genuinely need.

<menu>
{menu}
</menu>"""


def build(example):
    messages = [{"role": "system", "content": SYSTEM.format(
        brand=example["brand"], menu=example["menu"].rstrip())}]

    for turn in example["turns"]:
        if "user" in turn:
            messages.append({"role": "user", "content": turn["user"]})
            continue
        # YAML folds multi-line assistant text onto one line with stray spacing.
        messages.append({"role": "assistant",
                         "content": " ".join(turn["assistant"].split())})

    return {
        "messages": messages,
        "meta": {
            "category": example["category"],
            "brand": example["brand"],
            "voice": "friendly",
            "split": "hard",
        },
    }


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    examples = yaml.safe_load((root / "data" / "eval_hard.yaml").read_text(encoding="utf-8"))
    out = root / "data" / "eval_hard.jsonl"

    with out.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(build(example), ensure_ascii=False) + "\n")

    counts = {}
    for example in examples:
        counts[example["category"]] = counts.get(example["category"], 0) + 1

    print(f"wrote {len(examples)} records -> {out}")
    for category, count in sorted(counts.items()):
        print(f"  {category:22} {count}")


if __name__ == "__main__":
    main()
