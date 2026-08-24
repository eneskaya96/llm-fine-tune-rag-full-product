"""Emit notebooks/train.ipynb.

The notebook is generated rather than hand-written so its cells stay diffable
and reviewable as plain Python in this file.
"""

import json
import pathlib

REPO = "https://github.com/eneskaya96/llm-fine-tune-rag-full-product.git"
BRANCH = "claude/coffee-finetuned-order-system-pnendq"

CELLS = [
    ("md", """# Coffee ordering assistant — QLoRA fine-tune

Order of operations matters here: the base model is measured **before** any
adapter exists, so the final table has a real "before" column. Loss going down
is not evidence; beating the base model on held-out data is.

| | |
|---|---|
| Base | `unsloth/Qwen3-4B-Instruct-bnb-4bit` |
| Method | QLoRA, rank 16 |
| Hardware | Colab free T4 (16 GB, fp16 — T4 has no bf16) |
| Output | LoRA adapter pushed to the Hub, never merged |

Runtime → Change runtime type → **T4 GPU** before running."""),

    ("code", """!nvidia-smi -L
%pip install -q unsloth
%pip install -q --no-deps --upgrade "trl>=0.9" peft accelerate bitsandbytes"""),

    ("md", "## 1. Data and scoring code\n\nBoth come from the repo, so the "
           "notebook stays thin and the metrics match what CI checks."),

    ("code", f"""!git clone -q --branch {BRANCH} {REPO} repo

import sys
sys.path.append("/content/repo/scripts")

import evaluate as ev

train_records = ev.load("/content/repo/data/train.jsonl")
eval_records = ev.load("/content/repo/data/eval.jsonl")
print(f"train {{len(train_records)}}  eval {{len(eval_records)}}")"""),

    ("md", "## 2. Load the base model in 4-bit"),

    ("code", """import torch
from unsloth import FastLanguageModel

MAX_SEQ = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B-Instruct-bnb-4bit",
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
)"""),

    ("md", """## 3. Baseline — the base model with a strong prompt

The system prompt in every example already spells out the rules, so this is a
fair fight: prompt-only versus fine-tuned."""),

    ("code", """def generate(records, batch_size=8, max_new_tokens=160):
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                ev.prompt_messages(r), tokenize=False, add_generation_prompt=True)
            for r in chunk
        ]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        cut = batch.input_ids.shape[1]
        outputs += tokenizer.batch_decode(generated[:, cut:], skip_special_tokens=True)
        print(f"  {min(start + batch_size, len(records))}/{len(records)}", end="\\r")
    return outputs


baseline_outputs = generate(eval_records)
baseline = ev.summarise(eval_records, baseline_outputs)
print(ev.format_table(baseline, "BASE (prompt only)"))"""),

    ("md", """## 4. Attach the LoRA adapter

Rank 16 across attention and MLP projections. Higher rank overfits 800
examples; lower struggles with multi-turn tool-call discipline."""),

    ("code", """model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
model.print_trainable_parameters()"""),

    ("md", "## 5. Train\n\nLoss is computed on assistant turns only — the model "
           "should not be learning to predict the menu we hand it."),

    ("code", """from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

def to_text(record):
    return {"text": tokenizer.apply_chat_template(record["messages"], tokenize=False)}

train_ds = Dataset.from_list([to_text(r) for r in train_records])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ,
    args=SFTConfig(
        output_dir="outputs",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        logging_steps=10,
        optim="adamw_8bit",
        fp16=True,          # T4 has no bf16
        seed=42,
        report_to="none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\\n",
    response_part="<|im_start|>assistant\\n",
)

stats = trainer.train()"""),

    ("md", "## 6. Measure again on the same held-out set"),

    ("code", """tuned_outputs = generate(eval_records)
tuned = ev.summarise(eval_records, tuned_outputs)
print(ev.format_table(tuned, "FINE-TUNED"))"""),

    ("md", "## 7. Before / after\n\nThis table goes in the README."),

    ("code", """print(f"{'metric':14}{'base':>8}{'tuned':>8}{'delta':>9}")
for metric, before in baseline["overall"].items():
    after = tuned["overall"][metric]
    print(f"{metric:14}{before:>8.1%}{after:>8.1%}{after - before:>+9.1%}")

print("\\nExample generations:")
for record, base_out, tuned_out in list(zip(eval_records, baseline_outputs, tuned_outputs))[:3]:
    print(f"\\n[{record['meta']['category']}] {record['messages'][-2]['content']}")
    print(f"  BASE : {base_out.strip()[:200]}")
    print(f"  TUNED: {tuned_out.strip()[:200]}")"""),

    ("md", """## 8. Publish the adapter

Not merged: merging folds the weights into the base model, which kills the
hot-swap the serving layer depends on. ~200 MB of adapter, base stays shared."""),

    ("code", """from huggingface_hub import notebook_login
notebook_login()"""),

    ("code", """REPO_ID = "your-username/coffee-order-friendly"   # <- change this

model.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)
print("pushed:", REPO_ID)"""),
]


def cell(kind, source):
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def main():
    notebook = {
        "cells": [cell(kind, source) for kind, source in CELLS],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "train.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(CELLS)} cells -> {out}")


if __name__ == "__main__":
    main()
