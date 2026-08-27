# Fine-tuning

Two LoRA adapters over one base model, teaching a coffee shop's ordering
assistant a brand voice and a tool-call format — not product knowledge, which
belongs to the RAG layer.

| | |
|---|---|
| Base | `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit` (Apache 2.0) |
| Method | QLoRA, rank 16, alpha 16, all 7 attention + MLP projections |
| Hardware | Colab free T4 (16 GB, fp16 — T4 has no bf16) |
| Training | 900 dialogues per voice, 3 epochs, lr 2e-4, effective batch 8, ~31 min |
| Adapters | [`coffee-order-friendly`](https://huggingface.co/eneskaya96/coffee-order-friendly), [`coffee-order-blunt`](https://huggingface.co/eneskaya96/coffee-order-blunt) |

## Result

Hard set: 37 hand-written dialogues, no shared templates with training data.
The fine-tuned models are scored **without** the tool schema in their prompt;
the baseline gets it.

| metric | base + schema | friendly | blunt |
|---|---|---|---|
| format_ok | 78.4% | 100.0% | 100.0% |
| restraint | 94.6% | 97.3% | 100.0% |
| grounded | 100.0% | 100.0% | 100.0% |
| valid_slots | 88.2% | 95.8% | 100.0% |
| exact_match | 62.2% | **91.9%** | **94.6%** |

Across three runs: 81.1% → 86.5% → 91.9%. Each run's write-up in
[`results/`](results/) records what changed, what it fixed, and what it broke.

## Layout

```
data/
  templates/_shared.yaml     product vocabulary shared by every voice
  templates/friendly.yaml    what the friendly voice says
  templates/blunt.yaml       what the blunt voice says
  eval_hard.yaml             37 hand-written dialogues (edit this, not the jsonl)
  *.jsonl                    generated corpora
scripts/
  generate_dataset.py        templates -> training corpus
  build_eval_hard.py         eval_hard.yaml -> eval_hard.jsonl
  validate_dataset.py        every record checked against its own menu
  evaluate.py                scores model output; the notebook imports this
  tone_eval.py               scores prose rather than orders
notebooks/
  train.ipynb                Colab: baseline -> train -> measure -> push
  serve_check.ipynb          re-scores the adapters as the Space serves them
results/                     one file per run
```

## Regenerating the data

Everything is deterministic given `--seed`. The corpora are committed, but the
recipe is what matters:

```bash
python finetuning/scripts/generate_dataset.py --n 900 --seed 42
python finetuning/scripts/generate_dataset.py --n 900 --seed 42 \
    --templates finetuning/data/templates/blunt.yaml \
    --out finetuning/data/train_blunt.jsonl
python finetuning/scripts/generate_dataset.py --n 120 --seed 1337 \
    --exclude finetuning/data/train_friendly.jsonl \
    --out finetuning/data/eval.jsonl
python finetuning/scripts/build_eval_hard.py

for f in finetuning/data/*.jsonl; do
    python finetuning/scripts/validate_dataset.py "$f"
done
```

`--exclude` seeds the dedup table from an existing file so no eval dialogue
appears in training. Without it, 7 of 120 leaked.

## Training

Open `notebooks/train.ipynb` in Colab (Runtime → T4 GPU), set `VOICE` to
`friendly` or `blunt`, Run all. Restart the session between voices — training a
second adapter in the same session writes over the first.

The notebook measures the base model **before** attaching an adapter, so the
final table has a real before column. Loss going down is not evidence.

## Why the data looks the way it does

**Every example invents its own brand and menu.** No product catalog is stable
across the corpus, so the model cannot memorise one — it has to read the
`<menu>` block. That block is where retrieval output will be injected.

**Only assistant turns are scored during training.** `train_on_responses_only`
masks the system prompt, the menu, and the customer's words, so the model learns
what to say rather than what the menu contains.

**Voices differ only in wording.** Same categories, same shares, same rules
about when to order and when to ask. That is what makes the pair a usable test
of whether tone transferred rather than behaviour.

## Known limitations

- **The generated eval set is close to saturated** (98.3%) because it shares
  templates with training. The hard set is the honest number.
- **37 examples is small.** Per-category cells hold 3-6 examples, so one wrong
  answer moves a category by 20-33 points. Both adapters diverge by up to 40
  points per category while landing within one example of each other overall —
  most of that spread is noise. Read the overall figure as the result.
- **Tone is not scored yet.** `tone_eval.py` exists but has not been run against
  real generations; the tables above say nothing about whether either adapter
  sounds like its brand.
- **Customer turns are template-generated**, so they lack the variety of real
  transcripts even after the hard set's hand-written cases.
