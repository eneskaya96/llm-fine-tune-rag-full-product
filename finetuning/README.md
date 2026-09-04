# Fine-tuning

Two LoRA adapters over one base model, teaching a coffee shop's ordering
assistant a brand voice — not product knowledge, which belongs to the RAG
layer, and not which tools exist, which belongs to the prompt.

| | |
|---|---|
| Base | `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit` (Apache 2.0) |
| Method | QLoRA, rank 16, alpha 16, all 7 attention + MLP projections |
| Hardware | Colab free T4 (16 GB, fp16 — T4 has no bf16) |
| Training | 900 dialogues per voice, 3 epochs, lr 2e-4, effective batch 8, ~31 min |
| Adapters | [`coffee-order-friendly`](https://huggingface.co/eneskaya96/coffee-order-friendly), [`coffee-order-blunt`](https://huggingface.co/eneskaya96/coffee-order-blunt) |

## What is being taught

The corpus carries **no tool calls at all**. An instruct model already calls a
tool it is shown in the prompt, so teaching one is at best redundant and at
worst harmful: weights trained on a fixed list are a list you have to retrain
to change, and the shop adds a tool more often than it retrains a model.

So the split runs one layer deeper than "fine-tuning does not teach knowledge":

| | owned by |
|---|---|
| menu, prices, stock | RAG, at request time |
| which tools exist | the prompt, at request time |
| how the shop sounds, when it asks rather than assumes | **the adapter** |
| running the call, checking it, pricing it | code |

Adding a tool is an entry in `shared/tools.py`. Nothing is retrained.

## Result

Not yet measured. The adapters in production were trained on the earlier
corpus, which did teach `create_order`, and were scored on metrics that read
its arguments. Those numbers described a different question and are not carried
forward; the git history has them.

The metrics that replace them read the prose, which is the part an adapter
owns: `grounded` (names no product this menu lacks), `in_stock` (never offers
what the shop has run out of), `one_question` (asks one thing at a time),
`alternative` (when the ask cannot be met, names something real instead). Tone
is scored separately by `tone_eval.py` — a classifier that has to tell the two
voices apart from the words alone.

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
  validate_dataset.py        the rules, checked over the corpus
  evaluate.py                the same rules, over model output
  tone_eval.py               scores how it was said, not what was said
notebooks/
  train.ipynb                Colab: baseline -> train -> measure -> push
  serve_check.ipynb          re-scores the adapters as the Space serves them
```

## Regenerating the data

Everything is deterministic given `--seed`. The corpora are committed, but the
recipe is what matters:

```bash
python finetuning/scripts/generate_dataset.py --n 900 --seed 42
python finetuning/scripts/generate_dataset.py --n 900 --seed 42 \
    --templates finetuning/data/templates/blunt.yaml \
    --out finetuning/data/train_blunt.jsonl
python finetuning/scripts/build_eval_hard.py

for f in finetuning/data/*.jsonl; do
    python finetuning/scripts/validate_dataset.py "$f"
done
```

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

- **The metrics read prose, and prose can lie.** "We don't have cold brew" and
  "here is your cold brew" both name a product the customer already named, and
  the checker cannot tell them apart. Nothing downstream trusts prose — the
  order layer works from tool calls it validates itself — but the score is
  softer than the tool-call scores it replaces, and should be read that way.
- **37 examples is small.** Per-category cells hold 3-6 examples, so one wrong
  answer moves a category by 20-33 points. Read the overall figure as the
  result.
- **The voices differ in their system prompt, not only their weights.** That
  makes the tone score uninterpretable — a base model would follow "Be blunt.
  No pleasantries." on its own. The prompt belongs in `_shared.yaml` so every
  voice is generated against the same one; both adapters need retraining after
  that. Until then the corpora here match the published adapters rather than
  the fix.
- **Customer turns are template-generated**, so they lack the variety of real
  transcripts even after the hard set's hand-written cases.
- **Comparative sizing is undertaught.** 26 of 900 dialogues cover "the big
  one", identically in both voices, and neither handles it reliably. Raising
  that share needs more hard-set cases first, or a fix cannot be told from luck.
