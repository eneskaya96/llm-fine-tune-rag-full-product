# Coffee ordering assistant — fine-tuned voices over a RAG catalog

A conversational ordering system for a coffee company. The shop's staff pick a
brand voice from an admin panel; the same base model serves all of them, with
LoRA adapters swapped at runtime. Product facts come from retrieval, never from
the model's weights.

**Status:** fine-tuning done and measured, two adapters published and running
live on a Space. RAG and frontend not built yet.

**Live demo:** [one base model, two voices, swapped per request](https://huggingface.co/spaces/eneskaya96/coffee-order-voice-swap)

## The architecture, and why it splits this way

| Layer | Owns | Does not own |
|---|---|---|
| **RAG** | What exists, what it costs, what is in stock | How to talk |
| **Fine-tuned model** | Brand voice, when to ask vs order, tool-call format | Any product fact |
| **Code** | Actually placing the order | Anything conversational |

The load-bearing idea: **fine-tuning does not teach knowledge.** A menu changes
weekly; retraining a model on it would be absurd, and the model would still
hallucinate prices between retrains. So the menu is injected into the prompt at
request time, and fine-tuning is spent on the things that genuinely are
stable — how the brand sounds, when to stop and ask, and emitting a tool call
the code can execute.

Every training example therefore invents its own brand and menu. No product
catalog is stable across the corpus, so the model cannot memorise one. Swap the
vector database and the same adapter serves a different company.

### Serving

Adapters are **never merged** into the base model. Merging folds the weights in
permanently, which turns each voice into a separate 8 GB model and kills the
runtime swap the admin panel depends on:

| | VRAM for 5 voices | Switching cost |
|---|---|---|
| Adapters, hot-swapped | 1 base (2.5 GB) + 5 × ~130 MB | milliseconds |
| Merged models | 5 full models (~12.5 GB) | reload, 30-60 s |

Production would run vLLM with `--enable-lora`. The live demo runs Hugging Face
Spaces on ZeroGPU with `peft`, where `model.set_adapter("blunt")` is the whole
of the swap. Same adapter files, different loader.

Measured on the served stack: **14–17 ms** per swap, and the hard-set scores
move by at most one example against the 4-bit numbers the adapters were trained
and measured on — see [`serving-check.md`](finetuning/results/serving-check.md).

## Repository

```
finetuning/   data generation, training, evaluation      → Colab (T4)
rag/          catalog → ChromaDB → <menu> block          → imported by serving
serving/      base model + adapter hot-swap + API        → HF Space (ZeroGPU)
frontend/     chat, cart, admin panel                    → Vercel
shared/       order_schema.json — the contract all three obey
```

`shared/order_schema.json` is the single source of truth for what an order
looks like. The training data emits it, the evaluator checks against it, the
serving layer executes it, the frontend renders it.

## Results so far

Measured on 37 hand-written dialogues that share no templates with the training
data — messy phrasing, vague requests, unavailable sizes, negation, ambiguous
product names, compound failures, dialogues up to eight turns, customers who
push back after being told no.

The fine-tuned models are scored **without** the tool schema in their prompt.
The baseline gets it, which is the comparison worth winning: can a fine-tuned
model beat a well-prompted one while using a shorter prompt?

| metric | base + schema | friendly | blunt |
|---|---|---|---|
| format_ok | 78.4% | 100.0% | 100.0% |
| restraint | 94.6% | 97.3% | 100.0% |
| grounded | 100.0% | 100.0% | 100.0% |
| valid_slots | 88.2% | 95.8% | 100.0% |
| exact_match | 62.2% | **91.9%** | **94.6%** |

Every metric is computed in code against that example's own menu. No LLM judge,
no human rating.

Tone is scored separately, since `exact_match` compares order items and the two
voices are meant to agree on those. A cross-validated classifier tells the two
adapters' prose apart **81.1% ± 7.7%** of the time from identical prompts (50%
would be indistinguishable), driven mostly by politeness markers — 0.46 per
turn in friendly against 0.03 in blunt.

Three runs, each fixing what the last one measured:

| | hard-set exact_match | what changed |
|---|---|---|
| run 001 | 81.1% | first working adapter |
| run 002 | 86.5% | +3 ask-instead-of-order categories; broke negation |
| run 003 | 91.9% / 94.6% | +negation, closing phrases; rebalanced |

Run 002 is the interesting one: fixing over-eagerness produced under-eagerness,
and `negation` fell below the base model before the rebalance recovered it.
[`finetuning/results/`](finetuning/results/) records each run including the
regressions.

## What is not proven

- **The hard set is 37 examples.** Enough to expose failure modes, too small for
  confidence intervals on any single category. Both adapters diverge by up to 40
  points per category while landing within one example of each other overall —
  most of that spread is sampling noise.
- **The tone number does not prove brand fit.** 81.1% says the two adapters
  produce different prose, not that either sounds like a company a person would
  recognise. With template-generated training data the classifier may be
  separating memorised phrasings rather than a learned style.
- **Training dialogues are template-generated.** The hard set is hand-written to
  compensate, but the corpus does not have the variety of real transcripts.
- **Comparative sizes are thinly taught.** "The big one" is 2.9% of the corpus
  and neither voice handles it reliably; blunt misses one of the two hard-set
  cases.
- **No RAG layer yet**, so retrieval quality is untested end to end. The
  `<menu>` block is currently hand-assembled in the training data.

## Next

1. `rag/` — Ember & Oak catalog, ChromaDB index, retrieval that emits the exact
   `<menu>` format the model was trained on
2. `serving/` — order execution and a cart on top of the Space that already
   swaps voices
3. `frontend/` — React on Vercel, admin panel wired to the adapter swap

See [`finetuning/README.md`](finetuning/README.md) for how the data and the
adapters were made.
