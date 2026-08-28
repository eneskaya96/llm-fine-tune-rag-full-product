# Coffee ordering assistant — fine-tuned voices over a RAG catalog

A conversational ordering system for a coffee company. The shop's staff pick a
brand voice from an admin panel; the same base model serves all of them, with
LoRA adapters swapped at runtime. Product facts come from retrieval, never from
the model's weights.

**Status:** all three layers built. Two adapters serve a retrieved menu on a
Space, order calls are validated against that menu before they reach a cart,
and a React frontend puts a chat, the cart and an admin voice selector on top.

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
rag/          catalog → Chroma → <menu> block            → imported by serving
serving/      base model + adapter hot-swap + API        → HF Space (ZeroGPU)
frontend/     chat, cart, admin panel                    → Vercel
shared/       order_schema.json — the contract all three obey
```

Retrieval is why the catalog can be 28 products while the model only ever
reads eight: the adapters were trained on menus of 4-9 drinks and never saw a
longer one, so [`rag/`](rag/) chooses per turn — keeping anything named earlier
in the conversation, and leaving sold-out items listed as `OUT OF STOCK` so the
model can offer the alternative it was trained to offer.

`shared/order_schema.json` is the single source of truth for what an order
looks like, and `shared/tool_call.py` is how it is written into a reply and
read back out. The training data emits that shape, the evaluator checks against
it, the serving layer validates it, the frontend renders it.

**The model's output is not trusted.** The adapters are wrong about roughly one
order in twelve, so `serving/space/orders.py` checks every item against the
menu that was actually retrieved for that turn — the same rules the training
data is validated with — and drops what fails, with the reason shown in the
cart rather than swallowed. Prices are computed from the catalog, never read
out of the model's prose. This is what "code owns actually placing the order"
means in practice.

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

**Tone is not among the results.** A classifier separated the two adapters'
prose 81.1% of the time — but each voice was scored under its own training
prompt, and those prompts said *"Be warm and concise"* and *"Be blunt. No
pleasantries."* A base model follows those lines unaided, so the number
measured the prompts as much as the weights. Serving both adapters one prompt
that says nothing about tone, the difference largely goes.

So what fine-tuning is shown to have taught is the tool-call format and the
ordering discipline — neither of which was ever in the prompt. Teaching tone
needs a corpus where the voices differ *only* in what the model is trained to
say. [The write-up](finetuning/results/serving-check.md) has the details.

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
- **Tone is not proven at all.** See above: the voices differed in their system
  prompt as well as their weights, so the tone score could not be credited to
  fine-tuning. Fixing it means regenerating the corpus against one shared
  prompt and retraining both voices.
- **Training dialogues are template-generated.** The hard set is hand-written to
  compensate, but the corpus does not have the variety of real transcripts.
- **Comparative sizes are thinly taught.** "The big one" is 2.9% of the corpus
  and neither voice handles it reliably; blunt misses one of the two hard-set
  cases.
- **Retrieval is not measured.** Over a 28-product catalog, a benchmark would
  not mean much, so [`rag/`](rag/) carries regression tests rather than a
  score. Retrieval quality at a catalog size where it would matter is untested.

## Next

1. Retrain both voices against one shared system prompt, so tone can be
   measured for what the weights learned rather than what the prompt said
2. Take the catalog out of git — a barista cannot open a pull request to mark a
   product sold out. This becomes real once the admin panel writes products
3. Persist orders, which is the first thing that genuinely needs a store

See [`finetuning/README.md`](finetuning/README.md) for how the data and the
adapters were made.
