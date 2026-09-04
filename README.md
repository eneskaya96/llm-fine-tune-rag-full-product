# Coffee ordering assistant — fine-tuned voices over a RAG catalog

A conversational ordering system for a coffee company. The shop's staff pick a
brand voice from an admin panel; the same base model serves all of them, with
LoRA adapters swapped at runtime. Product facts come from retrieval, never from
the model's weights.

**Status:** all three layers built, and the ordering side is an agent. Two
adapters serve a retrieved menu on a Space; the model calls tools, is told what
each one did, and answers that; every item is checked against the retrieved
menu before it becomes a cart line; a React frontend puts the chat, the cart,
the agent's steps and an admin voice selector on top.

**Live demo:** [one base model, two voices, swapped per request](https://huggingface.co/spaces/eneskaya96/coffee-order-voice-swap)

## The architecture, and why it splits this way

| Layer | Owns | Does not own |
|---|---|---|
| **RAG** | What exists, what it costs, what is in stock | How to talk |
| **Prompt** | Which tools exist, and what each one takes | How to sound |
| **Fine-tuned model** | Brand voice, when to ask instead of assuming | Any product fact, any tool name |
| **Code** | Running the tools, and refusing what does not check out | Anything conversational |

The load-bearing idea: **fine-tuning does not teach knowledge.** A menu changes
weekly; retraining a model on it would be absurd, and the model would still
hallucinate prices between retrains. So the menu is injected at request time.

The same argument runs one layer further, and it is the one people skip: **it
does not teach tools either.** An instruct model already calls a tool it is
shown in the prompt. Bake the list into weights and every new tool becomes a
retraining job — and a shop adds a tool far more often than it retrains a
model. So the tool list is injected at request time too, from
`shared/tools.py`, and the corpus contains no tool calls at all.

What is left for fine-tuning is what a prompt genuinely cannot carry: how this
brand sounds, and when to stop and ask.

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
shared/       the contracts all three obey: order shape, tool list, prompt
```

Retrieval is why the catalog can be 28 products while the model only ever
reads eight: the adapters were trained on menus of 4-9 drinks and never saw a
longer one, so [`rag/`](rag/) chooses per turn — keeping anything named earlier
in the conversation, and leaving sold-out items listed as `OUT OF STOCK` so the
model can offer the alternative it was trained to offer.

`shared/order_schema.json` is the single source of truth for what an order
looks like, `shared/tool_call.py` is how a call is written into a reply and
read back out, and `shared/tools.py` lists the calls that exist. The serving
layer writes that list into the prompt and validates what comes back; the
frontend renders the result.

**The corpus teaches none of it.** An instruct model already calls a tool it is
shown, so a tool list baked into weights is a list you have to retrain to
change — and the shop adds a tool far more often than it retrains a model. What
the adapters own is the part a prompt cannot carry: how the shop sounds, and
when it asks instead of assuming. Adding a tool is an entry in
`shared/tools.py`.

**The model's output is not trusted.** Nothing trains it on what a valid order
looks like and nothing scores it on one, so `serving/space/orders.py` checks
every item against the menu that was actually retrieved for that turn —
product listed, size offered, milk on the list, nothing sold out — and drops
what fails, with the reason shown in the cart rather than swallowed. Prices are
computed from the catalog, never read out of the model's prose. This is what
"code owns actually placing the order" means in practice.

## Results so far

**None that still stand.** Three runs scored the adapters on the arguments of
the tool call they emitted — `format_ok`, `valid_slots`, `exact_match` — and
the best of them reached 91.9% / 94.6% on the hard set. Then the corpus stopped
teaching tool calls, because teaching them is what makes a new tool a
retraining job. Those numbers describe a model that was asked a different
question, so they are kept in [`finetuning/results/`](finetuning/results/) as
history and quoted nowhere as a result.

What replaces them reads the reply as prose, on the same 37 hand-written
dialogues — messy phrasing, vague requests, unavailable sizes, negation,
ambiguous product names, compound failures, dialogues up to eight turns,
customers who push back after being told no:

| metric | passes when |
|---|---|
| `grounded` | it names no product that example's menu lacks |
| `in_stock` | it never offers something the shop has run out of |
| `one_question` | it asks one thing at a time |
| `alternative` | when the ask cannot be met, it names something real instead |

Every metric is computed in code against that example's own menu. No LLM judge,
no human rating. The retrained adapters have not been measured yet.

**Tone is not among the results.** A classifier separated the two adapters'
prose 81.1% of the time — but each voice was scored under its own training
prompt, and those prompts said *"Be warm and concise"* and *"Be blunt. No
pleasantries."* A base model follows those lines unaided, so the number
measured the prompts as much as the weights. Serving both adapters one prompt
that says nothing about tone, the difference largely goes.

Teaching tone needs a corpus where the voices differ *only* in what the model
is trained to say. [The write-up](finetuning/results/serving-check.md) has the
details.

The three runs behind the retired numbers, each fixing what the last one
measured:

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

1. Retrain both voices on the voice-only corpus and measure them. Nothing in
   the shipped adapters matches what the corpus now teaches: they learned
   `create_order` and were scored on its arguments. Until they are replaced,
   `serving/space/tools.py` still executes `create_order` so the live Space
   keeps working, and the four prose metrics have no numbers behind them. The
   retrain also folds in the older item — one shared system prompt, so tone
   measures what the weights learned rather than what the prompt said
2. Take the catalog out of git — a barista cannot open a pull request to mark a
   product sold out. This becomes real once the admin panel writes products
3. Persist carts and orders. They are in memory today and a Space restart
   empties them, which is the first thing that genuinely needs a store
4. Voice. The agent is the half that transfers; a spoken one needs streaming
   generation and a GPU that answers in under a second, which ZeroGPU is not

See [`finetuning/README.md`](finetuning/README.md) for how the data and the
adapters were made.
