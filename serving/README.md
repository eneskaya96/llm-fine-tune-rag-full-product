# Serving

Hugging Face Space, Gradio SDK, ZeroGPU. Holds the base model once and swaps
LoRA adapters per request.

```
space/app.py       demo UI, and the REST endpoints the frontend calls
space/agent.py     the LangGraph loop: speak, act, speak again
space/tools.py     runs one tool call against a cart
space/orders.py    decides what a valid cart line is, and what it costs
tests/             all of the above, with a scripted model in place of weights
```

`.github/workflows/sync-space.yml` pushes `space/` to the Space, along with
`shared/` and `rag/`, which it imports at runtime.

## The API

`gr.api` registers a function as a REST endpoint without giving it a place in
the demo UI, so the two-voice comparison stays what it is and the product
endpoints sit beside it.

| endpoint | in | out |
|---|---|---|
| `chat` | message, history, voice, thread | reply, items, total, menu, chosen, rejections, steps, trace |
| `set_voice` | voice | the new shop state |
| `get_state` | — | voices, active, brand |

`get_state` reads its voice list from `ADAPTERS`, so publishing a third adapter
reaches the admin dropdown without a frontend change.

`thread` names a cart. The transcript stays with the client, because that is
what the client draws; the cart stays here, because a cart the client could
edit would undo the checking below. Carts are held in memory, capped, and lost
on a restart — the same limit the active voice has.

The active voice is a module-level variable: one value for everybody, because
it is a shop-wide setting rather than a per-visitor one. It returns to the
default when the Space restarts. Giving either a store means giving the shop a
store, which is the same decision as taking the catalog out of git.

## The agent

The model calls a tool, is told what the tool did, and answers that. Before
this it spoke once and read nothing back, so an item the shop refused became a
card on the screen rather than something the assistant noticed.

```
speak ──has a tool call?──► act ──terminal, or 3rd turn?──► end
  ▲                          │
  └──────────────────────────┘
```

| tool | what it does |
|---|---|
| `search_menu` | retrieval, run because the model asked rather than before it spoke |
| `add_item` | one line, checked by `orders.check_item` before it is one |
| `remove_item` | takes it back out |
| `confirm_order` | places the cart |

`create_order` is the fifth: one call carrying a whole order, which is what the
adapters in production were trained to emit back when the corpus taught tool
calls at all. It is executed but not advertised, so the shipped weights keep
working; it goes when they are replaced.

The corpus teaches no tool calls now. An instruct model calls a tool it is
shown in the prompt, and weights trained on a fixed list are a list you retrain
to change — the shop adds a tool far more often than it retrains a model. The
four tools above reach the model as prompt, from `shared/tools.py`, on every
turn. Adding a fifth is an entry in that file.

It is a `StateGraph` rather than a `while` loop because the stopping rules are
then edges you can read off the page, and because the conversation will want a
checkpointer when the cart outlives a browser tab.

Every step is another generation. `MAX_STEPS = 3` covers search → add → confirm
and is also a quota ceiling: on ZeroGPU a visitor's whole day is measured in
seconds.

## Why the model's output is not trusted

Nothing teaches the model what a valid order looks like any more, and nothing
scores it on one: the corpus is prose, and so are the metrics. That is not a
gap — it moves the whole burden here, which is where it always belonged.
`orders.py` checks every item against the menu that was actually retrieved for
that turn — product listed, size offered for that product, milk on the shop's
list, nothing sold out — and refuses what fails with a reason instead of
silently correcting it. Prices are computed from the catalog, never read out of
the model's prose.

This is the layer the architecture calls "code owns actually placing the
order", and the loop is what makes the refusal useful: the reason goes back to
the model as its next turn, so the assistant can offer an alternative rather
than the customer reading that the shop said no.

The tests cover it without the model. `test_orders.py`: invented products,
sizes a product does not offer, sold-out items, unlisted extras, bad
quantities. `test_tools.py`: what each tool does to a cart, and that a menu
widened by searching never grows past what the adapters were trained on.
`test_agent.py`: the loop itself, driven by a scripted model — a refusal
reaching the model, a terminal tool ending the turn, the step cap holding, and
the call the shipped adapters emit still working.

## Why Gradio and not FastAPI

ZeroGPU — the free GPU tier — requires the Gradio SDK. A Docker Space would
allow any stack but drops to CPU on the free tier, where a 4B model is
unusable. Gradio runs on FastAPI underneath and exposes REST endpoints
automatically, so the React frontend calls it over HTTP like any API.

## Why adapters are not merged

Merging folds LoRA weights into the base permanently, turning each voice into a
separate 8 GB model. Kept separate, one base serves all voices and switching is
`model.set_adapter("blunt")` — milliseconds, no reload. The admin panel's voice
selector maps directly onto that call.

Production would use vLLM `--enable-lora`; ZeroGPU allocates a GPU per function
call rather than holding one, so a persistent vLLM server does not fit. Same
adapter files either way.
