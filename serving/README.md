# Serving

Hugging Face Space, Gradio SDK, ZeroGPU. Holds the base model once and swaps
LoRA adapters per request.

```
space/app.py       demo UI, and the REST endpoints the frontend calls
space/orders.py    validates a create_order call before it becomes a cart
tests/             orders.py without the model
```

`.github/workflows/sync-space.yml` pushes `space/` to the Space, along with
`shared/` and `rag/`, which it imports at runtime.

## The API

`gr.api` registers a function as a REST endpoint without giving it a place in
the demo UI, so the two-voice comparison stays what it is and the product
endpoints sit beside it.

| endpoint | in | out |
|---|---|---|
| `chat` | message, history, voice | reply, items, total, menu, chosen, rejections |
| `set_voice` | voice | the new shop state |
| `get_state` | — | voices, active, brand |

`get_state` reads its voice list from `ADAPTERS`, so publishing a third adapter
reaches the admin dropdown without a frontend change.

The active voice is a module-level variable: one value for everybody, because
it is a shop-wide setting rather than a per-visitor one. It returns to the
default when the Space restarts. Giving it a store means giving the shop a
store, which is the same decision as taking the catalog out of git.

## Why the model's output is not trusted

The adapters score 100% on format and about 92% on exact match — another way of
saying they are wrong about one order in twelve. `orders.py` checks every item
against the menu that was actually retrieved for that turn, using the same
rules `validate_dataset.py` applies to the training data, and drops what fails
with a reason instead of silently correcting it. Prices are computed from the
catalog, never read out of the model's prose.

This is the layer the architecture calls "code owns actually placing the
order". `tests/test_orders.py` covers it without the model: invented products,
sizes a product does not offer, sold-out items, unlisted extras, bad
quantities, malformed JSON, and a plain question that should produce no order
at all.

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
