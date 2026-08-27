# Serving

Not built yet.

Hugging Face Space, Gradio SDK, ZeroGPU. Holds the base model once and swaps
LoRA adapters per request.

## Planned

```
app.py             Gradio interface + REST routes the frontend calls
adapters.py        load base 4-bit, load both adapters, set_adapter()
orders.py          execute the create_order tool call
requirements.txt
```

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
