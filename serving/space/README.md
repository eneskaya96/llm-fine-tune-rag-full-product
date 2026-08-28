---
title: Coffee Order Voice Swap
emoji: ☕
colorFrom: yellow
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: One base model, two fine-tuned voices, swapped per request
---

# Coffee order — voice swap

Both replies come from one 4B model in one process. The only difference between
them is which LoRA adapter is active: `model.set_adapter("blunt")`.

- Base: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- Adapters: [`coffee-order-friendly`](https://huggingface.co/eneskaya96/coffee-order-friendly),
  [`coffee-order-blunt`](https://huggingface.co/eneskaya96/coffee-order-blunt)
- Code: [github.com/eneskaya96/llm-fine-tune-rag-full-product](https://github.com/eneskaya96/llm-fine-tune-rag-full-product)

The ordering endpoint is an agent: the model calls a tool, is told what it did,
and answers that. An item the shop refuses — sold out, a size that product does
not come in — comes back to the model as its next turn, so it can offer
something else instead of the customer reading a refusal. `agent.py` holds the
loop, `tools.py` runs the calls, `orders.py` decides what a valid line is.

The menu is retrieved, not hardcoded. Ember & Oak has 28 products and the
adapters were trained on menus of 4-9 drinks, so `rag/` picks what each turn
needs and the UI shows why each line is on the list. The model was never
trained on a fixed catalog, so swapping the catalog serves a different shop.

## If it says the GPU quota is exceeded

ZeroGPU gives each visitor a few minutes of GPU a day, and it checks the
duration a function *declares* against what you have left before running
anything. So the declarations here are deliberately tight — 30 seconds for one
reply, 60 for the two-voice comparison — rather than padded. Signing in to
Hugging Face raises the allowance.

This file is pushed to the Space by `.github/workflows/sync-space.yml`. Edit it
here, not there — a Space-side edit is overwritten on the next sync.
