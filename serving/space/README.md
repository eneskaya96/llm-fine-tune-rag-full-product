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

The menu is editable in the UI and currently hand-written. A retrieval layer
will produce it later; the model reads it either way, since it was never
trained on a fixed catalog.

This file is pushed to the Space by `.github/workflows/sync-space.yml`. Edit it
here, not there — a Space-side edit is overwritten on the next sync.
