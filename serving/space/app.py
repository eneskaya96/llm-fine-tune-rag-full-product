"""Smoke test: does the adapter hot-swap work?

One question only -- can a single base model answer as both trained voices,
switching between them per request without a reload? No cart, no order
execution, no retrieval; those follow once this holds.

The file is self-contained so this folder uploads to a Space unchanged.
"""

import pathlib
import threading
import time

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Weights come from Qwen in bf16 -- ZeroGPU has VRAM to spare and skipping the
# 4-bit quantisation skips bitsandbytes with it. The tokenizer comes from the
# repo training used, so the chat template the adapters learned to read is the
# one they get served.
BASE_WEIGHTS = "Qwen/Qwen3-4B-Instruct-2507"
TOKENIZER = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"

ADAPTERS = {
    "friendly": "eneskaya96/coffee-order-friendly",
    "blunt": "eneskaya96/coffee-order-blunt",
}

# One prompt for both voices, and it says nothing about how to sound. Each
# adapter was trained under its own -- "Be warm and concise" against "Be blunt.
# No pleasantries." -- but serving them that way would prove nothing: a base
# model with no adapter follows those lines too, so the demo would be comparing
# two prompts rather than two adapters. Whatever difference shows up below is
# the adapter's.
SYSTEM = (pathlib.Path(__file__).parent / "shared" / "system_prompt.txt").read_text(
    encoding="utf-8")

BRAND = "Ember & Oak"

# Stands in for retrieval output. One item is deliberately out of stock, so the
# substitution behaviour is visible without having to hunt for it.
MENU = """Espresso | S | 2.60 | in stock
Americano | S,M,L | 2.90/3.40/3.90 | in stock
Flat White | S,M,L | 3.40/3.95/4.50 | in stock
Latte | S,M,L | 3.30/3.85/4.40 | in stock
Cold Brew | M,L | 4.10/4.70 | in stock
Iced Latte | M,L | 4.00/4.60 | OUT OF STOCK
Chai Latte | M,L | 4.20/4.80 | in stock
Matcha Latte | M,L | 4.50/5.10 | in stock
Croissant | - | 3.20 | in stock
Blueberry Muffin | - | 3.30 | in stock
Extras: extra shot (+0.80), vanilla syrup (+0.55), oat foam (+0.60)
Milk options: whole, skim, oat, almond (+0.50)"""


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
model = AutoModelForCausalLM.from_pretrained(
    BASE_WEIGHTS, dtype=torch.bfloat16, device_map="cuda"
)

# torch_device="cpu" is load-bearing on ZeroGPU. There is no GPU at startup --
# spaces intercepts torch calls and replays them once a @spaces.GPU function
# runs -- but peft reads adapter files through safetensors, which asks CUDA for
# memory directly and so never reaches the interception. Reading them onto CPU
# hands the weights back to torch, which does get intercepted.
LOAD = {"torch_device": "cpu"}

# from_pretrained attaches the first adapter and turns the plain model into a
# PeftModel; load_adapter stacks the rest alongside it. All of them sit in VRAM
# at once -- roughly 130 MB each against the base model's 8 GB.
(first_name, first_repo), *rest = ADAPTERS.items()
model = PeftModel.from_pretrained(model, first_repo, adapter_name=first_name, **LOAD)
for name, repo in rest:
    model.load_adapter(repo, adapter_name=name, **LOAD)
model.eval()

# set_adapter mutates one shared object, so two concurrent requests would fight
# over which voice is live. Serialising is the honest fix for a demo; production
# would use vLLM, which routes a per-request adapter without shared state.
_lock = threading.Lock()


def answer(menu, message):
    """One turn from whichever adapter is active, greedily decoded."""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM.format(brand=BRAND, menu=menu)},
            {"role": "user", "content": message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decoded the same way the evaluation harness does, so what a user reads
    # here is what the scores were computed on. <tool_call> is ordinary text in
    # Qwen3's vocabulary, not a special token, so it survives the strip.
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


@spaces.GPU(duration=120)
def compare(message, menu):
    """Answer as both voices, reporting what each swap cost."""
    replies, timings = [], []
    with _lock:
        for voice in ADAPTERS:
            start = time.perf_counter()
            model.set_adapter(voice)
            swap_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            replies.append(answer(menu, message))
            timings.append(f"**{voice}** — swap {swap_ms:.1f} ms, "
                           f"generate {time.perf_counter() - start:.1f} s")
    return (*replies, "  \n".join(timings))


EXAMPLES = [
    "large flat white with oat pls",
    "can I get an iced latte",
    "two croissants and a medium latte",
    "a latte please",
    "no milk in the americano, medium",
]

with gr.Blocks(title="Coffee order — voice swap") as demo:
    gr.Markdown(
        "# One base model, two voices\n"
        "Both replies below come from the same 4B model in the same process, "
        "reading the **same system prompt** — one that says nothing about how "
        "to sound. The only thing that differs is which LoRA adapter is active: "
        "`model.set_adapter(...)`, no reload. Any difference in tone is in the "
        "weights."
    )
    with gr.Row():
        message = gr.Textbox(label="Customer says", scale=3,
                             placeholder="large flat white with oat pls")
        run = gr.Button("Ask both", variant="primary", scale=1)
    with gr.Row():
        friendly_out = gr.Textbox(label="friendly", lines=8)
        blunt_out = gr.Textbox(label="blunt", lines=8)
    timing = gr.Markdown()
    menu = gr.Textbox(label="Menu (the RAG layer will produce this later)",
                      value=MENU, lines=13)
    gr.Examples(EXAMPLES, inputs=message)

    run.click(compare, [message, menu], [friendly_out, blunt_out, timing])
    message.submit(compare, [message, menu], [friendly_out, blunt_out, timing])

demo.queue(max_size=8).launch()
