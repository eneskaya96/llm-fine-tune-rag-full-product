"""Two trained voices over one base model, reading a retrieved menu.

Two questions in one demo. Can a single base model answer as both voices,
switching per request without a reload -- and does the menu it reads come from
the catalog rather than from a constant in this file? No cart or order
execution yet; those follow.

The file is self-contained apart from `rag/` and `shared/`, which the sync
workflow copies in beside it.
"""

import pathlib
import threading
import time

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.src import menu_format
from rag.src.retrieve import select

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

SHOP = "ember_and_oak"

# One prompt for both voices, and it says nothing about how to sound. Each
# adapter was trained under its own -- "Be warm and concise" against "Be blunt.
# No pleasantries." -- but serving them that way would prove nothing: a base
# model with no adapter follows those lines too, so the demo would be comparing
# two prompts rather than two adapters. Whatever difference shows up below is
# the adapter's.
SYSTEM = (pathlib.Path(__file__).parent / "shared" / "system_prompt.txt").read_text(
    encoding="utf-8")


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

# Seed the Chroma collection now rather than on the first request. Costs a
# couple of seconds at startup and takes them off whoever asks first.
select(SHOP, "warm up")


def answer(menu, brand, message):
    """One turn from whichever adapter is active, greedily decoded."""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM.format(brand=brand, menu=menu)},
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
def generate_both(menu, brand, message):
    """Answer as both voices, reporting what each swap cost."""
    replies, timings = [], []
    with _lock:
        for voice in ADAPTERS:
            start = time.perf_counter()
            model.set_adapter(voice)
            swap_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            replies.append(answer(menu, brand, message))
            timings.append(f"**{voice}** — swap {swap_ms:.1f} ms, "
                           f"generate {time.perf_counter() - start:.1f} s")
    return replies, timings


def compare(message, history_text):
    """Retrieve the menu, then answer from both voices.

    Retrieval runs out here rather than inside generate_both: embedding the
    query takes a few hundred milliseconds of CPU, and doing that inside a
    @spaces.GPU call would hold a GPU slot doing no GPU work.
    """
    history = [line for line in history_text.splitlines() if line.strip()]
    history.append(message)

    start = time.perf_counter()
    shop, products, reasons = select(SHOP, message, history)
    retrieval_ms = (time.perf_counter() - start) * 1000
    menu = menu_format.menu_block(products, shop)

    replies, timings = generate_both(menu, shop.brand, message)

    why = "\n".join(f"| {p.name} | {reasons[p.id]} |" for p in products)
    why = ("| product | why it is on the menu |\n|---|---|\n" + why +
           f"\n\n{len(products)} of 28 products, chosen in "
           f"{retrieval_ms:.0f} ms.")
    return (*replies, "  \n".join(timings), menu, why)


EXAMPLES = [
    "large flat white with oat pls",
    "can I get an iced latte",
    "something cold and refreshing",
    "do you have anything without milk",
    "two croissants and a medium latte",
]

with gr.Blocks(title="Coffee order — voice swap") as demo:
    gr.Markdown(
        "# One base model, two voices, one retrieved menu\n"
        "Both replies come from the same 4B model in the same process, reading "
        "the **same system prompt** — one that says nothing about how to sound. "
        "The only thing that differs is which LoRA adapter is active: "
        "`model.set_adapter(...)`, no reload.\n\n"
        "The menu is not hardcoded. Ember & Oak has **28 products** and the "
        "adapters were trained on menus of 4–9 drinks, so the RAG layer picks "
        "what this turn needs. Its working is shown below."
    )
    with gr.Row():
        message = gr.Textbox(label="Customer says", scale=3,
                             placeholder="large flat white with oat pls")
        run = gr.Button("Ask both", variant="primary", scale=1)
    with gr.Row():
        friendly_out = gr.Textbox(label="friendly", lines=8)
        blunt_out = gr.Textbox(label="blunt", lines=8)
    timing = gr.Markdown()

    with gr.Accordion("What the model was given to read", open=False):
        gr.Markdown(
            "Products named in the conversation stay on the menu even when "
            "this turn is about something else — otherwise a latte ordered "
            "three turns ago falls off and the model denies taking it. Add "
            "earlier turns here to see it hold."
        )
        history_box = gr.Textbox(label="Conversation so far (one turn per line)",
                                 lines=3, placeholder="a latte please")
        menu_out = gr.Textbox(label="<menu> block, retrieved", lines=12,
                              interactive=False)
        why_out = gr.Markdown()

    gr.Examples(EXAMPLES, inputs=message)

    outputs = [friendly_out, blunt_out, timing, menu_out, why_out]
    run.click(compare, [message, history_box], outputs)
    message.submit(compare, [message, history_box], outputs)

demo.queue(max_size=8).launch()
