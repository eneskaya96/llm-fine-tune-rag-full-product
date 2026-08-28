"""Two trained voices over one base model, reading a retrieved menu.

Two questions in one demo. Can a single base model answer as both voices,
switching per request without a reload -- and does the menu it reads come from
the catalog rather than from a constant in this file?

The customer-facing endpoint runs an agent: the model calls tools, sees what
they did, and answers that. agent.py holds the loop, tools.py runs the calls,
orders.py decides what a valid line is.

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

import agent
import tools
from rag.src import catalog, menu_format
from rag.src.retrieve import select
from shared.tools import tool_block

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
SHOP_INFO, _, _ = select(SHOP, "warm up")

# Which voice the shop is currently trading under. One value for everybody,
# because it is a shop-wide setting rather than a per-visitor one -- the admin
# panel changes what customers hear. It lives in memory, so a Space restart
# returns it to the default; giving it a store means giving the shop a store,
# which is the same decision as taking the catalog out of git.
_state = {"voice": next(iter(ADAPTERS))}


def answer(menu, brand, turns):
    """One reply from whichever adapter is active, greedily decoded.

    `turns` is the conversation so far, ending on the customer. Earlier
    assistant turns carry the text the customer saw, with the tool call
    stripped -- which is also how the training dialogues look before their
    final turn.
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "system",
          "content": SYSTEM.format(brand=brand, menu=menu, tools=tool_block())}]
        + list(turns),
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


# Built once, over `answer` rather than over the @spaces.GPU wrapper. A turn can
# generate up to MAX_STEPS times, and decorating the inner call would request a
# GPU slot -- and wait in the queue -- once per step instead of once per turn.
GRAPH = agent.build(answer)

# Every product the shop sells, to turn the ids a finished turn reports back
# into products. The agent may widen its own menu with search_menu, and what
# comes out of a @spaces.GPU call is a return value rather than a shared object.
ALL_PRODUCTS = {p.id: p for p in catalog.load(SHOP)[1]}

# One cart per conversation, keyed by whatever thread id the client sends.
# The transcript stays with the client -- it is what the client draws -- while
# the cart lives here, because a cart the client could edit would undo the
# checking that is the point of orders.py. Memory only: a Space restart empties
# it, the same limit the active voice has.
SESSIONS = {}
MAX_SESSIONS = 64


# ZeroGPU compares the *declared* duration against the caller's remaining quota
# before the function runs, and the platform pads what is declared: 120 here
# reached the scheduler as 180. An anonymous caller gets around 180 seconds a
# day, so every visitor was refused on their first message -- "180s requested
# vs. 174s left" -- without having spent any GPU time at all.
#
# Only the seconds actually spent are charged, so declaring a realistic worst
# case costs nothing and a generous one costs the whole demo. A smaller
# declaration also ranks higher in the queue. Two generations, so twice
# generate_one's.
@spaces.GPU(duration=60)
def generate_both(menu, brand, message):
    """Answer as both voices, reporting what each swap cost."""
    replies, timings = [], []
    with _lock:
        for voice in ADAPTERS:
            start = time.perf_counter()
            model.set_adapter(voice)
            swap_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            replies.append(answer(menu, brand,
                                  [{"role": "user", "content": message}]))
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



# ---------------------------------------------------------------------------
# The API the React frontend calls. gr.api registers a function as a REST
# endpoint without giving it a place in the demo UI, so the two-voice
# comparison above stays what it is and the product endpoints sit beside it.
#
# Every parameter of a gr.api function needs a type hint: gradio builds the
# endpoint's schema from them and raises at import time without them, which on
# a Space means the container dies on startup rather than failing a request.
# test_api_endpoints.py checks this without importing the module.
# ---------------------------------------------------------------------------


# Up to MAX_STEPS generations of at most 256 tokens each, in one allocation.
# See the note on generate_both for why the number is not padded.
@spaces.GPU(duration=60)
def run_agent(session, turns, voice):
    """A whole agent turn while the GPU is held.

    Takes and returns plain values rather than relying on `session` coming back
    mutated: what a @spaces.GPU call gives you is its return value, and whether
    the body shared memory with this process is a detail of the runtime that
    would only show up on the Space.
    """
    with _lock:
        model.set_adapter(voice)
        return agent.run_turn(GRAPH, session, turns)


def _session(thread, shop):
    """The cart this conversation has been building, or a new one.

    An empty thread is never stored. A client that sends "" gets single-turn
    behaviour, which is the honest reading of "I am not naming a cart" -- far
    better than every such client sharing one and adding to each other's order.
    """
    if not thread:
        return tools.Session(slug=SHOP, shop=shop)

    session = SESSIONS.get(thread)
    if session is None:
        # Oldest first, so the eviction is of whoever has been away longest.
        while len(SESSIONS) >= MAX_SESSIONS:
            SESSIONS.pop(next(iter(SESSIONS)))
        session = SESSIONS[thread] = tools.Session(slug=SHOP, shop=shop)
    return session


def chat(message: str, history: list, voice: str, thread: str) -> dict:
    """One customer turn: retrieve, then let the agent work.

    `voice` overrides the shop setting for this request only, which is what
    lets the admin screen show the same question answered two ways without
    changing what customers hear. Anything unrecognised, "" included, means
    the shop's current voice.

    `thread` names the cart. "" is not a name: it gets a cart of its own that
    is thrown away after the turn, so a client that does not track a
    conversation gets single-turn behaviour rather than a stranger's order.
    """
    # Plain str and list rather than `str | None`: gradio reads these
    # annotations to build the endpoint schema, and how a given version handles
    # a union is one more thing that can only be found out on the Space. The
    # frontend always sends all four, using "" for "whatever the shop is set
    # to", so nothing is lost by making them required.
    history = list(history or [])
    voice = voice if voice in ADAPTERS else _state["voice"]

    # Retrieval reads the whole conversation, not just this turn, so a product
    # ordered earlier stays on the menu. Run out here: embedding the query is
    # CPU work and doing it inside run_agent would hold a GPU slot idle.
    said = [m["content"] for m in history] + [message]
    shop, products, reasons = select(SHOP, message, said)

    session = _session(thread, shop)
    session.refresh(products)

    result = run_agent(session, history + [{"role": "user", "content": message}],
                       voice)

    session.cart = result["items"]
    session.confirmed = result["ordered"]
    session.products = [ALL_PRODUCTS[i] for i in result["menu_ids"]]

    result["voice"] = voice
    result["menu"] = menu_format.menu_block(session.products, shop)
    result["chosen"] = [{"name": p.name, "why": reasons[p.id]} for p in products]
    return result


def set_voice(voice: str) -> dict:
    """Point the shop at a different adapter. The admin panel's whole job."""
    if voice not in ADAPTERS:
        raise gr.Error(f"unknown voice {voice!r}")
    _state["voice"] = voice
    return get_state()


def get_state() -> dict:
    """What the frontend needs to render itself.

    `voices` comes from ADAPTERS rather than being listed in the frontend, so
    publishing a third adapter reaches the admin dropdown without a frontend
    change.
    """
    return {"voices": list(ADAPTERS), "active": _state["voice"],
            "brand": SHOP_INFO.brand, "shop": SHOP}


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
    gr.api(chat, api_name="chat")
    gr.api(set_voice, api_name="set_voice")
    gr.api(get_state, api_name="get_state")

    run.click(compare, [message, history_box], outputs)
    message.submit(compare, [message, history_box], outputs)

demo.queue(max_size=8).launch()
