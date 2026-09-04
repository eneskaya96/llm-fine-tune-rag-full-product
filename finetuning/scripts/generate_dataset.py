"""Generate the supervised fine-tuning set for the coffee ordering assistant.

All wording -- system prompt, product vocabulary, dialogue phrasing -- lives in
a templates file (see finetuning/data/templates/friendly.yaml). This module
only supplies the structure: how a menu is assembled, what each behaviour
category looks like as a sequence of turns, and how records are sampled and
deduplicated.

That split is what makes a second brand voice cheap: copy the YAML, rewrite the
phrasing, run with --templates pointing at it. No code changes.

Every example gets a freshly invented brand, menu, price ladder and stock state,
so no product catalog is stable across the corpus and the model must read the
<menu> block instead of memorising products. That block is where retrieval
output will be injected once the RAG layer lands.

Usage
-----
    python finetuning/scripts/generate_dataset.py --n 900 --seed 42
"""

import argparse
import hashlib
import json
import pathlib
import random

import yaml

SIZE_WORD = {"S": "small", "M": "medium", "L": "large"}


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

def with_article(noun):
    """'espresso' -> 'an espresso'. Templates that hardcode 'a {x}' otherwise
    produce 'a americano' for vowel-initial products."""
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def describe(name, size=None, milk=None, extras=None, quantity=1):
    """Natural-language rendering of one order line."""
    parts = []
    if quantity > 1:
        parts.append(str(quantity))
    if size:
        parts.append(SIZE_WORD[size])
    if milk:
        parts.append(milk)
    parts.append(name.lower() + ("s" if quantity > 1 else ""))
    text = " ".join(parts)
    if extras:
        text += " with " + " and ".join(extras)
    return text


def fill(rng, templates, **slots):
    """Pick a template and fill it. Every string slot also gets a capitalised
    twin ('d' -> 'D') so templates can start a sentence with any placeholder."""
    for key, value in list(slots.items()):
        if isinstance(value, str) and value:
            slots[key.capitalize()] = value[0].upper() + value[1:]
    return rng.choice(templates).format(**slots)


# --------------------------------------------------------------------------
# Menu construction
# --------------------------------------------------------------------------

def make_brand(rng, vocab):
    head = rng.choice(vocab["head"])
    roll = rng.random()
    if roll < 0.45:
        return f"{head} {rng.choice(vocab['tail'])} {rng.choice(vocab['suffix'])}"
    if roll < 0.75:
        return f"{head} & {rng.choice(vocab['tail'])}"
    return f"{head} {rng.choice(vocab['suffix'])}"


def price_ladder(rng, floor, sizes):
    """Prices rise with size, jittered per brand so no price is memorable."""
    prices, current = [], floor + rng.uniform(0.0, 0.9)
    for _ in sizes:
        prices.append(round(current, 2))
        current += rng.uniform(0.35, 0.75)
    return prices


def make_menu(rng, tpl):
    """Assemble one brand's menu and return (menu_text, menu_dict)."""
    groups = tpl["drinks"]
    drinks = (rng.sample(groups["espresso_group"], rng.randint(2, 4))
              + rng.sample(groups["brew_group"], rng.randint(1, 2))
              + rng.sample(groups["other_group"], rng.randint(1, 3)))
    rng.shuffle(drinks)

    food = rng.sample(tpl["food"], rng.randint(1, 3))
    milks = ["whole"] + rng.sample([m for m in tpl["milks"] if m != "whole"],
                                   rng.randint(1, 3))
    extras = rng.sample(tpl["extras"], rng.randint(2, 4))

    # One drink is out of stock roughly a third of the time.
    oos = rng.randrange(len(drinks)) if rng.random() < 0.35 else None

    lines, drink_rows = [], []
    for i, entry in enumerate(drinks):
        prices = price_ladder(rng, entry["floor"], entry["sizes"])
        stock = "OUT OF STOCK" if i == oos else "in stock"
        lines.append(f"{entry['name']} | {','.join(entry['sizes'])} | "
                     f"{'/'.join(f'{p:.2f}' for p in prices)} | {stock}")
        drink_rows.append({**entry, "prices": prices, "in_stock": i != oos})

    food_rows = []
    for entry in food:
        price = round(entry["floor"] + rng.uniform(-0.3, 0.6), 2)
        lines.append(f"{entry['name']} | - | {price:.2f} | in stock")
        food_rows.append({"name": entry["name"], "price": price})

    lines.append("Extras: " + ", ".join(
        f"{e} (+{rng.uniform(0.5, 1.0):.2f})" for e in extras))
    lines.append(f"Milk options: {', '.join(milks)} (+{rng.uniform(0.35, 0.65):.2f})")

    return "\n".join(lines), {
        "drinks": drink_rows, "food": food_rows,
        "milks": milks, "extras": extras,
        "neighbours": tpl["family_neighbours"],
        "off_menu": tpl["off_menu"],
    }


def available(menu):
    return [d for d in menu["drinks"] if d["in_stock"]]


def price_of(drink, size):
    return drink["prices"][drink["sizes"].index(size)]


def substitutes_for(family, menu, rng, limit=1):
    """In-stock drinks a customer would accept instead of `family`, walking the
    neighbour preference list so an iced drink is answered with another cold
    drink rather than whatever happens to be on the menu."""
    pool = available(menu)
    if not pool:
        return []
    for candidate in menu["neighbours"].get(family, []):
        matches = [d for d in pool if d["family"] == candidate]
        if matches:
            return rng.sample(matches, min(limit, len(matches)))
    return rng.sample(pool, min(limit, len(pool)))


# --------------------------------------------------------------------------
# Turns
# --------------------------------------------------------------------------

def say(text):
    """One assistant turn. Speech, and only speech.

    No tool call is ever written into this corpus. Which tools exist is read off
    the <tools> block the serving layer puts in the prompt, and a tool the
    weights have never seen is exactly what has to keep working the day the shop
    adds one. What the corpus teaches is the part a prompt cannot carry: how
    this brand sounds, and when it asks instead of assuming.
    """
    return ("assistant", text)


# --------------------------------------------------------------------------
# Dialogue builders -- one per behaviour category.
# Each returns a list of (role, content) turns, or None if this menu can't
# support the category (e.g. nothing is out of stock).
# --------------------------------------------------------------------------

def simple_order(rng, menu, cat):
    drink = rng.choice(available(menu))

    # Some customers name the size by comparison rather than by name.
    comparative = len(drink["sizes"]) > 1 and rng.random() < 0.15
    size = drink["sizes"][-1] if comparative else rng.choice(drink["sizes"])
    text = describe(drink["name"], size)
    slots = {"d": text, "ad": with_article(text)}

    if comparative:
        bare = drink["name"].lower()
        ask = fill(rng, cat["user_extreme"], d=bare, ad=with_article(bare))
    else:
        ask = fill(rng, cat["user"], **slots)

    return [
        ("user", ask),
        say(fill(rng, cat["assistant"], **slots)),
    ]


def negation(rng, menu, cat):
    """Customer rules something out. That is a complete request, not a question."""
    drink = rng.choice(available(menu))
    size = rng.choice(drink["sizes"])
    text = describe(drink["name"], size)
    slots = {"d": text, "ad": with_article(text)}

    modes = ["extra"] * bool(menu["extras"]) + ["food"] * bool(menu["food"])
    alternatives = [m for m in menu["milks"] if m != "whole"]
    modes += ["milk"] * bool(alternatives)
    if not modes:
        return None

    mode = rng.choice(modes)
    if mode == "extra":
        # The ruled-out extra must simply not come back in the reply.
        excluded = rng.choice(menu["extras"])
        user_key, reply_key = "user", "assistant"
        slots["no"] = excluded
    elif mode == "food":
        user_key, reply_key = "user_nofood", "assistant_nofood"
    else:
        # Turning a milk down still names one: the house milk, said out loud.
        slots["no_milk"] = rng.choice(alternatives)
        slots["milk"] = "whole"
        user_key, reply_key = "user_milk", "assistant_milk"

    return [
        ("user", fill(rng, cat[user_key], **slots)),
        say(fill(rng, cat[reply_key], **slots)),
    ]


def clarify_missing_size(rng, menu, cat):
    pool = [d for d in available(menu) if len(d["sizes"]) > 1]
    if not pool:
        return None
    drink = rng.choice(pool)
    size = rng.choice(drink["sizes"])
    bare = drink["name"].lower()
    full = describe(drink["name"], size)
    words = ", ".join(SIZE_WORD[s] for s in drink["sizes"])
    return [
        ("user", fill(rng, cat["user"], d=bare, ad=with_article(bare))),
        say(fill(rng, cat["question"], sizes=words)),
        ("user", fill(rng, cat["answer"], s=SIZE_WORD[size])),
        say(fill(rng, cat["close"], d=full, ad=with_article(full))),
    ]


def multi_item(rng, menu, cat):
    pool = available(menu)
    drink = rng.choice(pool)
    size = rng.choice(drink["sizes"])
    quantity = rng.choice([1, 1, 2, 2, 3])

    spoken = [describe(drink["name"], size, quantity=quantity)]

    if menu["food"] and rng.random() < 0.6:
        snack = rng.choice(menu["food"])
        spoken.append(with_article(snack["name"].lower()))
    else:
        others = [d for d in pool if d["name"] != drink["name"]] or pool
        second = rng.choice(others)
        spoken.append(describe(second["name"], rng.choice(second["sizes"])))

    joined = " and ".join(spoken)
    return [
        ("user", fill(rng, cat["user"], j=joined)),
        say(fill(rng, cat["assistant"], j=joined)),
    ]


def out_of_stock(rng, menu, cat):
    missing = [d for d in menu["drinks"] if not d["in_stock"]]
    if not missing:
        return None
    gone = missing[0]
    picks = substitutes_for(gone["family"], menu, rng)
    if not picks:
        return None
    alt, alt_size = picks[0], None
    alt_size = rng.choice(alt["sizes"])
    wanted = describe(gone["name"], rng.choice(gone["sizes"]))
    offer = describe(alt["name"], alt_size)
    return [
        ("user", fill(rng, cat["user"], d=wanted, ad=with_article(wanted))),
        say(fill(rng, cat["apology"], gone=gone["name"].lower(),
                 an=with_article(offer), alt=offer)),
        ("user", rng.choice(cat["accept"])),
        say(fill(rng, cat["close"], alt=offer, an=with_article(offer))),
    ]


def off_menu_request(rng, menu, cat):
    """The item does not exist on any menu and must not be invented."""
    wanted = rng.choice(menu["off_menu"])
    picks = substitutes_for(wanted["family"], menu, rng, limit=2)
    if not picks:
        return None
    names = " or ".join(d["name"].lower() for d in picks)
    key = "assistant_one" if len(picks) == 1 else "assistant_many"
    return [
        ("user", fill(rng, cat["user"], w=wanted["name"],
                      aw=with_article(wanted["name"]))),
        say(fill(rng, cat[key], n=names)),
    ]


def order_correction(rng, menu, cat):
    pool = available(menu)
    drink = rng.choice([d for d in pool if len(d["sizes"]) > 1] or pool)
    first, last = drink["sizes"][0], drink["sizes"][-1]
    others = [d for d in pool if d["name"] != drink["name"]] or pool
    second = rng.choice(others)
    second_size = rng.choice(second["sizes"])

    before = describe(drink["name"], first)
    after = describe(drink["name"], last)
    extra = describe(second["name"], second_size)
    return [
        ("user", fill(rng, cat["user"], d=before, ad=with_article(before))),
        say(fill(rng, cat["ack"], d=before)),
        ("user", fill(rng, cat["change"], s=SIZE_WORD[last], a2=with_article(extra))),
        say(fill(rng, cat["close"], d=after, ad=with_article(after),
                 a2=with_article(extra))),
    ]


def price_question_no_order(rng, menu, cat):
    """A question is not an order."""
    drink = rng.choice(available(menu))
    size = rng.choice(drink["sizes"])
    text = describe(drink["name"], size)
    slots = {"d": text, "ad": with_article(text), "p": f"{price_of(drink, size):.2f}"}
    return [
        ("user", fill(rng, cat["user"], **slots)),
        say(fill(rng, cat["assistant"], **slots)),
    ]


def customisation(rng, menu, cat):
    pool = [d for d in available(menu) if d["sizes"] != ["S"]]
    alt_milks = [m for m in menu["milks"] if m != "whole"]
    if not pool or not alt_milks:
        return None
    drink = rng.choice(pool)
    size = rng.choice(drink["sizes"])
    milk = rng.choice(alt_milks)
    chosen = rng.sample(menu["extras"], rng.randint(1, min(2, len(menu["extras"]))))
    text = describe(drink["name"], size, milk, chosen)
    slots = {"d": text, "ad": with_article(text)}
    return [
        ("user", fill(rng, cat["user"], **slots)),
        say(fill(rng, cat["assistant"], **slots)),
    ]


SIZE_ORDER = ["S", "M", "L"]


def invalid_size(rng, menu, cat):
    """Customer asks for a size the product does not come in."""
    pool = [d for d in available(menu) if len(d["sizes"]) < 3]
    if not pool:
        return None
    drink = rng.choice(pool)
    absent = [s for s in SIZE_ORDER if s not in drink["sizes"]]

    if absent and rng.random() > 0.25:
        asked_word = SIZE_WORD[rng.choice(absent)]
        asked_rank = SIZE_ORDER.index(next(s for s in absent
                                           if SIZE_WORD[s] == asked_word))
    else:
        asked_word, asked_rank = "extra large", len(SIZE_ORDER)

    closest = min(drink["sizes"], key=lambda s: abs(SIZE_ORDER.index(s) - asked_rank))
    offered = " and ".join(SIZE_WORD[s] for s in drink["sizes"])
    name = drink["name"].lower()
    full = describe(drink["name"], closest)

    turns = [
        ("user", fill(rng, cat["user"], size=asked_word, d=name)),
        say(fill(rng, cat["refuse"], d=name, size=asked_word,
                 offered=offered, closest=SIZE_WORD[closest])),
    ]
    # Half stop at the refusal, half carry through to the corrected order, so
    # the model does not learn that this shape always ends in agreement.
    if rng.random() < 0.5:
        turns += [
            ("user", rng.choice(cat["accept"])),
            say(fill(rng, cat["close"], full=full)),
        ]
    return turns


def ambiguous_match(rng, menu, cat):
    """Customer names a word that matches several products on this menu."""
    pool = available(menu)
    exact = {d["name"].lower() for d in pool}

    groups = {}
    for drink in pool:
        groups.setdefault(drink["name"].split()[-1].lower(), []).append(drink)
    # Two exclusions. A word that is itself a product is not ambiguous: "latte"
    # is unambiguous when a plain Latte is listed. And the offer must be
    # complete -- if the menu holds another drink of the same family whose name
    # lacks the word (Earl Grey alongside Green Tea and Iced Tea), listing only
    # the name matches teaches the model to leave options out.
    options = []
    for word, group in groups.items():
        if len(group) < 2 or word in exact:
            continue
        families = {d["family"] for d in group}
        if any(d["family"] in families and d not in group for d in pool):
            continue
        options.append((word, group))
    if not options:
        return None

    word, group = rng.choice(options)
    listed = ", ".join(d["name"].lower() for d in group[:-1]) + \
        " or " + group[-1]["name"].lower()

    turns = [
        ("user", fill(rng, cat["user"], word=word)),
        say(fill(rng, cat["clarify"], options=listed)),
    ]
    if rng.random() < 0.5:
        chosen = rng.choice(group)
        size = rng.choice(chosen["sizes"])
        full = describe(chosen["name"], size)
        turns += [
            ("user", fill(rng, cat["pick"], choice=chosen["name"].lower())),
            say(fill(rng, cat["close"], full=full)),
        ]
    return turns


def vague_request(rng, menu, cat):
    """Customer describes what they want rather than naming it."""
    trait = rng.choice(cat["traits"])
    matches = [d for d in available(menu) if d["family"] == trait["family"]]
    if not matches:
        return None
    picks = rng.sample(matches, min(2, len(matches)))
    listed = " or ".join(d["name"].lower() for d in picks)
    key = "suggest_one" if len(picks) == 1 else "suggest_many"
    return [
        ("user", fill(rng, cat["user"], ask=trait["ask"])),
        say(fill(rng, cat[key], options=listed)),
    ]


BUILDERS = {
    "negation": negation,
    "invalid_size": invalid_size,
    "ambiguous_match": ambiguous_match,
    "vague_request": vague_request,
    "simple_order": simple_order,
    "clarify_missing_size": clarify_missing_size,
    "multi_item": multi_item,
    "out_of_stock": out_of_stock,
    "off_menu_request": off_menu_request,
    "order_correction": order_correction,
    "price_question_no_order": price_question_no_order,
    "customisation": customisation,
}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_record(rng, name, tpl):
    brand = make_brand(rng, tpl["brand"])
    menu_text, menu = make_menu(rng, tpl)

    turns = BUILDERS[name](rng, menu, tpl["categories"][name])
    if turns is None:
        return None

    messages = [{"role": "system",
                 "content": tpl["system_prompt"].format(brand=brand, menu=menu_text)}]
    messages += [{"role": role, "content": content} for role, content in turns]

    # `brand` is here for the generator's own summary, which counts distinct
    # brands as a check that menus are not repeating. The voice is the file
    # name, and repeating it in every record would be a second place to be
    # wrong about it.
    return {"messages": messages,
            "meta": {"category": name, "brand": brand}}


def load_templates(path):
    """A voice file plus the shared product vocabulary beside it.

    Voices hold only what they say; the menu lives once in _shared.yaml so a new
    voice never re-lists it. Keys in the voice file win on collision.
    """
    path = pathlib.Path(path)
    voice = yaml.safe_load(path.read_text(encoding="utf-8"))
    shared_path = path.parent / "_shared.yaml"
    if not shared_path.exists():
        return voice
    shared = yaml.safe_load(shared_path.read_text(encoding="utf-8"))
    return {**shared, **voice}


def fingerprint(record):
    """Dedup over the conversation only -- menus are near-unique by design."""
    body = "|".join(m["content"] for m in record["messages"][1:])
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def generate(n, seed, tpl, exclude=()):
    """`exclude` holds fingerprints already used elsewhere, so an eval set can be
    generated with no overlap against the training file."""
    rng = random.Random(seed)
    records, seen = [], set(exclude)

    for name, cat in tpl["categories"].items():
        quota = max(1, round(n * cat["share"]))
        made, attempts = 0, 0
        while made < quota and attempts < quota * 60:
            attempts += 1
            record = build_record(rng, name, tpl)
            if record is None or fingerprint(record) in seen:
                continue
            seen.add(fingerprint(record))
            records.append(record)
            made += 1
        if made < quota:
            print(f"warning: {name} produced {made}/{quota} unique records")

    rng.shuffle(records)
    return records


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    # Defaults resolve against this file rather than the working directory, so
    # the script behaves the same from the repo root or from finetuning/.
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--templates", default=DATA_DIR / "templates" / "friendly.yaml")
    parser.add_argument("--out", default=DATA_DIR / "train_friendly.jsonl")
    parser.add_argument("--exclude", action="append", default=[],
                        help="existing jsonl whose dialogues must not reappear")
    args = parser.parse_args()

    tpl = load_templates(args.templates)

    exclude = set()
    for path in args.exclude:
        for line in pathlib.Path(path).open(encoding="utf-8"):
            exclude.add(fingerprint(json.loads(line)))

    records = generate(args.n, args.seed, tpl, exclude)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = {}
    for record in records:
        counts[record["meta"]["category"]] = counts.get(record["meta"]["category"], 0) + 1

    print(f"wrote {len(records)} records ({tpl['voice']} voice) -> {out_path}")
    for name in tpl["categories"]:
        print(f"  {name:26} {counts.get(name, 0)}")
    print(f"  unique brands              {len({r['meta']['brand'] for r in records})}")


if __name__ == "__main__":
    main()
