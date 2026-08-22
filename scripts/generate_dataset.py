"""Generate the supervised fine-tuning set for the coffee ordering assistant.

Scales the record shape approved in build_sample_dataset.py to a full training
set. The generator is deterministic given --seed.

What varies per example
-----------------------
Brand name, menu composition, prices, stock flags, milk and extra options,
customer phrasing, and assistant phrasing. Nothing about the *product catalog*
is stable across examples, so the model cannot memorise one -- it has to read
the <menu> block. That block is where retrieval output will be injected once
the RAG layer lands.

What stays fixed
----------------
The system-prompt rule set, the brand voice, and the create_order tool schema.
Those are exactly the things fine-tuning is meant to burn in.

Usage
-----
    python scripts/generate_dataset.py --n 800 --seed 42 --out data/train.jsonl
"""

import argparse
import hashlib
import json
import pathlib
import random

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

BRAND_HEAD = [
    "Copper", "Blue", "Willow", "Harbor", "Stone", "Marigold", "Ashfield",
    "Northgate", "Ember", "Cedar", "Iron", "Golden", "Rustic", "Wild",
    "Corner", "River", "Maple", "Birch", "Hollow", "Amber", "Slate",
    "Juniper", "Thistle", "Fern", "Larkspur", "Quiet", "Bright", "Old Town",
    "Westbrook", "Halden", "Rowan", "Alder", "Merrow", "Kestrel",
]

BRAND_TAIL = [
    "Kettle", "Lantern", "Bean", "Crow", "Fox", "Owl", "Mill", "Lane",
    "Anchor", "Ridge", "Grove", "Hill", "Press", "Barrel", "Sparrow",
    "Wren", "Post", "Stone", "Bridge", "Gate", "Well", "Hearth",
]

BRAND_SUFFIX = [
    "Coffee", "Coffee Co.", "Roasters", "Cafe", "Coffee Bar", "Brew House",
    "Coffee House", "Espresso Bar", "Coffee Roasters",
]

# (name, allowed sizes, price floor for the smallest size, family)
#
# `family` drives substitution: when something is out of stock or off menu, the
# assistant should propose a drink a customer would actually accept instead.
# Without it the generator suggests a hot espresso to someone asking for iced
# coffee, and the model learns to make irrelevant offers.
ESPRESSO_DRINKS = [
    ("Latte", ["S", "M", "L"], 3.0, "espresso"),
    ("Cappuccino", ["S", "M", "L"], 3.1, "espresso"),
    ("Flat White", ["S", "M", "L"], 3.0, "espresso"),
    ("Americano", ["S", "M", "L"], 2.6, "espresso"),
    ("Cortado", ["S", "M"], 2.9, "espresso"),
    ("Macchiato", ["S", "M"], 2.9, "espresso"),
    ("Mocha", ["S", "M", "L"], 3.5, "sweet"),
    ("Espresso", ["S"], 2.2, "espresso"),
    ("Doppio", ["S"], 2.7, "espresso"),
    ("Long Black", ["M", "L"], 3.0, "espresso"),
    ("Piccolo", ["S"], 2.8, "espresso"),
]

BREW_DRINKS = [
    ("Drip Coffee", ["S", "M", "L"], 2.0, "hot_brew"),
    ("Pour Over", ["M", "L"], 3.4, "hot_brew"),
    ("French Press", ["M", "L"], 3.6, "hot_brew"),
    ("Cold Brew", ["M", "L"], 3.8, "cold"),
    ("Nitro Cold Brew", ["M", "L"], 4.4, "cold"),
    ("Iced Coffee", ["M", "L"], 3.2, "cold"),
]

OTHER_DRINKS = [
    ("Chai Latte", ["M", "L"], 3.8, "flavoured"),
    ("Matcha Latte", ["M", "L"], 4.2, "flavoured"),
    ("Turmeric Latte", ["M", "L"], 4.0, "flavoured"),
    ("London Fog", ["M", "L"], 3.9, "flavoured"),
    ("Hot Chocolate", ["M", "L"], 3.4, "sweet"),
    ("Herbal Tea", ["M"], 2.7, "tea"),
    ("Green Tea", ["M"], 2.6, "tea"),
    ("Earl Grey", ["M"], 2.6, "tea"),
    ("Iced Tea", ["M", "L"], 2.7, "cold"),
]

# Which families a customer will accept as a substitute, in order of preference.
FAMILY_NEIGHBOURS = {
    "espresso": ["espresso", "hot_brew", "sweet"],
    "hot_brew": ["hot_brew", "espresso", "cold"],
    "cold": ["cold", "hot_brew", "espresso"],
    "sweet": ["sweet", "flavoured", "espresso"],
    "flavoured": ["flavoured", "sweet", "tea"],
    "tea": ["tea", "flavoured", "hot_brew"],
}

FOOD = [
    ("Almond Croissant", 3.5), ("Butter Croissant", 2.9),
    ("Banana Bread", 3.1), ("Blueberry Muffin", 3.2),
    ("Lemon Poppy Muffin", 3.3), ("Cinnamon Roll", 3.8),
    ("Chocolate Chip Cookie", 2.4), ("Sesame Bagel", 2.8),
    ("Avocado Toast", 6.2), ("Granola Bowl", 5.4), ("Brownie", 3.0),
    ("Cheese Scone", 3.4), ("Pain au Chocolat", 3.6),
]

MILKS = ["whole", "oat", "soy", "almond", "coconut", "skim"]

EXTRAS = [
    "extra shot", "vanilla syrup", "caramel syrup", "hazelnut syrup",
    "decaf", "extra hot", "cinnamon", "honey",
]

# Deliberately absent from every generated menu, paired with the family whose
# drinks make a sensible counter-offer.
OFF_MENU = [
    ("pumpkin spice frappuccino", "sweet"), ("caramel frappuccino", "sweet"),
    ("frozen hot chocolate", "sweet"), ("milkshake", "sweet"),
    ("bubble tea", "cold"), ("mango smoothie", "cold"),
    ("protein shake", "cold"), ("energy drink", "cold"),
    ("boba coffee", "cold"), ("iced matcha frappe", "flavoured"),
    ("espresso martini", "espresso"), ("affogato float", "sweet"),
]

SIZE_WORD = {"S": "small", "M": "medium", "L": "large"}

TOOL_OPEN = "<tool_call>\n"
TOOL_CLOSE = "\n</tool_call>"

SYSTEM = """You are the ordering assistant for {brand}.

Rules:
- Be warm and concise. Two sentences at most before acting.
- Only ever reference items present in <menu>. Never invent products, prices, or sizes.
- If a requested item is unavailable, say so plainly and offer the closest listed alternative.
- Ask at most one clarifying question at a time, and only for details you genuinely need.
- When the customer confirms, emit a create_order tool call.

<menu>
{menu}
</menu>"""


# --------------------------------------------------------------------------
# Menu construction
# --------------------------------------------------------------------------

def make_brand(rng):
    style = rng.random()
    head = rng.choice(BRAND_HEAD)
    if style < 0.45:
        return f"{head} {rng.choice(BRAND_TAIL)} {rng.choice(BRAND_SUFFIX)}"
    if style < 0.75:
        return f"{head} & {rng.choice(BRAND_TAIL)}"
    return f"{head} {rng.choice(BRAND_SUFFIX)}"


def price_ladder(rng, floor, sizes):
    """Prices rise with size, with per-brand jitter so no price is memorable."""
    base = round(floor + rng.uniform(0.0, 0.9), 2)
    prices, current = [], base
    for _ in sizes:
        prices.append(round(current, 2))
        current += rng.uniform(0.35, 0.75)
    return prices


def make_menu(rng):
    """Return (menu_text, drinks, food, milks, extras, milk_surcharge)."""
    drinks = []
    drinks += rng.sample(ESPRESSO_DRINKS, rng.randint(2, 4))
    drinks += rng.sample(BREW_DRINKS, rng.randint(1, 2))
    drinks += rng.sample(OTHER_DRINKS, rng.randint(1, 2))
    rng.shuffle(drinks)

    food = rng.sample(FOOD, rng.randint(1, 3))
    milks = ["whole"] + rng.sample([m for m in MILKS if m != "whole"], rng.randint(1, 3))
    extras = rng.sample(EXTRAS, rng.randint(2, 4))
    milk_surcharge = round(rng.uniform(0.35, 0.65), 2)

    # One drink is out of stock roughly a third of the time.
    oos_index = rng.randrange(len(drinks)) if rng.random() < 0.35 else None

    entries, drink_rows = [], []
    for i, (name, sizes, floor, family) in enumerate(drinks):
        prices = price_ladder(rng, floor, sizes)
        stock = "OUT OF STOCK" if i == oos_index else "in stock"
        entries.append(
            f"{name} | {','.join(sizes)} | {'/'.join(f'{p:.2f}' for p in prices)} | {stock}"
        )
        drink_rows.append({
            "name": name, "sizes": sizes, "prices": prices,
            "family": family, "in_stock": i != oos_index,
        })

    food_rows = []
    for name, floor in food:
        price = round(floor + rng.uniform(-0.3, 0.6), 2)
        entries.append(f"{name} | - | {price:.2f} | in stock")
        food_rows.append({"name": name, "price": price})

    entries.append(f"Extras: {', '.join(f'{e} (+{rng.uniform(0.5, 1.0):.2f})' for e in extras)}")
    entries.append(f"Milk options: {', '.join(milks)} (+{milk_surcharge:.2f})")

    return "\n".join(entries), drink_rows, food_rows, milks, extras


def price_of(drink, size):
    return drink["prices"][drink["sizes"].index(size)]


def in_stock_drinks(drinks):
    return [d for d in drinks if d["in_stock"]]


def substitutes_for(family, drinks, rng, limit=1):
    """In-stock drinks a customer would accept instead of `family`.

    Walks the neighbour list in preference order and returns from the first
    family that has anything available, so an iced drink is answered with
    another cold drink rather than whatever happens to be on the menu.
    """
    available = in_stock_drinks(drinks)
    if not available:
        return []
    for candidate_family in FAMILY_NEIGHBOURS.get(family, []):
        matches = [d for d in available if d["family"] == candidate_family]
        if matches:
            return rng.sample(matches, min(limit, len(matches)))
    return rng.sample(available, min(limit, len(available)))


# --------------------------------------------------------------------------
# Record helpers
# --------------------------------------------------------------------------

def order_call(items):
    return TOOL_OPEN + json.dumps({"name": "create_order", "arguments": {"items": items}}) + TOOL_CLOSE


def item(name, size=None, milk=None, extras=None, quantity=1):
    return {"name": name, "size": size, "milk": milk,
            "extras": extras or [], "quantity": quantity}


def with_article(noun):
    """'espresso' -> 'an espresso'. Templates that hardcode 'a {x}' produce
    'a americano' whenever the product name starts with a vowel."""
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def describe(drink_name, size=None, milk=None, extras=None, quantity=1):
    """Natural-language rendering used inside assistant confirmations."""
    parts = []
    if quantity > 1:
        parts.append(str(quantity))
    if size:
        parts.append(SIZE_WORD[size])
    if milk:
        parts.append(milk)
    parts.append(drink_name.lower() if quantity == 1 else drink_name.lower() + "s")
    text = " ".join(parts)
    if extras:
        text += " with " + " and ".join(extras)
    return text


# --------------------------------------------------------------------------
# Dialogue builders -- one per behaviour category
# --------------------------------------------------------------------------

def cat_simple_order(rng, drinks, food, milks, extras):
    drink = rng.choice(in_stock_drinks(drinks))
    size = rng.choice(drink["sizes"])
    ask = rng.choice([
        "Hey, can I get a {d} please?",
        "One {d}, thanks.",
        "I'll take a {d}.",
        "Could I have a {d}?",
        "Morning! A {d} for me.",
        "Just a {d} please.",
        "{d} to go, please.",
    ]).format(d=describe(drink["name"], size))
    reply = rng.choice([
        "Absolutely, one {d} coming right up.",
        "You got it — one {d}.",
        "Sure thing, one {d}.",
        "Of course, one {d} on the way.",
        "Perfect, one {d}.",
    ]).format(d=describe(drink["name"], size))
    return [
        ("user", ask),
        ("assistant", reply + "\n" + order_call([item(drink["name"], size=size)])),
    ]


def cat_clarify_missing_size(rng, drinks, food, milks, extras):
    drink = rng.choice([d for d in in_stock_drinks(drinks) if len(d["sizes"]) > 1])
    size = rng.choice(drink["sizes"])
    words = ", ".join(SIZE_WORD[s] for s in drink["sizes"])
    ask = rng.choice([
        "I'd like {ad}.",
        "Can I get {ad}?",
        "{Ad} please.",
        "{d}, thanks.",
    ]).format(
        d=drink["name"].lower(),
        ad=with_article(drink["name"].lower()),
        Ad=with_article(drink["name"].lower()).capitalize(),
    )
    question = rng.choice([
        "Happy to help — what size would you like? We have {w}.",
        "Sure! Which size — {w}?",
        "Of course. What size can I get you: {w}?",
        "Great choice — {w}?",
    ]).format(w=words)
    answer = rng.choice([
        "{s} please.", "{s} is fine.", "Make it {s}.", "The {s} one.", "{s}.",
    ]).format(s=SIZE_WORD[size])
    close = rng.choice([
        "One {d} it is.",
        "Perfect, one {d}.",
        "Got it — one {d}.",
    ]).format(d=describe(drink["name"], size))
    return [
        ("user", ask),
        ("assistant", question),
        ("user", answer),
        ("assistant", close + "\n" + order_call([item(drink["name"], size=size)])),
    ]


def cat_multi_item(rng, drinks, food, milks, extras):
    available = in_stock_drinks(drinks)
    drink = rng.choice(available)
    size = rng.choice(drink["sizes"])
    qty = rng.choice([1, 1, 2, 2, 3])

    picks = [item(drink["name"], size=size, quantity=qty)]
    spoken = [describe(drink["name"], size, quantity=qty)]

    if food and rng.random() < 0.6:
        snack = rng.choice(food)
        picks.append(item(snack["name"]))
        spoken.append(with_article(snack["name"].lower()))
    else:
        second = rng.choice([d for d in available if d["name"] != drink["name"]] or available)
        s2 = rng.choice(second["sizes"])
        picks.append(item(second["name"], size=s2))
        spoken.append(describe(second["name"], s2))

    joined = " and ".join(spoken)
    ask = rng.choice([
        "{j}, please.",
        "Can I get {j}?",
        "I'll have {j}.",
        "{j} — that's it.",
    ]).format(j=joined)
    reply = rng.choice([
        "{j} — got it.",
        "Perfect, {j}.",
        "Sure, {j} coming up.",
        "That's {j}.",
    ]).format(j=joined)
    return [("user", ask), ("assistant", reply + "\n" + order_call(picks))]


def cat_out_of_stock(rng, drinks, food, milks, extras):
    missing = [d for d in drinks if not d["in_stock"]]
    if not missing:
        return None
    gone = missing[0]
    alternatives = substitutes_for(gone["family"], drinks, rng)
    if not alternatives:
        return None
    alt = alternatives[0]
    alt_size = rng.choice(alt["sizes"])
    gone_size = rng.choice(gone["sizes"])

    ask = rng.choice([
        "One {d}, please.",
        "Can I get a {d}?",
        "I'd like a {d}.",
    ]).format(d=describe(gone["name"], gone_size))
    apology = rng.choice([
        "Sorry, the {g} is out of stock today. A {a} is the closest thing we have — want that instead?",
        "Afraid we're out of {g} right now. I could do a {a} instead — sound good?",
        "The {g} is unavailable today, unfortunately. Closest option is a {a} — shall I put that in?",
    ]).format(g=gone["name"].lower(), a=describe(alt["name"], alt_size))
    accept = rng.choice([
        "Yeah that works.", "Sure, go for it.", "That's fine.",
        "Okay, let's do that.", "Sounds good.",
    ])
    close = rng.choice([
        "Great, one {a}.",
        "Perfect — one {a}.",
        "Done, one {a}.",
    ]).format(a=describe(alt["name"], alt_size))
    return [
        ("user", ask),
        ("assistant", apology),
        ("user", accept),
        ("assistant", close + "\n" + order_call([item(alt["name"], size=alt_size)])),
    ]


def cat_off_menu_request(rng, drinks, food, milks, extras):
    """No tool call: the item does not exist and must not be invented."""
    wanted, family = rng.choice(OFF_MENU)
    picks = substitutes_for(family, drinks, rng, limit=2)
    if not picks:
        return None
    names = " or ".join(d["name"].lower() for d in picks)
    ask = rng.choice([
        "Do you have {aw}?",
        "Can I get {aw}?",
        "One {w} please.",
        "I'll take {aw}.",
        "Any chance you do {w}?",
    ]).format(w=wanted, aw=with_article(wanted))
    # Singular and plural offers need different phrasing; mixing them teaches
    # the model broken agreement ("our cold brew are the closest picks").
    if len(picks) == 1:
        reply = rng.choice([
            "We don't carry that one, sorry. Our {n} is the closest pick — would that work?",
            "Afraid that's not on our menu. I can offer our {n} instead — interested?",
            "Sorry, we don't make that here. The {n} is what I'd suggest — want one?",
        ]).format(n=names)
    else:
        reply = rng.choice([
            "We don't carry that one, sorry. Our {n} are the closest picks — would either work?",
            "Afraid that's not on our menu. I can offer {n} instead — interested?",
            "Sorry, we don't make that here. The {n} are what I'd suggest — want one?",
        ]).format(n=names)
    return [("user", ask), ("assistant", reply)]


def cat_order_correction(rng, drinks, food, milks, extras):
    available = in_stock_drinks(drinks)
    drink = rng.choice([d for d in available if len(d["sizes"]) > 1] or available)
    first_size = drink["sizes"][0]
    new_size = drink["sizes"][-1]
    second = rng.choice([d for d in available if d["name"] != drink["name"]] or available)
    second_size = rng.choice(second["sizes"])

    ask = rng.choice([
        "A {d} please.", "Can I have a {d}?", "One {d}.",
    ]).format(d=describe(drink["name"], first_size))
    ack = rng.choice([
        "One {d} — anything else?",
        "Sure, one {d}. Anything to go with it?",
        "Got it, one {d}. Anything else for you?",
    ]).format(d=describe(drink["name"], first_size))
    change = rng.choice([
        "Actually make it a {ns}, and add a {s2}.",
        "Sorry — change that to a {ns}, plus a {s2}.",
        "Hold on, I'd rather have a {ns}. And a {s2} too.",
    ]).format(ns=SIZE_WORD[new_size], s2=describe(second["name"], second_size))
    close = rng.choice([
        "Switched to a {d}, plus a {s2}.",
        "No problem — a {d} and a {s2}.",
        "Updated: a {d} and a {s2}.",
    ]).format(d=describe(drink["name"], new_size), s2=describe(second["name"], second_size))
    return [
        ("user", ask),
        ("assistant", ack),
        ("user", change),
        ("assistant", close + "\n" + order_call([
            item(drink["name"], size=new_size),
            item(second["name"], size=second_size),
        ])),
    ]


def cat_price_question(rng, drinks, food, milks, extras):
    """No tool call: a question is not an order."""
    drink = rng.choice(in_stock_drinks(drinks))
    size = rng.choice(drink["sizes"])
    price = price_of(drink, size)
    ask = rng.choice([
        "How much is a {d}?",
        "What does a {d} cost?",
        "Price on a {d}?",
        "How much for a {d}?",
    ]).format(d=describe(drink["name"], size))
    reply = rng.choice([
        "A {d} is {p:.2f}. Want me to add one to your order?",
        "That's {p:.2f} for a {d}. Shall I put one in?",
        "{p:.2f} for a {d}. Would you like one?",
    ]).format(d=describe(drink["name"], size), p=price)
    return [("user", ask), ("assistant", reply)]


def cat_customisation(rng, drinks, food, milks, extras):
    candidates = [d for d in in_stock_drinks(drinks) if d["sizes"] != ["S"]]
    if not candidates or len(milks) < 2:
        return None
    drink = rng.choice(candidates)
    size = rng.choice(drink["sizes"])
    milk = rng.choice([m for m in milks if m != "whole"])
    chosen = rng.sample(extras, rng.randint(1, min(2, len(extras))))

    ask = rng.choice([
        "{d}, please.",
        "Can I get a {d}?",
        "I'll have a {d}.",
    ]).format(d=describe(drink["name"], size, milk, chosen))
    reply = rng.choice([
        "One {d}.",
        "Coming up — one {d}.",
        "Sure, one {d}.",
    ]).format(d=describe(drink["name"], size, milk, chosen))
    return [
        ("user", ask),
        ("assistant", reply + "\n" + order_call([
            item(drink["name"], size=size, milk=milk, extras=chosen)
        ])),
    ]


# Category name -> (builder, share of the dataset)
CATEGORIES = [
    ("simple_order", cat_simple_order, 0.225),
    ("clarify_missing_size", cat_clarify_missing_size, 0.1375),
    ("multi_item", cat_multi_item, 0.1375),
    ("out_of_stock", cat_out_of_stock, 0.125),
    ("off_menu_request", cat_off_menu_request, 0.1125),
    ("order_correction", cat_order_correction, 0.1125),
    ("price_question_no_order", cat_price_question, 0.0875),
    ("customisation", cat_customisation, 0.0625),
]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_record(rng, category, builder):
    brand = make_brand(rng)
    menu_text, drinks, food, milks, extras = make_menu(rng)
    turns = builder(rng, drinks, food, milks, extras)
    if turns is None:
        return None

    messages = [{"role": "system", "content": SYSTEM.format(brand=brand, menu=menu_text)}]
    for role, content in turns:
        messages.append({"role": role, "content": content})

    return {
        "messages": messages,
        "meta": {"category": category, "brand": brand, "voice": "friendly"},
    }


def fingerprint(record):
    """Dedup key over the conversation only -- menus are near-unique by design."""
    body = "|".join(m["content"] for m in record["messages"][1:])
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def generate(n, seed):
    rng = random.Random(seed)
    quotas = {name: max(1, round(n * share)) for name, _, share in CATEGORIES}
    builders = {name: fn for name, fn, _ in CATEGORIES}

    records, seen = [], set()
    for category, quota in quotas.items():
        made, attempts = 0, 0
        while made < quota and attempts < quota * 60:
            attempts += 1
            record = build_record(rng, category, builders[category])
            if record is None:
                continue
            key = fingerprint(record)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
            made += 1
        if made < quota:
            print(f"warning: {category} produced {made}/{quota} unique records")

    rng.shuffle(records)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/train.jsonl")
    args = parser.parse_args()

    records = generate(args.n, args.seed)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = {}
    for record in records:
        counts[record["meta"]["category"]] = counts.get(record["meta"]["category"], 0) + 1

    print(f"wrote {len(records)} records -> {out_path}")
    for name, _, _ in CATEGORIES:
        print(f"  {name:26} {counts.get(name, 0)}")
    print(f"  unique brands              {len({r['meta']['brand'] for r in records})}")


if __name__ == "__main__":
    main()
