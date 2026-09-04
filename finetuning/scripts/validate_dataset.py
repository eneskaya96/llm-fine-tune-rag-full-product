"""Validate a coffee-ordering dataset file against its own <menu> context.

Every check here is a property the trained model is later scored on, so this
module is deliberately reusable: evaluate.py runs these same functions over
model output instead of over the corpus.

The corpus is speech. It carries no tool calls, because which tools exist is
read off the <tools> block the serving layer writes into the prompt -- a tool
added next month has to work without touching the weights. What is checked here
is what the weights are actually being taught: talk about this menu and no
other, do not offer what the shop has run out of, ask one thing at a time, and
answer an impossible request with a real alternative.

Checks
------
1. structural   -- system first, roles alternate, assistant speaks last
2. speech only  -- no <tool_call> block anywhere
3. grounded     -- an assistant turn names no product this menu lacks
4. in stock     -- it never offers something marked OUT OF STOCK
5. one question -- at most one question mark per assistant turn
6. alternative  -- when the ask cannot be met, a real product is named instead

A name the customer used first is exempt from 3 and 4: "we're out of cold brew"
has to be sayable. That is the limit of reading prose -- an assistant that
offered the missing item instead of refusing it would read the same to this
checker. Nothing downstream trusts prose, which is why the order layer works
from tool calls the code validates rather than from what the reply says.

Usage
-----
    python finetuning/scripts/validate_dataset.py finetuning/data/train_friendly.jsonl
"""

import json
import pathlib
import re
import sys

import yaml

MENU_RE = re.compile(r"<menu>\n(.*?)\n</menu>", re.DOTALL)
BRAND_RE = re.compile(r"You are the ordering assistant for (.+?)\.")

TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / "data" / "templates"

# The one case the menu cannot detect on its own: a customer who names no
# product at all ("something cold, not too sweet") is owed a suggestion, while
# one who says "yes please" is being served.
VAGUE = {"vague_request"}


def _product_names():
    """Every product name the generator can put in front of a model.

    The closed world matters: a name from this list that is not on this
    example's menu is a hallucination, and one that is off_menu is a product no
    brand in the corpus has ever stocked.
    """
    tpl = yaml.safe_load((TEMPLATES / "_shared.yaml").read_text(encoding="utf-8"))
    names = [d["name"] for group in tpl["drinks"].values() for d in group]
    names += [f["name"] for f in tpl["food"]]
    names += [o["name"] for o in tpl["off_menu"]]
    return sorted({n.lower() for n in names}, key=len, reverse=True)


PRODUCTS = _product_names()
# Longest first and non-overlapping, so "chai latte" is one product rather than
# also counting as "latte". The optional s catches "two lattes".
PRODUCT_RE = re.compile(
    r"(?<![a-z])(" + "|".join(re.escape(n) for n in PRODUCTS) + r")s?(?![a-z])")


def mentions(text, brand=""):
    """The products this text names.

    The brand is removed first: "Copper Kettle Espresso Bar" is a shop, not an
    order for an espresso.
    """
    text = text.lower()
    if brand:
        text = text.replace(brand.lower(), " ")
    return set(PRODUCT_RE.findall(text))


def menu_mentions(reply, menu):
    """Products on this menu the reply points at, including by their
    distinguishing word alone.

    "Chai, matcha, or turmeric?" offers three lattes without writing any of the
    three names out. A word shared by several products picks out nothing, so
    "latte" there counts for none of them. Deliberately generous: this is only
    ever used to ask whether an alternative was offered, never to accuse a reply
    of naming something it should not have.
    """
    text = reply.lower()
    seen, shared = set(), set()
    for name in menu["products"]:
        for word in name.split():
            if word in seen:
                shared.add(word)
            seen.add(word)

    found = set()
    for name in menu["products"]:
        keys = [name] + [w for w in name.split() if w not in shared]
        if any(re.search(rf"(?<![a-z]){re.escape(key)}s?(?![a-z])", text)
               for key in keys):
            found.add(name)
    return found


def parse_menu(system_prompt):
    """Parse the brand line and <menu> block into a lookup the checks query."""
    match = MENU_RE.search(system_prompt)
    if not match:
        return None

    brand = BRAND_RE.search(system_prompt)
    products, milks = {}, set()
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Milk options:"):
            body = line.split(":", 1)[1].split("(")[0]
            milks = {m.strip() for m in body.split(",") if m.strip()}
            continue
        if line.startswith("Extras:"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        name, sizes, prices, stock = parts
        products[name.lower()] = {
            "sizes": [] if sizes == "-" else sizes.split(","),
            "prices": [float(p) for p in prices.split("/")],
            "in_stock": stock == "in stock",
        }
    return {"products": products, "milks": milks,
            "brand": brand.group(1) if brand else ""}


def offers(reply, menu, asked):
    """In-stock products this reply puts on the table that nobody asked for."""
    return {name for name in menu_mentions(reply, menu) - asked
            if menu["products"][name]["in_stock"]}


def wants_alternative(category, menu, asked, already_offered=False):
    """Whether this turn still owes the customer a product they did not name.

    Read off the menu rather than the category: a customer is owed a
    suggestion exactly when what they have named is missing or out of stock.
    The label cannot decide this -- the hard set's compound_difficulty covers
    both a drink that is sold out, which owes an alternative, and one whose
    size is wrong, which owes a size.

    Discharged once any turn has put an in-stock product on the table. After
    that the conversation is settling it, and demanding a fresh suggestion
    every turn would score a correct reply as a failure.
    """
    if already_offered:
        return False
    if any(name not in menu["products"] or not menu["products"][name]["in_stock"]
           for name in asked):
        return True
    return category in VAGUE and not asked


def score_reply(reply, menu, asked, needs_alternative=False):
    """One assistant turn as (verdict per metric, sentence per failure).

    A verdict of None means the metric does not apply to this turn -- only a
    customer who cannot be given what they asked for can be owed an
    alternative, and counting the rest as passes would let a model that never
    suggests anything score full marks.

    Two readers, one set of rules: validate_dataset renders the sentences over
    the corpus, evaluate.py counts the verdicts over what a model generated.
    """
    problems = []

    offered = mentions(reply, menu["brand"]) - asked
    off_menu = sorted(name for name in offered if name not in menu["products"])
    out_of_stock = sorted(name for name in offered
                          if name in menu["products"]
                          and not menu["products"][name]["in_stock"])
    for name in off_menu:
        problems.append(f"names {name!r}, which is not on this menu")
    for name in out_of_stock:
        problems.append(f"offers {name!r}, which is out of stock")

    questions = reply.count("?")
    if questions > 1:
        problems.append(f"asks {questions} questions in one turn")

    alternative = None
    if needs_alternative:
        alternative = bool(offers(reply, menu, asked))
        if not alternative:
            problems.append("names no in-stock alternative")

    return {"grounded": not off_menu,
            "in_stock": not out_of_stock,
            "one_question": questions <= 1,
            "alternative": alternative}, problems


METRICS = ["grounded", "in_stock", "one_question", "alternative"]


def check_record(record, index):
    """Return a list of human-readable problems with this record."""
    problems = []
    messages = record.get("messages", [])
    category = record.get("meta", {}).get("category", "?")

    # 1. structural
    if not messages or messages[0]["role"] != "system":
        problems.append("no system message")
        return problems
    if messages[-1]["role"] != "assistant":
        problems.append("does not end on an assistant turn")
    expected = "user"
    for message in messages[1:]:
        if message["role"] != expected:
            problems.append(f"role order broken at '{message['role']}'")
            break
        expected = "assistant" if expected == "user" else "user"

    menu = parse_menu(messages[0]["content"])
    if menu is None:
        problems.append("system prompt has no <menu> block")
        return problems

    asked, offered = set(), False
    for message in messages[1:]:
        if message["role"] == "user":
            asked |= mentions(message["content"], menu["brand"])
            continue
        # A tool call in the corpus is a corpus bug, not a model failure: the
        # served prompt does advertise tools, so score_reply says nothing about
        # them. Here there is nothing to advertise and nothing to call.
        if "<tool_call>" in message["content"]:
            problems.append("assistant turn carries a tool call")
        problems += score_reply(
            message["content"], menu, asked,
            wants_alternative(category, menu, asked, offered))[1]
        offered = offered or bool(offers(message["content"], menu, asked))

    return problems


def main():
    default = pathlib.Path(__file__).resolve().parent.parent / "data" / "train_friendly.jsonl"
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default
    records = [json.loads(line) for line in path.open(encoding="utf-8")]

    failures, total_problems = 0, 0
    for index, record in enumerate(records):
        problems = check_record(record, index)
        if problems:
            failures += 1
            total_problems += len(problems)
            if failures <= 10:
                print(f"record {index} [{record.get('meta', {}).get('category')}]")
                for problem in problems:
                    print(f"    - {problem}")

    print(f"\n{len(records)} records checked")
    print(f"{failures} records with problems ({total_problems} problems total)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
