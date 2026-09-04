"""The rules the corpus is built to and the model is scored against.

One set of functions serves both, so these tests are the place the split
between the three layers is actually pinned down: the corpus teaches voice, the
prompt carries the tools, and code owns the order.
"""

import json
import pathlib

import evaluate as ev
import pytest
import validate_dataset as vd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

SYSTEM = """You are the ordering assistant for Copper Kettle Espresso Bar.

Rules:
- Only ever reference items present in <menu>.

<menu>
Chai Latte | M,L | 4.00/4.60 | in stock
Matcha Latte | M,L | 4.40/5.00 | in stock
Cold Brew | M,L | 4.10/4.75 | OUT OF STOCK
Almond Croissant | - | 3.55 | in stock
Extras: honey (+0.50)
Milk options: whole, oat (+0.50)
</menu>"""


@pytest.fixture
def menu():
    return vd.parse_menu(SYSTEM)


def verdicts(reply, menu, asked=(), needs=False):
    return vd.score_reply(reply, menu, set(asked), needs)[0]


# -- reading products out of prose ------------------------------------------

def test_a_longer_product_name_wins_over_the_one_inside_it(menu):
    assert vd.mentions("one chai latte please") == {"chai latte"}


def test_a_plural_is_the_same_product(menu):
    assert vd.mentions("two chai lattes") == {"chai latte"}


def test_the_brand_is_a_shop_not_an_order(menu):
    """'Copper Kettle Espresso Bar' contains a product name. It is a sign over
    a door, and counting it would fail every greeting."""
    assert vd.mentions("Welcome to Copper Kettle Espresso Bar!",
                       menu["brand"]) == set()


# -- what a reply may say ---------------------------------------------------

def test_naming_a_product_this_menu_lacks_is_not_grounded(menu):
    assert verdicts("We do a lovely americano.", menu)["grounded"] is False


def test_offering_something_out_of_stock_fails(menu):
    assert verdicts("Cold brew, coming up.", menu)["in_stock"] is False


def test_repeating_what_the_customer_asked_for_is_allowed(menu):
    """"We're out of cold brew" has to be sayable, and so does "we don't have
    bubble tea" -- the customer put the name there."""
    scores = verdicts("We're out of cold brew today.", menu, asked={"cold brew"})
    assert scores["in_stock"] and scores["grounded"]


def test_two_questions_in_one_turn_fails(menu):
    assert verdicts("What size? And which milk?", menu)["one_question"] is False


def test_one_question_passes(menu):
    assert verdicts("Medium or large?", menu)["one_question"] is True


# -- being owed an alternative ----------------------------------------------

def test_a_refusal_that_names_nothing_else_fails(menu):
    scores = verdicts("Sorry, we're out of cold brew.", menu,
                      asked={"cold brew"}, needs=True)
    assert scores["alternative"] is False


def test_a_refusal_that_names_a_real_product_passes(menu):
    scores = verdicts("Out of cold brew — the chai latte is good though.",
                      menu, asked={"cold brew"}, needs=True)
    assert scores["alternative"] is True


def test_the_distinguishing_word_alone_counts_as_an_offer(menu):
    """"Chai or matcha?" offers two lattes without writing either name out."""
    scores = verdicts("Chai or matcha?", menu, asked={"latte"}, needs=True)
    assert scores["alternative"] is True


def test_the_metric_does_not_apply_when_nothing_is_owed(menu):
    assert verdicts("One chai latte.", menu)["alternative"] is None


def test_an_unavailable_ask_owes_an_alternative(menu):
    assert vd.wants_alternative("anything", menu, {"cold brew"})
    assert vd.wants_alternative("anything", menu, {"bubble tea"})


def test_an_ask_the_shop_can_serve_owes_nothing(menu):
    """The hard set's compound_difficulty covers a sold-out drink and a wrong
    size under one label. A wrong size owes a size, not another product, so
    this is read off the menu rather than off the category."""
    assert not vd.wants_alternative("compound_difficulty", menu, {"chai latte"})


def test_the_debt_is_discharged_once_something_is_offered(menu):
    assert not vd.wants_alternative("out_of_stock", menu, {"cold brew"},
                                    already_offered=True)


def test_naming_nothing_at_all_owes_a_suggestion_only_when_vague(menu):
    assert vd.wants_alternative("vague_request", menu, set())
    assert not vd.wants_alternative("out_of_stock", menu, set())


# -- the corpus itself ------------------------------------------------------

@pytest.mark.parametrize("path", sorted(DATA.glob("*.jsonl")), ids=lambda p: p.name)
def test_every_committed_record_passes_its_own_checks(path):
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    problems = [(i, p) for i, r in enumerate(records)
                for p in vd.check_record(r, i)]
    assert not problems, problems[:5]


@pytest.mark.parametrize("path", sorted(DATA.glob("*.jsonl")), ids=lambda p: p.name)
def test_no_corpus_teaches_a_tool_call(path):
    """The load-bearing one. Tools are prompt, not weights: a tool added next
    month has to work without a retrain, which it cannot if the weights were
    taught one fixed list."""
    assert "<tool_call>" not in path.read_text(encoding="utf-8")


# -- scoring model output ---------------------------------------------------

def test_a_tool_call_in_the_output_is_stripped_not_punished():
    """The served prompt does advertise tools, so a reply may legitimately
    carry a call. What is scored is the prose around it."""
    record = {"messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": "a chai latte, large"},
                           {"role": "assistant", "content": "One large chai latte."}],
              "meta": {"category": "simple_order"}}
    reply = ('One large chai latte.\n<tool_call>\n'
             '{"name": "add_item", "arguments": {"name": "Chai Latte", "quantity": 1}}'
             '\n</tool_call>')
    assert ev.score_one(record, reply) == ev.score_one(record, "One large chai latte.")


def test_the_reference_answers_score_full_marks():
    """The corpus is the ceiling. If the answers we trained on do not pass, the
    metric is measuring something other than what was taught."""
    records = ev.load(DATA / "eval_hard.jsonl")
    summary = ev.summarise(records, [r["messages"][-1]["content"] for r in records])
    assert all(rate in (None, 1.0) for rate in summary["overall"].values()), \
        summary["overall"]
