"""What happens when the model calls a tool.

shared/tools.py says which tools exist; this runs them. Every one of them
returns a string, because that string is what goes back to the model as the
next turn -- including the refusals. That is the whole difference between this
and the single-shot layer it replaces: when an item is rejected the model finds
out and can offer something else, rather than the customer reading a card
saying the shop said no.

The cart lives in a Session rather than in module state. Two customers on one
Space must not share one, and a test must be able to make one without a model.
"""

from dataclasses import dataclass, field

import orders
from rag.src.retrieve import MAX_DRINKS, MAX_FOOD, select
from shared import tools as schema


@dataclass
class Session:
    """One conversation's menu and cart."""

    slug: str
    shop: object
    products: list = field(default_factory=list)
    cart: list = field(default_factory=list)
    rejections: list = field(default_factory=list)
    # What the agent did this turn, in order. The loop is the claim, so the UI
    # shows it; hiding it would hide the thing worth looking at.
    trace: list = field(default_factory=list)
    confirmed: bool = False

    @property
    def total(self):
        return orders.total(self.cart)

    def refresh(self, found):
        """Start a turn from what retrieval chose, keeping the cart orderable.

        `found` leads: a new turn's retrieval read the whole conversation and
        knows better than last turn's menu what belongs on this one. The
        standing menu trails it only so a cart line whose product has dropped
        out of retrieval still has something to pin.
        """
        self.products = _menu(self, list(found) + self.products)


def _menu(session, candidates):
    """Cut a candidate list down to a menu the model can be shown.

    The budget is the one retrieval already respects: the adapters were trained
    on menus of 4-9 drinks and 1-3 food, so an agent that searched four times
    could otherwise hand the model a menu longer than anything it has seen.

    Products the cart names are pulled to the front and so are never the ones
    dropped. A line whose product fell off the menu could not be re-checked,
    and check_item would refuse it as "not on the menu that was retrieved" --
    the cart would start rejecting its own contents. Order is otherwise the
    caller's: it decides whether a fresh search or the standing menu wins.
    """
    ordered = {line["name"].lower() for line in session.cart}
    ranked = sorted(candidates, key=lambda p: p.name.lower() not in ordered)

    kept, seen, room = [], set(), {"drink": MAX_DRINKS, "food": MAX_FOOD}
    for product in ranked:
        if product.id in seen or room.get(product.kind, 0) < 1:
            continue
        seen.add(product.id)
        room[product.kind] -= 1
        kept.append(product)
    return kept


def _search_menu(arguments, session):
    query = str(arguments.get("query", ""))
    _, found, _ = select(session.slug, query)
    before = {p.id for p in session.products}
    session.products = _menu(session, session.products + found)
    # Names, not menu lines. The next thing the model reads is a system prompt
    # built from session.products, so repeating the sizes and prices here would
    # put the menu in the context twice and disagree with itself the moment a
    # later search drops one.
    added = [p.name for p in session.products if p.id not in before]
    return ("added to the menu: " + ", ".join(added) if added
            else "nothing new; the menu already covers that")


def _add_item(arguments, session):
    line, reason = orders.check_item(arguments, session.products, session.shop)
    if reason:
        session.rejections.append(reason)
        return f"refused: {reason}"
    session.cart.append(line)
    return (f"added {line['quantity']}x {line['name']} at {line['line_total']:.2f}; "
            f"cart is now {session.total:.2f}")


def _remove_item(arguments, session):
    name = str(arguments.get("name", "")).lower()
    for index in reversed(range(len(session.cart))):
        if session.cart[index]["name"].lower() == name:
            removed = session.cart.pop(index)
            return (f"removed {removed['name']}; "
                    f"cart is now {session.total:.2f}")
    return f"refused: {arguments.get('name')!r} is not in the cart"


def _confirm_order(arguments, session):
    if not session.cart:
        return "refused: the cart is empty, nothing to confirm"
    session.confirmed = True
    return f"order placed, {len(session.cart)} line(s), {session.total:.2f}"


def _create_order(arguments, session):
    """The call the adapters were trained to emit, before these tools existed.

    It carries the whole order at once, so it replaces the cart rather than
    adding to it -- accumulating would double every item the model repeats on
    the confirmation turn. Accepted so the current weights still work; not
    advertised, so a retrained model learns the incremental tools instead.
    """
    entries = arguments.get("items")
    if not isinstance(entries, list) or not entries:
        return "refused: create_order arrived with no items"

    lines, refused = [], []
    for entry in entries:
        line, reason = orders.check_item(entry, session.products, session.shop)
        (lines if line else refused).append(line or reason)

    session.cart = lines
    session.rejections.extend(refused)
    session.confirmed = bool(lines)

    placed = f"order placed, {len(lines)} line(s), {session.total:.2f}"
    return placed + (f"; refused: {'; '.join(refused)}" if refused else "")


RUNNERS = {
    "search_menu": _search_menu,
    "add_item": _add_item,
    "remove_item": _remove_item,
    "confirm_order": _confirm_order,
    "create_order": _create_order,
}


def run(call, session):
    """Execute one tool call against `session`, and say what to tell the model."""
    name = call.get("name")
    runner = RUNNERS.get(name)
    if runner is None:
        reason = f"{name!r} is not a tool this shop has"
        session.rejections.append(reason)
        return f"refused: {reason}"

    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        reason = f"{name}: arguments must be an object"
        session.rejections.append(reason)
        return f"refused: {reason}"

    missing = [a for a in schema.TOOLS.get(name, {}).get("required", ())
               if arguments.get(a) is None]
    if missing:
        reason = f"{name}: {', '.join(missing)} missing"
        session.rejections.append(reason)
        return f"refused: {reason}"

    return runner(arguments, session)
