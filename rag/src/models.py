"""Shapes the rest of the layer passes around.

`sizes` and `prices` are held as the strings the menu line needs -- "S,M,L" and
"3.40/3.95/4.50" -- rather than lists of floats. Two reasons: Chroma metadata
only accepts scalars, and formatting a float on every render is a chance to
drift from the format the adapters were trained to read. Parsing happens once,
at seed time, and never again.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    kind: str          # "drink" or "food"
    sizes: str         # "S,M,L", or "-" for food
    prices: str        # "3.40/3.95/4.50", one entry per size
    in_stock: bool
    description: str   # the only field that gets embedded

    @classmethod
    def from_record(cls, record):
        """One catalog JSON entry."""
        sizes = ",".join(record["sizes"]) if record["sizes"] else "-"
        return cls(
            id=record["id"],
            name=record["name"],
            kind=record["kind"],
            sizes=sizes,
            prices="/".join(f"{p:.2f}" for p in record["prices"]),
            in_stock=record["in_stock"],
            description=record["description"],
        )

    @classmethod
    def from_metadata(cls, product_id, metadata):
        """One row read back out of Chroma."""
        return cls(id=product_id, **metadata)

    def as_metadata(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "sizes": self.sizes,
            "prices": self.prices,
            "in_stock": self.in_stock,
            "description": self.description,
        }


@dataclass(frozen=True)
class Shop:
    """Everything about a shop that is not a product.

    These lists go into every menu whole, so they never need retrieving.
    """

    slug: str
    brand: str
    milks: tuple
    milk_surcharge: float
    extras: tuple      # ((name, price), ...)
