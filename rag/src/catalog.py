"""The catalog, and the Chroma collection that holds it.

Chroma is the only store here. One record carries all three things a product
needs: the document (its description, which is what gets embedded), the
metadata (name, sizes, prices, stock) and the vector. Keeping the prices in a
second store would mean two things to synchronise, and they would not fail
loudly when they drifted -- they would quietly produce a menu with the wrong
price on it, which the model would then state as fact.

Only the description is embedded. Prices and stock are metadata, so marking a
product sold out is `collection.update(...)` with no re-embedding at all.
"""

import hashlib
import json
import pathlib

import chromadb

from .models import Product, Shop  # re-exported: retrieve.py reads rows back as Products

CATALOG_DIR = pathlib.Path(__file__).resolve().parent.parent / "catalog"
INDEX_DIR = pathlib.Path(__file__).resolve().parent.parent / "index"

_clients = {}


def load(slug):
    """Read one shop's catalog file. Returns (Shop, [Product], catalog_digest)."""
    path = CATALOG_DIR / f"{slug}.json"
    raw = path.read_bytes()
    data = json.loads(raw)
    shop = Shop(
        slug=data["slug"],
        brand=data["brand"],
        milks=tuple(data["milks"]),
        milk_surcharge=data["milk_surcharge"],
        extras=tuple((e["name"], e["price"]) for e in data["extras"]),
    )
    products = [Product.from_record(r) for r in data["products"]]
    return shop, products, hashlib.sha256(raw).hexdigest()


def collection(slug, client=None):
    """The shop's Chroma collection, seeded from its catalog file if needed.

    One collection per shop rather than one collection with a shop filter: a
    wrong `where` clause would put another shop's products into a prompt
    silently, while a missing collection raises. Isolation by construction.

    The digest guard reseeds whenever the catalog file changes, so an index
    left over from an older catalog cannot serve a stale price. When the admin
    panel starts writing products (stage 4), Chroma becomes the source of truth
    and this seeding step is what has to go -- reseeding would overwrite the
    edits it is meant to preserve.
    """
    shop, products, digest = load(slug)

    if client is None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if slug not in _clients:
            _clients[slug] = chromadb.PersistentClient(path=str(INDEX_DIR))
        client = _clients[slug]

    name = f"menu_{slug}"
    col = client.get_or_create_collection(name, metadata={"catalog_sha": digest})
    if col.metadata.get("catalog_sha") != digest or col.count() != len(products):
        client.delete_collection(name)
        col = client.get_or_create_collection(name, metadata={"catalog_sha": digest})
        col.add(
            ids=[p.id for p in products],
            documents=[f"{p.name}: {p.description}" for p in products],
            metadatas=[p.as_metadata() for p in products],
        )
    return shop, col
