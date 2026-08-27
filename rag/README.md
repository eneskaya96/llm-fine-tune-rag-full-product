# RAG layer

Owns the product catalog: what exists, what it costs, what is in stock. The
fine-tuned model owns none of that — it reads whatever this layer puts in the
`<menu>` block.

```
catalog/ember_and_oak.json    28 products, the seed catalog
src/models.py                 Product and Shop
src/catalog.py                catalog file -> Chroma collection
src/menu_format.py            the menu line — one place, on purpose
src/retrieve.py               (query, history) -> <menu> body
tests/test_retrieve.py
```

## Why retrieval is not optional here

The adapters were trained on menus of **4–9 drinks and 1–3 food items**
(`finetuning/scripts/generate_dataset.py:make_menu`) and never saw a longer
one. Ember & Oak has 28 products. Putting the catalog in the prompt would hand
the model a shape it has no training for, so something has to choose — and
choosing badly is worse than not retrieving at all, because the model then
correctly refuses a product the shop actually sells and the refusal looks
grounded.

## The format is not ours to improve

```
Flat White | S,M,L | 3.40/3.95/4.50 | in stock
Iced Latte | M,L | 4.00/4.60 | OUT OF STOCK
Banana Bread | - | 3.10 | in stock
Extras: extra shot (+0.80), vanilla syrup (+0.55)
Milk options: whole, skim, oat, almond, soy (+0.50)
```

`validate_dataset.py:parse_menu` is the reference parser and the round-trip
test drives retrieval output back through it, asserting every name, size,
price and stock flag survives. That test is what locks the format: a drifted
separator would otherwise be a silent regression, not an error.

## One store

Chroma holds all three things a product needs in one record — the document
(its description), the metadata (name, sizes, prices, stock) and the vector.
A separate catalog store would be a second thing to keep in sync, and drift
between them would not raise; it would quietly put a wrong price on the menu,
which the model would then state as fact.

**Only the description is embedded.** Prices and stock are metadata, so marking
a product sold out is a metadata update with no re-embedding. Embedding the
price would have made every price change an indexing job.

The description carries the words a customer actually uses — *hot*, *iced*,
*strong*, *sweet*, *no milk* — because the product name almost never does.
Nothing in "Cold Brew" matches *"something refreshing"*; its description does.

`rag/index/` is the Chroma directory, gitignored and rebuilt whenever the
catalog file's hash changes, so an index left over from an older catalog can
never serve a stale price.

## The four rules

Three of these exist because semantic search alone gets them wrong.

1. **Drinks and food are queried separately** — one ranked list lets a coffee
   query take every slot and leave the pastries invisible.
2. **Anything named earlier in the conversation stays on the menu.** Without
   this, a latte ordered on turn one is gone by turn three and the model denies
   an order it just took. It cannot fix this from its side: the menu is its
   only record of the product.
3. **Milks and extras go in whole, every time.** Two short lines, needed by any
   order, so there is nothing to retrieve.
4. **Sold-out products stay on the menu** as `OUT OF STOCK`. Filtering them
   would have the model say the shop does not sell the item, when it does and
   has simply run out — and the adapters were trained to offer the nearest
   listed alternative, which needs the alternative listed.

## Multi-tenant

One collection per shop, `menu_<slug>`. Isolation comes from the structure
rather than from getting a filter right: a wrong `where` clause cannot put
another shop's products into a prompt, and a missing collection raises.

## Cost

Measured on a 4-core box, so read these as an upper bound:

| | |
|---|---|
| ONNX MiniLM load | 1.2 s, once per process |
| Seeding 28 products | 1.9 s, only when the catalog changes |
| Opening an existing index | 0.01 s |
| One `retrieve()` call | ~0.6 s, of which ~0.3 s is embedding the query |

Chroma's default embedding function pads to a fixed length, so a short query
costs the same as a long one. Against 2–5 s of generation this is acceptable;
it is also the floor for any embedding model, not something this design added.

## Running it

```bash
python -m rag.src.retrieve --query "large flat white with oat"
python -m rag.src.retrieve --query "add a cookie" \
    --history "a latte please" "One latte. Anything else?"
pytest rag/tests
```

The CLI prints the menu and then why each line is on it.

## No benchmark here

Whether vector search beats substring matching over 28 products is not a
question worth a report — the answer is known and the catalog is far too small
for the comparison to mean anything. `tests/` holds regression tests instead:
each pins one rule that, if it broke, would produce a plausible-looking menu
with something wrong in it.

## Not yet

Chroma is seeded from a JSON file, which means the catalog is still in git.
That is the right shape for a demo and the wrong shape for a shop — stock
changes hourly and a barista cannot open a pull request. It becomes real when
the admin panel writes products (stage 4); at that point Chroma is the source
of truth and the seeding step in `catalog.py` is what has to go, since
reseeding would overwrite the edits it exists to preserve.
