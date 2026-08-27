# RAG layer

Not built yet.

Owns the product catalog: what exists, what it costs, what is in stock. The
fine-tuned model owns none of that — it reads whatever this layer puts in the
`<menu>` block.

## Planned

```
catalog/ember_and_oak.json    the demo shop's products
src/index.py                  catalog -> ChromaDB collection
src/retrieve.py               query -> <menu> block
src/schema.py                 catalog record shape
tests/
```

## The one hard constraint

`retrieve.py` must emit the **exact** `<menu>` format the adapters were trained
on. The model was fine-tuned to read this and nothing else:

```
Latte | S,M,L | 3.20/3.80/4.40 | in stock
Oat Flat White | M | 4.20 | OUT OF STOCK
Blueberry Muffin | - | 3.30 | in stock
Extras: extra shot (+0.80), vanilla syrup (+0.55)
Milk options: whole, oat, soy (+0.50)
```

Pipe-separated, food takes `-` for sizes, stock is `in stock` or
`OUT OF STOCK`. `finetuning/scripts/validate_dataset.py:parse_menu` is the
reference parser — retrieval output should round-trip through it.
