# Serving check — do the adapters survive the serving stack?

Training and every number before this one ran on a 4-bit base under Unsloth.
The Space serves them differently: unquantised weights, plain `transformers`
and `peft`, no Unsloth in the process at all. Nothing guaranteed the scores
carried over, so this re-runs the hard set through the serving path with the
same scorer.

Notebook: [`../notebooks/serve_check.ipynb`](../notebooks/serve_check.ipynb).

| | trained / measured | served |
|---|---|---|
| Base | `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit` | `Qwen/Qwen3-4B-Instruct-2507` |
| Precision | 4-bit NF4, fp16 compute | bf16 |
| Stack | Unsloth → PEFT → TRL | `transformers` + `peft` |
| Hardware | Colab T4 | Colab T4 (Space runs H200) |

Tokenizer came from the training repo in both cases. A different chat template
would mean the adapters were reading a prompt shape they had never seen, which
would confound the comparison with the thing being measured.

## Hard set, 4-bit against served

| metric | friendly 4-bit | friendly served | diff | blunt 4-bit | blunt served | diff |
|---|---|---|---|---|---|---|
| format_ok | 100.0% | 100.0% | — | 100.0% | 100.0% | — |
| restraint | 97.3% | 100.0% | +2.7 | 100.0% | 100.0% | — |
| grounded | 100.0% | 95.7% | −4.3 | 100.0% | 100.0% | — |
| valid_slots | 95.8% | 95.7% | −0.1 | 100.0% | 100.0% | — |
| exact_match | 91.9% | 89.2% | −2.7 | 94.6% | 91.9% | −2.7 |

One example is 2.7 points on a 37-dialogue set, so every cell here is within
one or two examples. Dropping the quantisation and the training framework did
not move behaviour measurably. The Space serves the adapters as measured.

## Hot-swap

`model.set_adapter()` between the two voices, both resident in one process:

```
friendly  17.1 ms
blunt     14.0 ms
```

That is the number the README's VRAM argument depends on. Merging the adapters
instead would make this a model reload — tens of seconds, and a separate 8 GB
of weights per voice.

## Tone

The first measurement of it. `exact_match` compares order items, which the two
voices are supposed to agree on, so it is tone-blind by construction;
`scripts/tone_eval.py` scores the prose instead.

```
discriminability  81.1% ± 7.7%   (50% = indistinguishable)
```

A cross-validated char-n-gram classifier separates the two voices' replies four
times in five. The interpretable features say what it is separating:

| feature | friendly | blunt |
|---|---|---|
| chars | 46.51 | 39.68 |
| words | 8.73 | 7.43 |
| sentences | 1.30 | 1.54 |
| contractions per 100w | 1.86 | 1.82 |
| dashes per turn | 0.41 | 0.30 |
| politeness markers per turn | 0.46 | **0.03** |
| questions per turn | 0.38 | 0.32 |

Politeness carries most of it — fifteen times more markers in friendly. The
second signal is shape rather than length: blunt says less but in *more*
sentences, which is fragments ("Large americano. Done.") against friendly's one
flowing clause.

**What this does not prove — and the confound in it.** Each voice was scored
under *its own* system prompt, because that is what it was trained with. Those
prompts are not neutral:

| friendly | blunt |
|---|---|
| Be warm and concise. Two sentences at most before acting. | Be blunt. No pleasantries, no filler. Fragments are fine. |
| — | One short line before acting, never more. |

A base model with no adapter follows those lines perfectly well. So 81.1%
measures two prompts *and* two adapters together and cannot separate them. It
is not yet evidence that fine-tuning taught tone.

`shared/system_prompt.txt` is the same prompt with every instruction about
tone stripped out — brand and menu kept, nothing about how to sound. Both the
Space and `prompt_messages(record, neutral=True)` now use it, and section 6b of
the notebook re-scores under it. Whatever survives there is the adapter's.

The adapters were trained *with* the tone lines, so this is a shift away from
what they saw.

**First reading, by eye: the difference largely goes.** Driving the Space by
hand under the neutral prompt, the two adapters are hard to tell apart. So most
of the 81.1% was the prompt. What fine-tuning demonstrably taught is the
tool-call format and the ordering discipline — `format_ok` and `exact_match`
beat the armed baseline by 22 and 30 points, and those were never in the
prompt. Tone is not yet among its results.

The design error underneath is worth naming, because it is the kind that
produces a number rather than an error message: **the variable under test was
put in the input.** Two voices that differ in their system prompt cannot be
compared for what their weights learned, since the prompt alone explains the
difference. Voices must differ only in what the model is trained to say, never
in what it is given to read.

The fix is a corpus change, not a serving change: the system prompt moves into
`data/templates/_shared.yaml` so every voice is generated against the same one,
and both adapters are retrained. Until that happens the published adapters
still carry the old prompt in their training distribution, so the repo's data
is left matching them rather than silently drifting ahead.

Beyond the confound: the training dialogues are template-generated, so a
classifier separating them may be picking out memorised phrasings rather than a
learned style, and no number here says whether either voice sounds like a brand
a person would recognise. That is still judged by eye.

## Superlative sizes: a thin patch, not a broken voice

Blunt read "chai latte pls — the big one — w oat" off a `Chai Latte | M,L`
menu and ordered `size: "M"`. Friendly got it right.

That looks like a blunt-specific defect and is not one. The two corpora teach
this pattern identically:

| | friendly | blunt |
|---|---|---|
| superlative-size examples | 26 / 900 | 26 / 900 |
| size chosen | L×22, M×3, S×1 | L×22, M×3, S×1 |
| menu shapes | M,L×14 · S,M,L×9 · S,M×3 | identical |

The hard set holds two such cases. Friendly got 2/2, blunt 1/2 — at n=2 that
is a coin toss, not a difference. The finding is that **2.9% of the corpus
teaches comparative sizing and neither voice learned it robustly**; friendly
happened to be lucky here.

The failing prompt is also compound — em-dashes, "w oat", and the superlative
together — so whether the superlative fails alone or only under noise is
unseparated. Fixing it properly means raising the share in both templates and
writing enough hard-set cases to tell the fix from luck. Retraining blunt alone
would be chasing noise.

Worth noting alongside it: run 003 recorded this same example failing
differently, with `milk: "whole"` when the customer said oat. The milk is right
now and the size is wrong. A marginal example failing in a different place each
time is itself the evidence that it sits on the model's decision boundary.
