"""Measure whether two adapters actually speak differently.

Behaviour metrics (evaluate.py) are tone-blind on purpose: exact_match compares
order items, not prose, so a voice swap must not move them. This module scores
the prose instead, on generations from the same prompts.

Two readings:

    discriminability  can a classifier tell the voices apart from the text
                      alone? Chance is 50%. High means the tone transferred.
    style features    interpretable proxies -- length, contractions, dashes,
                      politeness markers -- so the number has a shape.

A caveat worth stating in the write-up: a high score proves the outputs differ,
not that either matches a real brand. With template-generated training data,
a classifier may be separating memorised phrasings rather than a learned style.
"""

import re
import statistics

TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
CONTRACTION = re.compile(r"\b\w+'(s|t|re|ll|ve|d|m)\b", re.IGNORECASE)

POLITENESS = [
    "please", "thank", "of course", "sure thing", "happy to", "absolutely",
    "perfect", "great", "sorry", "afraid", "wonderful", "certainly",
]


def prose(text):
    """The spoken part of a turn, with any tool call removed."""
    return TOOL_CALL.sub("", text).strip()


def features(outputs):
    """Interpretable style statistics over one adapter's generations."""
    texts = [prose(o) for o in outputs]
    texts = [t for t in texts if t]
    if not texts:
        return {}

    words = [len(t.split()) for t in texts]
    return {
        "chars": statistics.mean(len(t) for t in texts),
        "words": statistics.mean(words),
        "sentences": statistics.mean(max(1, len(re.findall(r"[.!?]", t))) for t in texts),
        "contractions_per_100w": 100 * sum(len(CONTRACTION.findall(t)) for t in texts) / max(1, sum(words)),
        "dashes_per_turn": statistics.mean(t.count("—") + t.count(" - ") for t in texts),
        "politeness_per_turn": statistics.mean(
            sum(t.lower().count(marker) for marker in POLITENESS) for t in texts),
        "questions_per_turn": statistics.mean(t.count("?") for t in texts),
    }


def discriminability(outputs_a, outputs_b, folds=5):
    """Cross-validated accuracy of telling the two voices apart.

    Character n-grams rather than words: tone lives in punctuation and clipped
    endings as much as in vocabulary.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    texts = [prose(o) for o in outputs_a] + [prose(o) for o in outputs_b]
    labels = [0] * len(outputs_a) + [1] * len(outputs_b)
    keep = [i for i, t in enumerate(texts) if t]
    texts = [texts[i] for i in keep]
    labels = [labels[i] for i in keep]

    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
        LogisticRegression(max_iter=1000),
    )
    scores = cross_val_score(model, texts, labels, cv=folds, scoring="accuracy")
    return statistics.mean(scores), statistics.pstdev(scores)


def report(name_a, outputs_a, name_b, outputs_b):
    lines = []
    try:
        mean, spread = discriminability(outputs_a, outputs_b)
        lines.append(f"discriminability  {mean:.1%} ± {spread:.1%}   (50% = indistinguishable)")
    except ImportError:
        lines.append("discriminability  skipped (scikit-learn not installed)")

    a, b = features(outputs_a), features(outputs_b)
    lines += ["", f"  {'feature':24}{name_a:>12}{name_b:>12}"]
    for key in a:
        lines.append(f"  {key:24}{a[key]:>12.2f}{b[key]:>12.2f}")
    return "\n".join(lines)
