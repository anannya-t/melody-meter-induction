"""
Train meter classifiers on multi-lag autocorrelation features
==============================================================

Follows the design of Toiviainen & Eerola (2006), "Autocorrelation in Meter
Induction: The Role of Accent Structure" (JASA 119(2)), instead of the
hand-written "pick the strongest candidate lag" rule used in
onset_autocorrelation.py:

  * Each melody -> six onset impulse functions, one per accent type
    (constant, duration, interval size, pivotal, gross contour, melodic),
    sampled at 1/16 note.
  * Each impulse function -> normalized autocorrelation, keeping the values
    at lags of 1..16 eighth notes (16 features per accent type, 96 total).
  * A *trained*, cross-validated classifier maps those features to meter.

Tasks (as in the paper):
  Study 1: binary   duple vs. triple/compound
  Study 2: multi-class notated meter (classes with > 30 exemplars)
Extra (this project): beat-normalized features -> meter family (2/3/4/6).

Differences from the paper (so results are not over-claimed):
  * Paper used stepwise discriminant analysis + leave-one-out CV; here we use
    plain LDA / logistic regression / random forest with stratified 10-fold CV.
  * Paper's "melodic" accent is Thomassen's (1982) empirical model; here it is
    a simplified turning-point accent (onset_autocorrelation.contour_accent),
    a stand-in, not a reproduction.
  * Corpus: the Essen subset bundled with music21 (ABC), not the full Kern
    collection or the Finnish collection.

Usage:
    pip install music21 scikit-learn matplotlib
    python train_meter_classifier.py            # all 31 bundled Essen files
    python train_meter_classifier.py --max-files 3   # quick test
"""

import argparse
import collections
import os
import pickle
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import onset_autocorrelation as oa

warnings.filterwarnings("ignore")

RESOLUTION = oa.SIXTEENTH_NOTE            # 0.25 quarter-note beats
EIGHTH_LAGS = np.arange(1, 17)            # lags of 1..16 eighth notes
SAMPLE_LAGS = EIGHTH_LAGS * 2             # 1 eighth = 2 sixteenth-note samples
ACCENT_TYPES = ["constant", "duration", "interval", "pivotal", "contour", "melodic"]
MIN_EXEMPLARS = 30                        # paper: only meters with > 30 exemplars

# Classification rates reported in the paper for the Essen collection
PAPER_STUDY1_ESSEN = {"constant": 91.5, "duration": 90.5, "melodic": 86.8,
                      "interval": 85.9, "pivotal": 78.5, "contour": 73.1,
                      "ALL": 95.3}
PAPER_STUDY2_ESSEN = 83.2


# ---------------------------------------------------------------------------
# Data loading (cached, since parsing all Essen files takes a few minutes)
# ---------------------------------------------------------------------------

def _load_file(path, cache_dir):
    """Parse one Essen ABC file tune-by-tune (bounded memory) with a per-file
    cache, so an interrupted run resumes where it left off."""
    import re
    from music21 import converter
    cache_file = os.path.join(cache_dir, path.stem + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    text = path.read_text(encoding="utf-8", errors="replace")
    tunes = [t for t in re.split(r"(?m)^(?=X:)", text) if t.strip().startswith("X:")]
    melodies = []
    for tune in tunes:
        try:
            score = converter.parse(tune, format="abc")
            if hasattr(score, "scores") and len(score.scores):
                score = score.scores[0]
            title = (score.metadata.title if score.metadata and score.metadata.title
                     else None)
            number = re.match(r"X:\s*(\d+)", tune)
            name = str(title) if title else f"{path.stem}_{number.group(1) if number else len(melodies)}"
            melody = oa._score_to_melody(score, name=name)
            if melody is not None:
                melodies.append(melody)
        except Exception:
            continue  # skip any tune music21 cannot parse
    with open(cache_file, "wb") as fh:
        pickle.dump(melodies, fh)
    return melodies


def load_all_essen(max_files=None, cache_dir="essen_cache", use_cache=True):
    from music21 import corpus
    os.makedirs(cache_dir, exist_ok=True)
    paths = corpus.getComposer("essenFolksong")
    if max_files:
        paths = paths[:max_files]

    melodies = []
    for k, path in enumerate(paths, 1):
        if not use_cache:
            cached = os.path.join(cache_dir, path.stem + ".pkl")
            if os.path.exists(cached):
                os.remove(cached)
        melodies.extend(_load_file(path, cache_dir))
        print(f"  file {k}/{len(paths)} ({path.name}); "
              f"{len(melodies)} usable melodies so far", flush=True)
    return melodies


# ---------------------------------------------------------------------------
# Accent signals -> impulse functions -> autocorrelation features
# ---------------------------------------------------------------------------

def accent_signals(notes):
    """The six per-note accent values a_i used to weight each onset impulse."""
    pitches = np.array([n.pitch for n in notes], dtype=float)
    durations = np.array([n.duration for n in notes], dtype=float)
    n = len(notes)
    steps = np.diff(pitches, prepend=pitches[0])  # step[0] = 0 (no previous note)

    pivotal = np.zeros(n)
    for i in range(1, n - 1):
        if (pitches[i] - pitches[i - 1]) * (pitches[i + 1] - pitches[i]) < 0:
            pivotal[i] = 1.0  # melody changes direction at this note

    return {
        "constant": np.ones(n),            # a_i = 1
        "duration": durations,              # a_i = tone duration (Brown 1993)
        "interval": np.abs(steps),          # semitones from previous tone
        "pivotal": pivotal,                 # 1 if direction change, else 0
        "contour": np.sign(steps),          # +1 up, -1 down, 0 repeat
        "melodic": oa.contour_accent(notes),  # simplified stand-in for Thomassen
    }


def impulse_function(notes, weights):
    """f(n) = sum_i a_i * delta(n - round(t_i / dt)), summed (paper's Eq. 1)."""
    idx = np.rint(np.array([n.onset for n in notes]) / RESOLUTION).astype(int)
    f = np.zeros(idx.max() + 1)
    np.add.at(f, idx, weights)
    return f


def lag_features(f):
    """Normalized autocorrelation at lags of 1..16 eighth notes."""
    energy = float(np.dot(f, f))
    out = np.zeros(len(SAMPLE_LAGS))
    if energy == 0:
        return out
    ac = np.correlate(f, f, mode="full")[len(f) - 1:] / energy
    ok = SAMPLE_LAGS < len(ac)
    out[ok] = ac[SAMPLE_LAGS[ok]]
    return out


def build_feature_table(melodies, normalize=False):
    """Returns {accent_type: (n_melodies, 16) array}."""
    table = {a: [] for a in ACCENT_TYPES}
    for melody in melodies:
        m = melody.normalized() if normalize else melody
        notes = oa.extract_sorted_notes(m)
        signals = accent_signals(notes)
        for a in ACCENT_TYPES:
            table[a].append(lag_features(impulse_function(notes, signals[a])))
    return {a: np.vstack(v) for a, v in table.items()}


def stack(table, accents):
    return np.hstack([table[a] for a in accents])


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def notated_meter(m):
    return f"{m.meter[0]}/{m.meter[1]}"


def duple_or_triple(m):
    """Paper's grouping: duple = 2^n eighths/measure; triple/compound = 3n."""
    num = m.meter[0]
    if num in (2, 4, 8):
        return "duple"
    if num % 3 == 0:
        return "triple/compound"
    return None


def drop_rare(labels, min_count=MIN_EXEMPLARS):
    counts = collections.Counter(l for l in labels if l is not None)
    return np.array([l is not None and counts[l] > min_count for l in labels])


# ---------------------------------------------------------------------------
# Models + evaluation
# ---------------------------------------------------------------------------

def make_models():
    return {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "RandForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    }


def cv_predict(model, X, y, seed=0):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv)


def majority_baseline(y):
    counts = collections.Counter(y)
    label, n = counts.most_common(1)[0]
    return label, n / len(y)


def accuracy(y, pred):
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_binary(melodies, table):
    labels = [duple_or_triple(m) for m in melodies]
    keep = np.array([l is not None for l in labels])
    y = np.array(labels)[keep]
    base_label, base_acc = majority_baseline(y)

    print("\n" + "=" * 78)
    print(f"STUDY 1 (binary): duple vs triple/compound   n={len(y)}   "
          f"majority baseline ({base_label}) = {base_acc:.1%}")
    print("Raw (un-normalized) features, LDA, stratified 10-fold CV")
    print("=" * 78)
    print(f"{'Accent type':<22}{'Ours':>8}{'Paper (Essen)':>16}")

    lda = make_models()["LDA"]
    results = {}
    for a in ACCENT_TYPES:
        pred = cv_predict(lda, table[a][keep], y)
        results[a] = accuracy(y, pred)
        label = "melodic*" if a == "melodic" else a
        print(f"{label:<22}{results[a]:>8.1%}{PAPER_STUDY1_ESSEN[a]:>15.1f}%")

    X_all = stack(table, ACCENT_TYPES)[keep]
    for name, model in make_models().items():
        pred = cv_predict(model, X_all, y)
        results[f"ALL/{name}"] = accuracy(y, pred)
        paper = f"{PAPER_STUDY1_ESSEN['ALL']:.1f}%" if name == "LDA" else ""
        print(f"{'ALL 6 combined / ' + name:<22}{results[f'ALL/{name}']:>8.1%}{paper:>16}")
    print("* melodic = simplified stand-in for Thomassen's accent, not the original model")
    return results


def experiment_multiclass(melodies, table, outdir):
    labels = [notated_meter(m) for m in melodies]
    keep = drop_rare(labels)
    y = np.array(labels)[keep]
    classes = sorted(set(y), key=lambda c: -np.sum(y == c))
    base_label, base_acc = majority_baseline(y)

    print("\n" + "=" * 78)
    print(f"STUDY 2 (multi-class notated meter): {len(classes)} classes with >{MIN_EXEMPLARS} "
          f"exemplars, n={len(y)}")
    print(f"majority baseline ({base_label}) = {base_acc:.1%}   "
          f"paper (Essen, LDA, all accents) = {PAPER_STUDY2_ESSEN}%")
    print("Raw (un-normalized) features, all 96 features, stratified 10-fold CV")
    print("=" * 78)
    print(f"{'Model':<14}{'Accuracy':>10}{'Balanced':>10}{'Macro-F1':>10}")

    X = stack(table, ACCENT_TYPES)[keep]
    preds, results = {}, {}
    for name, model in make_models().items():
        pred = cv_predict(model, X, y)
        preds[name] = pred
        results[name] = {"acc": accuracy(y, pred),
                         "bal": balanced_accuracy_score(y, pred),
                         "f1": f1_score(y, pred, average="macro")}
        r = results[name]
        print(f"{name:<14}{r['acc']:>10.1%}{r['bal']:>10.1%}{r['f1']:>10.3f}")

    best = max(results, key=lambda k: results[k]["acc"])
    print(f"\nPer-class results for best model ({best}):")
    p, r, f, support = precision_recall_fscore_support(y, preds[best], labels=classes, zero_division=0)
    print(f"{'meter':<8}{'n':>6}{'recall':>9}{'precision':>11}{'F':>7}")
    for c, n_c, rc, pr, ff in zip(classes, support, r, p, f):
        print(f"{c:<8}{n_c:>6}{rc:>9.2f}{pr:>11.2f}{ff:>7.2f}")

    cm = confusion_matrix(y, preds[best], labels=classes)
    plot_confusion(cm, classes, f"Notated meter, {best} (acc {results[best]['acc']:.1%})",
                   os.path.join(outdir, "meter_confusion_matrix.png"))
    return results


def experiment_normalized_family(melodies, outdir):
    print("\n" + "=" * 78)
    print("EXTRA: beat-normalized features -> meter family (beats per measure)")
    print("=" * 78)
    table = build_feature_table(melodies, normalize=True)
    labels = [str(m.meter[0]) for m in melodies]
    keep = drop_rare(labels)
    y = np.array(labels)[keep]
    base_label, base_acc = majority_baseline(y)
    print(f"classes: {sorted(set(y))}  n={len(y)}   majority baseline ({base_label}) = {base_acc:.1%}")

    # Previous hand-written rule on the same melodies, for a before/after view
    kept = [m for m, k in zip(melodies, keep) if k]
    rule = oa.evaluate_corpus(kept, normalize=True, verbose=False)
    print(f"Previous rule (strongest candidate lag): strict {rule['strict_accuracy']:.1%}, "
          f"balanced {rule['balanced_accuracy']:.1%}")

    X = stack(table, ACCENT_TYPES)[keep]
    print(f"{'Model':<14}{'Accuracy':>10}{'Balanced':>10}{'Macro-F1':>10}")
    results = {}
    for name, model in make_models().items():
        pred = cv_predict(model, X, y)
        results[name] = {"acc": accuracy(y, pred),
                         "bal": balanced_accuracy_score(y, pred),
                         "f1": f1_score(y, pred, average="macro")}
        r = results[name]
        print(f"{name:<14}{r['acc']:>10.1%}{r['bal']:>10.1%}{r['f1']:>10.3f}")
    results["rule"] = rule
    return results


def plot_confusion(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted meter")
    ax.set_ylabel("notated meter")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="fraction of notated meter")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None,
                        help="limit number of Essen files parsed (default: all 31)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    melodies = load_all_essen(args.max_files, use_cache=not args.no_cache)
    print(f"\n{len(melodies)} melodies with a single, well-defined time signature")
    print("Meter counts:", dict(collections.Counter(notated_meter(m) for m in melodies).most_common()))

    print("\nBuilding autocorrelation features (6 accent types x 16 lags) ...")
    table = build_feature_table(melodies, normalize=False)

    experiment_binary(melodies, table)
    experiment_multiclass(melodies, table, args.outdir)
    experiment_normalized_family(melodies, args.outdir)
