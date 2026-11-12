"""
Experiment 2: do duration and melodic accents improve meter classification?
===========================================================================

Project note, experiment 2: "Add duration and melodic-accent functions and
quantify whether they improve meter classification."

Uses the same features and CV as train_meter_classifier.py, but compares
accent sets *incrementally* rather than one-at-a-time or all-at-once:

  A. constant only                      (plain onsets, the baseline)
  B. constant + duration                (Brown 1993's weighting added)
  C. constant + melodic accents         (interval, pivotal, contour, melodic)
  D. constant + duration + melodic      (= all six)

plus a leave-one-accent-out ablation from the full six, to see which accent
the full model would miss most.

Each configuration is run over several CV seeds so differences can be judged
against run-to-run noise (mean +/- std across seeds). All accents here are
the paper's six; "melodic" is our simplified stand-in for Thomassen's accent,
not a reproduction of it.

Usage:  python experiment2_accents.py [--seeds 5]
"""

import argparse
import collections

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import train_meter_classifier as tm

MELODIC = ["interval", "pivotal", "contour", "melodic"]

CONFIGS = [
    ("A  constant only", ["constant"]),
    ("B  constant + duration", ["constant", "duration"]),
    ("C  constant + melodic accents", ["constant"] + MELODIC),
    ("D  constant + duration + melodic (all six)", ["constant", "duration"] + MELODIC),
]

ABLATIONS = [(f"all six minus {a}", [x for x in tm.ACCENT_TYPES if x != a])
             for a in tm.ACCENT_TYPES]


def run(model_name, X, y, seeds):
    accs, bals = [], []
    for s in range(seeds):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)
        pred = tm.make_models()[model_name]
        pred = cross_val_predict(pred, X, y, cv=cv)
        accs.append(np.mean(pred == y))
        bals.append(balanced_accuracy_score(y, pred))
    return np.array(accs), np.array(bals)


def report(title, rows):
    print(f"\n{title}")
    print(f"{'configuration':<46}{'accuracy':>16}{'balanced':>16}")
    for name, (accs, bals) in rows:
        print(f"{name:<46}{accs.mean():>8.1%} ±{accs.std():.1%}{bals.mean():>9.1%} ±{bals.std():.1%}")


def main(seeds):
    melodies = tm.load_all_essen()
    print(f"{len(melodies)} melodies")
    table = tm.build_feature_table(melodies, normalize=False)

    tasks = {}
    labels = [tm.duple_or_triple(m) for m in melodies]
    keep = np.array([l is not None for l in labels])
    tasks["Duple vs triple/compound"] = (keep, np.array(labels)[keep])
    labels = [tm.notated_meter(m) for m in melodies]
    keep = tm.drop_rare(labels)
    tasks["12 notated meters"] = (keep, np.array(labels)[keep])

    for task, (keep, y) in tasks.items():
        base = tm.majority_baseline(y)
        print("\n" + "=" * 78)
        print(f"{task}: n={len(y)}, majority baseline {base[1]:.1%}, {seeds} CV seeds")
        print("=" * 78)
        for model_name in ["LDA", "LogReg"]:
            rows = []
            for name, accents in CONFIGS:
                rows.append((name, run(model_name, tm.stack(table, accents)[keep], y, seeds)))
            report(f"[{model_name}] incremental accent sets", rows)
            rows = []
            for name, accents in ABLATIONS:
                rows.append((name, run(model_name, tm.stack(table, accents)[keep], y, seeds)))
            report(f"[{model_name}] leave-one-accent-out from all six", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    main(ap.parse_args().seeds)
