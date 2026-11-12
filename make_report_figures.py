"""Figure 1 for the progress report: onset impulse train and autocorrelation for
one melody in each of three meters (4/4, 3/4, 6/8), from the cached Essen corpus.

Melodies are not hand-picked for clean peaks: for each meter we take the first
melody (in cache-file order) with 40 to 100 notes.
"""
import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import onset_autocorrelation as oa

METERS = [(4, 4), (3, 4), (6, 8)]
RES = oa.SIXTEENTH_NOTE          # quarter-note beats per sample
FEATURE_LAGS_EIGHTHS = (1, 16)   # lags used as classifier features
BLUE = "#2b78d6"

picked = {}
for path in sorted(glob.glob("essen_cache/*.pkl")):
    with open(path, "rb") as fh:
        for m in pickle.load(fh):
            if m.meter in METERS and m.meter not in picked and 40 <= len(m.notes) <= 100:
                picked[m.meter] = m
    if len(picked) == len(METERS):
        break

fig, axes = plt.subplots(len(METERS), 2, figsize=(11, 7.2),
                         gridspec_kw={"width_ratios": [1.15, 1]})
for row, meter in enumerate(METERS):
    m = picked[meter]
    onsets = oa.extract_onsets(m)
    imp = oa.build_impulse_train(onsets, m.total_beats(), RES)
    ac = oa.autocorrelation(imp)
    measure = m.measure_length_quarters()
    t = np.arange(len(imp)) * RES
    lags = np.arange(len(ac)) * RES

    ax = axes[row, 0]
    ax.vlines(t[imp > 0], 0, 1, color=BLUE, linewidth=1.2)
    for b in np.arange(0, t[-1] + 1e-9, measure):
        ax.axvline(b, color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    ax.set_title(f"{m.name} ({meter[0]}/{meter[1]}): note onsets", fontsize=10, loc="left")
    if row == len(METERS) - 1:
        ax.set_xlabel("time (quarter notes); grey lines = barlines")

    ax = axes[row, 1]
    keep = lags <= 12
    ax.axvspan(FEATURE_LAGS_EIGHTHS[0] / 2, FEATURE_LAGS_EIGHTHS[1] / 2,
               color="#eeeeee", zorder=0)
    ax.plot(lags[keep], ac[keep], color="#444444", linewidth=1.2)
    ax.axvline(measure, color=BLUE, linestyle="--", linewidth=1.3,
               label=f"one bar = {measure:g} quarter notes")
    ax.set_ylim(0, 1.05)
    ax.set_title("autocorrelation", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    if row == len(METERS) - 1:
        ax.set_xlabel("lag (quarter notes); shaded = lags used as features")

fig.tight_layout()
fig.savefig("fig_autocorr_examples.png", dpi=150)
print("saved", {f"{k[0]}/{k[1]}": v.name for k, v in picked.items()})
