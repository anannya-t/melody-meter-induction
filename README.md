# Melody-only meter induction: Stage 1 code

Symbolic-melody baseline for the "Melody-Only Meter Induction" project note.
It reproduces the autocorrelation approach of Toiviainen & Eerola (2006) on the
Essen Folksong Collection subset bundled with music21, then tests which accent
types add information.

Everything here works on symbolic scores (notes with onset, duration, pitch).
There is no audio and no tala yet.

## Setup

```
pip install -r requirements.txt
```

Run every command from inside this folder. The scripts import each other, and
the parsed corpus is cached in `./essen_cache/` (created on first run).

## What each file does

| File | Role |
| --- | --- |
| `onset_autocorrelation.py` | Core library: `Melody`/`Note` classes, onset impulse train at 1/16-note sampling, autocorrelation, music21 corpus loading, and the original single-peak classifier with accent experiments on 300 melodies. Run as a script to repeat those early experiments. |
| `train_meter_classifier.py` | Main model. Builds 96 features per melody (6 accent types x autocorrelation at 16 lags), then cross-validates LDA, logistic regression and random forest on three tasks: duple vs triple/compound, 12 notated meters, and beat-normalized meter family. Writes a confusion matrix PNG. |
| `experiment2_accents.py` | Experiment 2. Compares accent sets incrementally (constant only, + duration, + melodic accents, all six) and leave-one-accent-out, over several CV seeds. |
| `make_report_figures.py` | Figure 1 of the report: onset trains and autocorrelation for one melody each in 4/4, 3/4 and 6/8. Needs `essen_cache/` from a prior run of `train_meter_classifier.py`. |
| `results/` | Saved outputs from the runs reported: `train_meter_classifier_output.log`, `experiment2_accents_output.log`, `meter_confusion_matrix.png`. |

## Run order

```
python train_meter_classifier.py --outdir results   # parses corpus (first run), trains, plots
python experiment2_accents.py --seeds 5             # accent comparison
python make_report_figures.py                       # report Figure 1
```

Rough timing on 2 CPU cores: the first corpus parse takes several minutes, the
classifier run a few minutes after that, and `experiment2_accents.py` with 5
seeds about 40 minutes (use `--seeds 1` for a quick pass).
`train_meter_classifier.py --max-files 3` runs on a small subset.

## Known differences from the paper

- Classifiers and CV: LDA, logistic regression and random forest with stratified
  10-fold CV, not stepwise discriminant analysis with leave-one-out.
- Data: music21's bundled Essen subset (31 files, 7,369 single-meter melodies),
  not the full Essen and Finnish collections.
- The "melodic" accent is a simplified turning-point version, not Thomassen's
  model.
- Not yet checked: near-duplicate tune variants landing in both train and test
  folds, which could inflate accuracy.
