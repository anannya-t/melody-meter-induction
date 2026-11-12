"""
Onset Impulse Autocorrelation Analysis for Symbolic Melodies
==============================================================

This program:
  1. Takes symbolic melodies as input (lists of notes with onset times,
     durations, and a meter).
  2. Extracts note onset times from each melody.
  3. Builds an "onset impulse function": a signal sampled at a fixed
     interval of one sixteenth note, where every sample that coincides
     with a note onset gets a constant weight of 1 (all other samples
     are 0). No metrical/melodic accent weighting is applied at this
     stage -- every onset counts the same.
  4. Computes the autocorrelation of each impulse function to reveal
     recurring rhythmic periodicities.
  5. Plots the impulse trains and their autocorrelations, and compares
     the periodicities found across melodies written in different
     meters (e.g. 4/4, 3/4, 6/8).

Time convention
----------------
All onset/duration values are expressed in quarter-note beats (a
quarter note = 1.0 "beat unit"), regardless of the notated meter. The
sampling interval is fixed at one sixteenth note, i.e. 0.25 of a
quarter-note beat, so there are exactly 4 samples per quarter-note
beat for every melody. This keeps the impulse trains of melodies in
different meters directly comparable on a common time grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import collections
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Symbolic melody representation
# ---------------------------------------------------------------------------

@dataclass
class Note:
    onset: float      # onset time in quarter-note beats, measured from t=0
    duration: float   # duration in quarter-note beats
    pitch: int = 60    # MIDI pitch (not used for rhythm, kept for completeness)


@dataclass
class Melody:
    name: str
    meter: tuple                 # (beats_per_measure, beat_unit), e.g. (4, 4)
    notes: list = field(default_factory=list)

    def total_beats(self) -> float:
        """Length of the melody in quarter-note beats, from 0 to the end
        of the last note (rounded up to the next full measure)."""
        beats_per_measure, beat_unit = self.meter
        # convert "beats_per_measure of beat_unit" into quarter-note beats
        measure_len_in_quarters = beats_per_measure * (4.0 / beat_unit)
        last_offset = max(n.onset + n.duration for n in self.notes)
        n_measures = int(np.ceil(last_offset / measure_len_in_quarters))
        return n_measures * measure_len_in_quarters

    def measure_length_quarters(self) -> float:
        beats_per_measure, beat_unit = self.meter
        return beats_per_measure * (4.0 / beat_unit)

    def normalized(self) -> "Melody":
        """
        Rescale this melody so its notated beat (not the quarter note) is
        the unit of 1. This collapses mensural-notation artifacts: a 4/2
        piece (half-note beat) and a 4/4 piece (quarter-note beat) can
        represent the same felt rhythm written at different note-value
        scales -- in this corpus, mean note duration scales almost exactly
        with the denominator (confirmed empirically: ~1.17 quarter-lengths
        in 4/4 vs ~2.42 in 4/2 vs ~4.16 in 4/1). After normalization, every
        meter's measure length is simply its numerator (beats_per_measure),
        making different denominators directly comparable.
        """
        beats_per_measure, beat_unit = self.meter
        scale = beat_unit / 4.0  # converts quarter-length units into beat units
        new_notes = [Note(onset=n.onset * scale, duration=n.duration * scale, pitch=n.pitch)
                     for n in self.notes]
        return Melody(name=self.name, meter=(beats_per_measure, 4), notes=new_notes)


# ---------------------------------------------------------------------------
# 1b. Loading real-world melodies (music21 / Essen Folksong Collection)
# ---------------------------------------------------------------------------
#
# `music21` ships with a bundled subset of the real Essen Folksong
# Collection -- the same corpus used in Toiviainen & Eerola (2006), the
# paper this whole pipeline is modeled on. No download needed: it comes
# with the `music21` package (`pip install music21`).
#
# These loaders turn any music21-readable score (ABC, MusicXML, MIDI,
# **kern, etc.) into the same `Melody`/`Note` objects used above, so the
# rest of the pipeline (impulse train, autocorrelation, plotting) does
# not need to change at all.

def _score_to_melody(score, name: str):
    """
    Convert a single music21 Score/Part into a Melody, or return None if
    the score doesn't have a single, well-defined time signature (skips
    pieces with meter changes, which the simple periodicity model here
    isn't designed to handle).
    """
    from music21 import meter as m21_meter, note as m21_note, chord as m21_chord

    flat = score.flatten()
    time_sigs = list(flat.getElementsByClass(m21_meter.TimeSignature))
    if len(time_sigs) != 1:
        return None  # skip pieces with no / multiple time signatures

    ts = time_sigs[0]
    # music21 gives beat unit as a quarter-length fraction; convert to the
    # traditional "denominator" (4 = quarter, 8 = eighth, 2 = half...)
    beats_per_measure = ts.numerator
    beat_unit = int(round(4.0 / ts.beatDuration.quarterLength)) if False else ts.denominator

    notes = []
    for el in flat.notesAndRests:
        if isinstance(el, m21_note.Note):
            pitch_midi = el.pitch.midi
        elif isinstance(el, m21_chord.Chord):
            pitch_midi = max(p.midi for p in el.pitches)  # top note of the chord
        else:
            continue
        notes.append(Note(onset=float(el.offset), duration=float(el.quarterLength),
                           pitch=int(pitch_midi)))

    if len(notes) < 4:
        return None  # too short to say anything about periodicity

    return Melody(name=name, meter=(beats_per_measure, beat_unit), notes=notes)


def load_essen_folksong_corpus(max_files: int = 3, max_melodies: int = 200):
    """
    Load real melodies (with real, human-annotated meters) from the
    Essen Folksong Collection subset bundled with music21.

    Parameters
    ----------
    max_files : how many of the corpus's .abc files to read (each file
        contains dozens to hundreds of individual tunes).
    max_melodies : overall cap on the number of melodies returned, so a
        first run stays fast.
    """
    from music21 import corpus, converter

    paths = corpus.getComposer("essenFolksong")
    melodies = []

    for file_path in paths[:max_files]:
        opus = converter.parse(file_path)
        for i, score in enumerate(opus.scores):
            title = score.metadata.title if score.metadata and score.metadata.title else f"{file_path.stem}_{i}"
            melody = _score_to_melody(score, name=str(title))
            if melody is not None:
                melodies.append(melody)
            if len(melodies) >= max_melodies:
                return melodies

    return melodies


def load_melody_from_file(path: str):
    """
    Load a single melody from a symbolic music file on disk -- MIDI
    (.mid), MusicXML (.xml/.musicxml), **kern (.krn), ABC (.abc), or
    anything else music21 can parse. Returns a Melody, or None if the
    file has no single well-defined time signature.
    """
    from music21 import converter

    score = converter.parse(path)
    return _score_to_melody(score, name=path)


# ---------------------------------------------------------------------------
# 2. Onset extraction
# ---------------------------------------------------------------------------

def extract_sorted_notes(melody: Melody) -> list:
    """Return melody.notes sorted by onset time -- the canonical note order
    used everywhere else, so onsets and any per-note accent weights always
    line up index-for-index."""
    return sorted(melody.notes, key=lambda n: n.onset)


def extract_onsets(melody: Melody) -> np.ndarray:
    """Return the sorted array of note onset times (in quarter-note beats)."""
    return np.array([n.onset for n in extract_sorted_notes(melody)], dtype=float)


# ---------------------------------------------------------------------------
# 2b. Melodic-accent functions
# ---------------------------------------------------------------------------
#
# Each function below scores every note in a melody (in onset order) on
# ONE accent cue, normalized to the range [0, 1] (1 = most accented note
# of that type in the melody). These mirror the cues Toiviainen & Eerola
# and Hannon et al. identify as metrically informative even without
# percussion: note duration, melodic leaps, contour turning points
# (Thomassen's melodic accent model), and registral extremity.

def duration_accent(notes: list) -> np.ndarray:
    """Longer notes are more accented -- a held note tends to mark a
    metrically strong position."""
    durations = np.array([n.duration for n in notes], dtype=float)
    peak = durations.max() if len(durations) else 1.0
    return durations / peak if peak > 0 else np.zeros_like(durations)


def pitch_leap_accent(notes: list) -> np.ndarray:
    """A note arrived at by a large melodic leap (in semitones) tends to
    draw attention and often marks a metrically strong position."""
    pitches = np.array([n.pitch for n in notes], dtype=float)
    leaps = np.abs(np.diff(pitches, prepend=pitches[0] if len(pitches) else 0))
    leaps[0] = 0.0  # no previous note to leap from
    peak = leaps.max()
    return leaps / peak if peak > 0 else np.zeros_like(leaps)


def contour_accent(notes: list) -> np.ndarray:
    """Simplified Thomassen melodic-accent model: a note that is a local
    peak or valley (the melody changes direction there) is accented,
    scaled by how large the surrounding intervals are. Notes in the
    middle of a continuing run (no direction change) get zero accent."""
    pitches = np.array([n.pitch for n in notes], dtype=float)
    n = len(pitches)
    accent = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        before = pitches[i] - pitches[i - 1]
        after = pitches[i + 1] - pitches[i]
        is_turning_point = (before > 0 and after < 0) or (before < 0 and after > 0)
        if is_turning_point:
            accent[i] = (abs(before) + abs(after)) / 2.0
    peak = accent.max()
    return accent / peak if peak > 0 else accent


def registral_accent(notes: list) -> np.ndarray:
    """Notes far from the melody's average pitch (very high or very low)
    are registrally salient."""
    pitches = np.array([n.pitch for n in notes], dtype=float)
    deviation = np.abs(pitches - pitches.mean())
    peak = deviation.max()
    return deviation / peak if peak > 0 else deviation


def phrase_accent(notes: list) -> np.ndarray:
    """
    A note that starts a new phrase -- the very first note of the melody,
    or any note preceded by a silence (an inter-onset gap bigger than its
    predecessor's notated duration) -- is accented, scaled by the size of
    the gap. Phrase beginnings are natural candidates for metrically
    strong positions (compare the Indian concept of 'sam', the cycle's
    point of return, mentioned in the project note).
    """
    onsets = np.array([n.onset for n in notes], dtype=float)
    durations = np.array([n.duration for n in notes], dtype=float)
    n = len(notes)
    accent = np.zeros(n, dtype=float)
    accent[0] = 1.0  # the melody's own opening is always a phrase start
    for i in range(1, n):
        gap = onsets[i] - (onsets[i - 1] + durations[i - 1])
        if gap > 1e-6:
            accent[i] = gap
    peak = accent.max()
    return accent / peak if peak > 0 else accent


def motif_accent(notes: list, motif_length: int = 4) -> np.ndarray:
    """
    Parallelism cue (Temperley & Bartlette): find short melodic-interval
    patterns ("motifs") that recur more than once in the melody, and
    accent every note where a repetition of such a motif begins, scaled
    by how many times it recurs. The idea is that corresponding positions
    in repeated patterns tend to occupy corresponding -- often metrically
    strong -- positions.
    """
    pitches = np.array([n.pitch for n in notes], dtype=float)
    intervals = np.diff(pitches)
    n = len(notes)
    accent = np.zeros(n, dtype=float)

    if len(intervals) >= motif_length:
        pattern_starts = {}
        for i in range(len(intervals) - motif_length + 1):
            pattern = tuple(intervals[i:i + motif_length])
            pattern_starts.setdefault(pattern, []).append(i)
        for pattern, starts in pattern_starts.items():
            if len(starts) >= 2:
                repeat_count = len(starts)
                for start in starts:
                    accent[start] = max(accent[start], repeat_count)

    peak = accent.max()
    return accent / peak if peak > 0 else accent


ACCENT_FUNCTIONS = {
    "duration": duration_accent,
    "leap": pitch_leap_accent,
    "contour": contour_accent,
    "register": registral_accent,
    "phrase": phrase_accent,
    "motif": motif_accent,
}

# Minimum combined weight for any onset. Cues like `phrase` and `motif`
# are legitimately zero for most notes (most notes don't start a phrase
# or a repeated motif), so without a floor, a melody with few/no repeats
# could end up with most of its onsets weighted at exactly 0 -- which is
# equivalent to deleting those onsets from the impulse train entirely.
# The floor keeps every onset present (at reduced weight) regardless of
# which accent cues are selected.
ACCENT_WEIGHT_FLOOR = 0.05


def compute_accent_weights(melody: Melody, components=None) -> np.ndarray:
    """
    Combine one or more melodic-accent cues into a single per-note weight,
    in the same onset order as `extract_onsets`.

    `components`: list of cue names from ACCENT_FUNCTIONS to use (default:
    all six). Each cue contributes an equal-weighted average, then the
    result is rescaled into [ACCENT_WEIGHT_FLOOR, 1.0] so no onset is ever
    weighted at exactly 0 (see ACCENT_WEIGHT_FLOOR above).
    """
    if components is None:
        components = list(ACCENT_FUNCTIONS.keys())

    notes = extract_sorted_notes(melody)
    if not notes:
        return np.array([])

    cue_arrays = [ACCENT_FUNCTIONS[name](notes) for name in components]
    combined = np.mean(cue_arrays, axis=0)
    return ACCENT_WEIGHT_FLOOR + (1.0 - ACCENT_WEIGHT_FLOOR) * combined


# ---------------------------------------------------------------------------
# 3. Onset impulse function
# ---------------------------------------------------------------------------

SIXTEENTH_NOTE = 0.25  # a sixteenth note = 0.25 of a quarter-note beat
SAMPLES_PER_BEAT = int(round(1.0 / SIXTEENTH_NOTE))  # = 4


def build_impulse_train(onsets: np.ndarray, total_beats: float,
                         resolution: float = SIXTEENTH_NOTE,
                         weights: np.ndarray = None) -> np.ndarray:
    """
    Build a discretely sampled impulse function.

    The time axis is sampled every `resolution` quarter-note beats
    (default = one sixteenth note). Every sample that coincides with a
    note onset (rounded to the nearest sample) is set to that onset's
    weight; every other sample is 0.

    `weights`: optional array parallel to `onsets`. If omitted, every
    onset gets a constant weight of 1 (the original, accent-free
    baseline). Pass the output of `compute_accent_weights` to weight
    onsets by melodic accent instead. If two onsets quantize to the same
    sample (e.g. a chord), the larger weight wins.
    """
    n_samples = int(round(total_beats / resolution)) + 1
    impulse = np.zeros(n_samples, dtype=float)

    for i, onset in enumerate(onsets):
        idx = int(round(onset / resolution))
        w = 1.0 if weights is None else weights[i]
        if 0 <= idx < n_samples:
            impulse[idx] = max(impulse[idx], w)

    return impulse


# ---------------------------------------------------------------------------
# 4. Autocorrelation
# ---------------------------------------------------------------------------

def autocorrelation(impulse: np.ndarray) -> np.ndarray:
    """
    Compute the (biased) autocorrelation of a 1-D signal and return only
    the non-negative-lag half, normalized so that lag 0 = 1.0.
    """
    full = np.correlate(impulse, impulse, mode="full")
    mid = len(full) // 2
    non_negative_lags = full[mid:]
    if non_negative_lags[0] != 0:
        non_negative_lags = non_negative_lags / non_negative_lags[0]
    return non_negative_lags


def find_periodicities(autocorr: np.ndarray, resolution: float = SIXTEENTH_NOTE,
                        min_lag_beats: float = 0.2, top_n: int = 5):
    """
    Locate the strongest peaks in the autocorrelation (excluding lag 0)
    and report their lag in beats and in sixteenth-note units.
    """
    min_lag_samples = max(1, int(round(min_lag_beats / resolution)))
    search_region = autocorr[min_lag_samples:]
    peak_idx, props = find_peaks(search_region, height=0.0)
    peak_idx = peak_idx + min_lag_samples

    heights = autocorr[peak_idx]
    order = np.argsort(heights)[::-1][:top_n]
    top_lags_samples = peak_idx[order]
    top_heights = heights[order]

    results = []
    for lag_samples, height in zip(top_lags_samples, top_heights):
        lag_beats = lag_samples * resolution
        results.append({
            "lag_samples_16th": int(lag_samples),
            "lag_beats": lag_beats,
            "correlation": float(height),
        })
    # sort by lag for readable output
    results.sort(key=lambda r: r["lag_beats"])
    return results


def classify_meter(autocorr: np.ndarray, candidate_measure_lengths: list,
                    resolution: float = SIXTEENTH_NOTE) -> float:
    """
    A simple periodicity-based meter classifier, in the spirit of
    Toiviainen & Eerola: given a shortlist of candidate measure lengths
    (in quarter-note beats) seen across the corpus, predict whichever
    candidate has the strongest autocorrelation at that lag. This turns
    "where are the peaks" into an actual testable prediction instead of
    just a picture.
    """
    best_candidate, best_score = None, -np.inf
    for measure_len in candidate_measure_lengths:
        idx = int(round(measure_len / resolution))
        if idx < len(autocorr):
            score = autocorr[idx]
            if score > best_score:
                best_score, best_candidate = score, measure_len
    return best_candidate


def _match_category(pred_len: float, true_len: float, tol: float = 0.05) -> str:
    """Classify a prediction as exactly correct, or as the well-known
    half-measure / double-measure confusion, or as a genuine miss."""
    if true_len <= 0:
        return "other"
    ratio = pred_len / true_len
    if abs(ratio - 1.0) < tol:
        return "correct"
    if abs(ratio - 0.5) < tol:
        return "half"
    if abs(ratio - 2.0) < tol:
        return "double"
    return "other"


def majority_class_baseline(melodies: list) -> dict:
    """What accuracy would a classifier get by always guessing the single
    most common meter in the corpus? Any real method needs to beat this
    to be worth anything on an imbalanced dataset like this one."""
    counts = collections.Counter(f"{m.meter[0]}/{m.meter[1]}" for m in melodies)
    majority_class, majority_n = counts.most_common(1)[0]
    return {"majority_class": majority_class, "accuracy": majority_n / len(melodies) if melodies else 0.0}


def evaluate_corpus(melodies: list, resolution: float = SIXTEENTH_NOTE,
                     accent_components=None, label: str = "Onset-only",
                     normalize: bool = True, verbose: bool = True):
    """
    Run the full pipeline over a list of real melodies with known
    ground-truth meters, and report classification performance.

    `accent_components`: None (default) reproduces the original baseline,
    where every onset gets a constant weight of 1. Pass a list of cue
    names from ACCENT_FUNCTIONS to weight onsets by melodic accent instead.

    `normalize`: if True (default), rescale each melody to its own notated
    beat before classifying (see Melody.normalized()), so e.g. 4/2 and 4/4
    pieces are compared on the same footing instead of being penalized for
    a historical notation-scale difference.

    Reports four things, not just one pooled accuracy number, because
    pooled accuracy is misleading on an imbalanced corpus like this one:
      - strict accuracy   : exact match to the true measure length
      - tolerant accuracy : strict + the well-known half/double-measure
                             confusion counted as a distinct, named
                             near-miss category (not silently "wrong")
      - balanced accuracy : the average of each meter class's own
                             accuracy, so the result isn't dominated by
                             whichever meter happens to be most common
      - majority baseline : what a classifier gets by always guessing the
                             single most common meter (the bar any real
                             method needs to clear)
    """
    eval_melodies = [m.normalized() for m in melodies] if normalize else melodies
    candidate_lengths = sorted({m.measure_length_quarters() for m in eval_melodies})

    n_correct = 0
    match_counts = collections.Counter()
    per_meter_total = {}
    per_meter_correct = {}
    confusions = []

    for orig_melody, melody in zip(melodies, eval_melodies):
        onsets = extract_onsets(melody)
        total_beats = melody.total_beats()
        weights = (compute_accent_weights(melody, accent_components)
                   if accent_components is not None else None)
        impulse = build_impulse_train(onsets, total_beats, resolution, weights=weights)
        autocorr = autocorrelation(impulse)

        true_len = melody.measure_length_quarters()
        pred_len = classify_meter(autocorr, candidate_lengths, resolution)
        category = _match_category(pred_len, true_len)
        match_counts[category] += 1
        correct = category == "correct"

        key = f"{orig_melody.meter[0]}/{orig_melody.meter[1]}"
        per_meter_total[key] = per_meter_total.get(key, 0) + 1
        if correct:
            n_correct += 1
            per_meter_correct[key] = per_meter_correct.get(key, 0) + 1
        else:
            confusions.append((orig_melody.name, key, pred_len, true_len, category))

    n = len(melodies)
    strict_accuracy = n_correct / n if n else 0.0
    tolerant_accuracy = (match_counts["correct"] + match_counts["half"] + match_counts["double"]) / n if n else 0.0
    per_class_acc = [per_meter_correct.get(k, 0) / per_meter_total[k] for k in per_meter_total]
    balanced_accuracy = float(np.mean(per_class_acc)) if per_class_acc else 0.0
    baseline = majority_class_baseline(melodies)

    if verbose:
        norm_note = "normalized" if normalize else "raw (un-normalized)"
        print(f"\n=== {label} meter classification "
              f"({n} real melodies, {len(candidate_lengths)} candidate meters, {norm_note}) ===")
        print(f"Strict accuracy:   {strict_accuracy:.1%}  ({n_correct}/{n})")
        print(f"Tolerant accuracy: {tolerant_accuracy:.1%}  "
              f"(correct={match_counts['correct']}, half-measure={match_counts['half']}, "
              f"double-measure={match_counts['double']}, other-miss={match_counts['other']})")
        print(f"Balanced accuracy (avg across {len(per_meter_total)} meter classes): {balanced_accuracy:.1%}")
        print(f"Majority-class baseline (always guess {baseline['majority_class']}): {baseline['accuracy']:.1%}")
        print("\nAccuracy by meter:")
        for key in sorted(per_meter_total):
            correct_n = per_meter_correct.get(key, 0)
            total_n = per_meter_total[key]
            print(f"  {key:>6}: {correct_n:4d}/{total_n:<4d}  ({correct_n/total_n:.1%})")

        if confusions:
            print(f"\nExample misclassifications (up to 5 of {len(confusions)}):")
            for name, true_key, pred_len, true_len, category in confusions[:5]:
                print(f"  '{name}' — true meter {true_key} (measure={true_len:g} beats), "
                      f"predicted measure={pred_len:g} beats  [{category}]")

    return {"strict_accuracy": strict_accuracy, "tolerant_accuracy": tolerant_accuracy,
            "balanced_accuracy": balanced_accuracy, "majority_baseline": baseline["accuracy"],
            "n_correct": n_correct, "n_total": n,
            "per_meter_total": per_meter_total, "per_meter_correct": per_meter_correct,
            "match_counts": dict(match_counts)}


def compare_accent_configurations(melodies: list, resolution: float = SIXTEENTH_NOTE):
    """
    Experiment #2/#3 from the project note: run the classifier with
    onset-only weighting and with several melodic-accent configurations,
    and report accuracy side by side so the effect of each cue (and of
    combining them) is directly visible.
    """
    configs = [
        ("Onset-only (baseline)", None),
        ("+ duration accent", ["duration"]),
        ("+ leap accent", ["leap"]),
        ("+ contour accent", ["contour"]),
        ("+ register accent", ["register"]),
        ("+ all accents combined", ["duration", "leap", "contour", "register"]),
    ]

    summary = []
    for label, components in configs:
        result = evaluate_corpus(melodies, resolution, accent_components=components, label=label)
        summary.append((label, result))

    baseline = majority_class_baseline(melodies)["accuracy"]
    print("\n=== Summary: accuracy by representation ===")
    print(f"  {'Representation':<28}{'Strict':>9}{'Tolerant':>11}{'Balanced':>11}")
    print(f"  {'(majority-class baseline)':<28}{baseline:>9.1%}{'':>11}{'':>11}")
    for label, result in summary:
        print(f"  {label:<28}{result['strict_accuracy']:>9.1%}"
              f"{result['tolerant_accuracy']:>11.1%}{result['balanced_accuracy']:>11.1%}")

    return summary


def compare_phase3_representations(melodies: list, resolution: float = SIXTEENTH_NOTE):
    """
    Phase 3 of the project note's suggested first experiments:
    "Compare onset-only, onset+duration, onset+pitch/contour, and
    onset+phrase/motif representations."

    Each representation combines a specific, motivated bundle of accent
    cues (rather than testing every cue in isolation, as Phase 2 does):
      - onset-only:          no accent weighting (the original baseline)
      - onset+duration:      weight onsets by note duration
      - onset+pitch/contour: weight onsets by melodic leap size and by
                              contour turning points (Thomassen's model)
      - onset+phrase/motif:  weight onsets by phrase-boundary position
                              and by repeated-motif parallelism
                              (Temperley & Bartlette)
    """
    configs = [
        ("Onset-only", None),
        ("Onset + duration", ["duration"]),
        ("Onset + pitch/contour", ["leap", "contour"]),
        ("Onset + phrase/motif", ["phrase", "motif"]),
    ]

    summary = []
    for label, components in configs:
        result = evaluate_corpus(melodies, resolution, accent_components=components, label=label)
        summary.append((label, result))

    baseline = majority_class_baseline(melodies)["accuracy"]
    print("\n=== Phase 3 summary: accuracy by representation ===")
    print(f"  {'Representation':<24}{'Strict':>9}{'Tolerant':>11}{'Balanced':>11}")
    print(f"  {'(majority-class baseline)':<24}{baseline:>9.1%}{'':>11}{'':>11}")
    for label, result in summary:
        print(f"  {label:<24}{result['strict_accuracy']:>9.1%}"
              f"{result['tolerant_accuracy']:>11.1%}{result['balanced_accuracy']:>11.1%}")

    return summary


# ---------------------------------------------------------------------------
# 5. Example melodies (different meters)
# ---------------------------------------------------------------------------

def make_example_melodies():
    melodies = []

    # --- Melody 1: 4/4, a syncopated "son clave"-style pattern repeated
    # over 4 measures. Onsets in quarter-note beats.
    clave_pattern = [0.0, 0.75, 1.5, 2.0, 3.0]  # one measure of 4/4 (4 beats)
    measure_len = 4.0
    notes = []
    for m in range(4):  # 4 repetitions
        for onset in clave_pattern:
            notes.append(Note(onset=m * measure_len + onset, duration=0.25))
    melodies.append(Melody(name="4/4 clave-style riff", meter=(4, 4), notes=notes))

    # --- Melody 2: 3/4, waltz-like pattern (strong beat 1, offbeat pickups)
    waltz_pattern = [0.0, 1.0, 1.5, 2.0, 2.5]  # one measure of 3/4 (3 beats)
    measure_len = 3.0
    notes = []
    for m in range(5):  # 5 repetitions
        for onset in waltz_pattern:
            notes.append(Note(onset=m * measure_len + onset, duration=0.25))
    melodies.append(Melody(name="3/4 waltz-style riff", meter=(3, 4), notes=notes))

    # --- Melody 3: 6/8, compound-meter pattern (two dotted-quarter groups)
    # In quarter-note beats, a 6/8 measure = 6 * (4/8) = 3.0 quarter beats,
    # but its natural grouping is in two groups of three eighth notes.
    sixeight_pattern = [0.0, 0.5, 1.0, 1.5, 2.0]  # positions within the measure
    measure_len = 3.0  # 6/8 measure length in quarter-note beats
    notes = []
    for m in range(5):
        for onset in sixeight_pattern:
            notes.append(Note(onset=m * measure_len + onset, duration=0.25))
    melodies.append(Melody(name="6/8 compound-meter riff", meter=(6, 8), notes=notes))

    return melodies


# ---------------------------------------------------------------------------
# 6. Analysis pipeline + plotting
# ---------------------------------------------------------------------------

def analyze_melody(melody: Melody, accent_components=None):
    onsets = extract_onsets(melody)
    total_beats = melody.total_beats()
    weights = (compute_accent_weights(melody, accent_components)
               if accent_components is not None else None)
    impulse = build_impulse_train(onsets, total_beats, weights=weights)
    autocorr = autocorrelation(impulse)
    peaks = find_periodicities(autocorr)
    return {
        "melody": melody,
        "onsets": onsets,
        "impulse": impulse,
        "autocorr": autocorr,
        "peaks": peaks,
    }


def plot_results(results, out_path="onset_autocorrelation.png"):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for row, res in enumerate(results):
        melody = res["melody"]
        impulse = res["impulse"]
        autocorr = res["autocorr"]
        measure_len = melody.measure_length_quarters()

        # --- impulse train (time domain) ---
        t = np.arange(len(impulse)) * SIXTEENTH_NOTE
        ax0 = axes[row, 0]
        ax0.vlines(t[impulse > 0], 0, impulse[impulse > 0], color="tab:blue")
        ax0.set_ylim(0, 1.2)
        ax0.set_title(f"{melody.name} — onset impulse train")
        ax0.set_xlabel("time (quarter-note beats)")
        ax0.set_ylabel("weight")
        # mark measure boundaries
        for m_start in np.arange(0, t[-1] + 1e-9, measure_len):
            ax0.axvline(m_start, color="gray", linestyle=":", linewidth=0.7)

        # --- autocorrelation (lag domain) ---
        lags_beats = np.arange(len(autocorr)) * SIXTEENTH_NOTE
        ax1 = axes[row, 1]
        ax1.plot(lags_beats, autocorr, color="tab:red")
        ax1.set_title(f"{melody.name} — autocorrelation")
        ax1.set_xlabel("lag (quarter-note beats)")
        ax1.set_ylabel("normalized correlation")
        ax1.axvline(measure_len, color="gray", linestyle="--", linewidth=1,
                    label=f"1 measure = {measure_len:g} beats")
        for peak in res["peaks"]:
            ax1.plot(peak["lag_beats"], peak["correlation"], "ko", markersize=4)
        ax1.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def print_comparison(results):
    print("\n=== Periodicity comparison across melodies ===")
    for res in results:
        melody = res["melody"]
        measure_len = melody.measure_length_quarters()
        print(f"\n{melody.name}  (meter {melody.meter[0]}/{melody.meter[1]}, "
              f"1 measure = {measure_len:g} quarter-note beats)")
        print("  Strongest recurring lags (excluding lag 0):")
        for peak in res["peaks"]:
            ratio_to_measure = peak["lag_beats"] / measure_len
            print(f"    lag = {peak['lag_beats']:6.2f} beats "
                  f"({peak['lag_samples_16th']:3d} sixteenth-notes), "
                  f"corr = {peak['correlation']:.3f}, "
                  f"~{ratio_to_measure:.2f}x the measure length")


def pick_illustrative_sample(melodies: list, per_meter: int = 1):
    """Grab a few melodies per distinct meter, for a readable plot."""
    seen = {}
    sample = []
    for m in melodies:
        key = m.meter
        if seen.get(key, 0) < per_meter:
            sample.append(m)
            seen[key] = seen.get(key, 0) + 1
    return sample


if __name__ == "__main__":
    USE_REAL_DATA = True

    if USE_REAL_DATA:
        # Real, human-annotated melodies from the Essen Folksong Collection
        # (bundled with music21 -- `pip install music21`, no download needed).
        melodies = load_essen_folksong_corpus(max_files=5, max_melodies=300)
        print(f"Loaded {len(melodies)} real melodies with usable time signatures.")

        # Phase 2: does any single melodic-accent cue improve classification?
        compare_accent_configurations(melodies)

        # Phase 3: compare onset-only, onset+duration, onset+pitch/contour,
        # and onset+phrase/motif as four distinct representations.
        phase3_results = compare_phase3_representations(melodies)

        # Plot a small, readable sample: one melody per distinct meter,
        # using the best-performing Phase 3 representation.
        best_label, best_result = max(phase3_results, key=lambda lr: lr[1]["balanced_accuracy"])
        phase3_configs = {
            "Onset-only": None,
            "Onset + duration": ["duration"],
            "Onset + pitch/contour": ["leap", "contour"],
            "Onset + phrase/motif": ["phrase", "motif"],
        }
        print(f"\nBest Phase 3 representation by balanced accuracy: {best_label} "
              f"({best_result['balanced_accuracy']:.1%}) -- using it for the plot below.")

        sample = pick_illustrative_sample(melodies, per_meter=1)
        results = [analyze_melody(m, accent_components=phase3_configs[best_label]) for m in sample]
        print_comparison(results)
        plot_results(results, out_path="onset_autocorrelation_real_data.png")

    else:
        # Synthetic sanity-check melodies (useful for debugging the pipeline
        # itself, since their periodicity is known exactly by construction).
        melodies = make_example_melodies()
        results = [analyze_melody(m) for m in melodies]
        print_comparison(results)
        plot_results(results, out_path="onset_autocorrelation_synthetic.png")
