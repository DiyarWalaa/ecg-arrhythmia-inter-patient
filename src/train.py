# =========================================================
# ECG ARRHYTHMIA MULTICLASS CLASSIFICATION
# FINAL STABLE VERSION
#
# Classes:
# N = Normal
# S = Supraventricular
# V = Ventricular
#
# Features:
# - Inter-patient evaluation
# - CNN + BiLSTM
# - RR interval features
# - Targeted augmentation
# - Multiclass focal loss
# - GPU training enabled
# - Full visualization suite
#
# =========================================================

import os
import json
import random
import datetime
import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    Dense,
    Bidirectional,
    LSTM,
    Concatenate
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_recall_fscore_support
)

from sklearn.preprocessing import label_binarize


# =========================================================
# 1. REPRODUCIBILITY
# =========================================================

SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)

# GPU deterministic operations
tf.config.experimental.enable_op_determinism()


def reset_seeds(seed=SEED):
    """Re-seed every RNG so each sweep setting starts identically.

    Without this the later ratios would train from different weight
    initialisations and a difference in val macro-F1 could not be
    attributed to the sampling ratio alone.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    tf.keras.utils.set_random_seed(seed)



# =========================================================
# 2. GPU CHECK
# =========================================================

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("\nGPU detected and enabled.")
    print(gpus)

else:
    print("\nNo GPU detected. Running on CPU.")


# =========================================================
# 3. SETTINGS
# =========================================================

# This file lives in src/, so the project root is one level up.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get(
    "ECG_DATA_DIR",
    os.path.join(
        BASE_DIR,
        "data",
        "mit-bih-arrhythmia-database-1.0.0"
    )
)

if not os.path.exists(DATA_DIR):

    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATA_DIR}"
    )

RUN_NAME = os.environ.get(
    "ECG_RUN_NAME",
    "baseline"
)

OUT_ROOT = os.environ.get(
    "ECG_OUT_DIR",
    os.path.join(BASE_DIR, "runs")
)

RUN_DIR = os.path.join(
    OUT_ROOT,
    RUN_NAME
)

os.makedirs(
    RUN_DIR,
    exist_ok=True
)

print(f"\nDATA_DIR: {DATA_DIR}")
print(f"RUN_DIR : {RUN_DIR}")

# Inter-patient split
DS1 = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209', '215', '220', '223', '230'
]

DS2 = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234'
]

# Patient-wise validation split (step 1; enlarged at step 5, REVERTED at
# E1). Whole records are held out of training, never individual beats, so
# no patient can appear on both sides.
#
# E1 put this back to three records. The five-record set of step 5 added
# 106 (520 V beats, ZERO S beats) and 118 (16 V, 96 S), which made
# validation 85.3% N / 11.3% V / 3.4% S. Val V-F1 then peaks at epoch 2
# while val S-F1 is still climbing at epoch 6, so macro-F1 tracks V and
# stops early: E0 selected epoch 2 and scored test macro-F1 0.5591 against
# a required 0.6400, on training code identical to step 3.
DS1_VAL = ['207', '220', '223']

# Recorded so the choice is auditable and never quietly re-derived.
# Written into metrics.json as val_selection_rule.
VAL_SELECTION_RULE = {
    "records": ['207', '220', '223'],
    "history": "step 1 chose these three; step 5 enlarged to five by "
               "adding 106 and 118; E1 reverted to the original three",
    "reverted_at": "E1",
    "revert_reason": (
        "Record 106 contributes 520 V beats and ZERO S beats, and 118 "
        "contributes 16 V to 96 S. Together they made validation "
        "85.3% N / 11.3% V / 3.4% S. Val V-F1 peaks at epoch 2 while val "
        "S-F1 keeps rising to epoch 6, so macro-F1 follows V and selects "
        "early. E0 used training code identical to step 3 and differed "
        "only in this validation set: it selected epoch 2 instead of 8 "
        "and scored test macro-F1 0.5591 against a required 0.6400."
    ),
    "exclusions_still_binding": [
        {
            "record": "209",
            "reason": "holds 383 of the 943 DS1 S beats; moving it to "
                      "validation would starve training of S"
        },
        {
            "record": "201",
            "reason": "same subject as test record 202 (PhysioNet); "
                      "validating on it would stack a second leak"
        }
    ],
    "known_limitation": (
        "220 has zero V beats, so validation V-F1 rests on two records. "
        "That is accepted: the step 5 attempt to fix it cost more in "
        "selection bias than it bought in V coverage."
    )
}

DS1_TRAIN = [rec for rec in DS1 if rec not in DS1_VAL]

# Guard the hard constraint: validation comes from DS1 only, never DS2.
assert set(DS1_VAL).issubset(set(DS1)), "DS1_VAL must be a subset of DS1"
assert not set(DS1_VAL) & set(DS2), "DS1_VAL must never contain a DS2 record"
assert len(DS1_TRAIN) + len(DS1_VAL) == len(DS1)

PRE_SAMPLES = 90
POST_SAMPLES = 144

SEGMENT_LENGTH = PRE_SAMPLES + POST_SAMPLES

# Wavelet scalogram input (E2; scales re-spaced at E3).
#
# The beat window is a (234, 9) Mexican-hat (Ricker) scalogram rather than
# a raw (234, 1) waveform. Zahid et al. 2022 reach S-F1 0.8344 on this
# exact DS1/DS2 split with a structurally similar network (230-sample
# window, late-fused RR features, 23,619 parameters); the input
# representation is the one ingredient we can adopt directly.
#
# E3 tried re-spacing these 9 scales to LOG-spaced 3..90 Hz, reasoning
# that P-wave energy sits below 10 Hz. It scored macro-F1 0.6151 against
# E2's 0.7178 and lost on every class, so E4 REVERTS to E2's linear
# 10..90 Hz. Whatever limits S here, it is not the absence of
# sub-10 Hz channels.
#
# Widths are DERIVED from these target centre frequencies, never
# hardcoded - see section 6B.
SAMPLING_RATE_HZ = 360.0

# E9: E6's WINDOW, E8's BEAT POPULATION.
#
# E8 reached S-F1 0.4752 and S recall 0.6594 against E6's 0.3641 and
# 0.3388 - but on a different test set. Its 2-second span cap rejected
# 15.5% of N, 14.8% of S and 1.0% of V, so DS2 fell from 49,289 beats to
# 42,154. The gain cannot be attributed to the representation until both
# are scored on the same beats.
#
# E9 is the control. The model still sees E6's FIXED 234-sample window at
# 9 wavelet channels and 239,171 parameters - PRE_SAMPLES/POST_SAMPLES are
# untouched, there is no mask channel and no padding. The only thing
# borrowed from E8 is its ACCEPTANCE RULE: a beat is kept only if the span
# between its neighbouring R peaks is at most MAX_SPAN_SAMPLES.
#
# If E9 recovers most of E8's S gain, the gain was the easier population.
# If it does not, the gain was the representation.
MAX_SPAN_SAMPLES = int(2.0 * SAMPLING_RATE_HZ)

# E10: FIXED-WINDOW WIDTH SWEEP.
#
# On IDENTICAL beats, E8 (720-sample variable window plus a mask channel
# encoding the R-1..R+1 span) reached S-F1 0.4752 and V-F1 0.5661; E9
# (E6's fixed 234-sample window) reached 0.2976 and 0.8548. The wide
# window is worth +0.1776 S-F1 and costs -0.2887 V-F1.
#
# But E8 and E9 differ in TWO ways: how much CONTEXT the model sees, and
# whether an explicit LENGTH SIGNAL exists. A fixed wide window has the
# context and no length signal, so sweeping the fixed width separates
# them. If a wide fixed window recovers E8's S gain, context is what
# matters and the mask channel is incidental; if it does not, the length
# signal is doing the work.
#
# Each pair keeps E9's 0.385 / 0.615 pre/post split of the total width.
# The span cap is NOT lowered: DS1_TRAIN S beats average 407 samples and
# DS2 S beats 506, so a 400-500 cap would reject most of the S class.
WINDOW_GRID = [
    (90, 144),      # 234 - E9's window, the control arm
    (140, 220),     # 360
    (185, 295),     # 480
    (230, 370),     # 600
]

# POPULATION CONTROL.
#
# Every arm must train and test on IDENTICAL beats, otherwise a width
# difference and a population difference are confounded - which is exactly
# the mistake E8 made and E9 had to unpick. So acceptance is decided ONCE,
# by the LARGEST window in the grid, and applied to every arm regardless
# of the width that arm actually reads. A beat is kept only if it clears
# both the span cap and the largest window's bounds.
ACCEPT_PRE = max(pre for pre, _post in WINDOW_GRID)
ACCEPT_POST = max(post for _pre, post in WINDOW_GRID)

# The grid is monotone, so the widest pre and the widest post come from
# the same pair. If that ever stops being true the acceptance window would
# be wider than any arm actually trains on, which would silently drop
# beats no arm needed dropped.
assert (ACCEPT_PRE, ACCEPT_POST) in WINDOW_GRID, (
    "the widest pre and post must come from the same grid entry"
)

WINDOW_WIDTHS = [pre + post for pre, post in WINDOW_GRID]

WAVELET_TARGET_FREQS_HZ = [float(10 * k) for k in range(1, 10)]

# Patient-relative RR features (step 3).
# Raw RR intervals in samples are not comparable across patients: 280
# samples is early for a 60 bpm patient and late for a 100 bpm one. Every
# feature below is a ratio, so it is dimensionless and self-referenced to
# the patient's own rhythm. Uses only that patient's signal - never their
# labels, never another patient - so the inter-patient constraint holds.
RR_FEATURE_NAMES = [
    'pre_RR_over_median',
    'post_RR_over_median',
    'local_RR_over_median',
    'pre_RR_over_local',
    'post_RR_over_pre'
]

# Mean of the previous N RR intervals, excluding the beat's own pre_RR.
RR_LOCAL_WINDOW = 10

# A missed annotation produces one enormous interval. Clipping stops a
# single outlier from dominating a batch.
RR_CLIP_MIN = 0.2
RR_CLIP_MAX = 3.0

# Loss settings.
#
# E0 re-anchor: alpha is back to the SCALAR 0.50 that steps 0-3 used, and
# class balancing is back in the data via augment_training_data(). The
# step 4 per-class alpha vector and the step 6 BETA sweep are both
# reverted - see docs/ablation.md for why. A scalar alpha rebalances
# nothing; it simply scales the whole loss. That is intentional here: this
# run exists to re-establish the step 3 training configuration on the step
# 5 validation set, a combination that has never been run.
FOCAL_ALPHA = 0.50

FOCAL_GAMMA = 2.0

# Decision-threshold search (E1).
#
# Our S precision (0.402) is in line with de Chazal 2004 (0.385) and Zhou
# 2021 (0.415), but S recall is 0.128 against their 0.759 and 0.894. The
# model predicts S 584 times against 1836 true S beats - it under-calls
# the class about 6x. That is a decision-rule failure, not a feature
# failure, so it is fixed at the decision rule: prediction becomes
# argmax(w * p) for a per-class multiplier vector w tuned on VALIDATION.
#
# w_N is pinned to 1.0 - only the ratios matter, so one class must be the
# reference. The grid is log-spaced 0.25 .. 32 (2**-2 .. 2**5).
THRESHOLD_GRID = [
    float(v) for v in np.logspace(-2.0, 5.0, 15, base=2.0)
]

THRESHOLD_MAX_PASSES = 6

ADAM_LEARNING_RATE = 1e-4

LEAD_INDEX = 0

BATCH_SIZE = 128
EPOCHS = 40


# =========================================================
# 4. AAMI MAPPING
# =========================================================

AAMI_MAP = {

    'N': 'N',
    'L': 'N',
    'R': 'N',
    'e': 'N',
    'j': 'N',

    'A': 'S',
    'a': 'S',
    'J': 'S',
    'S': 'S',

    'V': 'V',
    'E': 'V'
}


# =========================================================
# 5. LABEL ENCODING
# =========================================================

LABEL_TO_INT = {
    'N': 0,
    'S': 1,
    'V': 2
}

INT_TO_LABEL = {
    0: 'N',
    1: 'S',
    2: 'V'
}

NUM_CLASSES = 3


# =========================================================
# 6. NORMALIZATION
# =========================================================

def normalize_segment(segment):

    mean = np.mean(segment)
    std = np.std(segment)

    if std == 0:
        return segment - mean

    return (segment - mean) / std


def fit_rr_norm(rr_array):
    """Fit per-column RR statistics on the TRAINING set only.

    RR is [prev_rr, next_rr], so the statistics are per column (axis=0),
    not one scalar over both columns as before. Returns (mean, std), each
    shape (2,). A zero-variance column gets std 1.0 so it is only centred,
    never divided by zero.
    """

    rr_array = np.array(rr_array, dtype=np.float32)

    mean = np.mean(rr_array, axis=0)
    std = np.std(rr_array, axis=0)

    std = np.where(std == 0.0, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def apply_rr_norm(rr_array, mean, std):
    """Apply already-fitted RR statistics.

    Never fits. Validation and test are scaled with the training set's
    mean and std so all three land on the same scale.
    """

    rr_array = np.array(rr_array, dtype=np.float32)

    return (rr_array - mean) / std


# =========================================================
# 6B. WAVELET SCALOGRAM
# =========================================================

# scipy.signal.ricker and scipy.signal.cwt were deprecated in SciPy 1.12
# and REMOVED in SciPy 1.15 (this machine runs 1.18, where neither
# exists). Both are reimplemented below from the SciPy source so the
# output is what scipy.signal.cwt(data, scipy.signal.ricker, widths)
# would have produced, with no scipy dependency at all - numpy only.
# That also removes any question of what SciPy version Kaggle ships.


def ricker_wavelet(points, a):
    """Mexican-hat wavelet - a faithful reimplementation of the removed
    scipy.signal.ricker(points, a)."""

    amplitude = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))

    wsq = a ** 2.0

    vec = np.arange(0, points) - (points - 1.0) / 2.0

    xsq = vec ** 2.0

    modulation = 1.0 - xsq / wsq

    gauss = np.exp(-xsq / (2.0 * wsq))

    return amplitude * modulation * gauss


def ricker_width_for_frequency(freq_hz, fs_hz):
    """Width `a` whose Ricker wavelet peaks at freq_hz.

    The Ricker spectrum is |psi(w)| proportional to w^2 exp(-a^2 w^2 / 2),
    which is maximal where d/dw of that is zero, i.e. 2 - a^2 w^2 = 0, so
    w_peak = sqrt(2) / a. Converting angular to cyclic frequency at a
    sampling rate of fs:

        f_peak = fs * sqrt(2) / (2 * pi * a)
        =>  a  = fs * sqrt(2) / (2 * pi * f_peak)
    """

    return fs_hz * np.sqrt(2.0) / (2.0 * np.pi * freq_hz)


def ricker_centre_frequency(a, fs_hz):
    """Inverse of the above - the centre frequency a given width yields."""

    return fs_hz * np.sqrt(2.0) / (2.0 * np.pi * a)


def cwt_ricker(data, widths):
    """Continuous wavelet transform - a faithful reimplementation of the
    removed scipy.signal.cwt(data, scipy.signal.ricker, widths).

    Returns (len(widths), len(data)).
    """

    output = np.empty(
        (len(widths), len(data)),
        dtype=np.float64
    )

    for index, width in enumerate(widths):

        # SciPy truncated each wavelet to 10 * width samples, capped at
        # the signal length.
        points = int(min(10.0 * width, len(data)))
        points = max(points, 1)

        wavelet = ricker_wavelet(points, width)

        # SciPy correlated via a reversed (conjugated) kernel; the ricker
        # wavelet is real and symmetric, so the reversal is a no-op, but
        # it is kept for exactness.
        output[index] = np.convolve(
            data,
            wavelet[::-1],
            mode='same'
        )

    return output


WAVELET_WIDTHS = [
    float(ricker_width_for_frequency(f, SAMPLING_RATE_HZ))
    for f in WAVELET_TARGET_FREQS_HZ
]

WAVELET_CENTRE_FREQS_HZ = [
    float(ricker_centre_frequency(a, SAMPLING_RATE_HZ))
    for a in WAVELET_WIDTHS
]

N_WAVELET_SCALES = len(WAVELET_WIDTHS)

print(f"\nWavelet scalogram: {N_WAVELET_SCALES} Ricker scales "
      f"at fs = {SAMPLING_RATE_HZ:.0f} Hz")
print(f"  {'target Hz':>10} {'width a':>10} {'centre Hz':>10} "
      f"{'support':>8}")

for _f, _a, _c in zip(WAVELET_TARGET_FREQS_HZ, WAVELET_WIDTHS,
                      WAVELET_CENTRE_FREQS_HZ):
    print(f"  {_f:>10.1f} {_a:>10.4f} {_c:>10.4f} "
          f"{int(min(10.0 * _a, max(WINDOW_WIDTHS))):>8}")


# =========================================================
# 7. EXTRACT ECG + RR FEATURES
# =========================================================

def empty_segmentation_stats():
    """Zeroed per-class accounting for one record or one split.

    `accepted` plus the four rejection buckets is every annotated beat
    whose symbol is in AAMI_MAP, so they always reconcile against the
    total. `spans` holds the accepted R-1..R+1 span lengths, which is what
    the per-class mean and standard deviation are computed from - the
    window the MODEL sees is this arm's (pre + post) regardless.
    """

    labels = sorted(set(AAMI_MAP.values()))

    return {
        "accepted": {lab: 0 for lab in labels},
        "rejected_max_span": {lab: 0 for lab in labels},
        "rejected_edge": {lab: 0 for lab in labels},
        "rejected_invalid_span": {lab: 0 for lab in labels},
        "rejected_accept_window": {lab: 0 for lab in labels},
        "spans": {lab: [] for lab in labels}
    }


def merge_segmentation_stats(into, other):
    """Accumulate one record's stats into a running total."""

    for key in ("accepted", "rejected_max_span", "rejected_edge",
                "rejected_invalid_span", "rejected_accept_window"):
        for lab, count in other[key].items():
            into[key][lab] += count

    for lab, spans in other["spans"].items():
        into["spans"][lab].extend(spans)

    return into


def summarize_segmentation_stats(stats):
    """Turn raw stats into the JSON-serialisable block metrics.json gets."""

    out = {}

    for lab in sorted(stats["accepted"]):

        spans = np.asarray(stats["spans"][lab], dtype=np.float64)

        out[lab] = {
            "accepted": int(stats["accepted"][lab]),
            "rejected_max_span": int(stats["rejected_max_span"][lab]),
            "rejected_edge": int(stats["rejected_edge"][lab]),
            "rejected_invalid_span": int(stats["rejected_invalid_span"][lab]),
            "rejected_accept_window": int(stats["rejected_accept_window"][lab]),
            "span_mean": float(spans.mean()) if spans.size else None,
            "span_std": float(spans.std()) if spans.size else None,
            "span_min": int(spans.min()) if spans.size else None,
            "span_max": int(spans.max()) if spans.size else None
        }

        total = (out[lab]["accepted"]
                 + out[lab]["rejected_max_span"]
                 + out[lab]["rejected_edge"]
                 + out[lab]["rejected_invalid_span"]
                 + out[lab]["rejected_accept_window"])

        out[lab]["total_annotated"] = int(total)

        out[lab]["rejection_rate"] = (
            float(total - out[lab]["accepted"]) / total if total else None
        )

    return out


def extract_beats_from_record(
    record_name,
    data_dir,
    pre_samples,
    post_samples,
    lead_index=0
):
    """A fixed (pre_samples + post_samples) window on a fixed population.

    Returns (beats, labels, rr_features, raw_beats, stats). Each beat is
    (pre_samples + post_samples, N_WAVELET_SCALES).

    ACCEPTANCE is independent of this arm's width: a beat is kept only if
    its R-1..R+1 span clears MAX_SPAN_SAMPLES and the LARGEST window in
    WINDOW_GRID fits inside the record. So `labels` and `rr_features` are
    identical for every window in the grid, and only `beats` differs -
    which is what makes the sweep a clean single-variable comparison.
    """

    record_path = os.path.join(
        data_dir,
        record_name
    )

    stats = empty_segmentation_stats()

    try:

        signal_record = wfdb.rdrecord(record_path)

        annotation = wfdb.rdann(
            record_path,
            'atr'
        )

    except Exception as e:

        print(f"Error reading {record_name}: {e}")

        return [], [], [], [], stats

    signal = signal_record.p_signal[:, lead_index]

    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    # Full RR series for this record, computed once.
    # rr_series[k] = ann_samples[k + 1] - ann_samples[k], so the interval
    # ending at beat i is rr_series[i - 1] and the one starting at beat i
    # is rr_series[i].
    rr_series = np.diff(ann_samples).astype(np.float64)

    median_rr = float(np.median(rr_series)) if len(rr_series) else 0.0

    if not np.isfinite(median_rr) or median_rr <= 0.0:
        median_rr = 1.0

    beats = []
    labels = []
    rr_features = []

    # The z-scored raw waveform is kept alongside the scalogram so that
    # augmentation can perturb the WAVEFORM and then re-enter the exact
    # same normalize -> CWT tail the originals went through. Augmentation
    # remains UNCALLED - class balance comes from the sampler.
    raw_beats = []

    for i in range(1, len(ann_samples) - 1):

        r_peak = ann_samples[i]

        symbol = ann_symbols[i]

        if symbol not in AAMI_MAP:
            continue

        label = AAMI_MAP[symbol]

        # --- E8's ACCEPTANCE RULE, applied first and identically --------
        #
        # This is the only thing E9 borrows from E8. It decides which
        # beats exist; it does not change what the model sees.
        span_start = int(ann_samples[i - 1])
        span_end = int(ann_samples[i + 1])

        if span_start < 0 or span_end > len(signal):
            stats["rejected_edge"][label] += 1
            continue

        span = span_end - span_start

        if span <= 0:
            stats["rejected_invalid_span"][label] += 1
            continue

        if span > MAX_SPAN_SAMPLES:
            stats["rejected_max_span"][label] += 1
            continue

        # --- ACCEPTANCE WINDOW: always the LARGEST in WINDOW_GRID -------
        #
        # Decided by ACCEPT_PRE/ACCEPT_POST, never by this arm's width, so
        # every arm sees the same beats. A beat the widest window cannot
        # read is dropped from ALL arms, including the narrow ones that
        # could have read it.
        if (r_peak - ACCEPT_PRE) < 0 or (r_peak + ACCEPT_POST) > len(signal):
            stats["rejected_accept_window"][label] += 1
            continue

        # --- THIS ARM'S WINDOW: what the model actually reads ------------
        start = r_peak - pre_samples
        end = r_peak + post_samples

        segment = signal[start:end]

        # Guaranteed by the acceptance test above for every grid entry,
        # since pre <= ACCEPT_PRE and post <= ACCEPT_POST. Kept as a guard
        # against a future grid whose widest entry is not the acceptance
        # window.
        if len(segment) != (pre_samples + post_samples):
            stats["rejected_accept_window"][label] += 1
            continue

        pre_rr = float(rr_series[i - 1])
        post_rr = float(rr_series[i])

        # The RR_LOCAL_WINDOW intervals immediately BEFORE this beat's own
        # pre_RR: rr_series[i - 1 - W : i - 1]. Excluding pre_RR is what
        # makes pre_RR / local_RR a prematurity measure rather than a
        # self-comparison. Too few preceding intervals -> fall back to the
        # record median.
        window_start = i - 1 - RR_LOCAL_WINDOW

        if window_start >= 0:
            local_rr = float(np.mean(rr_series[window_start:i - 1]))
        else:
            local_rr = median_rr

        if not np.isfinite(local_rr) or local_rr <= 0.0:
            local_rr = median_rr

        # pre_rr can only be <= 0 with corrupt annotations; fall back so
        # feature 5 never divides by zero.
        pre_rr_denom = pre_rr if pre_rr > 0.0 else median_rr

        rr = [
            pre_rr / median_rr,
            post_rr / median_rr,
            local_rr / median_rr,
            pre_rr / local_rr,
            post_rr / pre_rr_denom
        ]

        rr = [
            float(np.clip(value, RR_CLIP_MIN, RR_CLIP_MAX))
            for value in rr
        ]

        segment = normalize_segment(segment)

        raw_beats.append(segment.astype(np.float32))

        # (pre + post,) -> (pre + post, N_WAVELET_SCALES)
        segment = cwt_ricker(
            segment,
            WAVELET_WIDTHS
        ).T.astype(np.float32)

        beats.append(segment)

        labels.append(label)

        rr_features.append(rr)

        stats["accepted"][label] += 1
        stats["spans"][label].append(int(span))

    return beats, labels, rr_features, raw_beats, stats


# =========================================================
# 8. LOAD DATASET
# =========================================================

def load_dataset(
    record_list,
    data_dir,
    pre_samples,
    post_samples,
    lead_index=0
):

    all_beats = []
    all_labels = []
    all_rr = []
    all_raw = []

    stats = empty_segmentation_stats()

    for rec in record_list:

        print(f"Loading record {rec} ...")

        beats, labels, rr, raw, rec_stats = extract_beats_from_record(
            rec,
            data_dir,
            pre_samples,
            post_samples,
            lead_index
        )

        merge_segmentation_stats(stats, rec_stats)

        all_beats.extend(beats)

        all_labels.extend(labels)

        all_rr.extend(rr)

        all_raw.extend(raw)

    X = np.array(
        all_beats,
        dtype=np.float32
    )

    y = np.array(all_labels)

    RR = np.array(
        all_rr,
        dtype=np.float32
    )

    X_raw = np.array(
        all_raw,
        dtype=np.float32
    )

    return X, y, RR, X_raw, stats


# =========================================================
# 9. ECG AUGMENTATION  [DEPRECATED - unused since E6]
# =========================================================

# DEPRECATED. Reachable only from augment_training_data(), which is
# itself no longer called. Class balance is handled by the sampler in
# section 19B. Do not re-enable: synthetic beats violate hard
# constraint 2.

def augment_segment(segment):

    x = segment.copy()

    # Amplitude scaling
    if np.random.rand() < 0.8:

        scale = np.random.uniform(
            0.90,
            1.10
        )

        x = x * scale

    # Time shift
    if np.random.rand() < 0.8:

        shift = np.random.randint(-5, 6)

        if shift != 0:

            x = np.roll(x, shift)

            if shift > 0:
                x[:shift] = x[shift]

            else:
                x[shift:] = x[shift - 1]

    # Gaussian noise
    if np.random.rand() < 0.9:

        noise = np.random.normal(
            0,
            0.01,
            size=x.shape
        )

        x = x + noise

    x = normalize_segment(x)

    return x.astype(np.float32)


# =========================================================
# 10. TARGETED AUGMENTATION  [DEPRECATED - unused since E6]
# =========================================================

# DEPRECATED. augment_training_data() has zero call sites as of E6;
# section 18 passes the training arrays straight through and section 19B
# balances the classes by SAMPLING real beats with replacement instead of
# fabricating copies. Do not re-enable: it violates hard constraint 2,
# and it copied the RR feature vector unchanged across every duplicate.

def augment_training_data(
    X,
    X_raw,
    RR,
    y
):
    """Expand the minority classes with perturbed copies.

    X      - (n, SEGMENT_LENGTH, N_WAVELET_SCALES) scalograms, the
             originals, already produced by normalize -> CWT in section 7.
    X_raw  - (n, SEGMENT_LENGTH) z-scored waveforms for the same beats.

    Augmentation perturbs the WAVEFORM and then runs the identical
    normalize -> CWT tail the originals went through, so every sample the
    model sees - original or synthetic, train or test - reaches the
    network by the same route. Perturbing the scalogram instead put 85.7%
    of S training samples on a different scale from every test beat.
    """

    X_list = [X]
    RR_list = [RR]
    y_list = [y]

    for segment, rr, label in zip(X_raw, RR, y):

        # N
        if label == 0:
            multiplier = 0

        # S
        elif label == 1:
            multiplier = 6

        # V
        elif label == 2:
            multiplier = 2

        else:
            multiplier = 0

        for _ in range(multiplier):

            # Identical tail to the originals: normalize_segment is
            # applied inside augment_segment, then the same cwt_ricker.
            aug = augment_segment(segment)

            aug = cwt_ricker(
                aug,
                WAVELET_WIDTHS
            ).T.astype(np.float32)

            X_list.append(
                aug[np.newaxis, ...]
            )

            RR_list.append(
                rr[np.newaxis, ...]
            )

            y_list.append(
                np.array(
                    [label],
                    dtype=np.int32
                )
            )

    X_new = np.concatenate(
        X_list,
        axis=0
    )

    RR_new = np.concatenate(
        RR_list,
        axis=0
    )

    y_new = np.concatenate(
        y_list,
        axis=0
    )

    idx = np.random.permutation(
        len(y_new)
    )

    return (
        X_new[idx],
        RR_new[idx],
        y_new[idx]
    )


# =========================================================
# 11. MULTICLASS FOCAL LOSS  [DEPRECATED - unused since E6]
# =========================================================

# DEPRECATED. Not called anywhere: E6 compiles with plain
# 'categorical_crossentropy'. FOCAL_ALPHA and FOCAL_GAMMA are likewise
# retained but unused - metrics.json records loss and focal_loss_used so
# a run is never ambiguous about which was active.

def categorical_focal_loss(
    alpha,
    gamma=2.0
):
    """Multiclass focal loss.

    alpha may be a scalar or a per-class sequence of length NUM_CLASSES.
    y_pred is (batch, n_classes), so a (n_classes,) alpha broadcasts over
    the class axis and weights each class separately, while a scalar
    scales the whole loss uniformly and rebalances nothing.

    E0 passes the scalar 0.50, matching steps 0-3. alpha stays a REQUIRED
    argument so the choice is always explicit at the call site.
    """

    alpha = tf.constant(
        alpha,
        dtype=tf.float32
    )

    def loss(y_true, y_pred):

        y_true = tf.cast(
            y_true,
            tf.float32
        )

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        cross_entropy = -y_true * tf.math.log(y_pred)

        weight = alpha * tf.pow(
            1 - y_pred,
            gamma
        )

        focal_loss = weight * cross_entropy

        return tf.reduce_mean(
            tf.reduce_sum(
                focal_loss,
                axis=1
            )
        )

    return loss


# =========================================================
# 11B. DECISION-THRESHOLD SEARCH
# =========================================================

# Both functions below are PURE: they take probabilities and labels as
# arguments and read no module-level data. They cannot see the test set,
# and the AST of this section contains no test-related name. The caller
# in section 22C passes validation probabilities only.


def macro_f1_from_weights(y_prob, y_true_int, weights, labels):
    """Macro-F1 of argmax(w * p) for one weight vector."""

    y_pred = np.argmax(
        y_prob * np.asarray(weights, dtype=np.float64),
        axis=1
    )

    _, _, f1, _ = precision_recall_fscore_support(
        y_true_int,
        y_pred,
        labels=labels,
        zero_division=0
    )

    return float(np.mean(f1))


def tune_decision_weights(y_prob, y_true_int, num_classes, grid,
                          max_passes):
    """Coordinate ascent on per-class multipliers, maximising macro-F1.

    w[0] (class N) is pinned to 1.0; only the minority-class multipliers
    move, since scaling all three by a constant leaves argmax unchanged.
    One pass sweeps each remaining coordinate over the whole grid, keeping
    the best. Passes repeat until a full pass yields no improvement, or
    max_passes is reached.

    Returns (weights, best_macro_f1, search_log). The caller is
    responsible for passing VALIDATION data - this function has no way to
    reach anything else.
    """

    labels = list(range(num_classes))

    weights = [1.0] * num_classes

    best = macro_f1_from_weights(y_prob, y_true_int, weights, labels)

    search_log = [{
        "pass": 0,
        "coordinate": None,
        "weights": list(weights),
        "val_macro_f1": best
    }]

    for pass_index in range(1, max_passes + 1):

        improved = False

        for c in range(1, num_classes):

            best_w = weights[c]

            for candidate in grid:

                trial = list(weights)
                trial[c] = float(candidate)

                score = macro_f1_from_weights(
                    y_prob,
                    y_true_int,
                    trial,
                    labels
                )

                if score > best + 1e-12:
                    best = score
                    best_w = float(candidate)
                    improved = True

            weights[c] = best_w

            search_log.append({
                "pass": pass_index,
                "coordinate": c,
                "weights": list(weights),
                "val_macro_f1": best
            })

        if not improved:
            break

    return weights, best, search_log


# =========================================================
# 12. LOAD DS1 ONLY  (DS2 is not read until after selection)
# =========================================================

# E10 sweeps the window width, so the beats have to be re-extracted per
# arm. Only DS1 is loaded here, at the CONTROL window; each sweep arm
# re-extracts DS1 at its own width in section 22.
#
# DS2 IS NOT LOADED IN THIS FILE UNTIL SECTION 23B, which runs AFTER the
# winning window has been chosen on validation. That is a stronger
# guarantee than the BETA and sampler sweeps had: those kept X_test in
# memory throughout and relied on the loop not referring to it, whereas
# here the DS2 arrays do not exist while any selection decision is made.

CONTROL_PRE, CONTROL_POST = WINDOW_GRID[0]

print(f"Loading DS1_TRAIN (train) at the control window "
      f"({CONTROL_PRE}, {CONTROL_POST}) ...")

X_train, y_train, RR_train, X_raw_train, SEG_STATS_TRAIN = load_dataset(
    DS1_TRAIN,
    DATA_DIR,
    CONTROL_PRE,
    CONTROL_POST,
    LEAD_INDEX
)

print("\nLoading DS1_VAL (validation) ...")

# Loaded one record at a time so every validation beat keeps its record
# id. That is what makes the per-record breakdown after training possible.
# Concatenating in DS1_VAL order produces exactly the arrays a single
# load_dataset(DS1_VAL, ...) call would return - same records, same order,
# same beats.
X_valid_parts = []
y_valid_parts = []
RR_valid_parts = []
val_record_ids = []

SEG_STATS_VAL = empty_segmentation_stats()

for rec in DS1_VAL:

    X_rec, y_rec, RR_rec, _, _rec_stats = load_dataset(
        [rec],
        DATA_DIR,
        CONTROL_PRE,
        CONTROL_POST,
        LEAD_INDEX
    )

    merge_segmentation_stats(SEG_STATS_VAL, _rec_stats)

    X_valid_parts.append(X_rec)
    y_valid_parts.append(y_rec)
    RR_valid_parts.append(RR_rec)

    val_record_ids.extend([rec] * len(y_rec))

X_valid = np.concatenate(X_valid_parts, axis=0)
y_valid = np.concatenate(y_valid_parts, axis=0)
RR_valid = np.concatenate(RR_valid_parts, axis=0)

val_record_ids = np.array(val_record_ids)

del X_valid_parts, y_valid_parts, RR_valid_parts

print("\nTrain shape:", X_train.shape)
print("Val shape  :", X_valid.shape)

print(f"\nDS1_TRAIN records ({len(DS1_TRAIN)}): {DS1_TRAIN}")
print("Original Train Distribution:")
print(Counter(y_train))

print(f"\nDS1_VAL records ({len(DS1_VAL)}): {DS1_VAL}")
print("Original Validation Distribution:")
print(Counter(y_valid))


# Untouched references for the sweep. Acceptance does not depend on the
# window width, so every arm must return byte-identical labels and RR
# features; only the beats differ. These copies are what section 22
# asserts against, taken BEFORE section 15 standardises RR in place.
Y_TRAIN_REF = y_train.copy()
Y_VALID_REF = y_valid.copy()
RR_TRAIN_RAW = RR_train.copy()
RR_VALID_RAW = RR_valid.copy()


# --- E10 population accounting ------------------------------------------
#
# Acceptance is the LARGEST window in the grid plus the span cap, applied
# once and shared by every arm. That is strictly stricter than E9's rule,
# which only had to fit a 234-sample window, so E10 scores fewer beats
# than E9 - and the cost is reported here rather than discovered later.

SEGMENTATION_STATS = {
    "DS1_TRAIN": summarize_segmentation_stats(SEG_STATS_TRAIN),
    "DS1_VAL": summarize_segmentation_stats(SEG_STATS_VAL)
}

# E9's population, from results/E9_e6_window_e8_population/metrics.json.
E9_ACCEPTED = {
    "DS1_TRAIN": {"N": 35021, "S": 637, "V": 2878},
    "DS1_VAL": {"N": 5359, "S": 273, "V": 602},
    "DS2": {"N": 37395, "S": 1565, "V": 3188}
}

# Measured on all 44 records before the run, with the real extraction
# code. The wider acceptance window costs 17 DS1 beats and 19 DS2 beats
# relative to E9 - 34 N and 2 V in total, and NO S beats in any split, so
# the S class is identical to E9's at 637 / 273 / 1565.
E10_EXPECTED = {
    "DS1_TRAIN": {"N": 35006, "S": 637, "V": 2877},
    "DS1_VAL": {"N": 5358, "S": 273, "V": 602},
    "DS2": {"N": 37377, "S": 1565, "V": 3187}
}

print("\n" + "=" * 78)
print("E10 POPULATION: span cap + the LARGEST window in the grid")
print(f"  acceptance window ({ACCEPT_PRE}, {ACCEPT_POST}) = "
      f"{ACCEPT_PRE + ACCEPT_POST} samples, span cap {MAX_SPAN_SAMPLES}")
print(f"  widths swept: {WINDOW_WIDTHS}")
print("=" * 78)
print(f"  {'split':<10} {'cls':>3} {'annot':>7} {'accept':>7} {'E9':>7} "
      f"{'diff':>6} {'>cap':>6} {'window':>7} {'edge':>5} "
      f"{'span mean':>10}")

for _split, _block in SEGMENTATION_STATS.items():
    for _lab, _row in _block.items():
        _mean = _row["span_mean"]
        _e9 = E9_ACCEPTED[_split][_lab]
        print(f"  {_split:<10} {_lab:>3} {_row['total_annotated']:>7} "
              f"{_row['accepted']:>7} {_e9:>7} "
              f"{_row['accepted'] - _e9:>6} "
              f"{_row['rejected_max_span']:>6} "
              f"{_row['rejected_accept_window']:>7} "
              f"{_row['rejected_edge']:>5} "
              f"{_mean if _mean is None else round(_mean, 1):>10}")

# The stats must agree with what actually landed in the arrays.
for _split, _block, _y in (("DS1_TRAIN", SEGMENTATION_STATS["DS1_TRAIN"], y_train),
                           ("DS1_VAL", SEGMENTATION_STATS["DS1_VAL"], y_valid)):
    _counter = Counter(_y)
    for _lab, _row in _block.items():
        assert _row["accepted"] == _counter.get(_lab, 0), (
            f"{_split}/{_lab}: stats say {_row['accepted']} accepted but "
            f"the array holds {_counter.get(_lab, 0)}")

_pop_errors = []

for _split in ("DS1_TRAIN", "DS1_VAL"):
    for _lab, _row in SEGMENTATION_STATS[_split].items():
        _want = E10_EXPECTED[_split][_lab]
        if _row["accepted"] != _want:
            _pop_errors.append(
                f"{_split}/{_lab}: accepted {_row['accepted']}, "
                f"expected {_want}")

assert not _pop_errors, (
    "E10's shared acceptance rule is not producing the measured "
    "population: " + "; ".join(_pop_errors)
)

_cost_ds1 = sum(
    E9_ACCEPTED[sp][lb] - SEGMENTATION_STATS[sp][lb]["accepted"]
    for sp in ("DS1_TRAIN", "DS1_VAL") for lb in ("N", "S", "V"))

print(f"\n  the wider acceptance window costs {_cost_ds1} DS1 beats "
      f"relative to E9")


# =========================================================
# 13. CLASS DISTRIBUTION PLOT
# =========================================================

train_counter = Counter(y_train)

plt.figure(figsize=(6, 5))

plt.bar(
    train_counter.keys(),
    train_counter.values()
)

plt.title('Training Class Distribution')

plt.xlabel('Class')

plt.ylabel('Count')

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'class_distribution.png'),
    dpi=300
)

plt.show()


# =========================================================
# 14. ENCODE LABELS
# =========================================================

y_train_encoded = np.array([
    LABEL_TO_INT[label]
    for label in y_train
], dtype=np.int32)

y_valid_encoded = np.array([
    LABEL_TO_INT[label]
    for label in y_valid
], dtype=np.int32)

# y_test_encoded is built in section 23B, after the window is selected.


# =========================================================
# 15. NORMALIZE RR
# =========================================================

# Fitted on DS1_TRAIN only. Fitting on validation or test would leak
# their distribution into the pipeline; fitting each set separately (the
# step 1 behaviour) put the three on three different scales, which is why
# step 1 val_accuracy peaked at 0.7368, below the 0.8528 an all-N
# prediction scores on that validation set.
# Raw ratio ranges on DS1_TRAIN, before standardising. If a feature sits
# hard against RR_CLIP_MIN or RR_CLIP_MAX the clip is saturating and the
# bounds need revisiting.
print("\nDS1_TRAIN raw RR ratio features "
      f"(clipped to [{RR_CLIP_MIN}, {RR_CLIP_MAX}]):")
print(f"  {'feature':<24} {'min':>8} {'max':>8} {'mean':>8} "
      f"{'%at_min':>8} {'%at_max':>8}")

for _i, _name in enumerate(RR_FEATURE_NAMES):

    _col = RR_train[:, _i]

    print(f"  {_name:<24} {_col.min():>8.4f} {_col.max():>8.4f} "
          f"{_col.mean():>8.4f} "
          f"{100.0 * np.mean(_col <= RR_CLIP_MIN):>7.2f}% "
          f"{100.0 * np.mean(_col >= RR_CLIP_MAX):>7.2f}%")

RR_NORM_MEAN, RR_NORM_STD = fit_rr_norm(RR_train)

print("\nRR normalization fitted on DS1_TRAIN only:")
print(f"  mean (prev_rr, next_rr): {RR_NORM_MEAN.tolist()}")
print(f"  std  (prev_rr, next_rr): {RR_NORM_STD.tolist()}")

RR_train = apply_rr_norm(RR_train, RR_NORM_MEAN, RR_NORM_STD)
RR_valid = apply_rr_norm(RR_valid, RR_NORM_MEAN, RR_NORM_STD)

# RR_test is normalised in section 23B with these same DS1_TRAIN
# statistics. The RR features do not depend on the window width - they are
# built from the annotation samples - so they are identical for every arm
# and these statistics are fitted exactly once.


# =========================================================
# 16. PREPARE CNN INPUT
# =========================================================

# No expand_dims any more. Beats leave section 7 already shaped
# (pre + post, N_WAVELET_SCALES), so the Conv1D channel axis is the
# wavelet scale axis. build_model() takes its input shape from the data,
# so a sweep arm's width flows through without a code change - and because
# every parameter count in the network depends on CHANNELS and not on
# sequence length, all four arms have identical parameter counts.

def assert_cnn_input(arrays, expected_length):
    """Shape guard, reused for every sweep arm and for DS2 in 23B."""

    for _name, _arr in arrays:
        assert _arr.ndim == 3, f"{_name}: expected 3 dims, got {_arr.shape}"
        assert _arr.shape[1] == expected_length, \
            f"{_name}: expected length {expected_length}, got {_arr.shape}"
        assert _arr.shape[2] == N_WAVELET_SCALES, f"{_name}: {_arr.shape}"
        assert _arr.dtype == np.float32, f"{_name}: {_arr.dtype}"
        assert np.all(np.isfinite(_arr)), f"{_name}: non-finite values"


assert_cnn_input(
    (("train", X_train), ("valid", X_valid)),
    CONTROL_PRE + CONTROL_POST
)

print(f"\nCNN input shapes  train {X_train.shape}  "
      f"valid {X_valid.shape}   (control window)")


# =========================================================
# 17. TRAIN / VALIDATION SPLIT (patient-wise)
# =========================================================

# The split already happened at load time, by record: DS1_TRAIN and
# DS1_VAL were read as two separate datasets, so there is nothing to
# slice here and no beat from a validation patient can reach training.
#
# This previously used a stratified beat-level train_test_split over all
# of DS1, which put the same patient on both sides. Validation accuracy
# reached 0.9885 against a true DS2 accuracy of 0.9294, and EarlyStopping
# with restore_best_weights selected the final model on that leaked score.

X_tr = X_train
RR_tr = RR_train
y_tr = y_train_encoded

X_val = X_valid
RR_val = RR_valid
y_val = y_valid_encoded

print(f"\nTrain beats: {len(y_tr)}   Validation beats: {len(y_val)}")


# =========================================================
# 18. AUGMENT TRAINING DATA (REMOVED - E6)
# =========================================================

# Duplicate oversampling is gone again. Class balance now comes from the
# SAMPLER in section 19B: every minibatch holds an equal number of N, S
# and V beats, drawn WITH REPLACEMENT from the real beats. No synthetic
# data is created, so hard constraint 2 holds.
#
# The *_aug names are kept so nothing downstream needs to change.

X_tr_aug = X_tr
RR_tr_aug = RR_tr
y_tr_aug = y_tr

print(f"\nTraining samples (no oversampling): {len(y_tr_aug)}")
print(f"Training distribution: {Counter(y_tr_aug)}")


# =========================================================
# 19. ONE HOT ENCODING
# =========================================================

y_tr_aug_cat = tf.keras.utils.to_categorical(
    y_tr_aug,
    num_classes=NUM_CLASSES
)

y_val_cat = tf.keras.utils.to_categorical(
    y_val,
    num_classes=NUM_CLASSES
)

# y_test_cat is built in section 23B, after the window is selected.


# =========================================================
# 19B. BALANCED BATCH SAMPLING
# =========================================================

# De Waele et al. (2026) evaluate on MIT-BIH DS1->DS2, 3 classes, F and Q
# excluded - our exact protocol - and report S-F1 0.4861 with RR late
# fusion. Our S PRECISION (0.4214) already beats their 0.3327; the whole
# gap is recall (0.1972 against 0.9116). Their INCART training set holds
# 605 SVEB beats against our 670, so this is not a data-quantity effect.
# The mechanism is balanced batch sampling.
#
# Every minibatch holds an equal number of N, S and V beats. S beats are
# drawn WITH REPLACEMENT from the 670 real ones - repetition inside a
# stream, not fabricated data, so hard constraint 2 holds.

SAMPLER = "balanced_batch"

# E10 holds the sampler at E6's 1:1:1. E7 swept the ratio over [1, 2, 3, 4]
# and found it SATURATED: S recall moved +0.0005 between ratio 1 and ratio
# 2 while S precision fell 0.3934 -> 0.2789, and validation macro-F1
# spanned only 0.0178 across the whole grid. That question is settled, and
# the sweep in section 22 is over the WINDOW WIDTH instead.
#
# Ratio 1.0 gives weights [1/3, 1/3, 1/3] - exactly E6's sampler, and E7's
# ratio-1.0 arm reproduced E6's selection (0.5540 at epoch 6) exactly.
SAMPLER_RATIO = 1.0

SAMPLING_WEIGHTS = weights_for_ratio(SAMPLER_RATIO)

STEPS_PER_EPOCH = int(np.ceil(len(y_tr_aug) / BATCH_SIZE))


def weights_for_ratio(ratio, n_classes=NUM_CLASSES, minority_index=1):
    """Sampling weights over [N, S, V] for an S:N drawing ratio.

    Weights are proportional to [1, ratio, 1] and normalised to sum to 1,
    so ratio 1.0 gives exactly [1/3, 1/3, 1/3].
    """

    raw = [1.0] * n_classes
    raw[minority_index] = float(ratio)

    total = sum(raw)

    return [w / total for w in raw]


def make_balanced_dataset(weights):
    """One infinite, class-balanced, batched stream for the given weights.

    Each class becomes its own shuffled, repeated dataset; the classes are
    interleaved by sample_from_datasets according to `weights`. S beats
    repeat from the real 670 - nothing synthetic is created.
    """

    per_class = []

    for class_index in range(NUM_CLASSES):

        mask = (y_tr_aug == class_index)
        n_class = int(mask.sum())

        class_ds = tf.data.Dataset.from_tensor_slices((
            (X_tr_aug[mask], RR_tr_aug[mask]),
            y_tr_aug_cat[mask]
        ))

        class_ds = class_ds.shuffle(
            n_class,
            seed=SEED,
            reshuffle_each_iteration=True
        ).repeat()

        per_class.append(class_ds)

    stream = tf.data.Dataset.sample_from_datasets(
        per_class,
        weights=weights,
        seed=SEED
    )

    return stream.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


for _class_index in range(NUM_CLASSES):
    print(f"  sampler class {INT_TO_LABEL[_class_index]}: "
          f"{int((y_tr_aug == _class_index).sum())} real beats")

print(f"\nSampler ratio {SAMPLER_RATIO} (fixed), batch {BATCH_SIZE}, "
      f"{STEPS_PER_EPOCH} steps/epoch "
      f"({len(y_tr_aug)} training beats / {BATCH_SIZE})")
print(f"  weights [N, S, V] = {[round(v, 6) for v in SAMPLING_WEIGHTS]}, "
      f"sum {sum(SAMPLING_WEIGHTS):.4f}")


# =========================================================
# 20. BUILD MODEL
# =========================================================

def build_model(ecg_shape, rr_shape):

    # ECG INPUT
    ecg_input = Input(
        shape=ecg_shape,
        name='ecg_input'
    )

    # CNN BLOCK 1
    x = Conv1D(
        32,
        kernel_size=5,
        activation='relu',
        padding='same'
    )(ecg_input)

    x = BatchNormalization()(x)

    x = Conv1D(
        32,
        kernel_size=5,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.2)(x)

    # CNN BLOCK 2
    x = Conv1D(
        64,
        kernel_size=5,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = Conv1D(
        64,
        kernel_size=5,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.25)(x)

    # CNN BLOCK 3
    x = Conv1D(
        128,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = Conv1D(
        128,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.3)(x)

    # BiLSTM
    x = Bidirectional(
        LSTM(
            64,
            return_sequences=False
        )
    )(x)

    x = Dropout(0.4)(x)

    # RR INPUT
    rr_input = Input(
        shape=rr_shape,
        name='rr_input'
    )

    rr_branch = Dense(
        16,
        activation='relu'
    )(rr_input)

    rr_branch = Dropout(0.2)(rr_branch)

    # CONCATENATE
    combined = Concatenate()([
        x,
        rr_branch
    ])

    combined = Dense(
        128,
        activation='relu'
    )(combined)

    combined = Dropout(0.5)(combined)

    combined = Dense(
        64,
        activation='relu'
    )(combined)

    combined = Dropout(0.4)(combined)

    # OUTPUT
    output = Dense(
        NUM_CLASSES,
        activation='softmax'
    )(combined)

    model = Model(
        inputs=[ecg_input, rr_input],
        outputs=output
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=ADAM_LEARNING_RATE
        ),

        # E6: plain categorical cross-entropy. De Waele et al. reach
        # S-F1 0.4861 on this protocol with standard CE plus a balanced
        # sampler; stacking focal loss on top of balanced batches risks
        # over-firing the S head. categorical_focal_loss is now unused.
        loss='categorical_crossentropy',

        metrics=['accuracy']
    )

    return model


# The model is NOT built here any more. Section 22 builds one per ratio
# in the sweep and keeps the one validation selects.

# No ECG_INPUT_SHAPE constant any more: the sweep in section 22 builds one
# model per window and takes the shape from that arm's own array.
RR_INPUT_SHAPE = (len(RR_FEATURE_NAMES),)


# =========================================================
# 21. CALLBACKS
# =========================================================

class ValidationMetrics(tf.keras.callbacks.Callback):
    """Per-epoch macro-F1 and per-class metrics on the validation set.

    Aggregate val_accuracy is not a usable selection signal here: N is
    85.3% of DS1_VAL, so a model that never predicts S can still look
    fine. val_loss is worse still - under focal loss it climbs while
    accuracy stays flat, because the model becomes more confidently wrong
    rather than less correct.

    Writes 'val_macro_f1' and 'val_f1_N' / 'val_f1_S' / 'val_f1_V' into
    the logs dict, so EarlyStopping, ReduceLROnPlateau and History all
    see them.
    """

    def __init__(self, x_val, y_val_int, class_names):

        super().__init__()

        self.x_val = x_val
        self.y_val_int = np.asarray(y_val_int)
        self.class_names = list(class_names)
        self.records = []

    def on_epoch_end(self, epoch, logs=None):

        if logs is None:
            logs = {}

        y_pred = np.argmax(
            self.model.predict(self.x_val, verbose=0),
            axis=1
        )

        labels = list(range(len(self.class_names)))

        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_val_int,
            y_pred,
            labels=labels,
            zero_division=0
        )

        macro_f1 = float(np.mean(f1))

        # Into logs so the other callbacks can monitor them.
        logs['val_macro_f1'] = macro_f1

        for i, name in enumerate(self.class_names):
            logs[f'val_f1_{name}'] = float(f1[i])

        cm = confusion_matrix(
            self.y_val_int,
            y_pred,
            labels=labels
        )

        self.records.append({
            "epoch": int(epoch) + 1,
            "val_macro_f1": macro_f1,
            "val_f1": {
                n: float(f1[i]) for i, n in enumerate(self.class_names)
            },
            "val_recall": {
                n: float(recall[i]) for i, n in enumerate(self.class_names)
            },
            "val_precision": {
                n: float(precision[i]) for i, n in enumerate(self.class_names)
            },
            "val_support": {
                n: int(support[i]) for i, n in enumerate(self.class_names)
            },
            "val_confusion_matrix": cm.tolist()
        })

        print(
            f"  epoch {epoch + 1:>3}  val macro-F1 {macro_f1:.4f}   "
            + "  ".join(
                f"{n}-F1 {f1[i]:.4f}"
                for i, n in enumerate(self.class_names)
            )
        )


CLASS_NAMES = [INT_TO_LABEL[i] for i in range(NUM_CLASSES)]

def make_callbacks():
    """Fresh callback objects for one sweep run.

    EarlyStopping and ReduceLROnPlateau carry mutable state (wait, best,
    stopped_epoch, best_weights) and ValidationMetrics accumulates its
    records list, so they cannot be shared across ratios.

    Returns (callbacks, val_metrics_cb, early_stopping). val_metrics_cb
    MUST be first: it writes 'val_macro_f1' into the shared logs dict and
    the two below read that key in the same on_epoch_end pass.
    """

    val_cb = ValidationMetrics(
        [X_val, RR_val],
        y_val,
        CLASS_NAMES
    )

    early = EarlyStopping(
        monitor='val_macro_f1',
        mode='max',
        patience=10,
        restore_best_weights=True
    )

    plateau = ReduceLROnPlateau(
        monitor='val_macro_f1',
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    return [val_cb, early, plateau], val_cb, early


# =========================================================
# 22. TRAIN MODEL - FIXED-WINDOW WIDTH SWEEP, SELECTED ON VALIDATION
# =========================================================

# One model per window width, each from an identical seed reset and a
# fresh build_model(). Selection is on best val macro-F1 over DS1_VAL.
#
# THE TEST SET IS NOT TOUCHED IN THIS LOOP - and this time that is
# structural, not a promise. DS2 has not been READ yet: X_test, RR_test,
# y_test_encoded and y_test_cat do not exist as names until section 23B,
# which runs after SELECTED_WINDOW is fixed. The BETA and sampler sweeps
# kept DS2 in memory throughout and relied on the loop not mentioning it;
# here there is nothing to mention.
#
# Every arm draws from the SAME beats: acceptance was decided once, in
# section 7, by the largest window in the grid. The asserts below refuse
# to train if any arm's labels or RR features differ from the control's.

WINDOW_SWEEP = []
PARAM_COUNTS = {}

model = None
history = None
val_metrics_cb = None
early_stopping = None

SELECTED_WINDOW = None
SELECTED_INPUT_LENGTH = None
BEST_SWEEP_VAL = None

best_X_tr = None
best_X_val = None


def load_ds1_for_window(pre, post):
    """DS1_TRAIN and DS1_VAL beats at one window width.

    Labels and RR features are checked byte-identical to the control
    arm's - if they are not, the arms are not scoring the same beats and
    the sweep would confound width with population.
    """

    X_tr_w, y_tr_w, RR_tr_w, _raw, _st = load_dataset(
        DS1_TRAIN,
        DATA_DIR,
        pre,
        post,
        LEAD_INDEX
    )

    assert np.array_equal(y_tr_w, Y_TRAIN_REF), (
        f"window ({pre},{post}): DS1_TRAIN labels differ from the control"
    )
    assert np.array_equal(RR_tr_w, RR_TRAIN_RAW), (
        f"window ({pre},{post}): DS1_TRAIN RR features differ from the "
        f"control"
    )

    val_parts = []
    y_val_parts = []
    RR_val_parts = []

    for rec in DS1_VAL:

        X_rec_w, y_rec_w, RR_rec_w, _r, _s = load_dataset(
            [rec],
            DATA_DIR,
            pre,
            post,
            LEAD_INDEX
        )

        val_parts.append(X_rec_w)
        y_val_parts.append(y_rec_w)
        RR_val_parts.append(RR_rec_w)

    X_val_w = np.concatenate(val_parts, axis=0)

    assert np.array_equal(np.concatenate(y_val_parts, axis=0), Y_VALID_REF), (
        f"window ({pre},{post}): DS1_VAL labels differ from the control"
    )
    assert np.array_equal(np.concatenate(RR_val_parts, axis=0),
                          RR_VALID_RAW), (
        f"window ({pre},{post}): DS1_VAL RR features differ from the control"
    )

    return X_tr_w, X_val_w


for sweep_pre, sweep_post in WINDOW_GRID:

    sweep_width = sweep_pre + sweep_post

    print("\n" + "=" * 60)
    print(f"SWEEP: window ({sweep_pre}, {sweep_post}) -> {sweep_width} "
          f"samples")
    print("=" * 60)

    if (sweep_pre, sweep_post) == (CONTROL_PRE, CONTROL_POST):
        # Already loaded in section 12; do not pay for it twice.
        X_tr_sweep = X_train
        X_val_sweep = X_valid
    else:
        X_tr_sweep, X_val_sweep = load_ds1_for_window(sweep_pre, sweep_post)

    assert_cnn_input(
        (("train", X_tr_sweep), ("valid", X_val_sweep)),
        sweep_width
    )

    # The helpers close over these module-level names, so rebinding them
    # is what points the sampler and the callbacks at this arm's beats.
    # RR_tr_aug, y_tr_aug_cat, RR_val and y_val are window-independent and
    # are deliberately NOT rebound.
    X_tr_aug = X_tr_sweep
    X_val = X_val_sweep

    # Identical initialisation for every width, so any difference in val
    # macro-F1 is attributable to the width and not to weight init.
    reset_seeds()

    sweep_dataset = make_balanced_dataset(SAMPLING_WEIGHTS)

    sweep_callbacks, sweep_val_cb, sweep_early = make_callbacks()

    sweep_model = build_model(
        ecg_shape=X_tr_sweep.shape[1:],
        rr_shape=RR_INPUT_SHAPE
    )

    PARAM_COUNTS[sweep_width] = int(sweep_model.count_params())

    print(f"  input shape {X_tr_sweep.shape[1:]}, "
          f"{PARAM_COUNTS[sweep_width]:,} parameters")

    if (sweep_pre, sweep_post) == WINDOW_GRID[0]:
        sweep_model.summary()

    sweep_history = sweep_model.fit(

        sweep_dataset,

        validation_data=(
            [X_val, RR_val],
            y_val_cat
        ),

        epochs=EPOCHS,

        steps_per_epoch=STEPS_PER_EPOCH,

        callbacks=sweep_callbacks,

        verbose=1
    )

    sweep_curve = [
        float(rec["val_macro_f1"])
        for rec in sweep_val_cb.records
    ]

    if sweep_curve:
        sweep_best = float(max(sweep_curve))
        sweep_best_epoch = int(sweep_curve.index(sweep_best)) + 1
    else:
        sweep_best = float("-inf")
        sweep_best_epoch = None

    WINDOW_SWEEP.append({
        "pre": int(sweep_pre),
        "post": int(sweep_post),
        "input_length": int(sweep_width),
        "total_parameters": PARAM_COUNTS[sweep_width],
        "best_val_macro_f1": sweep_best,
        "best_epoch": sweep_best_epoch,
        "epochs_run": len(sweep_curve),
        "early_stopping_fired": int(sweep_early.stopped_epoch) > 0,
        "val_macro_f1_curve": sweep_curve
    })

    print(f"\n  window {sweep_width}: best val macro-F1 {sweep_best:.4f} "
          f"at epoch {sweep_best_epoch} of {len(sweep_curve)}")

    if BEST_SWEEP_VAL is None or sweep_best > BEST_SWEEP_VAL:

        BEST_SWEEP_VAL = sweep_best
        SELECTED_WINDOW = (int(sweep_pre), int(sweep_post))
        SELECTED_INPUT_LENGTH = int(sweep_width)

        # Rebinding drops the previous winner's arrays.
        best_X_tr = X_tr_sweep
        best_X_val = X_val_sweep

        # Keep this run's objects; sections 22B onward use these names.
        model = sweep_model
        history = sweep_history
        val_metrics_cb = sweep_val_cb
        early_stopping = sweep_early

    # Drop this arm's local handles. If it won, best_X_* still holds them;
    # if it lost, this is what frees up to 0.83 GB before the next arm.
    X_tr_sweep = None
    X_val_sweep = None


# The control arm's arrays were held by X_train / X_valid for the whole
# sweep. If it lost, release them now.
X_train = None
X_valid = None

# Point the module-level names at the SELECTED arm, so sections 22B
# onward - validation diagnostics, threshold tuning, DS2 evaluation - all
# operate on the window that was actually chosen.
X_tr_aug = best_X_tr
X_val = best_X_val


# --- selection, still without touching DS2 ------------------------------

WINDOW_SELECTION_CRITERION = (
    "highest best val macro-F1 on DS1_VAL across WINDOW_GRID; DS2 was not "
    "loaded at all during the sweep - it is read for the first time in "
    "section 23B, after this selection, and scored exactly once"
)

# Conv1D, BatchNormalization and Dense parameter counts depend on the
# CHANNEL count, and the BiLSTM runs with return_sequences=False so it
# depends on the feature dimension and its units - none of them on the
# sequence length. Every arm must therefore have identical parameters; if
# it does not, something width-dependent crept into build_model.
_distinct_params = sorted(set(PARAM_COUNTS.values()))

assert len(_distinct_params) == 1, (
    f"window width changed the parameter count: {PARAM_COUNTS}. The "
    f"architecture must be shape-agnostic for this sweep to be a "
    f"single-variable comparison."
)

print("\n" + "=" * 60)
print("WINDOW WIDTH SWEEP RESULT")
print("=" * 60)
print(f"  {'window':>14} {'length':>7} {'params':>9} "
      f"{'best val macro-F1':>19} {'best epoch':>11} {'epochs':>7}")

for entry in WINDOW_SWEEP:
    marker = ("  <-- selected"
              if (entry["pre"], entry["post"]) == SELECTED_WINDOW else "")
    print(f"  ({entry['pre']:>3}, {entry['post']:>3})   "
          f"{entry['input_length']:>7} {entry['total_parameters']:>9,} "
          f"{entry['best_val_macro_f1']:>19.4f} "
          f"{str(entry['best_epoch']):>11} {entry['epochs_run']:>7}"
          f"{marker}")

print(f"\n  selected window : {SELECTED_WINDOW} "
      f"-> {SELECTED_INPUT_LENGTH} samples")
print(f"  parameters      : {_distinct_params[0]:,} (identical for all "
      f"{len(WINDOW_GRID)} arms)")
print(f"  criterion: {WINDOW_SELECTION_CRITERION}")


# =========================================================
# 22B. VALIDATION DIAGNOSTICS
# =========================================================

val_epoch_records = val_metrics_cb.records

if val_epoch_records:

    best_record = max(
        val_epoch_records,
        key=lambda r: r["val_macro_f1"]
    )

    BEST_EPOCH = int(best_record["epoch"])
    BEST_VAL_MACRO_F1 = float(best_record["val_macro_f1"])

else:

    BEST_EPOCH = None
    BEST_VAL_MACRO_F1 = None

# restore_best_weights only actually restores when EarlyStopping fires.
# If training runs to the final epoch the last weights are kept, so record
# what happened rather than assuming BEST_EPOCH is what got evaluated.
EARLY_STOPPED_EPOCH = int(early_stopping.stopped_epoch)
WEIGHTS_RESTORED = EARLY_STOPPED_EPOCH > 0

print("\nValidation model selection:")
print(f"  epochs run                 : {len(val_epoch_records)}")
print(f"  best epoch by val macro-F1 : {BEST_EPOCH}")
print(f"  best val macro-F1          : {BEST_VAL_MACRO_F1}")
print(f"  early stopping fired       : {WEIGHTS_RESTORED}"
      f"  (stopped_epoch={EARLY_STOPPED_EPOCH})")

if not WEIGHTS_RESTORED:
    print("  NOTE: training reached the last epoch, so the FINAL weights "
          "are in the model, not the best-epoch weights.")

# --- per-record breakdown on the model actually being evaluated ---------
# Is one validation record an outlier, or do all three behave the same?

y_val_pred = np.argmax(
    model.predict([X_val, RR_val], verbose=0),
    axis=1
)

val_per_record = {}

print("\nPer-record validation breakdown:")

for rec in DS1_VAL:

    mask = (val_record_ids == rec)

    y_true_rec = y_val[mask]
    y_pred_rec = y_val_pred[mask]

    rec_labels = list(range(NUM_CLASSES))

    p_rec, r_rec, f_rec, sup_rec = precision_recall_fscore_support(
        y_true_rec,
        y_pred_rec,
        labels=rec_labels,
        zero_division=0
    )

    entry = {
        "n_beats": int(mask.sum()),
        "accuracy": float(np.mean(y_true_rec == y_pred_rec)),
        "macro_f1": float(np.mean(f_rec)),
        "support": {
            INT_TO_LABEL[i]: int(sup_rec[i]) for i in rec_labels
        },
        "recall": {
            INT_TO_LABEL[i]: float(r_rec[i]) for i in rec_labels
        },
        "precision": {
            INT_TO_LABEL[i]: float(p_rec[i]) for i in rec_labels
        },
        "f1": {
            INT_TO_LABEL[i]: float(f_rec[i]) for i in rec_labels
        },
        "confusion_matrix": confusion_matrix(
            y_true_rec,
            y_pred_rec,
            labels=rec_labels
        ).tolist()
    }

    val_per_record[rec] = entry

    print(
        f"  record {rec}: {entry['n_beats']:>5} beats  "
        f"acc {entry['accuracy']:.4f}  macro-F1 {entry['macro_f1']:.4f}   "
        + "  ".join(
            f"{INT_TO_LABEL[i]} n={sup_rec[i]:>4} "
            f"rec={r_rec[i]:.4f}"
            for i in rec_labels
        )
    )


# =========================================================
# 22C. DECISION-THRESHOLD TUNING (VALIDATION ONLY)
# =========================================================

# DS2 is NOT involved here. The only probabilities passed to the search
# come from X_val / RR_val.

val_prob_for_threshold = model.predict(
    [X_val, RR_val],
    batch_size=BATCH_SIZE,
    verbose=0
)

THRESHOLD_WEIGHTS, THRESHOLD_VAL_MACRO_F1, THRESHOLD_SEARCH_LOG = \
    tune_decision_weights(
        val_prob_for_threshold,
        y_val,
        NUM_CLASSES,
        THRESHOLD_GRID,
        THRESHOLD_MAX_PASSES
    )

THRESHOLD_VAL_MACRO_F1_ARGMAX = macro_f1_from_weights(
    val_prob_for_threshold,
    y_val,
    [1.0] * NUM_CLASSES,
    list(range(NUM_CLASSES))
)

print("\nDecision-threshold tuning (validation only):")
print(f"  grid: {[round(g, 4) for g in THRESHOLD_GRID]}")
print(f"  passes run: {THRESHOLD_SEARCH_LOG[-1]['pass']} "
      f"of {THRESHOLD_MAX_PASSES}")
print(f"  plain argmax val macro-F1 : "
      f"{THRESHOLD_VAL_MACRO_F1_ARGMAX:.4f}")
print(f"  tuned        val macro-F1 : {THRESHOLD_VAL_MACRO_F1:.4f}")

for _i in range(NUM_CLASSES):
    print(f"    w_{INT_TO_LABEL[_i]} = {THRESHOLD_WEIGHTS[_i]:.4f}")

print("  weights are now FROZEN and applied to DS2 exactly once.")


# =========================================================
# 23. TRAINING CURVES
# =========================================================

plt.figure(figsize=(12, 5))

# LOSS
plt.subplot(1, 2, 1)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

# ACCURACY
plt.subplot(1, 2, 2)

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Training and Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'training_curves.png'),
    dpi=300
)

plt.show()


# =========================================================
# 23B. LOAD DS2  (first read of the test set, after selection)
# =========================================================

# This is the first time DS2 is read in this file. The window was chosen
# on DS1_VAL in section 22 and the decision thresholds were tuned on
# DS1_VAL in section 22C; both are already fixed by the time these arrays
# exist, so nothing downstream can leak back into a choice.
#
# DS2 is extracted at the SELECTED window, under the same acceptance rule
# every arm used - the largest window in the grid plus the span cap - so
# the test population matches the training population exactly.

print("\n" + "=" * 60)
print(f"LOADING DS2 at the selected window {SELECTED_WINDOW} "
      f"({SELECTED_INPUT_LENGTH} samples)")
print("=" * 60)

X_test, y_test, RR_test, _x_raw_test, SEG_STATS_TEST = load_dataset(
    DS2,
    DATA_DIR,
    SELECTED_WINDOW[0],
    SELECTED_WINDOW[1],
    LEAD_INDEX
)

SEGMENTATION_STATS["DS2"] = summarize_segmentation_stats(SEG_STATS_TEST)

print("\nTest shape :", X_test.shape)
print("Original Test Distribution:")
print(Counter(y_test))

print(f"\n  {'split':<10} {'cls':>3} {'annot':>7} {'accept':>7} {'E9':>7} "
      f"{'diff':>6} {'>cap':>6} {'window':>7}")
for _lab, _row in SEGMENTATION_STATS["DS2"].items():
    _e9 = E9_ACCEPTED["DS2"][_lab]
    print(f"  {'DS2':<10} {_lab:>3} {_row['total_annotated']:>7} "
          f"{_row['accepted']:>7} {_e9:>7} {_row['accepted'] - _e9:>6} "
          f"{_row['rejected_max_span']:>6} "
          f"{_row['rejected_accept_window']:>7}")

_counter = Counter(y_test)
for _lab, _row in SEGMENTATION_STATS["DS2"].items():
    assert _row["accepted"] == _counter.get(_lab, 0), (
        f"DS2/{_lab}: stats say {_row['accepted']} accepted but the array "
        f"holds {_counter.get(_lab, 0)}")

for _lab, _row in SEGMENTATION_STATS["DS2"].items():
    assert _row["accepted"] == E10_EXPECTED["DS2"][_lab], (
        f"DS2/{_lab}: accepted {_row['accepted']}, expected "
        f"{E10_EXPECTED['DS2'][_lab]}")

y_test_encoded = np.array([
    LABEL_TO_INT[label]
    for label in y_test
], dtype=np.int32)

# Same DS1_TRAIN statistics fitted in section 15. Never refitted.
RR_test = apply_rr_norm(RR_test, RR_NORM_MEAN, RR_NORM_STD)

assert_cnn_input((("test", X_test),), SELECTED_INPUT_LENGTH)

y_test_cat = tf.keras.utils.to_categorical(
    y_test_encoded,
    num_classes=NUM_CLASSES
)

TEST_POPULATION = {
    str(_lab): int(_count) for _lab, _count in Counter(y_test).items()
}

print(f"\n  DS2 beats scored: {len(y_test):,} "
      f"(E9 scored {sum(E9_ACCEPTED['DS2'].values()):,})")


# =========================================================
# 24. TEST EVALUATION
# =========================================================

y_pred_prob = model.predict(
    [X_test, RR_test],
    batch_size=BATCH_SIZE
)

y_pred_enc = np.argmax(
    y_pred_prob,
    axis=1
)

acc = accuracy_score(
    y_test_encoded,
    y_pred_enc
)

print(f"\nTest Accuracy: {acc:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test_encoded,
    y_pred_enc,
    target_names=[
        'N',
        'S',
        'V'
    ],
    digits=4
))

cm = confusion_matrix(
    y_test_encoded,
    y_pred_enc
)

print("\nConfusion Matrix:\n")
print(cm)


# --- the frozen validation-tuned decision rule, applied once ------------

y_pred_enc_tuned = np.argmax(
    y_pred_prob * np.asarray(THRESHOLD_WEIGHTS, dtype=np.float64),
    axis=1
)

acc_tuned = accuracy_score(
    y_test_encoded,
    y_pred_enc_tuned
)

cm_tuned = confusion_matrix(
    y_test_encoded,
    y_pred_enc_tuned
)

report_argmax = classification_report(
    y_test_encoded,
    y_pred_enc,
    target_names=['N', 'S', 'V'],
    digits=4,
    output_dict=True
)

report_tuned = classification_report(
    y_test_encoded,
    y_pred_enc_tuned,
    target_names=['N', 'S', 'V'],
    digits=4,
    output_dict=True
)

print(f"\nTuned weights {[round(w, 4) for w in THRESHOLD_WEIGHTS]} "
      f"(frozen on validation)")

print(f"Tuned Test Accuracy: {acc_tuned:.4f}")

print("\nClassification Report (tuned):\n")

print(classification_report(
    y_test_encoded,
    y_pred_enc_tuned,
    target_names=['N', 'S', 'V'],
    digits=4
))

print("\nConfusion Matrix (tuned):\n")
print(cm_tuned)

print("\nargmax vs tuned on DS2:")
print(f"  {'metric':<16} {'argmax':>10} {'tuned':>10}")
for _k in ('macro avg', 'S', 'V', 'N'):
    print(f"  {_k + ' F1':<16} "
          f"{report_argmax[_k]['f1-score']:>10.4f} "
          f"{report_tuned[_k]['f1-score']:>10.4f}")
print(f"  {'S recall':<16} {report_argmax['S']['recall']:>10.4f} "
      f"{report_tuned['S']['recall']:>10.4f}")
print(f"  {'S precision':<16} {report_argmax['S']['precision']:>10.4f} "
      f"{report_tuned['S']['precision']:>10.4f}")
print(f"  {'accuracy':<16} {acc:>10.4f} {acc_tuned:>10.4f}")


# =========================================================
# 25. CONFUSION MATRIX VISUALIZATION
# =========================================================

classes = ['N', 'S', 'V']

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)

plt.title('Confusion Matrix')

plt.xlabel('Predicted Label')

plt.ylabel('True Label')

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'confusion_matrix.png'),
    dpi=300
)

plt.show()


# =========================================================
# 26. ROC CURVES
# =========================================================

y_test_bin = label_binarize(
    y_test_encoded,
    classes=[0, 1, 2]
)

plt.figure(figsize=(8, 6))

per_class_roc_auc = {}

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_pred_prob[:, i]
    )

    roc_auc = auc(fpr, tpr)

    per_class_roc_auc[INT_TO_LABEL[i]] = float(roc_auc)

    plt.plot(
        fpr,
        tpr,
        label=f'Class {i} AUC = {roc_auc:.3f}'
    )

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curves')

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'roc_curves.png'),
    dpi=300
)

plt.show()


# =========================================================
# 27. PRECISION-RECALL CURVES
# =========================================================

plt.figure(figsize=(8, 6))

for i in range(NUM_CLASSES):

    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i],
        y_pred_prob[:, i]
    )

    plt.plot(
        recall,
        precision,
        label=f'Class {i}'
    )

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision-Recall Curves')

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'precision_recall_curves.png'),
    dpi=300
)

plt.show()


# =========================================================
# 28. SAMPLE ECG VISUALIZATION
# =========================================================

plt.figure(figsize=(12, 6))

for i in range(3):

    plt.subplot(3, 1, i + 1)

    plt.plot(
        X_test[i].squeeze()
    )

    plt.title(
        f'True Label: {INT_TO_LABEL[y_test_encoded[i]]}'
    )

plt.tight_layout()

plt.savefig(
    os.path.join(RUN_DIR, 'sample_ecg_signals.png'),
    dpi=300
)

plt.show()


# =========================================================
# 29. SAVE MODEL
# =========================================================

MODEL_PATH = os.path.join(
    RUN_DIR,
    "best_ecg_multiclass_model.keras"
)

model.save(MODEL_PATH)

print(
    f"\nModel saved as {MODEL_PATH}"
)


# =========================================================
# 30. SAVE METRICS AND HISTORY
# =========================================================

metrics = {

    "run_name": RUN_NAME,

    "timestamp": datetime.datetime.now().isoformat(),

    "tensorflow_version": tf.__version__,

    "gpus": [str(gpu) for gpu in gpus],

    "config": {

        "SEED": SEED,

        "EPOCHS": EPOCHS,

        "BATCH_SIZE": BATCH_SIZE,

        "PRE_SAMPLES": PRE_SAMPLES,

        "POST_SAMPLES": POST_SAMPLES,

        "SEGMENT_LENGTH": SEGMENT_LENGTH,   # E9's width, for reference

        "LEAD_INDEX": LEAD_INDEX,

        # E9: E6's fixed window, E8's acceptance rule. The span cap
        # E10: acceptance is decided once by the LARGEST window in
        # WINDOW_GRID plus the span cap, so every swept width trains and
        # tests on identical beats. No mask, no padding.
        "segmentation": "fixed_window_sweep_on_shared_population",

        "max_span_samples": MAX_SPAN_SAMPLES,

        "max_span_seconds": MAX_SPAN_SAMPLES / SAMPLING_RATE_HZ,

        "input_length": SELECTED_INPUT_LENGTH,

        "n_input_channels": N_WAVELET_SCALES,

        "n_wavelet_scales": N_WAVELET_SCALES,

        "mask_channel_index": None,

        "loss": "categorical_crossentropy",

        "focal_loss_used": False,

        "focal_loss_alpha": FOCAL_ALPHA,

        "focal_loss_gamma": FOCAL_GAMMA,

        "oversampling": False,

        "sampler": SAMPLER,

        "sampler_ratio": SAMPLER_RATIO,

        "sampling_weights": SAMPLING_WEIGHTS,

        # E10: the swept variable.
        "window_grid": [list(w) for w in WINDOW_GRID],

        "window_widths": WINDOW_WIDTHS,

        "selected_window": list(SELECTED_WINDOW),

        "selected_input_length": SELECTED_INPUT_LENGTH,

        "window_selection_criterion": WINDOW_SELECTION_CRITERION,

        "accept_window": [ACCEPT_PRE, ACCEPT_POST],

        "accept_window_length": ACCEPT_PRE + ACCEPT_POST,

        "steps_per_epoch": STEPS_PER_EPOCH,

        "total_parameters": int(model.count_params()),

        "adam_learning_rate": ADAM_LEARNING_RATE,

        "DS1": DS1,

        "DS2": DS2,

        "ds1_train": DS1_TRAIN,

        "ds1_val": DS1_VAL,

        "val_selection_rule": VAL_SELECTION_RULE,

        "wavelet_scales": WAVELET_WIDTHS,

        "wavelet_centre_frequencies": WAVELET_CENTRE_FREQS_HZ,

        "wavelet_target_frequencies": WAVELET_TARGET_FREQS_HZ,

        "sampling_rate_hz": SAMPLING_RATE_HZ,

        "rr_feature_names": RR_FEATURE_NAMES,

        "rr_local_window": RR_LOCAL_WINDOW,

        "rr_clip": [RR_CLIP_MIN, RR_CLIP_MAX],

        "rr_norm_mean": RR_NORM_MEAN.tolist(),

        "rr_norm_std": RR_NORM_STD.tolist()
    },

    # Per class and per split: accepted, rejected by E8's 2-second span
    # cap, rejected because E6's fixed window runs off the record start,
    # and the mean/std of the R-1..R+1 span. E9 scores 42,148 DS2 beats
    # against E8's 42,154; the 6 missing beats are 5 N and 1 V, all
    # first-of-record, and the S class is identical at 1,565 both sides.
    "segmentation_stats": SEGMENTATION_STATS,

    "e9_population": E9_ACCEPTED,

    "expected_population": E10_EXPECTED,

    "population_rule": (
        "a beat is accepted iff its R-1..R+1 span is at most "
        "MAX_SPAN_SAMPLES AND the LARGEST window in WINDOW_GRID "
        "(ACCEPT_PRE, ACCEPT_POST) fits inside the record. Applied once, "
        "before the sweep, so all arms train and test on identical beats. "
        "This is stricter than E9's rule, which only had to fit a "
        "234-sample window, so E10 scores fewer beats than E9."
    ),

    "train_distribution": {
        str(label): int(count)
        for label, count in Counter(y_train).items()
    },

    "val_distribution": {
        str(label): int(count)
        for label, count in Counter(y_valid).items()
    },

    "test_distribution": {
        str(label): int(count)
        for label, count in Counter(y_test).items()
    },

    "test_accuracy": float(acc),

    "classification_report": classification_report(
        y_test_encoded,
        y_pred_enc,
        target_names=[
            'N',
            'S',
            'V'
        ],
        digits=4,
        output_dict=True
    ),

    "confusion_matrix": cm.tolist(),

    "per_class_roc_auc": per_class_roc_auc,

    "window_sweep": WINDOW_SWEEP,

    "parameter_counts_by_width": {
        str(_w): int(_n) for _w, _n in sorted(PARAM_COUNTS.items())
    },

    "threshold_weights": [float(w) for w in THRESHOLD_WEIGHTS],

    "threshold_class_order": [INT_TO_LABEL[i] for i in range(NUM_CLASSES)],

    "threshold_val_macro_f1": float(THRESHOLD_VAL_MACRO_F1),

    "threshold_val_macro_f1_argmax": float(THRESHOLD_VAL_MACRO_F1_ARGMAX),

    "threshold_grid": THRESHOLD_GRID,

    "threshold_search_log": THRESHOLD_SEARCH_LOG,

    "test_argmax": {
        "accuracy": float(acc),
        "classification_report": report_argmax,
        "confusion_matrix": cm.tolist()
    },

    "test_tuned": {
        "accuracy": float(acc_tuned),
        "classification_report": report_tuned,
        "confusion_matrix": cm_tuned.tolist(),
        "weights": [float(w) for w in THRESHOLD_WEIGHTS]
    },

    "best_epoch": BEST_EPOCH,

    "best_val_macro_f1": BEST_VAL_MACRO_F1,

    "early_stopping_fired": WEIGHTS_RESTORED,

    "val_per_record": val_per_record
}

METRICS_PATH = os.path.join(
    RUN_DIR,
    "metrics.json"
)

with open(METRICS_PATH, "w") as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

print(f"Metrics saved as {METRICS_PATH}")

HISTORY_PATH = os.path.join(
    RUN_DIR,
    "history.json"
)

history_serializable = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

# The scalar Keras logs above already carry val_macro_f1 and val_f1_N/S/V
# because the callback wrote them into logs. This adds the rest: per-class
# recall, precision, support and the validation confusion matrix per epoch.
history_serializable["val_metrics_per_epoch"] = val_epoch_records

with open(HISTORY_PATH, "w") as f:

    json.dump(
        history_serializable,
        f,
        indent=2
    )

print(f"History saved as {HISTORY_PATH}")