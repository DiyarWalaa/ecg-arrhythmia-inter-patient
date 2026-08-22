# Last report

**Task:** step 3 - replace raw RR features with patient-relative ratios.
Also: added the step 2 run to `docs/ablation.md`.

**Date:** 2026-08-22

---

## What changed

`extract_beats_from_record` now returns **five dimensionless ratios** instead
of two raw sample counts.

Per record, computed once before the beat loop:

```python
rr_series = np.diff(ann_samples)          # interval ending at beat k is rr_series[k-1]
median_rr = np.median(rr_series)          # falls back to 1.0 if empty or <= 0
```

Per beat `i`:

| # | feature | definition |
|---|---|---|
| 1 | `pre_RR_over_median` | `pre_RR / median_RR_record` |
| 2 | `post_RR_over_median` | `post_RR / median_RR_record` |
| 3 | `local_RR_over_median` | `local_RR / median_RR_record` |
| 4 | `pre_RR_over_local` | `pre_RR / local_RR` |
| 5 | `post_RR_over_pre` | `post_RR / pre_RR` |

`local_RR` is the mean of the `RR_LOCAL_WINDOW = 10` intervals immediately
**before** this beat's own `pre_RR` - slice `rr_series[i-1-W : i-1]`. Beats
with fewer than 10 preceding intervals fall back to `median_RR_record`.

**On that interpretation:** "the previous 10 RR intervals" excludes the
beat's own `pre_RR`. That is what makes feature 4 a prematurity measure -
current interval against recent local rhythm. Including `pre_RR` in its own
denominator would dilute exactly the signal the feature exists to capture. On
record 209 the fallback branch is taken by 11 beats.

Every feature is clipped to `[0.2, 3.0]` after computation. Division guards:
`median_rr <= 0` or non-finite falls back to `1.0`; `local_rr <= 0` or
non-finite falls back to `median_rr`; `pre_rr <= 0` falls back to `median_rr`
as feature 5's denominator.

**Downstream**

- `fit_rr_norm` / `apply_rr_norm` unchanged, still fitted on DS1_TRAIN only.
  Both are already `axis=0` and shape-agnostic, so they went from 2 columns
  to 5 with no edit.
- `build_model` call site: `rr_shape=(2,)` -> `rr_shape=(len(RR_FEATURE_NAMES),)`.
  I used the length of the names list rather than a literal `(5,)` so the two
  cannot drift apart in a later step. It evaluates to `(5,)` today.
- New section 3 constants: `RR_FEATURE_NAMES`, `RR_LOCAL_WINDOW = 10`,
  `RR_CLIP_MIN = 0.2`, `RR_CLIP_MAX = 3.0`.
- `metrics.json` config gains `rr_feature_names`, plus `rr_local_window` and
  `rr_clip` so a run's feature definition is fully recoverable from its own
  metrics file.
- Section 15 prints min / max / mean / median and the percentage of values
  sitting at each clip bound, per feature, for DS1_TRAIN.

`augment_training_data` needed no change: it copies the RR row verbatim, so it
is shape-agnostic. (It still copies RR unchanged across synthetic duplicates -
the known defect logged in CLAUDE.md.)

Nothing else changed: no architecture change (the RR branch is still
`Dense(16)`, verified), no loss, learning-rate, augmentation or split change.
`DS1_VAL` is still `['207','220','223']`.

---

## Step 2 predictions - scorecard

All three passed.

| # | prediction | outcome |
|---|---|---|
| 1 | `best_epoch >= 5`; if it returns 1 the problem is upstream of selection | **PASS** - `best_epoch = 14`, ran 24 epochs, `early_stopping_fired = true` |
| 2 | `val_per_record` shows record 220 with `support.V = 0`, `recall.V = 0.0` | **PASS** - exactly that |
| 3 | test macro-F1 beats step 1b's 0.4742 | **PASS** - 0.6800 |

Prediction 3 came in higher than I framed it: 0.6800 also clears the
baseline's 0.6358, which I explicitly declined to predict. Selecting on
`val_macro_f1` instead of `val_loss` was worth +0.2058 macro-F1 on its own.

**What the instrumentation found.** Record **207** is a severe outlier:
accuracy 0.1647, N recall 0.0700, S recall 0.0000, while 220 and 223 score
0.9780 and 0.9602. The model calls almost everything in 207 a V beat (V
recall 0.9429). One of the three validation records is dragging the selection
signal, which is precisely the question `val_per_record` was added to answer.
That is a step 4 conversation, not something I changed here.

---

## Verification

**py_compile** - passed, exit code 0.

**AST line-span check** (90 added lines, 4 deleted):

| function | lines (new) | added | deleted | expectation |
|---|---|---|---|---|
| `build_model` | 876-1021 | CLEAN | CLEAN | must NOT change |
| `categorical_focal_loss` | 600-634 | CLEAN | CLEAN | must NOT change |
| `augment_segment` | 468-510 | CLEAN | CLEAN | must NOT change |
| `augment_training_data` | 517-593 | CLEAN | CLEAN | must NOT change |
| `fit_rr_norm` | 264-280 | CLEAN | CLEAN | must NOT change |
| `apply_rr_norm` | 283-292 | CLEAN | CLEAN | must NOT change |
| `extract_beats_from_record` | 299-412 | 44 lines | 3 lines | **MAY change** |

Whole-file `ast.dump` comparison against the pre-edit file:
**`extract_beats_from_record` is the only function that changed.** No
functions added, none removed.

`build_model`'s RR branch still contains `Dense(16, ...)`; the only edit near
the model is the call-site `rr_shape`.

**Constants unchanged**: `alpha=0.50` (2), `gamma=2.0` (2),
`learning_rate=1e-4` (1), `multiplier = 6` (1), `PRE_SAMPLES = 90` (1),
`POST_SAMPLES = 144` (1).

**Record literals unchanged**:

```
DS1      IDENTICAL  sha256 9f20e3ac1758a312...
DS2      IDENTICAL  sha256 b8a3e6bbdeeec72a...
DS1_VAL  IDENTICAL  sha256 0d9df3612a6111a1...
```

Same hashes as steps 1, 1b and 2.

**Behavioural test on real MIT-BIH data.** `extract_beats_from_record`,
`load_dataset` and `normalize_segment` were lifted out of the edited
`src/train.py` by AST, so the shipped code is what runs. The five features
were then recomputed by an independent reference written from your spec
rather than from the code, and compared element-wise.

```
feature names (5):
   1. pre_RR_over_median
   2. post_RR_over_median
   3. local_RR_over_median
   4. pre_RR_over_local
   5. post_RR_over_pre
local window 10, clip [0.2, 3.0]

A) independent recomputation on record 209
   beats 3004   rr shape (3004, 5)
   reference shape (3004, 5)
   independent recomputation matches shipped code: True -> PASS
   beats using the median fallback (fewer than 10 preceding intervals): 11

B) clip saturation across all of DS1_TRAIN
   total beats 44076   feature matrix (44076, 5)

   feature                       min      max     mean   median   %at_min   %at_max
   pre_RR_over_median         0.2000   3.0000   1.0009   1.0000    0.000%    0.005%
   post_RR_over_median        0.2000   2.4932   1.0027   1.0030    0.000%    0.000%
   local_RR_over_median       0.3212   1.7618   0.9925   1.0016    0.000%    0.000%
   pre_RR_over_local          0.2000   3.0000   1.0124   1.0014    0.000%    0.011%
   post_RR_over_pre           0.2000   3.0000   1.0461   0.9963    0.000%    0.658%

   values at a clip bound: 297 of 220380 = 0.1348% of all cells
   beats with >=1 clipped feature: 0.6693%
   saturating (>5%% of cells at a bound): False -> PASS

C) do the ratios separate S from N the way they should?
   S beats are premature, so pre_RR_over_local should sit BELOW 1
   for S and near 1 for N.

   feature                      N mean     S mean     V mean
   pre_RR_over_median           1.0271     0.6728     0.7323
   post_RR_over_median          0.9882     0.9386     1.2041
   local_RR_over_median         0.9933     0.8815     1.0065
   pre_RR_over_local            1.0377     0.8065     0.7276
   post_RR_over_pre             0.9874     1.4246     1.7267

   S pre_RR/local 0.8065 < N pre_RR/local 1.0377 : True -> PASS
   best single-threshold S F1 from pre_RR/local alone: 0.1515 (threshold 0.8365)
   step 2 model achieved S F1 0.1942 on DS2 using all features.

OVERALL: PASS
```

Three results worth pulling out:

1. **The independent recomputation matches exactly** on all 3,004 beats x 5
   features of record 209, to 1e-9. Two implementations from the same spec
   agreeing is much stronger evidence than the code merely running.

2. **The clip is nowhere near saturating.** Across all 44,076 DS1_TRAIN beats,
   **0.1348% of feature values** sit at a bound, and **0.6693% of beats** have
   at least one clipped feature. The worst single feature is
   `post_RR_over_pre` at 0.658% hitting the upper bound - which is the
   compensatory pause after an ectopic beat, so it is real signal being
   bounded, not noise. No feature touches the lower bound at all.

3. **The features separate S from N in the direction the physiology predicts.**
   `pre_RR_over_median` averages **0.6728** on S beats against **1.0271** on N -
   S beats arrive early, as they should. `post_RR_over_pre` averages **1.4246**
   on S against 0.9874 on N - the compensatory pause. As a sanity check on how
   much signal that is: a single threshold on `pre_RR_over_local` alone reaches
   **S F1 0.1515**, against the step 2 model's 0.1942 using the whole network
   and the old raw features.

---

## git diff - extract_beats_from_record

```diff
@@ -307,6 +329,17 @@ def extract_beats_from_record(
     ann_samples = annotation.sample
     ann_symbols = annotation.symbol
 
+    # Full RR series for this record, computed once.
+    # rr_series[k] = ann_samples[k + 1] - ann_samples[k], so the interval
+    # ending at beat i is rr_series[i - 1] and the one starting at beat i
+    # is rr_series[i].
+    rr_series = np.diff(ann_samples).astype(np.float64)
+
+    median_rr = float(np.median(rr_series)) if len(rr_series) else 0.0
+
+    if not np.isfinite(median_rr) or median_rr <= 0.0:
+        median_rr = 1.0
+
     beats = []
     labels = []
     rr_features = []@@ -331,10 +364,40 @@ def extract_beats_from_record(
         if len(segment) != SEGMENT_LENGTH:
             continue
 
-        prev_rr = ann_samples[i] - ann_samples[i - 1]
-        next_rr = ann_samples[i + 1] - ann_samples[i]
+        pre_rr = float(rr_series[i - 1])
+        post_rr = float(rr_series[i])
+
+        # The RR_LOCAL_WINDOW intervals immediately BEFORE this beat's own
+        # pre_RR: rr_series[i - 1 - W : i - 1]. Excluding pre_RR is what
+        # makes pre_RR / local_RR a prematurity measure rather than a
+        # self-comparison. Too few preceding intervals -> fall back to the
+        # record median.
+        window_start = i - 1 - RR_LOCAL_WINDOW
+
+        if window_start >= 0:
+            local_rr = float(np.mean(rr_series[window_start:i - 1]))
+        else:
+            local_rr = median_rr
+
+        if not np.isfinite(local_rr) or local_rr <= 0.0:
+            local_rr = median_rr
+
+        # pre_rr can only be <= 0 with corrupt annotations; fall back so
+        # feature 5 never divides by zero.
+        pre_rr_denom = pre_rr if pre_rr > 0.0 else median_rr
+
+        rr = [
+            pre_rr / median_rr,
+            post_rr / median_rr,
+            local_rr / median_rr,
+            pre_rr / local_rr,
+            post_rr / pre_rr_denom
+        ]
 
-        rr = [prev_rr, next_rr]
+        rr = [
+            float(np.clip(value, RR_CLIP_MIN, RR_CLIP_MAX))
+            for value in rr
+        ]
 
         segment = normalize_segment(segment)
 
```

---

## Falsifiable predictions for the next Kaggle run

1. **`config.rr_feature_names` will list the five names in the order above,
   and `rr_norm_mean` / `rr_norm_std` will each be 5-element lists** whose
   means are all near 1.0 (my local values: 1.0009, 1.0027, 0.9925, 1.0124,
   1.0461). Any deviation means Kaggle loaded different data.
2. **The printed clip table will show under 1% at either bound for every
   feature.** If any feature exceeds 5%, the bounds are wrong and the
   comparison against step 2 is contaminated.
3. **Test S recall will exceed step 2's 0.1280.** This is the actual bet.
   Raw RR was the one input that could not generalise across patients, and S
   detection depends on it more than N or V do. **If S recall does not move,
   the RR representation was not the bottleneck** and the next suspect is the
   6x synthetic oversampling of S with RR copied unchanged - which makes six
   of every seven S training examples carry identical rhythm context.

I am not predicting macro-F1 will rise. V F1 is already 0.8718 and may give
back a little as the RR branch changes meaning; the S column is what this
step targets.

---

## Commit

```
1a2509c  step 3: patient-relative RR ratio features
```

Pushed to `origin/main`. This report lands in a small follow-up commit.

---

## Tree

```
.
├── CLAUDE.md
├── .gitignore
├── data/                                   (gitignored, not in repo)
│   └── mit-bih-arrhythmia-database-1.0.0/
├── docs/
│   ├── ablation.md
│   ├── ds1_beat_counts.txt
│   └── last_report.md
├── notebooks/
│   └── .gitkeep
├── results/
│   ├── baseline/metrics.json
│   ├── step1_patient_val/{history,metrics}.json
│   ├── step1b_rr_trainstats/{history,metrics}.json
│   └── step2_macrof1_selection/{history,metrics}.json
├── src/
│   └── train.py
└── tools/
    ├── ds1_beat_counts.json
    └── inspect_ds1.py
```

---

## Problems

1. **This step changes DS2's feature representation too**, in the same way
   step 1b changed its scaling. That is unavoidable - features must be
   computed identically on train and test - and it uses only each patient's
   own signal, never labels, so the inter-patient constraint holds. But it
   means step 3 versus step 2 is a comparison across a feature-space change,
   not a pure model change. Same caveat as step 1b, worth the same sentence
   in the methods section.

2. **I used `rr_shape=(len(RR_FEATURE_NAMES),)` rather than the literal
   `(5,)` you specified.** Same value, but it cannot silently disagree with
   the feature list if a later step adds a sixth feature. Flagging it because
   it is a deviation from the letter of the instruction.

3. **Feature 3 is nearly constant and may carry little information.**
   `local_RR_over_median` has mean 0.9925 and median 1.0016 across DS1_TRAIN,
   with a much narrower spread than the others (min 0.3212, max 1.7618 -
   never clipped). It describes the patient's recent rhythm relative to their
   own median, so by construction it hovers near 1. It is not harmful, and
   after standardisation its variance is rescaled, but do not expect it to
   contribute much.

4. **Record 207 remains a severe validation outlier** (accuracy 0.1647, N
   recall 0.0700 in step 2). Patient-relative RR may or may not help it; 207
   is a record with sustained ventricular flutter, so its *median* RR is
   itself computed over abnormal rhythm. If step 3 does not fix 207, the
   validation set composition needs revisiting - and note that changing
   `DS1_VAL` would break comparability with steps 1 through 3, so it is a
   decision to take deliberately rather than by drift.

5. Carried over, unchanged: augmentation still active (S x7, V x3, still
   violating hard constraint 2, and still copying RR unchanged across
   duplicates - now more clearly a defect, since the copied RR is what this
   step just made meaningful); record 114 lead swap still unfixed;
   `tools/inspect_ds1.py` still writes its JSON into `tools/` and that JSON is
   now **stale** - it was generated before this change and still describes the
   old feature set's beat counts (the counts themselves are unaffected);
   stale root `__pycache__/` still present.
