# Last report

**Task:** step 1b - fit RR normalization on the training set only.

**Date:** 2026-08-22

---

## What changed

- Replaced `normalize_rr()` with two functions in section 6:
  - `fit_rr_norm(rr_array)` - computes mean and std **per column**
    (`axis=0`, so `prev_rr` and `next_rr` get their own statistics rather
    than one scalar over both columns and all rows), returns a `(2,)` mean
    and a `(2,)` std, and replaces any zero std with `1.0` so a
    zero-variance column is centred but never divided by zero.
  - `apply_rr_norm(rr_array, mean, std)` - applies already-fitted
    statistics. It cannot fit, by construction.
- Section 15 now fits **once on `RR_train` (DS1_TRAIN) only** and applies
  those same two vectors to `RR_train`, `RR_valid` and `RR_test`. All three
  sets now land on one scale, and no validation or test statistic enters the
  pipeline.
- The fitted values are printed, labelled by column.
- Added `config.rr_norm_mean` and `config.rr_norm_std` to `metrics.json`, as
  lists.
- `normalize_rr` is gone entirely - `grep` returns no occurrences, so no dead
  function is left behind.
- `normalize_segment()` (the per-beat ECG normalizer used inside
  `extract_beats_from_record` and `augment_segment`) was **not** touched.
  Verified by AST comparison, not by eye.

Nothing else changed: no model, loss, hyperparameter, augmentation, or split
change. `DS1_VAL` is still `['207','220','223']`.

---

## Files touched

- `src/train.py` - section 6 (normalization functions), section 15, and the
  metrics config block
- `docs/last_report.md` - this file

`docs/ablation.md` was **not** touched: see Problems item 3.

---

## Verification

**py_compile**

```
$ python -m py_compile src/train.py
$ echo $?
0
```

Passed, exit code 0.

**No changed line falls inside a protected function** (AST line-span method,
applied to both sides of the diff - 42 added lines, 9 deleted lines):

| function | lines (new) | added-side | deleted-side |
|---|---|---|---|
| `build_model` | 771-916 | CLEAN | CLEAN |
| `categorical_focal_loss` | 536-570 | CLEAN | CLEAN |
| `augment_segment` | 404-446 | CLEAN | CLEAN |
| `augment_training_data` | 453-529 | CLEAN | CLEAN |
| `extract_beats_from_record` | 276-348 | CLEAN | CLEAN |

Additionally `normalize_segment` was compared by `ast.dump` against the
pre-edit file: **identical**. It sits next to the function I replaced and
feeds two protected functions, so eyeballing it was not good enough.

**Constants unchanged**

| constant | occurrences |
|---|---|
| `alpha=0.50` | 2 |
| `gamma=2.0` | 2 |
| `learning_rate=1e-4` | 1 |
| `multiplier = 6` | 1 |
| `PRE_SAMPLES = 90` | 1 |
| `POST_SAMPLES = 144` | 1 |

**Record literals unchanged** (extracted and hashed against the pre-edit file)

```
DS1      IDENTICAL  sha256 9f20e3ac1758a312...
DS2      IDENTICAL  sha256 b8a3e6bbdeeec72a...
DS1_VAL  IDENTICAL  sha256 0d9df3612a6111a1...

DS1_TRAIN = [rec for rec in DS1 if rec not in DS1_VAL]
```

The DS1 and DS2 hashes are the same values recorded in the step 1 report, so
the lists are unchanged across both steps.

**Numerical test on the real data.** `fit_rr_norm` and `apply_rr_norm` were
lifted out of the edited `src/train.py` by AST (so the test exercises the
shipped code, not a copy), and run against RR features extracted from the
actual MIT-BIH records with the same beat-selection rule as
`extract_beats_from_record`. No TensorFlow needed.

```
lifted: ['apply_rr_norm', 'fit_rr_norm']
extracting RR features ...
shapes: (44076, 2) (6494, 2) (49289, 2)

OLD (step 1) - each set fitted on itself, scalar stats:
  train mean/std:  275.015   76.299
  val   mean/std:  289.605   60.672   <- different scale
  test  mean/std:  282.492   87.781   <- different scale
  val mean is 5.3% off train; val std is -20.5% off train

NEW (step 1b) - fitted once on DS1_TRAIN, per column:
  mean: [274.881103515625, 275.1480712890625]
  std : [77.20227813720703, 75.38490295410156]

  after applying TRAIN stats to all three:
    train col means [-0.0, -0.0]  col stds [1.0, 1.0]
    val   col means [0.1890999972820282, 0.19339999556541443]  col stds [0.7914000153541565, 0.7990999817848206]
    test  col means [0.10970000177621841, 0.08609999716281891]  col stds [1.1380000114440918, 1.1634000539779663]
  same-transform check: PASS

zero-variance guard: std -> [1.0, 1.0] output all zeros & finite: True -> PASS
mixed constant/varying columns: finite: True  varying col std -> 1.0 -> PASS

per-column vs scalar - the reason axis=0 matters:
  col means (raw train): [274.8810119628906, 275.14801025390625]
  col stds  (raw train): [77.2020034790039, 75.38500213623047]
  scalar mean/std over both cols: 275.015 76.299

OVERALL: PASS
```

**This confirms the diagnosis.** Under the step 1 behaviour the validation
set was scaled by its own std of **60.672** against training's **76.299** -
**20.5% smaller**. Every validation RR feature was therefore inflated by
roughly 1.26x relative to the scale the model was trained on, which is a
feature distribution mismatch, exactly as you called it. After the fix,
validation sits at mean +0.19 and std 0.79 *on the training scale* - a real,
modest patient-to-patient difference rather than a fabricated rescaling.

---

## Falsifiable predictions for the next Kaggle run

1. **Steps/epoch must stay at 425.** Nothing about the split, the record
   lists, or the augmentation changed, so the training set is still
   40301 N + 670x7 S + 3105x3 V = 54,306 samples = 425 steps/epoch at batch
   128. **If steps/epoch is not 425, this change did something it should not
   have.**
2. **The run must print exactly**
   `mean (prev_rr, next_rr): [274.881103515625, 275.1480712890625]` and
   `std  (prev_rr, next_rr): [77.20227813720703, 75.38490295410156]`.
   I computed these locally from the same records the script will load. Any
   deviation means Kaggle is reading different data than this machine.
3. **val_accuracy should now clear 0.8528**, the all-N rate on this
   validation set (5538 N of 6494 beats). Step 1 peaked at 0.7368, i.e. below
   trivial. If it stays below 0.8528, the scale mismatch was not the whole
   story and the next suspect is the record 114 lead swap.

Prediction 3 is the real test of the hypothesis. Predictions 1 and 2 are
guards that the change was surgical.

---

## git diff

```diff
diff --git a/src/train.py b/src/train.py
index 82a5d22..25ac4df 100644
--- a/src/train.py
+++ b/src/train.py
@@ -238,15 +238,33 @@ def normalize_segment(segment):
     return (segment - mean) / std
 
 
-def normalize_rr(rr_array):
+def fit_rr_norm(rr_array):
+    """Fit per-column RR statistics on the TRAINING set only.
+
+    RR is [prev_rr, next_rr], so the statistics are per column (axis=0),
+    not one scalar over both columns as before. Returns (mean, std), each
+    shape (2,). A zero-variance column gets std 1.0 so it is only centred,
+    never divided by zero.
+    """
 
     rr_array = np.array(rr_array, dtype=np.float32)
 
-    mean = np.mean(rr_array)
-    std = np.std(rr_array)
+    mean = np.mean(rr_array, axis=0)
+    std = np.std(rr_array, axis=0)
 
-    if std == 0:
-        return rr_array - mean
+    std = np.where(std == 0.0, 1.0, std)
+
+    return mean.astype(np.float32), std.astype(np.float32)
+
+
+def apply_rr_norm(rr_array, mean, std):
+    """Apply already-fitted RR statistics.
+
+    Never fits. Validation and test are scaled with the training set's
+    mean and std so all three land on the same scale.
+    """
+
+    rr_array = np.array(rr_array, dtype=np.float32)
 
     return (rr_array - mean) / std
 
@@ -655,9 +673,20 @@ y_test_encoded = np.array([
 # 15. NORMALIZE RR
 # =========================================================
 
-RR_train = normalize_rr(RR_train)
-RR_valid = normalize_rr(RR_valid)
-RR_test = normalize_rr(RR_test)
+# Fitted on DS1_TRAIN only. Fitting on validation or test would leak
+# their distribution into the pipeline; fitting each set separately (the
+# step 1 behaviour) put the three on three different scales, which is why
+# step 1 val_accuracy peaked at 0.7368, below the 0.8528 an all-N
+# prediction scores on that validation set.
+RR_NORM_MEAN, RR_NORM_STD = fit_rr_norm(RR_train)
+
+print("\nRR normalization fitted on DS1_TRAIN only:")
+print(f"  mean (prev_rr, next_rr): {RR_NORM_MEAN.tolist()}")
+print(f"  std  (prev_rr, next_rr): {RR_NORM_STD.tolist()}")
+
+RR_train = apply_rr_norm(RR_train, RR_NORM_MEAN, RR_NORM_STD)
+RR_valid = apply_rr_norm(RR_valid, RR_NORM_MEAN, RR_NORM_STD)
+RR_test = apply_rr_norm(RR_test, RR_NORM_MEAN, RR_NORM_STD)
 
 
 # =========================================================
@@ -1247,7 +1276,11 @@ metrics = {
 
         "ds1_train": DS1_TRAIN,
 
-        "ds1_val": DS1_VAL
+        "ds1_val": DS1_VAL,
+
+        "rr_norm_mean": RR_NORM_MEAN.tolist(),
+
+        "rr_norm_std": RR_NORM_STD.tolist()
     },
 
     "train_distribution": {
```

---

## Commit

```
10e06f2  step 1b: fit RR normalization on training set only
```

Pushed to `origin/main`; `git status -sb` reports `## main...origin/main` with
no divergence. This report lands in a small follow-up commit, as before.

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
│   └── baseline/
│       └── metrics.json
├── src/
│   └── train.py
└── tools/
    ├── ds1_beat_counts.json
    └── inspect_ds1.py
```

---

## Problems

1. **The per-column change barely moves the numbers here, and I want that on
   the record rather than overclaimed.** The two RR columns turn out to have
   nearly identical statistics (mean 274.88 vs 275.15, std 77.20 vs 75.38,
   against the old scalar 275.02 / 76.30). `axis=0` is the correct thing to
   do and it is now correct, but on this dataset it is a rounding-level
   difference. **The fix that matters is fitting on training only** - that is
   what removes the 20.5% scale mismatch. If step 1b improves val_accuracy,
   credit the train-only fit, not the per-column split.

2. **Test data is now scaled by training statistics, which is a change to how
   DS2 is processed** - previously DS2 was normalized by its own statistics.
   This is strictly more correct and is *required* by the no-tuning-against-
   DS2 rule (fitting a scaler on DS2 is fitting a parameter on the test set),
   but it does mean baseline and step 1 test numbers were produced under a
   different preprocessing than step 1b's will be. Worth one sentence in the
   paper's methods section.

3. **There is still no `metrics.json` for step 1 or step 1b in `results/`.**
   The step 1 run happened - you quoted val_accuracy 0.7368 over 8 epochs -
   but its metrics never came back into the repo, so `docs/ablation.md` still
   holds only the baseline row. Under the "every number must trace to a
   `results/<run>/metrics.json`" rule I did not add rows for either step. Drop
   `results/step1/metrics.json` and `results/step1b/metrics.json` in and I
   will fill both rows.

4. **Record 114's lead swap is still unfixed** and is the leading remaining
   suspect if prediction 3 fails. It is in DS1_TRAIN, so it is currently
   feeding V5 into a model that sees MLII from the other 18 training records.
   Already logged in CLAUDE.md under Known facts.

5. Carried over, unchanged: augmentation is still active (S x7, V x3, still
   violating hard constraint 2, still queued for removal);
   `tools/inspect_ds1.py` still writes its JSON into `tools/`; the stale root
   `__pycache__/` is still there.
