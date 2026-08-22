# Last report

**Tasks:** (A) correct a factual error in CLAUDE.md, (B) step 1 - patient-wise
validation split.

**Date:** 2026-08-22

---

## What changed

### Task A - CLAUDE.md correction

- The "Landmine" note claimed `augment_training_data()` was dead code. **That
  was wrong.** It is called at `src/train.py` section 18 (line 711 after this
  change), and its output `X_tr_aug / RR_tr_aug / y_tr_aug` is what actually
  feeds `build_model` and `model.fit`.
- My earlier grep pattern matched only `def .*augment` and the bare names
  `augment_segment` / `augment_dataset`, so it never looked for
  `augment_training_data` as a call. The distribution cross-check I offered as
  corroboration was worthless: `train_distribution` is computed from
  `y_train`, before the split and before augmentation, so it cannot detect
  augmentation at all.
- Replaced the note with a **"Known facts"** section recording three verified
  items: augmentation is active (S x7, V x3, to be removed later because it
  violates hard constraint 2 and copies RR features unchanged across
  duplicates); record 114 has leads `['V5','MLII']` so `LEAD_INDEX = 0` reads
  V5 (to be fixed later); records 201 and 202 are the same subject, 201 in
  DS1 and 202 in DS2, which must be disclosed as a limitation and never
  "fixed" by editing the lists, and 201 must never go in the validation set.

### Task B - step 1, patient-wise validation split

- Added `DS1_VAL = ['207', '220', '223']` in section 3, plus
  `DS1_TRAIN = [rec for rec in DS1 if rec not in DS1_VAL]` - computed from
  DS1, not hardcoded, so it can never drift out of sync with the list.
- Added three assertions next to it guarding the hard constraint: `DS1_VAL`
  is a subset of DS1, shares nothing with DS2, and partitions DS1 exactly.
- Section 12 now makes **three** `load_dataset` calls - `DS1_TRAIN`,
  `DS1_VAL`, `DS2` - in the same style as the existing DS1/DS2 calls.
- Sections 14, 15 and 16 gained the matching validation lines
  (`y_valid_encoded`, `normalize_rr(RR_valid)`, `expand_dims(X_valid)`).
- Section 17 no longer splits anything. The split happens at load time by
  record, so the section just binds the downstream names.
- **Deleted** the `train_test_split` call, `test_size=0.1`, and the
  `from sklearn.model_selection import train_test_split` import, which is now
  unused. `sklearn.metrics` and `sklearn.preprocessing` imports are untouched.
- Every downstream name is preserved exactly: `X_tr`, `RR_tr`, `y_tr`,
  `X_val`, `RR_val`, `y_val`. Nothing after section 17 needed changing.
- Section 12 now prints the record lists and the class distribution of both
  `DS1_TRAIN` and `DS1_VAL`; section 17 prints the two beat counts.
- Added `"ds1_train": DS1_TRAIN` and `"ds1_val": DS1_VAL` to the `config`
  block written into `metrics.json`.

**Why 207 / 220 / 223.** They are the three records with the most S beats
(106, 94, 73) after 209, 201 and 118. Together they give the validation set
**273 S beats, a 4.20% S share** against DS1's overall 1.9% - dense enough for
val S-F1 to be a usable selection signal, which was the whole point of running
`tools/inspect_ds1.py` first. They are 12.8% of DS1 beats, close to the 10%
the old `test_size=0.1` took. **201 was deliberately excluded** despite having
128 S beats, because it is the same subject as DS2's record 202.

---

## Files touched

- `CLAUDE.md` - Landmine section replaced with Known facts
- `src/train.py` - sections 3, 12, 14, 15, 16, 17, and the metrics config
  block; one import removed
- `docs/last_report.md` - this file

`docs/ablation.md` was **not** touched. Step 1 has not been trained yet, so
there is no `results/<run>/metrics.json` to source a row from, and inventing
one would break the "every number must trace to a metrics.json" rule. The row
gets added when the Kaggle run comes back.

---

## Verification

**py_compile**

```
$ python -m py_compile src/train.py
$ echo $?
0
```

Passed, exit code 0.

**Constants that must still be present**

| constant | occurrences | status |
|---|---|---|
| `alpha=0.50` | 2 | present |
| `gamma=2.0` | 2 | present |
| `learning_rate=1e-4` | 1 | present |
| `multiplier = 6` | 1 | present |
| `PRE_SAMPLES = 90` | 1 | present |
| `POST_SAMPLES = 144` | 1 | present |

(`alpha` and `gamma` appear twice each - once in the loss call, once in the
metrics config block. That was already true before this change.)

**`test_size=0.1` must be GONE**

```
$ grep -n "test_size" src/train.py
(no output)
```

Confirmed absent. `grep -n train_test_split` returns exactly one line - line
691, inside the explanatory comment in section 17. The import and the call are
both gone.

**DS1 and DS2 byte-identical**

Extracted both list literals from the pre-edit and post-edit files and
compared:

```
DS1 IDENTICAL   sha256 9f20e3ac1758a312...
DS2 IDENTICAL   sha256 b8a3e6bbdeeec72a...
```

**No changed line touches a protected function**

Parsed the AST to get each protected function's line span, then mapped every
added line (new side) and every deleted line (old side) of the diff onto those
spans:

| function | lines | added-side | deleted-side |
|---|---|---|---|
| `extract_beats_from_record` | 258-330 | CLEAN | CLEAN |
| `augment_segment` | 386-428 | CLEAN | CLEAN |
| `augment_training_data` | 435-511 | CLEAN | CLEAN |
| `categorical_focal_loss` | 518-552 | CLEAN | CLEAN |
| `build_model` | 742-887 | CLEAN | CLEAN |

66 added lines and 24 deleted lines, none of them inside any of the five.

**Split logic dry-run** (executed standalone, no TensorFlow needed - the
record lists were extracted from the edited file and the beat counts came from
`tools/ds1_beat_counts.json`):

```
asserts pass
DS1_TRAIN (19): ['101','106','108','109','112','114','115','116','118','119',
                 '122','124','201','203','205','208','209','215','230']
DS1_VAL   (3): ['207', '220', '223']
201 in val? False  (must be False)

predicted train dist: {'N': 40301, 'S': 670, 'V': 3105}  total 44076
predicted val   dist: {'N': 5538,  'S': 273, 'V': 683}   total 6494
val S share: 4.20%   val is 12.8% of DS1 beats
```

No record appears on both sides, and the two sets partition DS1 exactly.

**Falsifiable prediction for the Kaggle run.** Post-augmentation the training
set should be 40301 + 670x7 + 3105x3 = **54,306 samples = 425 steps/epoch** at
batch 128. The baseline ran 449. If the next run does not print 425
steps/epoch, something in this change did not do what I think it did.

**Augmentation still training-only**

Section 18 consumes `X_tr / RR_tr / y_tr`, which are now DS1_TRAIN alone.
`model.fit` receives `[X_tr_aug, RR_tr_aug]` for training and
`validation_data=([X_val, RR_val], y_val_cat)` - the held-out records,
un-augmented. Unchanged behaviour, correct data.

**Variable reachability**

Traced every name introduced or rebound; each is assigned before first use:
`X_valid` (571 -> 590, 672, 700), `y_valid` (571 -> 599, 645), `RR_valid`
(571 -> 659, 701), `y_valid_encoded` (643 -> 702), `X_val`/`RR_val`
(700/701 -> 929), `y_val` (702 -> 704, 728). `SEED` is still referenced in six
places, so dropping `random_state=SEED` left nothing dangling.

---

## git diff - section 17

```diff
@@ -641,29 +681,27 @@ X_test = np.expand_dims(
 
 
 # =========================================================
-# 17. TRAIN / VALIDATION SPLIT
+# 17. TRAIN / VALIDATION SPLIT (patient-wise)
 # =========================================================
 
-(
-    X_tr,
-    X_val,
-    RR_tr,
-    RR_val,
-    y_tr,
-    y_val
-
-) = train_test_split(
-
-    X_train,
-    RR_train,
-    y_train_encoded,
+# The split already happened at load time, by record: DS1_TRAIN and
+# DS1_VAL were read as two separate datasets, so there is nothing to
+# slice here and no beat from a validation patient can reach training.
+#
+# This previously used a stratified beat-level train_test_split over all
+# of DS1, which put the same patient on both sides. Validation accuracy
+# reached 0.9885 against a true DS2 accuracy of 0.9294, and EarlyStopping
+# with restore_best_weights selected the final model on that leaked score.
 
-    test_size=0.1,
+X_tr = X_train
+RR_tr = RR_train
+y_tr = y_train_encoded
 
-    random_state=SEED,
+X_val = X_valid
+RR_val = RR_valid
+y_val = y_valid_encoded
 
-    stratify=y_train_encoded
-)
+print(f"\nTrain beats: {len(y_tr)}   Validation beats: {len(y_val)}")
 
 
 # =========================================================
```

---

## Commit

Two commits, so that the step 1 hash is the code change alone and can be cited
cleanly in `docs/ablation.md` later:

```
403c0ab  docs: correct CLAUDE.md - augmentation is active, not dead code
ae9a91c  step 1: patient-wise validation split (DS1_VAL = 207,220,223)
```

Both pushed to `origin/main`; `git status -sb` reports `## main...origin/main`
with no divergence. This report lands in a small follow-up commit, as before.

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

1. **RR normalisation statistics changed as an unavoidable consequence of
   splitting at load time.** `normalize_rr()` computes mean and std from
   whatever array it is handed. Before, `RR_train` was all of DS1, so training
   and validation RR were scaled by shared DS1 statistics. Now `DS1_TRAIN` and
   `DS1_VAL` are separate arrays and each is normalised by its own statistics.
   I chose this deliberately: it is exactly the convention the script already
   uses for DS1 versus DS2, and it keeps `normalize_rr()` untouched. The
   alternative - normalising validation with training statistics - is arguably
   more correct, but it would have meant changing section 6 or threading stats
   through, which is beyond "change nothing else". **Flagging it because it is
   a real semantic change that the instructions did not explicitly authorise.**
   Worth its own step later if you want validation scaled by training stats.

2. **`train_distribution` in `metrics.json` now means DS1_TRAIN, not DS1.**
   It is still `Counter(y_train)`, but `y_train` is now the 19 training
   records rather than all 22. Expect **N=40301 / S=670 / V=3105** in the next
   run instead of the baseline's 45839 / 943 / 3788. That is the honest
   number - it is what the model trains on - but the field is not comparable
   across the baseline and step 1, so do not read a drop there as data loss.
   There is no `val_distribution` field; adding one was not in scope, and it
   is recoverable from `ds1_val` plus `tools/ds1_beat_counts.json`.

3. **The step 1 numbers are not comparable to the baseline on validation.**
   Baseline val accuracy of 0.9885 was measured on a leaked split. Step 1 val
   accuracy will look much worse, and that is the fix working, not a
   regression. Only the DS2 test numbers are comparable between the two runs.

4. **Augmentation is still active** (S x7, V x3), exactly as instructed for
   this step. It still violates hard constraint 2 and still copies RR features
   unchanged across every duplicate. Now recorded in CLAUDE.md under Known
   facts and queued for a later step.

5. Carried over from the previous task, unchanged: `tools/inspect_ds1.py`
   writes its JSON into `tools/` rather than `docs/`, and a stale root
   `__pycache__/` from before the reorganisation is still present. Both
   harmless, both untouched.
