# Last report

**Task:** step 5 - enlarge DS1_VAL to 5 records. Also: correct CLAUDE.md
(augmentation was removed at step 4), record the DS1_VAL change in both
CLAUDE.md and `docs/ablation.md`.

**Date:** 2026-08-22

---

## Blocking problem: there is no step 4 run

`results/step4_class_alpha/metrics.json` is **byte-identical** to
`results/step3_rr_ratios/metrics.json`:

```
metrics.json   step3 sha256 f6e91c1dcaf43d0b...   step4 sha256 f6e91c1dcaf43d0b...   identical True
history.json   step3 sha256 3da5fb67e3e916b4...   step4 sha256 3da5fb67e3e916b4...   identical True
```

Both files carry `run_name: step3_rr_ratios`, timestamp
`2026-08-22T16:14:13.768739`, scalar `focal_loss_alpha: 0.5`, and **no**
`oversampling`, `focal_loss_beta` or `focal_alpha_class_counts` keys - all of
which step 4's code writes. Its `history.json` peaks at val macro-F1 0.5618 at
epoch 8, which is step 3's curve, not the 0.7288-at-epoch-3 curve you
described. It is a copy of the step 3 run placed in a step 4 folder.

**I did not fill the step 4 ablation row from it.** Reading it
programmatically, as instructed, would have written step 3's numbers under
step 4's name and commit hash - the exact failure the "every number must trace
to a `results/<run>/metrics.json`" rule exists to prevent. The row is present
but marked `pending`, with a note recording the sha256 collision and your
verbally-reported figures (val 0.7288 -> 0.5791, plateau 0.5661, test 0.5599)
as provenance, explicitly flagged as not yet traceable.

**I also did not commit `results/step4_class_alpha/`** - it is left untracked
rather than enshrining a mislabelled artefact in the results tree. Drop the
real `metrics.json` in and I will fill the row and commit it.

Everything else in the task is complete.

---

## What changed

**`DS1_VAL` grew from 3 records to 5:**

```python
DS1_VAL = ['106', '118', '207', '220', '223']
```

`DS1_TRAIN` is unchanged in form - still
`[rec for rec in DS1 if rec not in DS1_VAL]`, computed rather than hardcoded -
and now resolves to 17 records.

**The selection rule is recorded in code and written to `metrics.json` under
`val_selection_rule`**, as a structured object rather than prose, so it is
auditable and cannot be quietly re-derived:

- **keep** `207, 220, 223` - the original step 1 set, so the signal stays as
  comparable as possible
- **exclude 209** - holds 383 of the 943 DS1 S beats; moving it to validation
  would starve training of S
- **exclude 201** - same subject as test record 202
- **add 118** - most S beats of the remaining candidates (96)
- **add 106** - most V beats among records with <= 5 S beats (520), which
  fixes validation V-F1 resting on only two records

**CLAUDE.md corrected.** The "Known facts" augmentation entry said
augmentation is active and "will be removed in a later step" - that step was
step 4. It now records that augmentation is **removed**, that both functions
are dead code with zero call sites, that class balancing happens in the loss
via `FOCAL_ALPHA`, and - kept as history - that baseline through step 3 were
produced *with* augmentation so their numbers are read correctly. Added a note
that `config.oversampling` or the steps/epoch count is how to check, since
`train_distribution` structurally cannot reveal it.

A new "Known facts" entry records the DS1_VAL change and its comparability
consequence. The stale "Current state" block (baseline-only) now points at
`docs/ablation.md` and carries the step 2 / step 3 rows, and the "Blocking
problem" paragraph is updated: the failure is no longer purely S->N, since
step 3 moved those beats to V.

**`docs/ablation.md`** gains a blockquote at the top recording the validation
change and that baseline..step 4 are not like-for-like with step 5 onward.

Nothing else changed: no architecture, loss, alpha, beta, learning-rate, RR
feature, or augmentation change.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` comparison - exactly which functions changed:**

```
changed  : NONE
added    : NONE
removed  : NONE
unchanged: 12 of 12
classes changed: NONE
```

Step 5 touches module-level constants only, so **zero functions changed** -
which is what a pure split change should look like.

**Constants unchanged** (count identical before and after, one occurrence
each): `PRE_SAMPLES = 90`, `POST_SAMPLES = 144`, `FOCAL_BETA = 0.5`,
`FOCAL_GAMMA = 2.0`, `ADAM_LEARNING_RATE = 1e-4`, `RR_LOCAL_WINDOW = 10`,
`RR_CLIP_MIN = 0.2`, `RR_CLIP_MAX = 3.0`.

**The 5 RR feature definitions**: the `rr = [...]` source block and
`RR_FEATURE_NAMES` are both byte-identical to the pre-edit file.

**Record literals**: `DS1` and `DS2` identical (sha256 `9f20e3ac1758a312...`
and `b8a3e6bbdeeec72a...`, the same hashes as steps 1 through 4). `DS1_VAL`
changed, as intended.

**`augment_training_data` still dead** - AST call-graph reports zero call
sites, unchanged by this step.

**Split and counts:**

```
A) split membership
   DS1_VAL   (5): ['106', '118', '207', '220', '223']
   DS1_TRAIN (17): ['101', '108', '109', '112', '114', '115', '116', '119', '122', '124', '201', '203', '205', '208', '209', '215', '230']
   [PASS] DS1_VAL has 5 records
   [PASS] DS1_VAL == ['106','118','207','220','223']
   [PASS] DS1_VAL subset of DS1
   [PASS] DS1_VAL disjoint from DS2
   [PASS] partitions DS1
   [PASS] no overlap train/val
   [PASS] 209 in DS1_TRAIN, NOT in DS1_VAL
   [PASS] 201 in DS1_TRAIN, NOT in DS1_VAL
   [PASS] 202 (same subject as 201) is in DS2, untouched
   [PASS] DS1_TRAIN computed, not hardcoded

B) class counts from tools/ds1_beat_counts.json
   set               N      S      V    total    S share
   train         36631    574   2569    39774      1.44%
   val            9208    369   1219    10796      3.42%

   validation grew: 6494 -> 10796 beats (12.8% -> 21.3% of DS1)
   val S beats : 273 -> 369
   val V beats : 683 -> 1219
   records contributing V to val: 2 -> 4
   records contributing S to val: 3 -> 4

   training shrank: 44076 -> 39774 beats (S 670 -> 574)

C) FOCAL_ALPHA recomputes automatically from the new counts
   step 4 counts [40301, 670, 3105] -> alpha [0.2428, 1.8827, 0.8746]
   step 5 counts [36631, 574, 2569] -> alpha [0.235, 1.8775, 0.8875]
   alpha sums to 3: 3.000000
   NOTE: alpha shifts because it is derived from DS1_TRAIN counts,
         which changed. Max per-class shift: 0.0129

D) steps/epoch (no oversampling since step 4)
   step 4: 44076 samples / 128 = 345 steps
   step 5: 39774 samples / 128 = 311 steps

============================================================
   [PASS] alpha still sums to NUM_CLASSES
   [PASS] alpha ordering still N < V < S
OVERALL: PASS
```

**209 and 201 are both in DS1_TRAIN and neither is in DS1_VAL** - checked
explicitly, along with 202 still being in DS2.

---

## What this costs and what it buys

| | step 4 (3 records) | step 5 (5 records) |
|---|---|---|
| validation beats | 6,494 (12.8% of DS1) | **10,796 (21.3%)** |
| validation S beats | 273 | **369** |
| validation V beats | 683 | **1,219** |
| records contributing S | 3 | 4 |
| records contributing V | **2** | **4** |
| training beats | 44,076 | 39,774 |
| training S beats | 670 | **574** |
| steps/epoch @128 | 345 | **311** |

The cost is real: training loses 96 S beats, 14% of an already tiny class.
The buy is that validation V-F1 no longer rests on two records, and the
selection signal is computed over 66% more beats.

**`FOCAL_ALPHA` shifts as an automatic consequence**, because it is derived
from DS1_TRAIN counts: `[0.2428, 1.8827, 0.8746]` -> `[0.2350, 1.8775,
0.8875]`, max per-class shift 0.0129. This is the code doing what it should,
not a manual alpha edit - but it does mean step 5 is not a *pure* validation
change. Flagging it because the task said change nothing else.

---

## Falsifiable predictions for the next Kaggle run

1. **311 steps/epoch** (39,774 samples at batch 128), down from 345.
   `train_distribution` will read **N=36631 / S=574 / V=2569** and
   `config.ds1_val` will list the five records.
2. **`config.focal_loss_alpha` will be approximately
   `[0.2350, 1.8775, 0.8875]`**, summing to 3.0, with
   `focal_alpha_class_counts = [36631, 574, 2569]`.
3. **The epoch-to-epoch swing in val macro-F1 will be smaller than step 4's
   0.1497.** This is the actual bet - the whole point of the change. I expect
   the largest single-epoch swing to come in under 0.10.
4. **`val_per_record` will show non-zero V support for four of the five
   records** (all but 220), so validation V-F1 stops depending on two
   patients.

Prediction 3 is the one that decides whether this step worked. Note it is a
statement about *validation stability*, not about test macro-F1 - a
lower-variance selector should produce a more trustworthy checkpoint, but
with 14% fewer S training beats the test score could legitimately go either
way.

---

## git diff

```diff
diff --git a/src/train.py b/src/train.py
index 92ad64c..238373a 100644
--- a/src/train.py
+++ b/src/train.py
@@ -157,13 +157,50 @@ DS2 = [
     '222', '228', '231', '232', '233', '234'
 ]
 
-# Patient-wise validation split (step 1).
+# Patient-wise validation split (step 1, enlarged at step 5).
 # Whole records are held out of training, never individual beats, so no
-# patient can appear on both sides. 207 / 220 / 223 are chosen because
-# they carry enough S beats (106 / 94 / 73) for val S-F1 to be a usable
-# selection signal. Record 201 is deliberately NOT used here: it is the
-# same subject as 202, which lives in DS2.
-DS1_VAL = ['207', '220', '223']
+# patient can appear on both sides.
+#
+# Step 5 grew this from 3 records to 5. With 3 records, val macro-F1 swung
+# 0.7288 -> 0.5791 between two consecutive epochs, and the selected peak
+# sat 0.1627 above the surrounding plateau for exactly one epoch. That
+# checkpoint then scored 0.5599 on test - we were selecting noise. More
+# records means a lower-variance selection signal.
+DS1_VAL = ['106', '118', '207', '220', '223']
+
+# Why these five, recorded so the choice is auditable and never quietly
+# re-derived. Written into metrics.json as val_selection_rule.
+VAL_SELECTION_RULE = {
+    "keep": {
+        "records": ['207', '220', '223'],
+        "reason": "the original step 1 validation set, kept so the "
+                  "selection signal stays comparable where possible"
+    },
+    "exclude": [
+        {
+            "record": "209",
+            "reason": "holds 383 of the 943 DS1 S beats; moving it to "
+                      "validation would starve training of S"
+        },
+        {
+            "record": "201",
+            "reason": "same subject as test record 202 (PhysioNet); "
+                      "validating on it would stack a second leak"
+        }
+    ],
+    "add": [
+        {
+            "record": "118",
+            "reason": "most S beats of the remaining candidates (96)"
+        },
+        {
+            "record": "106",
+            "reason": "most V beats among records with <= 5 S beats "
+                      "(520); 220 has zero V, so validation V-F1 rested "
+                      "on only two records"
+        }
+    ]
+}
 
 DS1_TRAIN = [rec for rec in DS1 if rec not in DS1_VAL]
 
@@ -1673,6 +1710,8 @@ metrics = {
 
         "ds1_val": DS1_VAL,
 
+        "val_selection_rule": VAL_SELECTION_RULE,
+
         "rr_feature_names": RR_FEATURE_NAMES,
 
         "rr_local_window": RR_LOCAL_WINDOW,
```

---

## Commit

```
7b0236d  step 5: enlarge DS1_VAL to 5 records
```

Pushed to `origin/main`. This report lands in a small follow-up commit.

---

## Problems

1. **No step 4 metrics.json** - see the top of this report. The step 4 row is
   `pending`, `results/step4_class_alpha/` is uncommitted, and step 4's
   effect on the S->V leak (its whole purpose) is still unmeasured. This also
   means the step 3 stated condition - "if step 4 does not recover macro-F1
   above 0.6800, the ratio features get reconsidered" - **cannot yet be
   evaluated**. Your verbal 0.5599 would fail that condition, but I am not
   treating an unverified number as grounds for reconsidering the features.

2. **Step 5 is not perfectly single-variable.** Enlarging validation
   necessarily shrinks training, which shifts `FOCAL_ALPHA` (max 0.0129) and
   removes 96 S beats. Unavoidable given the fix, but it means a step 5 vs
   step 4 delta mixes three things: selection variance, training-set size,
   and a small alpha shift.

3. **Training S beats are down to 574.** That is a thin class to learn from,
   and it is the class the whole project is about. If step 5's test S recall
   drops, the 14% reduction in S training data is a live explanation before
   "the larger validation set selected a worse model".

4. **Record 207 remains a severe outlier** (step 3: accuracy 0.1733, N recall
   0.0824, S recall 0.0000) and it is still in DS1_VAL. Enlarging the set
   dilutes its influence from 1/3 to 1/5 of the records, which helps, but it
   does not fix it. If 207 continues to dominate, removing it is a separate
   deliberate decision - and it would break comparability again.

5. Carried over: record 114 lead swap still unfixed;
   `tools/inspect_ds1.py` still writes its JSON into `tools/`; stale root
   `__pycache__/` still present.
