# Last report

**Task:** E1 - revert `DS1_VAL` to 3 records, add validation-tuned decision
thresholds. Also: correct the stale CLAUDE.md "Known facts".

**Date:** 2026-08-23

---

## E0 prediction - scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | 378 steps/epoch; `train_distribution` N=36631 / S=574 / V=2569; `oversampling: true`; scalar alpha `0.5`; no `beta_sweep` key | **PASS** - distribution exact, `oversampling: true`, `alpha 0.5`, `oversampling_multipliers {N:1, S:7, V:3}` |

E0's diagnosis is confirmed directly by its own history. The per-class
validation curves show exactly the mechanism you described:

```
val_f1_V : 0.3114  0.6617  0.5867  0.5663  0.4157  0.3944 ...   peaks epoch 2
val_f1_S : 0.2629  0.1852  0.2466  0.2658  0.2661  0.2869 ...   still rising at 6
val_macro: 0.5112  0.5911  0.5774  0.5768  0.4958  0.4872 ...   follows V
```

macro-F1 peaks at epoch 2 because V peaks at epoch 2. S never gets the
chance to mature.

---

## Part 1 - revert

`DS1_VAL = ['207', '220', '223']`, byte-identical to step 3's literal.
`DS1_TRAIN` is still computed and now resolves to 19 records, with **106 and
118 back in training**.

`VAL_SELECTION_RULE` in `metrics.json` is rewritten to record the revert
rather than the step 5 rationale: the records, the history, `reverted_at:
"E1"`, the full `revert_reason` (106 has 520 V / **zero** S, 118 has 16 V /
96 S, validation became 85.3% N / 11.3% V / 3.4% S, V-F1 peaks epoch 2 while
S-F1 rises to epoch 6, E0 selected epoch 2 and scored 0.5591 against a
required 0.6400), the still-binding exclusions of 209 and 201, and the
accepted limitation that 220 has zero V beats.

---

## Part 2 - decision-threshold tuning

New section **11B** defines two functions; new section **22C** runs the
search; section **24** reports both rules.

`tune_decision_weights(y_prob, y_true_int, num_classes, grid, max_passes)`
does coordinate ascent on `w`, with `w_N` pinned to 1.0 (scaling all three
classes leaves the argmax unchanged, so one must be the reference). Each pass
sweeps `w_S` then `w_V` over the whole grid keeping the best; passes repeat
until one yields no improvement, or `max_passes` is reached. It returns
`(weights, best_macro_f1, search_log)`.

`THRESHOLD_GRID` is 15 log-spaced points from **0.25 to 32** (2^-2 .. 2^5,
constant ratio sqrt(2)). `THRESHOLD_MAX_PASSES = 6`.

`metrics.json` gains `threshold_weights`, `threshold_class_order`,
`threshold_val_macro_f1`, `threshold_val_macro_f1_argmax`, `threshold_grid`,
`threshold_search_log`, and two complete blocks **`test_argmax`** and
**`test_tuned`**, each with accuracy, classification report and confusion
matrix. The existing top-level `classification_report` / `confusion_matrix` /
`test_accuracy` are left as the **argmax** values so `tools/make_ablation.py`
and every prior row stay comparable.

---

## Proof that the search never touches DS2

Same method as the BETA-sweep isolation, but stronger, because the search is
a **pure function**. I computed each function's free variables - names
referenced but neither an argument nor bound locally:

```
tune_decision_weights (lines 764-829)
  args       : ['grid', 'max_passes', 'num_classes', 'y_prob', 'y_true_int']
  free names : ['float', 'list', 'macro_f1_from_weights', 'range']
  test names : NONE                                        -> CLEAN

macro_f1_from_weights (lines 746-761)
  args       : ['labels', 'weights', 'y_prob', 'y_true_int']
  free names : ['float', 'np', 'precision_recall_fscore_support']
  test names : NONE                                        -> CLEAN
```

Every name either arrives as an argument or is a builtin / module import.
**Neither function can reach `X_test`, `RR_test` or `y_test_encoded` even in
principle** - there is no path, not merely no current use.

The call site passes validation probabilities only:

```
line 1491: tune_decision_weights(val_prob_for_threshold, y_val,
                                 NUM_CLASSES, THRESHOLD_GRID,
                                 THRESHOLD_MAX_PASSES)

val_prob_for_threshold = model.predict([X_val, RR_val], ...)
```

Ordering: weights frozen at line 1491, first applied to DS2 at line 1623.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` against E0 HEAD:**

```
changed : NONE
added   : ['macro_f1_from_weights', 'tune_decision_weights']
removed : NONE
```

**Zero existing functions changed.** Against step 3 the only differences are
`build_model` and `categorical_focal_loss`, and those are the hoisted
constant names carried over from E0 - identical values.

**Split membership** - all pass: `DS1_VAL == ['207','220','223']`, 106 and
118 back in `DS1_TRAIN`, 209 and 201 in training, disjoint from DS2, and the
two sets partition DS1.

```
DS1_TRAIN (19): 101 106 108 109 112 114 115 116 118 119 122
                124 201 203 205 208 209 215 230
train {'N': 40301, 'S': 670, 'V': 3105}  raw 44076 -> aug 54306
val   {'N': 5538,  'S': 273, 'V': 683}   total 6494
val composition: 85.3% N / 4.2% S / 10.5% V
```

Compare E0's validation: 85.3% N / **3.4% S / 11.3% V**. The revert roughly
swaps the S and V shares, which is the whole point.

**Constants unchanged** (one occurrence each): `PRE_SAMPLES = 90`,
`POST_SAMPLES = 144`, `FOCAL_ALPHA = 0.50`, `FOCAL_GAMMA = 2.0`,
`ADAM_LEARNING_RATE = 1e-4`, `RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`,
`RR_CLIP_MAX = 3.0`.

**Augmentation multipliers** inside `augment_training_data` are
`[0, 6, 2, 0]` - identical to step 3. `augment_training_data` is still called
from `<module>` (AST call-graph).

**The 5 RR features**: `rr = [...]` block byte-identical to both HEAD and
step 3; `RR_FEATURE_NAMES` unchanged. `DS1` `9f20e3ac1758a312...`, `DS2`
`b8a3e6bbdeeec72a...` unchanged; `DS1_VAL` now matches step 3's literal
exactly.

**Coordinate ascent, behavioural test** on synthetic data shaped like the
real DS1_VAL (5538 N / 273 S / 683 V) with a deliberately under-calling S
model, using the functions lifted from the edited source:

```
plain argmax macro-F1 : 0.6648
tuned        macro-F1 : 0.6718   (weights [1.0, 1.0, 0.7071])
passes run: 2 of 6
S recall 0.3443 -> 0.3590   S predicted 397 -> 425 (true 273)

[PASS] w_N pinned to exactly 1.0
[PASS] tuned >= argmax
[PASS] macro-F1 non-decreasing across the search log
[PASS] terminated within max_passes
[PASS] every returned weight came from the grid
[PASS] reported best matches an independent recomputation
[PASS] S recall improved
[PASS] repeat run gives identical weights (deterministic)
[PASS] perfect classifier -> macro-F1 1.0000, no weight change needed
```

The synthetic model under-calls S far less severely than ours does, so the
gain there is small; the test validates the mechanics, not the magnitude.

---

## Falsifiable predictions

1. **425 steps/epoch** - 40301 + 670x7 + 3105x3 = 54,306 at batch 128.
   That is exactly step 3's count, which is the point: E1 restores step 3's
   training configuration.
2. **The plain-argmax confusion matrix will reproduce step 3's**
   `[[43233,479,521],[692,278,866],[138,8,3074]]`, and `best_epoch` will be
   8. The pipeline is deterministic and nothing on the training path changed.
   **Any mismatch is a bug, and I will report it as one rather than as a
   result.**
3. **`threshold_weights[1]` (w_S) will be greater than 1.0** - substantially
   so, given the model under-calls S about 6x. `w_N` will be exactly 1.0.
4. **Tuned S recall will exceed argmax S recall by a wide margin**, with S
   precision falling. Whether tuned *macro-F1* beats argmax is genuinely
   open: the search maximises validation macro-F1, but 273 validation S beats
   is a thin basis and the weights may not transfer to DS2's 1836.

Prediction 2 is the integrity check on the revert. Prediction 4 is the
experiment.

---

## CLAUDE.md corrections

- The augmentation entry said it was removed at step 4 and is dead code.
  Corrected: **active again since E0** (S x7, V x3, called from section 18),
  violating hard constraint 2 deliberately, with the RR-duplication defect
  now worse than at step 3 because the duplicated vector is the five
  patient-relative ratios. Also records that `FOCAL_ALPHA` is back to the
  scalar 0.50 and that balancing lives in the data, not the loss.
- The DS1_VAL entry now records the E1 revert with the V-heavy-validation
  reason, the accepted zero-V limitation of record 220, and that steps 5, 6
  and E0 are not comparable to the rest.
- Current state gains the E0 row and the statement that E1's argmax result
  must reproduce step 3's confusion matrix.
- Blocking problem rewritten around the real finding: our S precision matches
  the literature while S recall is 6x worse, which points at the decision
  rule rather than the features.

---

## Commit

```
3d3494b  E1: revert DS1_VAL to 3 records, add validation-tuned decision thresholds
```

Pushed to `origin/main`. This report lands in a small follow-up commit.

---

## Problems

1. **Tuning thresholds on 273 validation S beats is a thin basis.** The
   weights are selected to maximise macro-F1 over a validation set whose S
   class is 4.2% of 6,494 beats. That is a real overfitting risk, and it is
   why both results are reported rather than only the tuned one. If tuned
   macro-F1 beats argmax on validation but not on DS2, that gap is the
   finding, not a failure.

2. **The threshold search adds a second selection layer on the same
   validation set** that already picks the epoch. Validation macro-F1 is now
   doubly optimistic. DS2 is untouched so the test number stays honest, but
   the validation figure should not be quoted as an estimate of test
   performance.

3. **Plots in sections 25-28 still use the argmax predictions.** I left them
   comparable to previous runs rather than silently switching them to the
   tuned rule. The tuned confusion matrix is in `metrics.json` under
   `test_tuned` but is not plotted.

4. **Hard constraint 2 is still violated** - augmentation remains on, and the
   RR duplication defect is live. E1 does not address it; that is a later
   step and it needs a replacement balancing mechanism.

5. **`docs/ablation.md` has no E0 or E1 rows yet.** E0's results are now
   committed, so its row can be added; I did not add one because
   `tools/make_ablation.py` needs a `RUNS` entry and E0's headline number
   only makes sense alongside E1's. Worth doing in one pass once E1 returns.

6. Carried over: record 114 lead swap still unfixed; record 207 still a
   validation outlier; `tools/inspect_ds1.py` writes its JSON into `tools/`
   and that JSON is stale; stale root `__pycache__/` present.
