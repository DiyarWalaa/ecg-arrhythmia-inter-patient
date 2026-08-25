# Last report

**Task:** E7 - sweep the balanced-sampler class ratio, selected on validation.

**Date:** 2026-08-25

---

## E6 predictions - scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | 345 steps, sampler config keys, **`total_parameters == 239171`** | **PASS - all exact.** First run where a parameter claim was checkable from the artefact, and it holds. |
| 2 | S recall rises well past 0.1972, above 0.60 | **FAIL** - 0.3388 |
| 3 | S precision falls below 0.20 | **FAIL** - 0.3934 |
| 4 | N-F1 drops from 0.9787; above 0.95 means under-firing | **PASS** - 0.9646 |

**Predictions 2 and 3 failed in the same direction**, and prediction 4 says
why: the sampler moved everything less than I expected. I wrote that N-F1
holding above 0.95 would mean the sampler was under-firing. It held at 0.9646.
It is under-firing - which is precisely the premise of E7.

**E6 is nonetheless the best run so far**: macro-F1 0.7263 argmax, **0.7372
tuned**, S-F1 0.3641 (nearly double E2's 0.2686). Two firsts: `best_epoch` is
**6** rather than 1, and **threshold tuning finally transferred** (+0.0109 on
test, after -0.0495 at E1 and -0.0846 at E2). Hard constraint 2 also holds
again for the first time since E0.

---

## What changed

**`SAMPLER_RATIO_GRID = [1.0, 2.0, 3.0, 4.0]`**, with weights derived rather
than hardcoded:

```python
def weights_for_ratio(ratio, n_classes=NUM_CLASSES, minority_index=1):
    raw = [1.0] * n_classes
    raw[minority_index] = float(ratio)
    total = sum(raw)
    return [w / total for w in raw]
```

A ratio `r` draws S `r` times as often as each of N and V - weights
proportional to `[1, r, 1]`, normalised.

**Three new helpers**: `reset_seeds()` (identical initialisation per setting),
`make_balanced_dataset(weights)` (the per-class stream factory, lifted out of
the old inline section 19B), and `make_callbacks()` (EarlyStopping,
ReduceLROnPlateau and ValidationMetrics all carry mutable state, so they
cannot be shared across runs).

**Section 22 is the sweep.** Per ratio: reset seeds, build the dataset, build
the model, train with `steps_per_epoch = 345`, record the best val macro-F1,
best epoch, epochs run, early-stopping flag and the full per-epoch curve. The
winner is kept under the names `model` / `history` / `val_metrics_cb` /
`early_stopping`, so sections 22B through 30 - the diagnostics, the threshold
tuning and the test evaluation - needed **no changes at all**.

**metrics.json** gains `sampler_ratio_grid`, `selected_ratio`,
`sampler_selection_criterion` and `sampler_sweep` (per ratio: weights, best
val macro-F1, best epoch, epochs run, full curve). `sampling_weights` now
records the *selected* vector. `total_parameters` was added at E6.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` against E6 HEAD:**

```
changed : NONE
added   : ['make_balanced_dataset', 'make_callbacks', 'reset_seeds',
           'weights_for_ratio']
removed : NONE
```

**Zero existing functions changed**, and **`build_model` is byte-identical to
E6's** - so the architecture, the loss and the 239,171 parameters are provably
untouched.

**The test set is not evaluated inside the sweep.** There are two loops over
`SAMPLER_RATIO_GRID`; the sweep is the one with target `sweep_ratio`:

```
`for _ratio in SAMPLER_RATIO_GRID:`     lines 1353-1356  <- printing weights
`for sweep_ratio in SAMPLER_RATIO_GRID:` lines 1666-1744 <- the sweep
```

Walking every AST node inside lines 1666-1744 for seventeen test-related
names - `X_test`, `RR_test`, `y_test`, `y_test_encoded`, `y_test_cat`, `DS2`,
`y_pred_prob`, `y_pred_enc`, `y_pred_enc_tuned`, `acc`, `acc_tuned`, `cm`,
`cm_tuned`, `report_argmax`, `report_tuned`, `per_class_roc_auc`,
`THRESHOLD_WEIGHTS`:

```
found: NONE          RESULT: CLEAN
```

Every `predict` / `evaluate` / `fit` call in the file, with position:

```
line 1691  INSIDE sweep    fit      obj=sweep_model
line 1552  outside sweep   predict  self.x_val        (ValidationMetrics)
line 1814  outside sweep   predict  X_val, RR_val     (per-record diagnostics)
line 1881  outside sweep   predict  X_val, RR_val     (threshold tuning)
line 1979  outside sweep   predict  X_test, RR_test   <- the only DS2 call
```

Code path in order: **sweep 1666-1744 -> selection line 1741
(`model = sweep_model`) -> threshold tuning on validation line 1888 -> DS2
scored once line 1980.**

**Weight vectors** - every one matches the specification exactly:

| ratio | weights [N, S, V] | sum | matches spec |
|---|---|---|---|
| 1.0 | `[0.333333, 0.333333, 0.333333]` | 1.000000 | yes |
| 2.0 | `[0.25, 0.50, 0.25]` | 1.000000 | yes |
| 3.0 | `[0.20, 0.60, 0.20]` | 1.000000 | yes |
| 4.0 | `[0.166667, 0.666667, 0.166667]` | 1.000000 | yes |

Ratio 1 equals E6's `[1/3, 1/3, 1/3]` to 1e-15. N and V weights are always
equal; the S weight is strictly increasing across the grid.

**Unchanged**: `PRE_SAMPLES = 90`, `POST_SAMPLES = 144`,
`ADAM_LEARNING_RATE = 1e-4`, `RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`,
`RR_CLIP_MAX = 3.0`, `SAMPLING_RATE_HZ = 360.0`, `BATCH_SIZE = 128` - one
occurrence each. `loss='categorical_crossentropy'`;
`categorical_focal_loss` and `augment_training_data` both have **zero call
sites**. `rr = [...]` and `RR_FEATURE_NAMES` byte-identical;
`WAVELET_TARGET_FREQS_HZ` identical, widths 8.1028 ... 0.9003. `DS1`
`9f20e3ac1758a312...`, `DS2` `b8a3e6bbdeeec72a...`, `DS1_VAL`
`0d9df3612a6111a1...` = `['207','220','223']`.

---

## What the grid actually costs

Worth seeing before reading the result. With 345 steps of 128:

| ratio | S draws/epoch | each S beat seen | N pool seen/epoch |
|---|---|---|---|
| 1.0 | 14,720 | 22.0x | 37% |
| 2.0 | 22,080 | 33.0x | 27% |
| 3.0 | 26,496 | 39.5x | 22% |
| 4.0 | 29,440 | 43.9x | 18% |

At ratio 4 the model sees each of the 670 S beats **44 times per epoch** and
only **18% of the 40,301 N beats**. Pushing S harder is not free: it buys S
exposure by starving N coverage, and at some point the N class will degrade
faster than S improves. That crossover is what the sweep is measuring.

---

## Falsifiable predictions

1. **`sampler_sweep` will have 4 entries** with the weight vectors tabulated
   above, `total_parameters == 239171`, `steps_per_epoch == 345`,
   `loss == "categorical_crossentropy"`, and `selected_ratio` equal to
   whichever entry has the highest `best_val_macro_f1`.
2. **Ratio 1.0 will not be selected.** E6 showed the 1:1:1 point under-firing
   with N-F1 still at 0.9646; there should be headroom above it. I expect
   **2.0 or 3.0**.
3. **S recall will rise monotonically with the ratio, and S precision will
   fall monotonically.** This is the mechanism, and if the sweep does not show
   both trends across the four settings then the ratio is not doing what I
   think it is.
4. **Ratio 4.0 will have the worst N-F1 of the four**, and I expect it below
   0.93 - at 18% N coverage per epoch the N class starts to suffer.

Prediction 2 is the bet. Prediction 3 is the mechanism check. I am **not**
predicting the selected model beats E6's 0.7372 tuned: a better operating
point on validation need not transfer, and validation is only 273 S beats.

---

## Commit

```
1339261  E7: validation-selected sampler ratio sweep
```

Pushed to `origin/main`. `docs/ablation.md` now carries 14 runs including E6.

---

## Problems

1. **Four trainings per run.** With `EPOCHS = 40` and E6 stopping around
   epoch 16, expect roughly 4x E6's runtime. If Kaggle times out, drop ratio
   1.0 from the grid - E6 already measured that point.

2. **Selecting the ratio on DS1_VAL adds a third selection layer.** We now
   choose a checkpoint, a sampling ratio, and threshold weights against the
   same 6,494 validation beats, 273 of them S. DS2 stays untouched so the test
   number is honest, but the validation figure is now triply optimistic and
   should not be quoted as a performance estimate.

3. **The sampler still cannot be executed locally.** No TensorFlow here, so
   `make_balanced_dataset` is verified structurally and by the numpy
   simulation from E6, not by running `tf.data`. E6's run confirmed the
   mechanism works end to end, which raises confidence, but each new weight
   vector is still unexecuted until Kaggle.

4. **Higher ratios amplify the repetition concern from E6.** 44 repeats per
   epoch at ratio 4 is a lot of exposure to 670 beats. If the sweep selects a
   high ratio and S-F1 improves on validation but not on DS2, memorisation of
   the training S beats is the first thing to suspect.

5. **`X_raw_train` is still unused** (carried from E6), and `SAMPLER` is still
   the literal `"balanced_batch"` even though the ratio now varies - the ratio
   is recorded separately in `selected_ratio`, so nothing is ambiguous, but
   the name no longer implies 1:1:1.

6. Carried over: record 114 lead swap unfixed; record 207 still a validation
   outlier; `tools/inspect_ds1.py` JSON stale; stale root `__pycache__/`.
