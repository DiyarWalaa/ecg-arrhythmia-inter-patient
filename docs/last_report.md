# Last report

**Task:** step 6 - sweep the focal-loss BETA and select on validation. Also:
move the ablation generator into `tools/`.

**Date:** 2026-08-23

---

## Step 5 predictions - scorecard

All four passed.

| # | prediction | outcome |
|---|---|---|
| 1 | 311 steps/epoch; `train_distribution` N=36631 / S=574 / V=2569 | **PASS** - exact (39,774 samples at batch 128 = 311) |
| 2 | alpha approximately `[0.2350, 1.8775, 0.8875]`, counts `[36631, 574, 2569]` | **PASS** - exact |
| 3 | largest single-epoch val macro-F1 swing under 0.10 (step 4: 0.1497) | **PASS** - **0.0724** |
| 4 | four of five validation records carry V beats | **PASS** - 4 of 5 |

**Step 5 did what it was for.** Selection variance halved and the
test-minus-validation gap narrowed from -0.1689 to **-0.0617**. But test
macro-F1 fell again to 0.5408, and the enlarged validation set made the
actual problem legible: **6273 V predictions against 3220 true V beats
(+95%)**. The model floods V. That is exactly what step 6 targets.

---

## What changed

**`BETA_GRID = [0.0, 0.25, 0.41, 0.50]`** added in section 3.

**The alpha derivation became a function.** `compute_focal_alpha(class_counts,
beta)` replaces the inline module-level block, so the sweep can call it per
BETA. Section 12 now prints the alpha vector for every BETA in the grid before
training starts.

**`reset_seeds(seed=SEED)`** added in section 1. Called at the top of each
sweep iteration so every BETA trains from an identical initialisation -
without it, a difference in val macro-F1 could be weight init rather than
BETA.

**`make_callbacks()`** added in section 21, returning fresh
`(callbacks, val_metrics_cb, early_stopping)`. EarlyStopping and
ReduceLROnPlateau carry mutable state (`wait`, `best`, `stopped_epoch`,
`best_weights`) and `ValidationMetrics` accumulates a records list, so they
cannot be reused across runs.

**Section 20 no longer builds a model.** It just captures `ECG_INPUT_SHAPE`
and `RR_INPUT_SHAPE`. Section 22 is now the sweep: for each BETA it resets
seeds, computes alpha, assigns the module-level `FOCAL_ALPHA` (which
`build_model` reads when it compiles), builds, trains, and records the result.
The winning run's `model`, `history`, `val_metrics_cb` and `early_stopping`
are kept under those exact names, so **sections 22B through 30 required no
changes at all**.

**metrics.json** gains `beta_grid`, `beta_sweep` (per BETA: alpha, best val
macro-F1, best epoch, epochs run, early-stopping flag, full val macro-F1
curve), `selected_beta`, and `selection_criterion`. `focal_loss_beta` now
reports the *selected* beta; `focal_loss_beta_default` keeps the 0.5 constant.

**`tools/make_ablation.py`** - the generator now lives in the repo with its
gate assertions, so `docs/ablation.md` is reproducible without me. It supports
`--check` (verify current, write nothing) for CI. The gate refuses to emit a
row unless the metrics file identifies itself as the run the table claims, and
it hashes every file to catch the duplicate-copy failure mode directly.

Nothing else changed: no architecture, RR feature, learning-rate, split, or
augmentation change. `DS1_VAL` is still `['106','118','207','220','223']`.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` comparison:**

```
changed  : NONE
added    : ['compute_focal_alpha', 'make_callbacks', 'reset_seeds']
removed  : NONE
unchanged: 12 of 12
classes changed: NONE
```

**Zero existing functions changed.** `build_model` and
`categorical_focal_loss` are untouched - the sweep drives them through the
module-level `FOCAL_ALPHA` global rather than by editing either one.

**The test set is not evaluated inside the sweep.** There are two loops over
`BETA_GRID`; the sweep is the one with target `sweep_beta`:

```
`for _beta in BETA_GRID:`      lines  839-842   <- printing alphas only
`for sweep_beta in BETA_GRID:` lines 1331-1412  <- the sweep
```

Walking every AST node inside lines 1331-1412 for the names `X_test`,
`RR_test`, `y_test`, `y_test_encoded`, `y_test_cat`, `DS2`, `y_pred_prob`,
`y_pred_enc`, `acc`, `cm`:

```
found: NONE          RESULT: CLEAN
```

Every `predict`/`evaluate` call in the file, with its position relative to the
sweep:

```
line 1606  outside sweep  args: ['model', 'X_test', 'RR_test', 'BATCH_SIZE']
line 1484  outside sweep  args: ['model', 'X_val', 'RR_val']
line 1216  outside sweep  args: ['self', 'self']   <- inside ValidationMetrics, val only
```

The code path in order:

```
1. sweep trains 4 models, validation only   lines 1331-1412
2. selection keeps the winning model        line 1409  (model = sweep_model)
3. FOCAL_ALPHA reset to the selected beta   line 1417
4. DS2 scored ONCE on that model            line 1607  ([X_test, RR_test],)
```

**Constants unchanged** (one occurrence each, before and after):
`PRE_SAMPLES = 90`, `POST_SAMPLES = 144`, `ADAM_LEARNING_RATE = 1e-4`,
`FOCAL_GAMMA = 2.0`, `FOCAL_BETA = 0.5`, `RR_LOCAL_WINDOW = 10`,
`RR_CLIP_MIN = 0.2`, `RR_CLIP_MAX = 3.0`. The `rr = [...]` source block and
`RR_FEATURE_NAMES` are byte-identical. `DS1` `9f20e3ac1758a312...`, `DS2`
`b8a3e6bbdeeec72a...`, `DS1_VAL` `f6759845186d6324...` - all unchanged.

**Alpha verified numerically** for every BETA against DS1_TRAIN counts
`[36631, 574, 2569]`, each checked against an independent recomputation of
`(1/count)**beta` rescaled to sum 3:

| beta | alpha | sum | S:N | V:N |
|---|---|---|---|---|
| 0.00 | `[1.0000, 1.0000, 1.0000]` | 3.0000 | 1.000 | 1.000 |
| 0.25 | `[0.5200, 1.4696, 1.0104]` | 3.0000 | 2.826 | 1.943 |
| 0.41 | `[0.3168, 1.7412, 0.9419]` | 3.0000 | 5.496 | **2.973** |
| 0.50 | `[0.2350, 1.8775, 0.8875]` | 3.0000 | 7.989 | 3.776 |

All four sum to exactly 3.0, all match the independent recomputation to 1e-6,
and `beta=0.0` gives exactly `[1, 1, 1]`.

**`BETA=0.41` reproduces V:N = 2.973**, essentially the 3.000 the old
oversampling produced - so the grid contains a point that undoes precisely the
over-weighting identified in the problem statement.

**`tools/make_ablation.py`** regenerates `docs/ablation.md` and then reports
`up to date (7 runs)` on a second invocation with `--check`, so it is
idempotent. The gate immediately earned its place: it rejected my own first
draft, where I had asserted step 1b's `rr_norm_mean` was a 3-vector when that
run predates the 5 ratio features and has 2. That is the failure the tool
exists to prevent, caught on the tool's first run.

---

## Falsifiable predictions for the next Kaggle run

1. **`beta_sweep` will have 4 entries** with alphas exactly as tabulated
   above, and `selected_beta` will be whichever has the highest
   `best_val_macro_f1`. `config.focal_loss_beta` will equal `selected_beta`,
   while `focal_loss_beta_default` stays 0.50.
2. **`BETA=0.50` will not win.** It is the current setting and it produces the
   V flooding this step exists to fix. I expect **0.25 or 0.41** to be
   selected.
3. **V predictions will fall well below step 5's 6273** for the selected
   model - I expect under 4,500 against 3,220 true V beats. This is the
   mechanistic test.
4. **Runtime roughly 4x step 5's**, since four models train instead of one.

I am **not** predicting macro-F1 clears 0.6800. If `BETA=0.0` wins, that is
the informative outcome: it would mean class reweighting was never helping and
the S problem is not a weighting problem at all.

---

## Commit

```
25229d9  step 6: validation-selected focal BETA sweep
```

Pushed to `origin/main`. This report lands in a small follow-up commit.

---

## Problems

1. **Four trainings per run, one selection - the sweep multiplies runtime by
   4** and, with `EPOCHS=40`, could reach 160 epochs. If Kaggle times out, the
   grid is the thing to trim, not the epoch budget.

2. **Selecting BETA on DS1_VAL adds a selection degree of freedom.** We now
   choose both a checkpoint *and* a hyperparameter against 10,796 validation
   beats, 369 of them S. That is legitimate - DS2 is untouched - but the
   validation estimate is optimistically biased for the winner, and the
   test-minus-validation gap should be read with that in mind. Worth one
   sentence in the methods section.

3. **Four models are held in memory at peak**, and discarded ones are left to
   GC rather than freed explicitly. I deliberately did not call
   `tf.keras.backend.clear_session()`, which would destroy the retained winner
   too. If memory becomes a problem on the P100, saving each model to disk and
   reloading the winner is the fix.

4. **The step 3 condition remains unresolved for a third step.** Step 4's
   selection was confounded, step 5's was clean but V flooding dominates. If
   step 6 does not clear 0.6800 either, the honest reading is that the ratio
   features need reconsidering on their own terms rather than waiting for
   another confounder to clear.

5. Carried over: record 114 lead swap still unfixed; `tools/inspect_ds1.py`
   still writes its JSON into `tools/` and that JSON is now stale relative to
   the current DS1_VAL; stale root `__pycache__/` still present.
