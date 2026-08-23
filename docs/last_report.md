# Last report

**Task:** E0 re-anchor - restore the step 3 training configuration, keep the
step 5 validation set.

**Date:** 2026-08-23

---

## What changed

Reverted from HEAD (step 6), with the prior code **spliced verbatim out of
commit `1a2509c`** rather than retyped. The patch script asserts each source
region is unique before substituting, so the reverted blocks are byte-exact
step 3 code.

**1. Augmentation re-enabled.** Section 18 is step 3's block again:

```python
X_tr_aug, RR_tr_aug, y_tr_aug = augment_training_data(
    X_tr,
    RR_tr,
    y_tr
)
```

**2. Scalar alpha, no sweep.** `FOCAL_ALPHA = 0.50` in section 3. Removed
`BETA_GRID`, `FOCAL_BETA`, `compute_focal_alpha`, `make_callbacks`,
`reset_seeds`, and the whole sweep loop - sections 20/21/22 are step 3's
single build / single callback set / single `model.fit`. `FOCAL_GAMMA = 2.0`
and `ADAM_LEARNING_RATE = 1e-4` are kept as named constants.

**3. Everything else stays at HEAD:** the 5 RR ratio features,
`DS1_VAL = ['106','118','207','220','223']`, macro-F1 model selection, the
`ValidationMetrics` callback, the per-record breakdown, train-only RR
normalization.

**Section headers 9 and 10** no longer say `[DEPRECATED]`. They now say the
functions are live again, and record the two known defects that come back with
them: the RR feature vector is copied unchanged across every duplicate, and
the `np.roll` shift moves the R-peak off its aligned position.

**metrics.json config** reverts to `focal_loss_alpha: 0.50` and drops
`beta_grid` / `focal_loss_beta` / `focal_alpha_class_counts` / `beta_sweep` /
`selected_beta` / `selection_criterion`. `oversampling` is now `true`, and I
added `oversampling_multipliers: {"N": 1, "S": 7, "V": 3}` so the expansion
factors are recoverable from the run's own metrics rather than only from the
code - flagged below as an addition beyond a pure revert.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` against step 6 HEAD:**

```
changed  : ['categorical_focal_loss']
added    : NONE
removed  : ['compute_focal_alpha', 'make_callbacks', 'reset_seeds']
```

**`ast.dump` against step 3 (`1a2509c`) - the revert target:**

```
differ from step 3: ['build_model', 'categorical_focal_loss']
only in HEAD      : NONE
only in step 3    : NONE
```

Two functions differ from step 3, and **both differences are the step 4
constant-hoisting you told me to keep, not behaviour**:

```diff
build_model (6 diff lines, all in model.compile):
-            learning_rate=1e-4          +            learning_rate=ADAM_LEARNING_RATE
-            alpha=0.50,                 +            alpha=FOCAL_ALPHA,
-            gamma=2.0                   +            gamma=FOCAL_GAMMA

categorical_focal_loss:
-    alpha=0.50,                         +    alpha,          (required, no default)
                                         +    alpha = tf.constant(alpha, dtype=tf.float32)
                                         +    (docstring rewritten)
```

`ADAM_LEARNING_RATE` is 1e-4, `FOCAL_ALPHA` is 0.50, `FOCAL_GAMMA` is 2.0 -
the exact values step 3 had inline. `tf.constant(0.50)` broadcasts identically
to step 3's bare Python float. **The training configuration is numerically
identical to step 3.** I kept `alpha` required rather than restoring the
`alpha=0.50` default, so a scalar can never be reached by accident again; the
call site now states it explicitly.

**`augment_training_data` IS called** - AST call-graph, not grep:

```
augment_training_data    called from: ['<module>']
augment_segment          called from: ['augment_training_data']
build_model              called from: ['<module>']
```

**Scalar alpha, no BETA:**

```
FOCAL_ALPHA = 0.50                      <- scalar
BETA_GRID           occurrences: 0
FOCAL_BETA          occurrences: 0
compute_focal_alpha occurrences: 0
sweep_beta          occurrences: 0
BETA_SWEEP          occurrences: 0
SELECTED_BETA       occurrences: 0
make_callbacks      occurrences: 0
reset_seeds         occurrences: 0
```

**One model, one fit:** `build_model` called once (line 1105), `model.fit`
called once (line 1240). No module-level loop over any BETA grid remains.

**Constants unchanged** (one occurrence each, before and after):
`PRE_SAMPLES = 90`, `POST_SAMPLES = 144`, `ADAM_LEARNING_RATE = 1e-4`,
`FOCAL_GAMMA = 2.0`, `RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`,
`RR_CLIP_MAX = 3.0`, `multiplier = 6`.

**The 5 RR features** - the `rr = [...]` source block is byte-identical to
both HEAD and step 3; `RR_FEATURE_NAMES` unchanged.

**Record literals:** `DS1` `9f20e3ac1758a312...`, `DS2` `b8a3e6bbdeeec72a...`,
`DS1_VAL` `f6759845186d6324...` - all unchanged, and `DS1_VAL` is still
`['106', '118', '207', '220', '223']`.

---

## Falsifiable prediction: steps/epoch

E0 is the first run to combine augmentation with the 17-record training pool,
so it should produce a step count no previous run has had:

```
DS1_TRAIN (17 records): N=36631, S=574, V=2569   raw total 39,774
post-augmentation: 36631*1 + 574*7 + 2569*3 = 48,356
steps/epoch @ batch 128 = 378
```

| run | training pool | augmentation | samples | steps/epoch |
|---|---|---|---|---|
| step 3 | 19 records | on | 54,306 | 425 |
| step 5 | 17 records | off | 39,774 | 311 |
| **E0** | **17 records** | **on** | **48,356** | **378** |

**Prediction: 378 steps/epoch.** `config.oversampling` will be `true`,
`config.focal_loss_alpha` will be the scalar `0.5`, and there will be no
`beta_sweep` key. `train_distribution` will still read N=36631 / S=574 /
V=2569 - that field is computed before augmentation, so it does not change.

**If steps/epoch is not 378, the revert did not do what I think it did.**

I am deliberately **not** predicting a macro-F1 value. E0 exists to establish
a baseline for a configuration that has never been run, not to beat one.

---

## Commit

```
042597b  E0: re-anchor - step 3 training config with 5-record validation
```

Pushed to `origin/main`. This report lands in a small follow-up commit.

---

## Problems

1. **Hard constraint 2 is violated again, deliberately.** Augmentation is back
   on: S x7, V x3, with the RR feature vector copied unchanged across every
   duplicate. That last part is worse now than it was at step 3, because the
   RR vector is no longer two raw intervals but five patient-relative ratios -
   the features this project spent step 3 making meaningful are the ones being
   duplicated verbatim. CLAUDE.md's "Known facts" entry currently says
   augmentation was removed at step 4 and is dead code; **that is now stale
   again**. I have not edited it, because you did not ask and E0's scope was
   the revert - but it should be corrected before the next task or the next
   reader will act on a wrong fact. Say the word.

2. **`docs/ablation.md` has no E0 row and cannot have one yet** - no
   `results/E0.../metrics.json` exists. `tools/make_ablation.py` will need an
   entry appended when the run comes back; its gate will reject a folder that
   does not identify itself correctly.

3. **One addition beyond a pure revert:** `oversampling_multipliers` in the
   config block. Step 3 recorded nothing about augmentation at all, which is
   precisely why detecting it later required counting steps/epoch. Recording
   the multipliers costs nothing and makes the run self-describing. Flagging
   it as a deviation from the letter of "revert".

4. **This is a two-variable change relative to both neighbours.** Against step
   3 it changes the validation set; against step 5 it changes the training
   configuration. That is the intent - the combination has never been run -
   but E0's number is a new anchor, not a delta against anything in the
   existing table.

5. Carried over: record 114 lead swap still unfixed; record 207 still a
   validation outlier; `tools/inspect_ds1.py` still writes its JSON into
   `tools/` and that JSON is stale relative to the current `DS1_VAL`; stale
   root `__pycache__/` still present.
