# Last report

**Task:** step 4 - remove duplicate oversampling, replace the scalar focal
alpha with a per-class vector. Also: added the step 3 run to
`docs/ablation.md`.

**Date:** 2026-08-22

---

## What changed

**Oversampling removed.** Section 18 no longer calls
`augment_training_data()`. It assigns the training arrays straight through:

```python
X_tr_aug = X_tr
RR_tr_aug = RR_tr
y_tr_aug = y_tr
```

The `*_aug` names are kept so nothing downstream changes. Both
`augment_training_data()` and `augment_segment()` remain defined but are
**unreachable**, and their section headers are now marked
`[DEPRECATED - unused since step 4]` with a comment saying not to re-enable
them.

**Focal alpha is now per-class.** `categorical_focal_loss(alpha, gamma=2.0)`
takes a length-3 vector, converted once to a `tf.constant` and broadcast over
the class axis of `y_pred` (batch, 3). `alpha` is now a **required** argument
with no default - the old signature defaulted to the scalar `0.50`, and I did
not want that silently reachable again.

Derived at runtime in section 12 from DS1_TRAIN counts only:

```
alpha_c = (1 / count_c) ** FOCAL_BETA,  rescaled so sum(alpha) == NUM_CLASSES
```

**Constants hoisted** into section 3: `FOCAL_BETA = 0.5`,
`FOCAL_GAMMA = 2.0`, `ADAM_LEARNING_RATE = 1e-4`. The learning-rate *value* is
unchanged - only the literal moved.

**metrics.json config** now carries `focal_loss_alpha` (the computed vector),
`focal_loss_beta`, `focal_loss_gamma`, `focal_alpha_class_counts`,
`oversampling: false`, and `adam_learning_rate`, so a run's loss configuration
is fully recoverable from its own metrics file.

The alpha vector and the per-class counts are printed at load time; section 18
prints the training size and distribution.

Nothing else changed: no architecture change, no learning-rate value change,
no RR feature change, no split change. `DS1_VAL` is still
`['207','220','223']`.

---

## Step 3 predictions - scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | `rr_feature_names` in order; `rr_norm_mean` a 5-list near 1.0 | **PASS** - names exact; mean `[1.0009, 1.0027, 0.9925, 1.0124, 1.0461]`, identical to my local computation |
| 2 | printed clip table under 1% at either bound | **PASS by construction, not directly observed** - the console print is not captured in the artefacts; `rr_clip` is `[0.2, 3.0]` in config and the same data locally gives 0.1348% of cells. I am marking this as not independently verified from the run output. |
| 3 | test S recall exceeds 0.1280 | **PASS** - 0.1514 |

**But macro-F1 regressed 0.6800 -> 0.6645**, and that matters more than
prediction 3 passing. What actually happened: the S->N leak nearly halved
(1350 -> 692 of 1836), which is what the ratio features were supposed to do -
but the freed beats went to **V**, not to S (S->V 251 -> 866). V F1 fell
0.8718 -> 0.8004. The features carry real prematurity signal; the model is
spending it on the wrong boundary. That is a class-balancing symptom, which
is exactly what this step addresses.

**Your stated condition is recorded in `docs/ablation.md`:** if step 4 does
not recover macro-F1 above 0.6800, the ratio features get reconsidered.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` comparison - exactly which functions changed:**

```
changed  : ['build_model', 'categorical_focal_loss']
added    : NONE
removed  : NONE
unchanged: 10 of 12
```

Both changes were expected; there are no others. `build_model` changed **only**
in its `model.compile()` call - `model.compile` lives inside `build_model`, so
the loss and optimiser edits necessarily land there. The complete diff of that
function is four lines:

```diff
-            learning_rate=1e-4
+            learning_rate=ADAM_LEARNING_RATE
-            alpha=0.50,
-            gamma=2.0
+            alpha=FOCAL_ALPHA,
+            gamma=FOCAL_GAMMA
```

`Dense(16)` in the RR branch is still present. No layer, unit count, kernel
size, dropout rate or activation changed.

**Is `augment_training_data` actually dead?** Checked by AST call-graph, not
grep - grep is what got this wrong once before. Every `ast.Call` node in the
file was collected and attributed to the function containing it:

```
augment_training_data        called from: NOWHERE
augment_segment              called from: ['augment_training_data']
extract_beats_from_record    called from: ['load_dataset']
load_dataset                 called from: ['<module>']
categorical_focal_loss       called from: ['build_model']
build_model                  called from: ['<module>']
fit_rr_norm                  called from: ['<module>']
apply_rr_norm                called from: ['<module>']
```

`augment_training_data` has zero call sites, and `augment_segment` is
reachable only through it, so both are dead. The three `*_tr_aug` names are
still assigned at module level, so the model still receives data.

**Constants and definitions unchanged:** `PRE_SAMPLES = 90` (1),
`POST_SAMPLES = 144` (1), `ADAM_LEARNING_RATE = 1e-4` present, the literal
`alpha=0.50` gone. The five RR feature definitions were compared as a source
block against the pre-edit file: **identical**, as are `RR_FEATURE_NAMES`,
`RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`, `RR_CLIP_MAX = 3.0`.

**Record literals unchanged:**

```
DS1      IDENTICAL  sha256 9f20e3ac1758a312...
DS2      IDENTICAL  sha256 b8a3e6bbdeeec72a...
DS1_VAL  IDENTICAL  sha256 0d9df3612a6111a1...
```

Same hashes as steps 1 through 3.

**Numerical verification of the alpha computation.** The derivation block was
extracted from the shipped source and executed against a label array built to
the real DS1_TRAIN counts (taken from `tools/ds1_beat_counts.json`, not
hardcoded), then compared against both your reference values and an
independent recomputation of the formula.

```
A) AST call-graph check (not grep - grep missed this once)
   augment_training_data        called from: NOWHERE
   augment_segment              called from: ['augment_training_data']
   extract_beats_from_record    called from: ['load_dataset']
   load_dataset                 called from: ['<module>']
   categorical_focal_loss       called from: ['build_model']
   build_model                  called from: ['<module>']
   fit_rr_norm                  called from: ['<module>']
   apply_rr_norm                called from: ['<module>']

   augment_training_data is dead: True -> PASS
   augment_segment only reachable via the dead function: True -> PASS
   *_tr_aug still assigned at module level: 3 -> PASS

B) alpha derivation, executed from the shipped source
   extracted 22 lines of derivation code
   DS1_TRAIN counts from tools/ds1_beat_counts.json: {'N': 40301, 'S': 670, 'V': 3105}
   FOCAL_CLASS_COUNTS: [40301, 670, 3105]
   FOCAL_ALPHA       : [0.2428, 1.8827, 0.8746]
   sum               : 3.000000
   [PASS] counts are N/S/V in class-index order
   [PASS] length 3
   [PASS] sums to NUM_CLASSES=3
   [PASS] matches reference [0.2428, 1.8827, 0.8746] to 1e-3
   [PASS] matches independent recomputation to 1e-6
   [PASS] rarest class gets the largest weight
   [PASS] most common class gets the smallest weight
   [PASS] ordering follows inverse frequency (N < V < S)

C) what the change actually does to the loss
   old: scalar alpha 0.50 applied to every class
   new: [0.2428, 1.8827, 0.8746]

   class     count  old alpha  new alpha ratio new/old
   N         40301     0.5000     0.2428        0.486x
   S           670     0.5000     1.8827        3.765x
   V          3105     0.5000     0.8746        1.749x

   total loss mass per class (count x copies x alpha):
   class             old            new     change
   N             20150.5         9783.1      0.49x
   S              2345.0         1261.4      0.54x
   V              4657.5         2715.5      0.58x

   effective per-beat emphasis vs N (copies x alpha, normalised):
   S:N   old 7.000x -> new 7.756x   (+10.8%)
   V:N   old 3.000x -> new 3.603x   (+20.1%)

   mean alpha per sample  old 0.5000 -> new 0.3122  (62% of old)
   training samples  old (S x7, V x3): 54306 -> 425 steps/epoch @128
   training samples  new (no oversample): 44076 -> 345 steps/epoch @128

OVERALL: PASS
```

`FOCAL_ALPHA = [0.2428, 1.8827, 0.8746]`, matching your reference to four
decimals and an independent recomputation to 1e-6, summing to exactly 3.

---

## The swap is close to weight-neutral - worth knowing before reading the result

This is the part I would not have predicted, and it changes how step 4's
outcome should be read.

Old scheme: S got its emphasis from **duplication** (7 copies, each weighted
0.50). New scheme: S gets it from **alpha** (1 copy, weighted 1.8827). Those
nearly cancel:

| | S:N emphasis | V:N emphasis |
|---|---|---|
| old (copies x alpha) | 7.000x | 3.000x |
| new (copies x alpha) | 7.756x | 3.603x |
| change | **+10.8%** | **+20.1%** |

So this step is **not** a large rebalancing toward S. It is mostly a
*cleaner* rebalancing - the same relative emphasis, achieved without
fabricating data or duplicating RR vectors.

What does change substantially is the **absolute gradient scale**:

- mean alpha per sample: 0.5000 -> 0.3122 (**62% of before**)
- samples per epoch: 54,306 -> 44,076 (**425 -> 345 steps/epoch**)

Combined, an epoch now delivers roughly half the loss mass it used to. That
behaves like a lower learning rate. If step 4 underperforms, "needs more
epochs" is a live explanation before "the rebalancing failed" - check
`best_epoch` and whether early stopping fired before concluding anything.

---

## Falsifiable predictions for the next Kaggle run

1. **345 steps/epoch** (44,076 samples at batch 128), down from 425.
   `config.oversampling` will be `false` and `train_distribution` will still
   read N=40301 / S=670 / V=3105 - that field is computed before the split,
   so it does not change.
2. **`config.focal_loss_alpha` will be `[0.2428..., 1.8827..., 0.8746...]`**
   summing to 3.0, with `focal_alpha_class_counts = [40301, 670, 3105]`.
3. **S->V misclassification will fall from 866.** The V weight rose only 1.75x
   against N's 0.49x, so V should stop absorbing S beats. This is the
   mechanism the step targets.
4. **macro-F1 will land above step 3's 0.6645.** Whether it clears step 2's
   0.6800 - your stated condition - I genuinely do not know: the relative
   rebalancing is only +10.8% for S, so a large jump would surprise me.

Prediction 3 is the mechanistic test. Prediction 4 is the decision gate.

---

## git diff

```diff
diff --git a/src/train.py b/src/train.py
index 8296cc6..92ad64c 100644
--- a/src/train.py
+++ b/src/train.py
@@ -199,6 +199,17 @@ RR_LOCAL_WINDOW = 10
 RR_CLIP_MIN = 0.2
 RR_CLIP_MAX = 3.0
 
+# Loss settings (step 4).
+# Focal alpha is a PER-CLASS vector, derived at runtime from the DS1_TRAIN
+# counts as (1 / count_c) ** FOCAL_BETA, rescaled to sum to NUM_CLASSES.
+# It was previously the scalar 0.50, which multiplies every class by the
+# same factor and rebalances nothing - it just halved the effective loss.
+FOCAL_BETA = 0.5
+
+FOCAL_GAMMA = 2.0
+
+ADAM_LEARNING_RATE = 1e-4
+
 LEAD_INDEX = 0
 
 BATCH_SIZE = 128
@@ -462,9 +473,13 @@ def load_dataset(
 
 
 # =========================================================
-# 9. ECG AUGMENTATION
+# 9. ECG AUGMENTATION  [DEPRECATED - unused since step 4]
 # =========================================================
 
+# DEPRECATED. Not called anywhere. Kept only so earlier runs in
+# docs/ablation.md remain readable against this file. Do not re-enable:
+# synthetic beats violate hard constraint 2 (no data expansion).
+
 def augment_segment(segment):
 
     x = segment.copy()
@@ -511,9 +526,14 @@ def augment_segment(segment):
 
 
 # =========================================================
-# 10. TARGETED AUGMENTATION
+# 10. TARGETED AUGMENTATION  [DEPRECATED - unused since step 4]
 # =========================================================
 
+# DEPRECATED. augment_training_data() is no longer called; section 18
+# passes the training arrays straight through. It expanded S x7 / V x3 with
+# duplicated RR vectors, which violates hard constraint 2. Class balancing
+# is done in the loss now (FOCAL_ALPHA). Do not re-enable.
+
 def augment_training_data(
     X,
     RR,
@@ -598,9 +618,24 @@ def augment_training_data(
 # =========================================================
 
 def categorical_focal_loss(
-    alpha=0.50,
+    alpha,
     gamma=2.0
 ):
+    """Multiclass focal loss with a PER-CLASS alpha vector.
+
+    alpha must be a sequence of length NUM_CLASSES. y_pred is
+    (batch, n_classes), so a (n_classes,) alpha broadcasts over the class
+    axis and each class gets its own weight.
+
+    alpha is deliberately required, not defaulted: the old signature
+    defaulted to the scalar 0.50, which applied the same factor to all
+    three classes and therefore performed no rebalancing at all.
+    """
+
+    alpha = tf.constant(
+        alpha,
+        dtype=tf.float32
+    )
 
     def loss(y_true, y_pred):
 
@@ -708,6 +743,38 @@ print("\nOriginal Test Distribution:")
 print(Counter(y_test))
 
 
+# --- per-class focal alpha, derived from DS1_TRAIN counts only ----------
+# Uses training counts only - no validation or test information.
+
+_train_counts = Counter(y_train)
+
+FOCAL_CLASS_COUNTS = [
+    int(_train_counts[INT_TO_LABEL[_i]])
+    for _i in range(NUM_CLASSES)
+]
+
+_alpha_raw = np.array(
+    [
+        (1.0 / _count) ** FOCAL_BETA if _count > 0 else 0.0
+        for _count in FOCAL_CLASS_COUNTS
+    ],
+    dtype=np.float64
+)
+
+FOCAL_ALPHA = (
+    _alpha_raw * (NUM_CLASSES / _alpha_raw.sum())
+).astype(np.float32)
+
+print(f"\nFocal alpha (per class, beta={FOCAL_BETA}, "
+      f"rescaled to sum {NUM_CLASSES}):")
+
+for _i in range(NUM_CLASSES):
+    print(f"  {INT_TO_LABEL[_i]}: count {FOCAL_CLASS_COUNTS[_i]:>6}  "
+          f"alpha {FOCAL_ALPHA[_i]:.4f}")
+
+print(f"  vector: {FOCAL_ALPHA.tolist()}  (sum {FOCAL_ALPHA.sum():.4f})")
+
+
 # =========================================================
 # 13. CLASS DISTRIBUTION PLOT
 # =========================================================
@@ -839,14 +906,27 @@ print(f"\nTrain beats: {len(y_tr)}   Validation beats: {len(y_val)}")
 
 
 # =========================================================
-# 18. AUGMENT TRAINING DATA
+# 18. AUGMENT TRAINING DATA (REMOVED - step 4)
 # =========================================================
 
-X_tr_aug, RR_tr_aug, y_tr_aug = augment_training_data(
-    X_tr,
-    RR_tr,
-    y_tr
-)
+# Duplicate oversampling is gone. It expanded S x7 and V x3 while copying
+# the RR feature vector UNCHANGED across every copy, so six of every seven
+# S training examples carried identical rhythm context - teaching the model
+# to memorise specific training rhythms instead of the relative-timing
+# rule the step 3 features encode. The np.roll time shift also moved the
+# R-peak off its aligned position. It violated hard constraint 2.
+#
+# Class balancing now happens in the loss instead, via the per-class
+# FOCAL_ALPHA vector derived in section 12.
+#
+# The *_aug names are kept so nothing downstream needs to change.
+
+X_tr_aug = X_tr
+RR_tr_aug = RR_tr
+y_tr_aug = y_tr
+
+print(f"\nTraining samples (no oversampling): {len(y_tr_aug)}")
+print(f"Training distribution: {Counter(y_tr_aug)}")
 
 
 # =========================================================
@@ -1007,12 +1087,12 @@ def build_model(ecg_shape, rr_shape):
     model.compile(
 
         optimizer=tf.keras.optimizers.Adam(
-            learning_rate=1e-4
+            learning_rate=ADAM_LEARNING_RATE
         ),
 
         loss=categorical_focal_loss(
-            alpha=0.50,
-            gamma=2.0
+            alpha=FOCAL_ALPHA,
+            gamma=FOCAL_GAMMA
         ),
 
         metrics=['accuracy']
@@ -1573,11 +1653,17 @@ metrics = {
 
         "LEAD_INDEX": LEAD_INDEX,
 
-        "focal_loss_alpha": 0.50,
+        "focal_loss_alpha": FOCAL_ALPHA.tolist(),
+
+        "focal_loss_beta": FOCAL_BETA,
+
+        "focal_loss_gamma": FOCAL_GAMMA,
+
+        "focal_alpha_class_counts": FOCAL_CLASS_COUNTS,
 
-        "focal_loss_gamma": 2.0,
+        "oversampling": False,
 
-        "adam_learning_rate": 1e-4,
+        "adam_learning_rate": ADAM_LEARNING_RATE,
 
         "DS1": DS1,
 
```

---

## Commit

```
3b67016  step 4: remove oversampling, per-class focal alpha
```

Pushed to `origin/main`. This report lands in a small follow-up commit.

---

## Problems

1. **Hard constraint 2 is now satisfied, and CLAUDE.md is out of date.** The
   "Known facts" entry still says augmentation is active and will be removed
   "in a later step". That later step is this one. I have not edited CLAUDE.md
   because you did not ask and it is not part of the commit's stated scope -
   but it should be updated before the next task, or the next reader will act
   on a stale fact. Say the word and I will fix it.

2. **The two augmentation functions are now genuinely dead code**, kept
   deliberately at your instruction. Verified dead by call-graph, and marked
   DEPRECATED in two places. Note this is the same shape of claim I got wrong
   before, which is why it is AST-verified rather than grepped this time.

3. **Effective gradient scale roughly halved** - see the section above. This
   is a real confound in comparing step 4 against step 3, and it was not
   flagged in the task. It is a consequence of doing the swap correctly, not
   an error, but it means step 4 is not a perfectly clean single-variable
   comparison either.

4. **Record 207 is still a severe validation outlier** - step 3 gave accuracy
   0.1733, N recall 0.0824, S recall 0.0000, essentially unchanged from step
   2's 0.1647 / 0.0700 / 0.0000. Patient-relative RR did not help it. 207 has
   sustained ventricular flutter, so its median RR is itself computed over
   abnormal rhythm - the ratio denominator is unreliable for exactly this
   record. If step 4 does not move it, the validation set composition is worth
   a deliberate decision (changing `DS1_VAL` would break comparability with
   steps 1-4, so it is not a change to make casually).

5. Carried over: record 114 lead swap still unfixed; `tools/inspect_ds1.py`
   still writes its JSON into `tools/`; stale root `__pycache__/` still
   present.
