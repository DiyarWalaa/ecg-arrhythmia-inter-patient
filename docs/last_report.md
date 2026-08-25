# Last report

**Task:** E6 - revert the E5 skip; balanced batch sampling with plain
cross-entropy.

**Date:** 2026-08-25

---

## E5 predictions - scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | total parameters 239,186 | **still not verifiable** - E5's `metrics.json` has no parameter count. Fixed this step. |
| 2 | 425 steps/epoch, frequencies `[10, ..., 90]` | **PASS** |
| 3 | **S-F1 will exceed E2's 0.2686** | **FAIL** - 0.2321 |
| 4 | N-F1 and V-F1 move very little | **PASS** - N 0.9787 -> 0.9734, V 0.9111 -> 0.8982 |

**Prediction 3 was the experiment and it failed.** Fifteen parameters giving
the output layer an un-mediated view of the five RR ratios changed nothing
useful. S recall went 0.1972 -> 0.1574 and S precision 0.4214 -> 0.4419: the
model became slightly *more conservative*, not more sensitive. `best_epoch`
was 1 again.

So the RR signal is **not** attenuated by network depth. I flagged in E5's
report that a 5-wide skip against a 64-wide vector was a weak intervention and
that a null result should not be over-read - that caveat stands, but the
direction of the change (recall down, precision up) argues against attenuation
rather than merely failing to confirm it.

**Two hypotheses are now eliminated.** E4: S is not limited by overfitting -
cutting capacity 14.7x moved `best_epoch` off 1 and destroyed S completely
(11 predictions in 1,836). E5: S is not limited by RR attenuation. What
remains is that the model is never *asked* to predict S: N outnumbers S 60:1
in every minibatch it has ever seen.

---

## What changed

**Part 1 - the E5 skip is gone.** `build_model`'s layer graph is byte-identical
to E2's again.

**Part 2 - balanced batch sampling:**

- Section 18 no longer calls `augment_training_data()`. `X_tr_aug` etc. pass
  straight through; the function has **zero call sites** and sections 9 and 10
  are marked `[DEPRECATED]`.
- New section **19B** builds three per-class `tf.data.Dataset`s, each
  `.shuffle(n_class, seed=SEED).repeat()`, combined by
  `tf.data.Dataset.sample_from_datasets(weights=[1/3, 1/3, 1/3], seed=SEED)`,
  then `.batch(BATCH_SIZE).prefetch(AUTOTUNE)`.
- `STEPS_PER_EPOCH = ceil(44076 / 128) = 345`, passed explicitly to
  `model.fit`, which now receives `train_dataset` positionally and **no**
  `batch_size` (illegal alongside a Dataset).
- Loss is plain `'categorical_crossentropy'`. `categorical_focal_loss` has
  zero call sites and its section is marked `[DEPRECATED]`.
- `metrics.json` config gains `sampler`, `sampling_weights`,
  `steps_per_epoch`, `loss`, `focal_loss_used` and **`total_parameters`**.

**No synthetic data is created.** S beats repeat from the 670 real ones inside
an infinite stream, so hard constraint 2 holds - the first time it has since
E0.

---

## One instruction I could not satisfy literally

You asked that `build_model` "must return to being byte-identical to E2's",
and separately that the focal loss be replaced with cross-entropy.
**`model.compile()` lives inside `build_model`**, so both cannot hold.

The layer graph *is* byte-identical - I verified it by slicing everything
before `model.compile(` from both versions and comparing:

```
layer graph (everything before model.compile) byte-identical to E2: True
```

The entire remaining diff inside `build_model` is the loss argument:

```diff
-        loss=categorical_focal_loss(
-            alpha=FOCAL_ALPHA,
-            gamma=FOCAL_GAMMA
-        ),
+        loss='categorical_crossentropy',
```

Parameter count is unaffected by the loss: **239,171**, confirmed by the
same analytic method that reproduces E2's and E4's known totals.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` against E5 HEAD:** `changed: ['build_model']`, nothing added or
removed. Sections 18, 19B and the fit call are module-level, so they do not
appear in a function diff.

**`augment_training_data` call graph** (AST, not grep): **called from
NOWHERE**. `categorical_focal_loss`: **zero call sites**. `compile` uses
`'categorical_crossentropy'`.

**Sampler structure, read from the AST:** one `sample_from_datasets` call with
`weights=SAMPLING_WEIGHTS, seed=SEED`; `SAMPLING_WEIGHTS` evaluates to
`[0.333333, 0.333333, 0.333333]` summing to 1.0; three `.shuffle`, three
`.repeat`, one `.batch(BATCH_SIZE)`; `model.fit(train_dataset, ...,
steps_per_epoch=..., ...)` with **no** `batch_size` kwarg.

**Empirical sampler check - and its limitation.** TensorFlow is not installed
locally, so `tf.data` cannot be executed here. I simulated the documented
semantics of `sample_from_datasets` in numpy - a categorical draw over three
infinite shuffled streams with weights 1/3 - and drew 20 batches of 128:

```
   20 batches of 128, per-class counts (target 42.7 each):
    batch      N      S      V   max deviation
        1     43     43     42        0.7
        2     45     40     43        2.7
        3     45     42     41        2.3
        4     44     42     42        1.3
        5     37     43     48        5.7
        6     52     38     38        9.3
        7     41     41     46        3.3
        8     45     42     41        2.3
        9     37     43     48        5.7
       10     52     40     36        9.3
       11     39     49     40        6.3
       12     39     47     42        4.3
       13     46     37     45        5.7
       14     39     35     54       11.3
       15     43     47     38        4.7
       16     41     47     40        4.3
       17     48     42     38        5.3
       18     43     41     44        1.7
       19     49     38     41        6.3
       20     37     55     36       12.3

   mean per class over 20 batches: N 43.25  S 42.60  V 42.15
   overall share: N 0.3379  S 0.3328  V 0.3293
   multinomial std per class: 5.33  (observed 4.45)
   worst single-batch deviation from 42.7: 12 (2.3 sd)

C) what balanced sampling does to how often each beat is seen
```

Mean per class 43.25 / 42.60 / 42.15 against a target of 42.67; overall shares
0.3379 / 0.3328 / 0.3293; no class ever absent from a batch. The spread is
ordinary multinomial noise (theoretical sd 5.33, worst single-batch deviation
2.3 sd).

**This validates the sampling scheme, not `tf.data`'s implementation of it.**
The Kaggle run is the first execution of the real pipeline, and the printed
per-class counts should be compared against the table above.

**Constants and literals unchanged**: `PRE_SAMPLES = 90`, `POST_SAMPLES = 144`,
`ADAM_LEARNING_RATE = 1e-4`, `RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`,
`RR_CLIP_MAX = 3.0`, `SAMPLING_RATE_HZ = 360.0`, `BATCH_SIZE = 128` - one
occurrence each. `rr = [...]` and `RR_FEATURE_NAMES` byte-identical.
`WAVELET_TARGET_FREQS_HZ` identical to E2's, widths 8.1028 ... 0.9003.
`DS1` `9f20e3ac1758a312...`, `DS2` `b8a3e6bbdeeec72a...`, `DS1_VAL`
`0d9df3612a6111a1...` = `['207','220','223']`.

---

## What balanced sampling actually does to exposure

This is the part worth pausing on before reading the result.

```
C) what balanced sampling does to how often each beat is seen
   samples per epoch: 345 steps x 128 = 44160
   cls    beats  draws/epoch     times seen
   N      40301        14720           0.37
   S        670        14720          21.97
   V       3105        14720           4.74

   For comparison, the augmentation E6 removes gave each S beat
   7 copies per epoch. The sampler shows each S beat ~22 times.
   And only 37% of the N pool is seen in a given epoch.

D) results
```

**Each S beat is now seen about 22 times per epoch** - three times more
repetition than the sevenfold augmentation E6 just removed. And **only 37% of
the N pool is seen in any given epoch**: the model now sees roughly 14,720 N
beats per epoch instead of 40,301.

Both follow directly from equal weights plus a fixed 345-step epoch, and both
are intended. But "no synthetic data" is not the same as "no repetition", and
the constraint-2 argument for this change should not be read as an argument
that repetition has gone away - it has increased for S.

---

## Falsifiable predictions

1. **345 steps/epoch**, `config.sampler == "balanced_batch"`,
   `sampling_weights == [1/3, 1/3, 1/3]`, `loss == "categorical_crossentropy"`,
   `focal_loss_used == false`, `oversampling == false`, and
   **`total_parameters == 239171`** - the first run where that last claim is
   checkable from the artefact.
2. **S recall will rise sharply, well past E2's 0.1972** - I expect above 0.60.
   This is the mechanism De Waele et al. attribute their 0.9116 to, and it is
   the first time our model will see S in a third of its training signal.
3. **S precision will fall well below E2's 0.4214** - probably under 0.20.
   Their S precision is 0.3327 *with* RR fusion; a balanced sampler trades
   precision for recall, and our N pool is 60x larger than theirs relative to S.
   **Whether S-F1 beats 0.2686 depends entirely on which moves further**, and I
   am not confident in the direction.
4. **N-F1 will drop from 0.9787**, because N is now a third of the training
   signal rather than 91% of it. If N-F1 holds above 0.95 the sampler is
   under-firing; if it falls below 0.90 it is over-firing.

Prediction 2 is the mechanism test. Prediction 3 is the cost. Prediction 1 is
the integrity check.

---

## Commit

```
5b3b203  E6: balanced batch sampling with plain cross-entropy
```

Pushed to `origin/main`. `docs/ablation.md` now carries 13 runs including E5.

---

## Problems

1. **The sampler is untested against `tf.data` itself.** No TensorFlow
   locally means the first real execution is on Kaggle. The AST check
   guarantees the call shape; the numpy simulation guarantees the scheme is
   sound; neither guarantees `sample_from_datasets` behaves as documented in
   TF 2.20. If the printed per-class counts are not near 42.7, that is where
   to look first.

2. **S repetition went up, not down** - ~22 times per epoch against
   augmentation's 7. Constraint 2 is satisfied because nothing synthetic is
   created, but if E6 overfits S, the repetition rate is the reason, and the
   fix would be fewer steps per epoch rather than a different sampler.

3. **Only 37% of N is seen per epoch.** With `EPOCHS = 40` and early stopping
   typically firing well before that, some N beats may never be sampled in a
   short run. That is a real change to what the model is trained on, not just
   how often.

4. **Three things changed at once**: the E5 skip removed, augmentation
   removed, loss changed, sampler added. Against E2 that is three variables
   (sampler, loss, no augmentation - the skip is absent from both). If E6
   moves S sharply, attributing the gain between the sampler and the loss will
   require a follow-up run.

5. **`X_raw_train` is now unused.** `load_dataset` still computes and returns
   the z-scored raw waveforms for every split, but nothing consumes them since
   augmentation stopped. Harmless (~46 MB for DS2) and left in place because
   removing it would touch `load_dataset`, which was out of scope.

6. Carried over: threshold tuning has failed to transfer twice; record 114
   lead swap unfixed; record 207 still a validation outlier;
   `tools/inspect_ds1.py` JSON stale; stale root `__pycache__/`.
