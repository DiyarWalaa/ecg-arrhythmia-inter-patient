# Last report

**Task:** E4 - revert to E2's wavelet scales, reduce model capacity ~10x.

**Date:** 2026-08-25

---

## E3 predictions - scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | 425 steps/epoch, 239,171 parameters, both unchanged | **PASS** |
| 2 | centre frequencies `[3.00, 4.59, ..., 90.00]`, widths 27.0095 -> 0.9003 | **PASS** - exact |
| 3 | **S-F1 will exceed E2's 0.2686** | **FAIL** - 0.1498, it nearly halved |
| 4 | V-F1 will drop somewhat from 0.9111 | **PASS**, but understated - 0.7321, a 0.18 drop, not "a few points" |

**Prediction 3 was the experiment and it failed cleanly.** Four channels at or
below 10.74 Hz did not help S; S-F1 went 0.2686 -> 0.1498. The
missing-P-wave-frequency hypothesis is refuted, which is a useful negative
result: whatever limits S here, it is not the absence of sub-10 Hz coverage.
E3 lost on every class (N 0.9787 -> 0.9635, S 0.2686 -> 0.1498,
V 0.9111 -> 0.7321) for macro-F1 0.6151 against 0.7178.

I also under-called prediction 4. I said "I would be surprised by a collapse,
but a few points of V-F1 is the price". The actual cost was 18 points. Trading
five high-frequency channels for four low-frequency ones hurt the wide-QRS
morphology far more than I allowed for.

**E3's real contribution is diagnostic.** Its history shows the overfitting
plainly:

```
train accuracy : 0.8011 0.9096 0.9328 0.9439 0.9536 ... 0.9717   (11 epochs)
val macro-F1   : 0.6629 0.5910 0.5763 0.5633 0.5761 ... 0.5406
```

Validation peaks at **epoch 1** and declines monotonically while training
accuracy climbs 17 points. `best_epoch` is 1, exactly as in E2. That is what
E4 addresses.

---

## Part 1 - scales reverted

`WAVELET_TARGET_FREQS_HZ` is back to E2's exact line, byte-identical:

```python
WAVELET_TARGET_FREQS_HZ = [float(10 * k) for k in range(1, 10)]
```

No E3 leftovers - `WAVELET_F_MIN_HZ`, `WAVELET_F_MAX_HZ` and
`N_WAVELET_SCALES_TARGET` are all gone (0 occurrences).

**Widths match E2's run to 4 decimals**, all nine:

| ch | target Hz | width (E4) | width (E2) |
|---|---|---|---|
| 0 | 10 | 8.1028 | 8.1028 |
| 1 | 20 | 4.0514 | 4.0514 |
| 2 | 30 | 2.7009 | 2.7009 |
| 3 | 40 | 2.0257 | 2.0257 |
| 4 | 50 | 1.6206 | 1.6206 |
| 5 | 60 | 1.3505 | 1.3505 |
| 6 | 70 | 1.1575 | 1.1575 |
| 7 | 80 | 1.0129 | 1.0129 |
| 8 | 90 | 0.9003 | 0.9003 |

Independently confirmed on record 209: per-channel std is back to E2's
`[2.6169, 2.2094, 1.4953, 0.9834, 0.6684, 0.4742, 0.3495, 0.2659, 0.2086]`
and the max off-diagonal correlation is back to **0.9968**.

---

## Part 2 - capacity

Filters 32/32/64/64/128/128 -> 8/8/16/16/32/32, BiLSTM 64 -> 16, head Dense
128/64 -> 32/16. The RR branch Dense stays 16. Kernel sizes (5, 5, 5, 5, 3, 3),
pool sizes, dropout rates (0.2, 0.25, 0.3, 0.4, 0.2, 0.5, 0.4), activations and
layer order are all untouched.

```
C) E4 layer table (8/16/32, LSTM 16, head 32/16)
   layer                      output             params
   ----------------------------------------------------
   Input ecg                  (234, 9)                -
   Conv1D k5                  (234, 8)              368
   BatchNorm                  (234, 8)               32
   Conv1D k5                  (234, 8)              328
   BatchNorm                  (234, 8)               32
   MaxPool /2                 (117, 8)                -
   Dropout 0.2                (117, 8)                -
   Conv1D k5                  (117, 16)             656
   BatchNorm                  (117, 16)              64
   Conv1D k5                  (117, 16)           1,296
   BatchNorm                  (117, 16)              64
   MaxPool /2                 (58, 16)                -
   Dropout 0.25               (58, 16)                -
   Conv1D k3                  (58, 32)            1,568
   BatchNorm                  (58, 32)              128
   Conv1D k3                  (58, 32)            3,104
   BatchNorm                  (58, 32)              128
   MaxPool /2                 (29, 32)                -
   Dropout 0.3                (29, 32)                -
   Bidirectional LSTM 16      (32,)               6,272
   Dropout 0.4                (32,)                   -
   Input rr                   (5,)                    -
   Dense 16 (rr)              (16,)                  96
   Dropout 0.2 (rr)           (16,)                   -
   Concatenate                (48,)                   -
   Dense 32                   (32,)               1,568
   Dropout 0.5                (32,)                   -
   Dense 16                   (16,)                 528
   Dropout 0.4                (16,)                   -
   Dense 3 softmax            (3,)                   51
   ----------------------------------------------------
   TOTAL                                         16,283

   first Conv1D: 5 * 9 * 8 + 8 = 368  (was 5*9*32+32 = 1472)
   reduction: 239171 -> 16283  = 14.69x
   Zahid et al. reference: 23,619 parameters
   E4 is 0.69x Zahid's size
```

**Parameter counts are computed analytically, and the method is validated
first**: applying the same formulas to the pre-E4 architecture reproduces
**239,171 exactly**, the number Keras reported. Only then is 16,283 trusted.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` against E3 HEAD:**

```
changed : ['build_model']
added   : NONE
removed : NONE
```

**`build_model` is the only function that changed, and its complete diff is
nine integers:**

```diff
-        32,     +        8,      (block 1 conv 1)
-        32,     +        8,      (block 1 conv 2)
-        64,     +       16,      (block 2 conv 1)
-        64,     +       16,      (block 2 conv 2)
-       128,     +       32,      (block 3 conv 1)
-       128,     +       32,      (block 3 conv 2)
-            64, +           16,  (BiLSTM units)
-       128,     +       32,      (head dense 1)
-        64,     +       16,      (head dense 2)
```

Nothing else in the file changed except the wavelet constant. Compared against
**E2 (post-fix)**, `build_model` is the only differing function - E3 changed no
functions and E4 reverted its constant, so the pipeline outside the model is
now identical to the configuration that produced 0.7178.

**Byte-identical to E3**: `tune_decision_weights`, `macro_f1_from_weights`,
`cwt_ricker`, `ricker_width_for_frequency`, `categorical_focal_loss`.

**Constants unchanged** (one occurrence each): `PRE_SAMPLES = 90`,
`POST_SAMPLES = 144`, `FOCAL_ALPHA = 0.50`, `FOCAL_GAMMA = 2.0`,
`ADAM_LEARNING_RATE = 1e-4`, `RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`,
`RR_CLIP_MAX = 3.0`, `SAMPLING_RATE_HZ = 360.0`. Augmentation multipliers
`[0, 6, 2, 0]` identical. `rr = [...]` block and `RR_FEATURE_NAMES`
byte-identical. `DS1` `9f20e3ac1758a312...`, `DS2` `b8a3e6bbdeeec72a...`,
`DS1_VAL` `0d9df3612a6111a1...` = `['207','220','223']`.

---

## The reduction is 14.7x, not 8-10x

You asked for "roughly 8-10x" and then gave exact filter numbers. **The exact
numbers produce 239,171 -> 16,283, a 14.69x reduction.** I followed the
explicit numbers, since they are unambiguous, but the two parts of the
instruction do not agree and you should know which one I honoured.

This also puts E4 at **0.69x Zahid et al.'s 23,619 parameters** - smaller than
the reference architecture, not comparable to it. If the intent was to land
near Zahid's scale, the head is where to give capacity back: going to
`Dense 64/32` instead of `32/16` would add roughly 3,400 parameters, and
BiLSTM 24 instead of 16 would add roughly 4,700. Reaching ~24,000 is a
one-line change if you want it.

---

## Falsifiable predictions

1. **Total parameters: 16,283.** First Conv1D `5*9*8 + 8 = 368`; BiLSTM
   `2 * 4 * 16 * (32+16+1) = 6,272`; head `1,568 + 528 + 51`. If Keras reports
   a different total, my layer table is wrong and every parameter claim in
   this report should be rechecked.
2. **425 steps/epoch**, `train_distribution` N=40301 / S=670 / V=3105, and
   `wavelet_centre_frequencies` exactly `[10, 20, ..., 90]`.
3. **`best_epoch` will be greater than 1.** This is the real bet. E2 and E3
   both peaked at epoch 1 because a 239k-parameter network fits 670 unique S
   beats within one pass. At 16k parameters it should take several epochs to
   reach the same point. **If `best_epoch` is still 1, capacity was not the
   cause of the early peak** and the next suspect is the learning rate or the
   augmentation duplicating S beats sevenfold.
4. **Train accuracy at the final epoch will be below E3's 0.9717.** A model
   with 14.7x fewer parameters should not memorise the training set as
   completely.

I am not predicting macro-F1 beats E2's 0.7178. Less capacity may simply mean
a worse model; the point of the run is to find out whether the epoch-1 peak is
a capacity artefact.

---

## Commit

```
cb9224e  E4: revert to E2 wavelet scales, reduce capacity ~10x
```

Pushed to `origin/main`. `docs/ablation.md` now carries 11 runs including E3.

---

## Problems

1. **The 14.7x / "8-10x" discrepancy above.** Flagged rather than silently
   resolved.

2. **E4 changes two things at once** - the scale revert and the capacity cut.
   Against E3 that is two variables; against **E2** it is one, because the
   scales are now byte-identical to E2's and `build_model` is the only
   differing function. **E2 is the correct comparison for E4**, not E3, and
   the ablation table's ordering should not be read as implying otherwise.

3. **Capacity may not be the cause of the epoch-1 peak.** The competing
   explanation is that augmentation creates 4,020 near-duplicate S beats from
   670 originals, so one epoch already shows the network each real S beat
   seven times. Reducing parameters does not change that. If prediction 3
   fails, that is the thing to test next.

4. **Dropout rates were kept at values tuned for a 239k network.** Dropout 0.5
   on a 32-unit dense layer removes 16 units; the same rate on the old
   128-unit layer removed 64 of 128. The relative regularisation is unchanged
   but the absolute capacity removed is much smaller, and heavy dropout on a
   small layer can under-fit. Deliberate - you said keep them identical - but
   worth revisiting if E4 under-performs.

5. Carried over: hard constraint 2 still violated (augmentation on); threshold
   tuning has failed to transfer twice; record 114 lead swap unfixed; record
   207 still a validation outlier; `tools/inspect_ds1.py` JSON stale; stale
   root `__pycache__/`.
