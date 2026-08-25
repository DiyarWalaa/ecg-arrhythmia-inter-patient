# Last report

**Task:** E5 - add a direct skip connection from the raw RR features to the
output layer.

**Date:** 2026-08-25

---

## E4 predictions - scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | total parameters exactly 16,283 | **not verifiable from the artefacts** - `metrics.json` records no parameter count |
| 2 | 425 steps/epoch, centre frequencies `[10, ..., 90]` | **PASS** - frequencies exact |
| 3 | **`best_epoch` will be greater than 1** | **PASS** - 4 |
| 4 | final train accuracy below E3's 0.9717 | **PASS** - 0.9003 |

**Predictions 3 and 4 both landed, and the model still got worse.** That
combination is the useful result. Reducing capacity 14.7x did fix the
epoch-1 peak - selection moved to epoch 4 and the network stopped memorising
(train accuracy 0.9717 -> 0.9003). So the early peak *was* a capacity
artefact.

**But S vanished entirely.** E4 predicts S **11 times in total** against 1,836
true S beats: S-F1 0.0000, S recall 0.0000. macro-F1 0.5602, second-worst of
any run. 16,283 parameters cannot represent the S class at all.

Taken together: overfitting was never the binding constraint on S. Shrinking
the network removed the symptom I was chasing and made the actual problem
worse. E5 goes back to E2's capacity and attacks the RR attenuation instead.

**Prediction 1 is unverifiable and that is a gap I should close.** No run
records its parameter count, so I have been predicting a number that no
artefact can confirm or refute. Worth adding `model_parameters` to
`metrics.json` in a future step - flagged rather than done, since E5's brief
was one change to `build_model`.

---

## What changed

**Capacity restored to E2's sizes** - conv 32/32/64/64/128/128, BiLSTM 64,
head Dense 128/64. Read back out of the AST to confirm:

```
Conv1D (filters, kernel): [(32, 5), (32, 5), (64, 5), (64, 5), (128, 3), (128, 3)]
LSTM units             : [64]
Dense units in order   : [16, 128, 64, NUM_CLASSES]
```

The `16` is the RR branch, untouched.

**The skip connection** - four lines, immediately after the last dropout and
before the output:

```python
    combined = Dropout(0.4)(combined)

    # DIRECT RR SKIP (E5)
    combined = Concatenate()([
        combined,
        rr_input
    ])

    # OUTPUT
    output = Dense(NUM_CLASSES, activation='softmax')(combined)
```

`rr_input` is the raw 5-feature tensor, not `rr_branch`. The existing branch -
`Dense(16)` then `Dropout(0.2)` into the 144-wide concatenation - is
completely unaltered. This is an addition, not a replacement, so the output
layer now sees the RR features by two routes: one deeply transformed, one
untouched.

---

## Layer table

```
E) E5 layer table
   layer                      output             params
   ----------------------------------------------------
   Input ecg                  (234, 9)                -
   Conv1D k5                  (234, 32)           1,472
   BatchNorm                  (234, 32)             128
   Conv1D k5                  (234, 32)           5,152
   BatchNorm                  (234, 32)             128
   MaxPool /2                 (117, 32)               -
   Dropout 0.2                (117, 32)               -
   Conv1D k5                  (117, 64)          10,304
   BatchNorm                  (117, 64)             256
   Conv1D k5                  (117, 64)          20,544
   BatchNorm                  (117, 64)             256
   MaxPool /2                 (58, 64)                -
   Dropout 0.25               (58, 64)                -
   Conv1D k3                  (58, 128)          24,704
   BatchNorm                  (58, 128)             512
   Conv1D k3                  (58, 128)          49,280
   BatchNorm                  (58, 128)             512
   MaxPool /2                 (29, 128)               -
   Dropout 0.3                (29, 128)               -
   Bidirectional LSTM 64      (128,)             98,816
   Dropout 0.4                (128,)                  -
   Input rr                   (5,)                    -
   Dense 16 (rr branch)       (16,)                  96
   Dropout 0.2 (rr)           (16,)                   -
   Concatenate                (144,)                  -
   Dense 128                  (128,)             18,560
   Dropout 0.5                (128,)                  -
   Dense 64                   (64,)               8,256
   Dropout 0.4                (64,)                   -
   Concatenate (RR skip)      (69,)                   -
   Dense 3 softmax            (3,)                  210
   ----------------------------------------------------
   TOTAL                                        239,186

   output Dense inputs: 64 + 5 = 69
   output Dense params: 69*3 + 3 = 210   (was 64*3+3 = 195)
   E2 239171 -> E5 239186  = +15 parameters
```

**Total: 239,186 = E2's 239,171 + 15.** The only parameter change is the
output layer: `69*3 + 3 = 210` against `64*3 + 3 = 195`.

**The counting method is validated against two known totals before being
trusted**: the same formulas reproduce E2's **239,171** and E4's **16,283**
exactly.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` against E4 HEAD:**

```
changed : ['build_model']
added   : NONE
removed : NONE
```

`build_model` is the **only** changed function, as required.

**Against E2 (post-fix), the comparison that matters:** `build_model` is again
the only differing function, and its entire code diff is the four-line
concatenate above - every other diff line is comment. Outside `build_model`,
the source differs from E2 in **18 lines, 0 of them code** (the comment I
added at E4 recording E3's failure).

**So E5 is exactly E2 plus a skip connection.** Not approximately - the module
level is code-identical.

**Wavelet widths still match E2 to 4 decimals**: 8.1028, 4.0514, 2.7009,
2.0257, 1.6206, 1.3505, 1.1575, 1.0129, 0.9003. Confirmed independently on
record 209: per-channel std is `[2.6169, 2.2094, 1.4953, 0.9834, 0.6684,
0.4742, 0.3495, 0.2659, 0.2086]`, identical to E2's.

**Skip placement checked structurally**, not by eye: two `Concatenate()` call
sites, one with `[x, rr_branch]` (the original) and one with
`[combined, rr_input]` (the skip); the skip's source position sits after the
last `Dropout(0.4)` and before `output = Dense(`.

**Constants unchanged** (one occurrence each): `PRE_SAMPLES = 90`,
`POST_SAMPLES = 144`, `FOCAL_ALPHA = 0.50`, `FOCAL_GAMMA = 2.0`,
`ADAM_LEARNING_RATE = 1e-4`, `RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`,
`RR_CLIP_MAX = 3.0`, `SAMPLING_RATE_HZ = 360.0`. Augmentation multipliers
`[0, 6, 2, 0]` identical. `rr = [...]` and `RR_FEATURE_NAMES` byte-identical.
`DS1` `9f20e3ac1758a312...`, `DS2` `b8a3e6bbdeeec72a...`, `DS1_VAL`
`0d9df3612a6111a1...` = `['207','220','223']`.

---

## Falsifiable predictions

1. **Total parameters: 239,186.** Output Dense reports 210 rather than 195;
   every other layer matches E2's summary line for line. (Not checkable from
   `metrics.json` today - see the E4 scorecard note.)
2. **425 steps/epoch**, `train_distribution` N=40301 / S=670 / V=3105,
   `wavelet_centre_frequencies` exactly `[10, 20, ..., 90]`.
3. **S-F1 will exceed E2's 0.2686.** This is the bet. The skip gives the
   output layer an un-attenuated view of the one feature block where the S
   signal demonstrably lives. **If S-F1 does not move, the RR signal is not
   being attenuated by depth** - it is either already sufficient at the
   output, or the 5 ratios simply do not separate S from N well enough, and
   the next step is a better S feature rather than a better path to it.
4. **N-F1 and V-F1 will move very little** from E2's 0.9787 and 0.9111. Fifteen
   extra parameters cannot restructure the ECG pathway; if V-F1 shifts by more
   than a point or two, something other than the skip is in play and the run
   is not the clean ablation it looks like.

Prediction 3 is the experiment; prediction 4 is the control.

---

## Commit

```
986dc29  E5: direct RR skip connection to the output layer
```

Pushed to `origin/main`. `docs/ablation.md` now carries 12 runs including E4.

---

## Problems

1. **No run records its parameter count**, so my parameter predictions cannot
   be checked against any artefact. I have now made three of them. Adding
   `model_parameters: model.count_params()` to the metrics config would close
   this in one line - flagged, not done, because E5's brief was one change to
   `build_model`.

2. **A 5-wide skip against a 64-wide feature vector is a weak intervention.**
   The output layer sees 69 inputs of which 5 are the raw RR features. If the
   attenuation hypothesis is right but the effect is small, this may not be
   enough to show it. A stronger test - had it been in scope - would widen the
   RR path rather than add a thin bypass. Worth knowing before reading a null
   result as a refutation.

3. **E5 is two changes against E4** (capacity restore + skip) but **one change
   against E2**. E2 is the correct comparison and the report treats it that
   way; the ablation table's row order should not be read as implying E4 is
   the baseline.

4. **The dropout before the skip still applies to the CNN/LSTM path only.**
   The raw `rr_input` bypasses every dropout in the network, so those 5
   features are the only inputs to the output layer that are never dropped.
   That is the intent, but it does mean the output layer can learn to lean on
   them heavily without the usual regularisation pressure - a plausible
   overfitting route on 670 unique S beats.

5. Carried over: hard constraint 2 still violated (augmentation on); threshold
   tuning has failed to transfer twice; record 114 lead swap unfixed; record
   207 still a validation outlier; `tools/inspect_ds1.py` JSON stale; stale
   root `__pycache__/`.
