# Last report

**Task:** E2 - replace the raw beat waveform with a 9-scale Mexican-hat
wavelet scalogram.

**Date:** 2026-08-25

---

## E1 predictions - scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | 425 steps/epoch | **PASS** (train distribution N=40301 / S=670 / V=3105, augmented 54,306) |
| 2 | argmax reproduces step 3's confusion matrix exactly, `best_epoch` 8 | **PASS - exactly** |
| 3 | `w_S > 1.0`, `w_N` exactly 1.0 | **PASS** - `w = [1.0, 4.0, 4.0]` |
| 4 | tuned S recall far above argmax, S precision down; macro-F1 open | **PARTIAL** - recall and precision as predicted, macro-F1 got **worse** |

**Prediction 2 is the important one and it landed exactly:**

```
step 3 confusion : [[43233, 479, 521], [692, 278, 866], [138, 8, 3074]]
E1 argmax        : [[43233, 479, 521], [692, 278, 866], [138, 8, 3074]]
identical: True     best_epoch 8 both     macro-F1 0.6645 both
```

E1's val macro-F1 (0.5618) and test-minus-val gap (+0.1027) are also identical
to step 3's. The revert restored the configuration bit for bit, and the
pipeline is deterministic. No bug to report.

**Threshold tuning did not transfer, and that is the finding.** The search
picked `w = [1.0, 4.0, 4.0]`, lifting *validation* macro-F1 0.5618 -> 0.6207.
On DS2 the same weights made things worse:

| metric | argmax | tuned |
|---|---|---|
| macro-F1 | **0.6645** | 0.6150 |
| S recall | 0.1514 | **0.3028** |
| S precision | 0.3634 | 0.1199 |
| N F1 | 0.9793 | 0.9372 |
| accuracy | 0.9451 | 0.8697 |

S recall doubled exactly as intended, but upweighting S by 4x pushed **3,990
true N beats** into the S column, and the search also raised `w_V` to 4.0, so
N F1 fell too. Tuning two coordinates against 273 validation S beats
overfitted the validation set - the risk I flagged in E1's report, now
measured. Validation gained +0.0589 while test lost -0.0495.

**E1's argmax result is the current best-comparable anchor: macro-F1 0.6645.**
Rows for E0 and E1 are now in `docs/ablation.md` (9 runs, gate passed).

---

## What changed in E2

**New section 6B.** Four functions: `ricker_wavelet`,
`ricker_width_for_frequency`, `ricker_centre_frequency`, `cwt_ricker`.

**Widths are derived, not hardcoded.** The Ricker spectrum is
`|psi(w)| ~ w^2 exp(-a^2 w^2 / 2)`, maximal where `2 - a^2 w^2 = 0`, so
`w_peak = sqrt(2)/a` and therefore

```
a = fs * sqrt(2) / (2 * pi * f_target)
```

**Extraction** now emits `(234, 9)`: `normalize_segment(raw)` first, exactly as
before, then `cwt_ricker(...).T`.

**`build_model` is byte-identical to E1.** The Conv1D branch widens from 1 to
9 channels on its own, because the call site already passes
`ecg_shape=X_tr_aug.shape[1:]`. No architecture edit was needed or made.

**Section 16** no longer calls `expand_dims` - the channel axis is now the
wavelet-scale axis. It asserts the three arrays are `(n, 234, 9)` instead.

**metrics.json config** gains `wavelet_scales`, `wavelet_centre_frequencies`,
`wavelet_target_frequencies`, `sampling_rate_hz`.

---

## scipy: the requested API no longer exists

`scipy.signal.cwt` and `scipy.signal.ricker` were **deprecated in SciPy 1.12
and removed in SciPy 1.15**. This machine runs **SciPy 1.18.0**, where
`hasattr(signal, 'cwt')` and `hasattr(signal, 'ricker')` are both **False**.

So the instruction could not be followed literally. Rather than pin an old
SciPy, I reimplemented both from the SciPy source, in numpy only:

- `ricker_wavelet` reproduces `scipy.signal.ricker` including its exact
  amplitude term `2 / (sqrt(3a) * pi^0.25)`.
- `cwt_ricker` reproduces `scipy.signal.cwt`, including its truncation of each
  wavelet to `min(10 * width, len(data))` samples and its reversed kernel.

**The answer to "check Kaggle and add a pip install line if needed" is: no pip
line, and none would have helped.** SciPy ships in the Kaggle image, but any
recent image has >= 1.15, where these names are gone. Pinning `scipy<1.15`
would risk conflicting with the rest of the image. **`src/train.py` now imports
no scipy at all** - verified, no `import scipy` / `from scipy` anywhere - so
there is nothing to install and no version to worry about.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` against E1 HEAD:**

```
changed : ['augment_segment', 'augment_training_data',
           'extract_beats_from_record']
added   : ['cwt_ricker', 'ricker_centre_frequency', 'ricker_wavelet',
           'ricker_width_for_frequency']
removed : NONE
unexpected: NONE
```

`extract_beats_from_record` changed to emit the scalogram. The two
augmentation functions changed **only** to survive a real channel axis - the
complete diffs are:

```diff
augment_segment:
-            x = np.roll(x, shift)
+            x = np.roll(x, shift, axis=0)      (+ 2 comment lines)

augment_training_data:
-        segment = sample.squeeze(-1)
+        segment = sample                       (+ 2 comment lines)
-            aug = np.expand_dims(aug, axis=-1)
```

Without those, `squeeze(-1)` raises on a size-9 axis and `np.roll` without
`axis` would flatten the array and mix scales. **The multipliers are
untouched:** `[0, 6, 2, 0]`, identical to E1 and to step 3.

**Widths and centre frequencies** - the derivation round-trips exactly, and I
checked each discretised wavelet empirically by FFT:

| target Hz | width a | centre Hz | FFT peak Hz | support |
|---|---|---|---|---|
| 10 | 8.1028 | 10.0000 | 10.02 | 81 |
| 20 | 4.0514 | 20.0000 | 20.04 | 40 |
| 30 | 2.7009 | 30.0000 | 29.97 | 27 |
| 40 | 2.0257 | 40.0000 | 39.99 | 20 |
| 50 | 1.6206 | 50.0000 | 50.01 | 16 |
| 60 | 1.3505 | 60.0000 | 60.03 | 13 |
| 70 | 1.1575 | 70.0000 | 69.96 | 11 |
| 80 | 1.0129 | 80.0000 | 79.98 | 10 |
| 90 | 0.9003 | 90.0000 | 90.35 | 9 |

FFT peaks match the analytic centres to within one bin.

**Scalogram on record 209** (3,004 beats), computed with the functions lifted
out of the edited source:

```
X shape (3004, 234, 9)  dtype float32

 ch  centre Hz        min        max       mean        std
  0       10.0    -8.7485    12.8062    -0.0282     2.6169
  1       20.0   -10.0534    13.4625    -0.0093     2.2094
  2       30.0    -8.7389    11.5800    -0.0052     1.4953
  3       40.0    -7.3527     9.1375    -0.0033     0.9834
  4       50.0    -5.8266     6.6779    -0.0024     0.6684
  5       60.0    -4.6605     5.2050    -0.0020     0.4742
  6       70.0    -3.6755     3.9632    -0.0016     0.3495
  7       80.0    -2.9482     4.1499    -0.0011     0.2659
  8       90.0    -2.3716     3.3636    -0.0012     0.2086

[PASS] shape is (n_beats, 234, 9)
[PASS] no NaN            [PASS] no inf
[PASS] RR still 5 features
[PASS] no channel is constant
[PASS] no two channels are duplicates (|corr| < 0.999)
[PASS] channel std spans a real range (max/min = 12.55)
[PASS] independent CWT recomputation matches to 1e-12
```

**Constants unchanged** (one occurrence each): `PRE_SAMPLES = 90`,
`POST_SAMPLES = 144`, `FOCAL_ALPHA = 0.50`, `FOCAL_GAMMA = 2.0`,
`ADAM_LEARNING_RATE = 1e-4`, `RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`,
`RR_CLIP_MAX = 3.0`. The `rr = [...]` block and `RR_FEATURE_NAMES` are
byte-identical. `DS1` `9f20e3ac1758a312...`, `DS2` `b8a3e6bbdeeec72a...`,
`DS1_VAL` `0d9df3612a6111a1...` = `['207','220','223']` - all unchanged.
`build_model` is byte-identical to E1, so the RR branch is untouched.

---

## Falsifiable predictions

1. **425 steps/epoch, unchanged.** The split, the augmentation multipliers and
   the beat count are all identical to E1; only the per-beat tensor shape
   changed. `train_distribution` will still read N=40301 / S=670 / V=3105.
   **If steps/epoch is not 425, something other than the representation
   changed.**
2. **The first Conv1D layer will report 1,472 parameters**, up from 192:
   `5 * 9 * 32 + 32 = 1472`. Total model parameters should be
   **239,171** (237,891 + 1,280). Every other layer is unchanged, so the
   delta must be exactly 1,280.
3. `config.wavelet_centre_frequencies` will be `[10, 20, ..., 90]` to within
   1e-9, and `wavelet_scales` will start `8.1028` and end `0.9003`.

I am not predicting an S-F1 value. Zahid et al. reach 0.8344 with this
representation, but they also have a different architecture and 10x fewer
parameters; adopting one ingredient is not adopting the result.

---

## Commit

```
6bf4d9e  E2: 9-scale wavelet scalogram input representation
```

Pushed to `origin/main`. `docs/ablation.md` now carries E0 and E1 (9 runs).

---

## Problems

1. **Augmented beats are now normalised differently from original beats.**
   Originals are `CWT(z-scored raw)` with no post-CWT normalisation.
   `augment_segment` ends with `normalize_segment(x)`, which on a `(234, 9)`
   array z-scores across **all scales at once**, so an augmented copy sits on a
   different scale from the original it came from. This is a genuine defect
   introduced by combining augmentation with a multi-channel input, and it
   affects 6 of every 7 S training samples. I did not fix it because the fix
   is a real design choice - augment in the waveform domain and recompute the
   CWT, or normalise per channel everywhere - and either is more than a
   one-variable ablation. **Flagging it as the most likely thing to undermine
   E2's result.**

2. **The scales are not equally scaled and the top ones are near-duplicates.**
   Channel std falls monotonically 2.62 -> 0.21 (12.55x), so the 90 Hz channel
   contributes about a twelfth of the 10 Hz channel's dynamic range to the
   first convolution. Worse, channels 5 and 6 (60 and 70 Hz) correlate at
   **|r| = 0.9968** - their widths differ by only 0.19 samples. The top three
   scales carry little independent information. If E2 helps, a coarser
   high-frequency spacing is worth testing; if it does not, per-channel
   normalisation is the first thing to try.

3. **The 90 Hz wavelet is marginally sampled** - width 0.90 samples, support 9
   samples. At fs = 360 Hz that is half Nyquist, and the discretised wavelet is
   barely resolved. Its FFT peak lands at 90.35 Hz rather than 90.00.

4. **Memory grows 9x.** Beat arrays go from ~145 MB to ~1.3 GB total
   (train augmented 457 MB, test 415 MB, train raw 371 MB, val 55 MB). Should
   fit a Kaggle P100 session, but it is no longer negligible, and the CWT adds
   9 convolutions per beat at extraction time.

5. **E1's threshold weights are not carried forward.** E2 will run its own
   search and pick its own `w`. Given the tuning hurt on DS2, the argmax column
   remains the honest comparison, and that is what `docs/ablation.md` reports.

6. Carried over: hard constraint 2 still violated (augmentation on); record 114
   lead swap still unfixed; record 207 still a validation outlier;
   `tools/inspect_ds1.py` JSON stale; stale root `__pycache__/`.
