# Last report

**Task:** E3 - log-space the wavelet scales down to 3 Hz to capture P-wave
frequencies.

**Date:** 2026-08-25

---

## E2 predictions - scorecard

E2's results arrived and are now committed with an ablation row.

| # | prediction | outcome |
|---|---|---|
| 1 | 425 steps/epoch, `train_distribution` N=40301 / S=670 / V=3105 | **PASS** |
| 2 | first Conv1D 1,472 params, total 239,171 | **PASS** (9 channels as designed) |
| 3 | `wavelet_centre_frequencies` = `[10, 20, ..., 90]` | **PASS** - exact |
| 4 | E2 is now a valid one-variable ablation against E1 | **PASS** - see below |

**Prediction 4 needed checking, not assuming.** The E2 run is stamped
`2026-08-24T21:43:12`; the normalization fix was committed at 00:31:22 +0300
= **21:31:22 UTC**. Kaggle stamps UTC, so the run started **12 minutes after
the fix landed** and therefore includes it. E2's 0.7178 is the clean number,
not the mis-scaled one. That reasoning is recorded in the ablation note rather
than left implicit.

**E2 is the best run so far: macro-F1 0.7178** against E1's 0.6645, and the
gain is almost entirely V - V-F1 0.8004 -> 0.9111, V precision 0.6891 ->
0.9495, N-called-V 521 -> 95, S-called-V 866 -> 55. S moved very little,
0.2138 -> 0.2686. Two caveats now in the table: `best_epoch` was **1**, and
threshold tuning failed to transfer for the second run running (validation
0.5557 -> 0.5822, test 0.7178 -> 0.6332).

---

## What changed in E3

One constant. `WAVELET_TARGET_FREQS_HZ` goes from linear to log-spaced:

```python
WAVELET_F_MIN_HZ = 3.0
WAVELET_F_MAX_HZ = 90.0
N_WAVELET_SCALES_TARGET = 9

WAVELET_TARGET_FREQS_HZ = [
    float(
        WAVELET_F_MIN_HZ
        * (WAVELET_F_MAX_HZ / WAVELET_F_MIN_HZ)
        ** (k / (N_WAVELET_SCALES_TARGET - 1.0))
    )
    for k in range(N_WAVELET_SCALES_TARGET)
]
```

Computed, not hardcoded. The widths come from the **same**
`ricker_width_for_frequency` already in the code - that function is
byte-identical, as is `cwt_ricker`. The section 6B print already reports
targets, widths, centre frequencies and support lengths, so no print change
was needed.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` against E2 HEAD:**

```
changed : NONE
added   : NONE
removed : NONE
classes changed: NONE
```

**Zero functions changed.** The entire source diff outside comments is one
deleted line and the eleven-line replacement above:

```diff
-WAVELET_TARGET_FREQS_HZ = [float(10 * k) for k in range(1, 10)]
+WAVELET_F_MIN_HZ = 3.0
+WAVELET_F_MAX_HZ = 90.0
+N_WAVELET_SCALES_TARGET = 9
+WAVELET_TARGET_FREQS_HZ = [ ... ]
```

This is as clean as a one-variable ablation gets.

**Measurements on record 209:**

```
A) scale set
    ch   target Hz    width a   centre Hz   support truncated
     0      3.0000    27.0095      3.0000       234       YES
     1      4.5895    17.6553      4.5895       176        no
     2      7.0210    11.5408      7.0210       115        no
     3     10.7409     7.5439     10.7409        75        no
     4     16.4317     4.9312     16.4317        49        no
     5     25.1375     3.2234     25.1375        32        no
     6     38.4558     2.1071     38.4558        21        no
     7     58.8305     1.3773     58.8305        13        no
     8     90.0000     0.9003     90.0000         9        no

   empirical FFT peak of each discretised wavelet:
    ch   target Hz   FFT peak Hz      err %
     0      3.0000        3.0103      0.34%
     1      4.5895        4.5923      0.06%
     2      7.0210        7.0312      0.15%
     3     10.7409       10.7446      0.03%
     4     16.4317       16.4355      0.02%
     5     25.1375       25.1367      0.00%
     6     38.4558       38.4521      0.01%
     7     58.8305       58.8208      0.02%
     8     90.0000       90.3735      0.42%

B) the 3 Hz channel, where the support is truncated
   width a = 27.0095   untruncated support = 270   used = 234  (cap 234)
   envelope at the truncation edge, relative to peak: 9.122e-05
   wavelet sum (admissibility, ideally 0):
     untruncated +1.691902e-04   truncated +3.285266e-03
     truncated sum as a fraction of peak |amplitude|: 1.970e-02
   energy retained by truncation: 99.999958%

C) scalogram on record 209
   X (3004, 234, 9)  dtype float32
   NaN: False   inf: False
   channel 0 (3 Hz) NaN: False  inf: False  min -10.9393  max 12.8413

    ch   centre Hz        min        max        std
     0        3.00   -10.9393    12.8413     2.6516
     1        4.59   -10.1253    11.0011     2.8981
     2        7.02    -7.5971    11.4178     2.6765
     3       10.74    -9.0580    13.1331     2.6118
     4       16.43   -10.1064    13.9030     2.4403
     5       25.14    -9.4066    12.8038     1.8286
     6       38.46    -7.6175     9.6044     1.0478
     7       58.83    -4.7937     5.3866     0.4926
     8       90.00    -2.3716     3.3636     0.2086

   9x9 channel correlation matrix:
            3.0     4.6     7.0    10.7    16.4    25.1    38.5    58.8    90.0
     3.0  1.0000  0.8623  0.5417  0.2480  0.1050  0.0526  0.0339  0.0268  0.0241
     4.6  0.8623  1.0000  0.8183  0.4491  0.2082  0.1077  0.0690  0.0531  0.0454
     7.0  0.5417  0.8183  1.0000  0.7987  0.4712  0.2736  0.1856  0.1444  0.1221
    10.7  0.2480  0.4491  0.7987  1.0000  0.8426  0.6012  0.4542  0.3689  0.3159
    16.4  0.1050  0.2082  0.4712  0.8426  1.0000  0.8977  0.7747  0.6685  0.5874
    25.1  0.0526  0.1077  0.2736  0.6012  0.8977  1.0000  0.9374  0.8598  0.7775
    38.5  0.0339  0.0690  0.1856  0.4542  0.7747  0.9374  1.0000  0.9731  0.9096
    58.8  0.0268  0.0531  0.1444  0.3689  0.6685  0.8598  0.9731  1.0000  0.9755
    90.0  0.0241  0.0454  0.1221  0.3159  0.5874  0.7775  0.9096  0.9755  1.0000

   max |off-diagonal correlation| : 0.9755  (ch 7 @ 58.8 Hz vs ch 8 @ 90.0 Hz)
   E2 (linear 10-90 Hz) was          : 0.9968  (60 Hz vs 70 Hz)
   change: -0.0213
   top 3 correlated pairs:
     0.9755  ch7 (58.8 Hz) - ch8 (90.0 Hz)
     0.9731  ch6 (38.5 Hz) - ch7 (58.8 Hz)
     0.9374  ch5 (25.1 Hz) - ch6 (38.5 Hz)
   mean |off-diagonal|: 0.4607

   std ratio max/min across channels: 13.89  (E2: 12.55)
```

**Constants unchanged** (one occurrence each): `PRE_SAMPLES = 90`,
`POST_SAMPLES = 144`, `FOCAL_ALPHA = 0.50`, `FOCAL_GAMMA = 2.0`,
`ADAM_LEARNING_RATE = 1e-4`, `RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`,
`RR_CLIP_MAX = 3.0`, `SAMPLING_RATE_HZ = 360.0`. Augmentation multipliers
`[0, 6, 2, 0]` identical. The `rr = [...]` block and `RR_FEATURE_NAMES`
byte-identical. `DS1` `9f20e3ac1758a312...`, `DS2` `b8a3e6bbdeeec72a...`,
`DS1_VAL` `0d9df3612a6111a1...` = `['207','220','223']`. `build_model`
byte-identical, so the model is unchanged: first Conv1D `5*9*32 + 32 = 1472`,
total **239,171**.

---

## The truncation question

You flagged it correctly, and it engages on exactly one channel.

The 3 Hz scale has width `a = 27.0095`, so its untruncated support would be
`10a = 270` samples against a 234-sample window. `cwt_ricker` caps it at 234.
The consequences, measured:

- **Envelope at the truncation edge: 9.12e-05 of peak.** The cut lands far out
  in the Gaussian tail.
- **Energy retained: 99.999958%.**
- The one real side effect is admissibility. A Ricker wavelet should sum to
  zero; truncation breaks that slightly. The sum goes from `+1.69e-04`
  (untruncated) to `+3.29e-03` (truncated) - a 19x increase, but still only
  **1.97e-02 of peak amplitude**, and the mean per sample is 1.4e-05 against a
  peak of 0.167. Since every beat is z-scored to zero mean before the CWT, the
  DC that this non-zero sum would pass is already approximately zero.

**The channel is not degenerate:** std 2.6516, range -10.94 to 12.84, no NaN,
no inf, and it is the *least* correlated channel with the rest (0.0241 against
90 Hz).

---

## Does log spacing decorrelate the set?

**Yes, but the improvement is at the low end, not the top.**

- **Max off-diagonal correlation: 0.9755** (58.8 vs 90.0 Hz), down from E2's
  **0.9968** (60 vs 70 Hz). A change of **-0.0213**.
- Mean off-diagonal correlation 0.4607, against a much more uniform structure
  in E2.
- The correlation matrix now has clear band structure: adjacent scales
  correlate, distant ones do not. 3 Hz against 90 Hz is **0.0241** - nearly
  independent. In E2 the low and high ends were far more entangled.

**But the top three pairs are still above 0.93** (0.9755, 0.9731, 0.9374).
Log spacing compresses the high end - 38.5, 58.8 and 90 Hz sit closer in
*width* terms (1.49, 0.98, 0.90 samples) than the linear set did. So the
redundancy you identified has moved rather than vanished: it is no longer two
channels at 60/70 Hz, it is a smoother gradient across the top three.

Channel std is now much flatter across the informative band: 2.65, 2.90, 2.68,
2.61, 2.44, 1.83, 1.05, 0.49, 0.21. The **4.59 Hz channel now carries the most
energy of any scale (std 2.8981)** - which is where P- and T-wave content
lives, and is the point of the change. The max/min std ratio is 13.89 against
E2's 12.55, slightly worse only because the 90 Hz channel is unchanged while
the low end got stronger.

---

## Falsifiable predictions

1. **425 steps/epoch and 239,171 total parameters**, both unchanged from E2.
   The channel count did not move, so if either differs, something other than
   the scale set changed.
2. **`config.wavelet_centre_frequencies` will be**
   `[3.00, 4.59, 7.02, 10.74, 16.43, 25.14, 38.46, 58.83, 90.00]` to 2 dp,
   with `wavelet_scales` running 27.0095 down to 0.9003.
3. **S-F1 will exceed E2's 0.2686.** This is the actual bet, and the
   reasoning is the physiological one: four channels now sit at or below 10.74
   Hz where P-wave energy lives, against zero in E2. **If S-F1 does not move,
   the missing-P-wave hypothesis is wrong and the S problem is not a
   frequency-coverage problem.**
4. **V-F1 will drop somewhat from 0.9111.** E2 spent five channels between 50
   and 90 Hz on the wide-QRS morphology that V detection relies on; E3 spends
   two. I would be surprised by a collapse, but a few points of V-F1 is the
   price being paid for the low-frequency coverage.

Prediction 3 is the experiment. Prediction 4 is the cost. If both land, the
right follow-up is more scales rather than re-spaced ones.

---

## Commit

```
fe14c34  E3: log-spaced wavelet scales 3-90 Hz
```

Pushed to `origin/main`. `docs/ablation.md` now carries 10 runs including E2.

---

## Problems

1. **The high end is still redundant.** 0.9755 between 58.8 and 90 Hz is
   better than 0.9968 but not good. If E3's V-F1 drops as predicted, the
   diagnosis is that we are simultaneously under-resolving the QRS band and
   wasting channels inside it - which argues for more than 9 scales rather
   than a different 9.

2. **The 3 Hz wavelet is mildly non-admissible** after truncation (sum
   3.29e-03 rather than 0). Harmless here because the input is z-scored, but
   worth knowing if the pre-CWT normalization is ever removed.

3. **E2's `best_epoch` was 1 and it still produced the best test score.** That
   is the third run selected at a very early epoch. It worked this time, but
   the validation signal is clearly not tracking test performance well - the
   test-minus-validation gap is now **+0.1621**, the largest of any run.
   Selection remains the weakest part of this pipeline and is worth a step of
   its own.

4. **Threshold tuning has now failed to transfer twice** (E1: -0.0495; E2:
   -0.0846). Two failures on the same mechanism is a pattern, not noise. The
   argmax column remains the honest comparison, which is what the table
   reports. Worth considering whether to stop running the search.

5. Carried over: hard constraint 2 still violated (augmentation on); record
   114 lead swap unfixed; record 207 still a validation outlier;
   `tools/inspect_ds1.py` JSON stale; stale root `__pycache__/`.
