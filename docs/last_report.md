# Last report

**Task:** E2 fix - one normalization path for all beats.

**Date:** 2026-08-25

---

## What changed

Augmentation now perturbs the **raw waveform** and re-enters the same tail the
originals use. Both routes are now:

```
raw 234-sample window -> normalize_segment -> cwt_ricker -> (234, 9)
```

- `extract_beats_from_record` keeps the z-scored raw waveform alongside the
  scalogram and returns it as a fourth value; `load_dataset` passes it
  through as `X_raw`.
- `augment_training_data(X, X_raw, RR, y)` now iterates `X_raw`, perturbs the
  waveform via `augment_segment`, and then applies `cwt_ricker` itself. The
  originals in `X_list[0]` are the scalograms section 7 already produced.
- The post-CWT global z-score is **gone**. `augment_segment` is back to
  operating on a 1-D waveform, so its `normalize_segment` call is the same
  per-beat z-score the originals get - not a z-score over 234x9 values.
- The three `load_dataset` call sites take the extra return value;
  validation and test discard it with `_` since only training augments.

Amplitude scaling, time shift and Gaussian noise are unchanged and still
applied before normalization, exactly as before E2.

---

## The two code paths, from the AST

```
ORIGINAL beat  (extract_beats_from_record):
   line 603   normalize_segment
   line 608   cwt_ricker

AUGMENTED beat (augment_training_data -> augment_segment):
   line 730   normalize_segment   (inside augment_segment)
   line 790   augment_segment
   line 792   cwt_ricker
```

Both end `normalize_segment -> cwt_ricker -> (234, 9)`. Verified by counting
real `ast.Call` nodes rather than substrings:

- `augment_training_data` makes **no** `normalize_segment` call of its own
- `augment_segment` makes **exactly one**
- `extract_beats_from_record` calls them in the order
  `[normalize_segment, cwt_ricker]`, once each
- `cwt_ricker` does not appear inside `augment_segment` at all

*(My first version of this check tested `"normalize_segment" not in
source`, which failed on the explanatory comment that names the function.
The substring test was wrong, not the code; it now counts call nodes.)*

---

## Per-channel statistics: original vs augmented

Record 209, 3,004 originals plus 2,300 synthetic beats, using the functions
lifted out of the edited source:

```
A) the code path, from AST
   ORIGINAL beat  (extract_beats_from_record):
      line 603   normalize_segment
      line 608   cwt_ricker
   AUGMENTED beat (augment_training_data -> augment_segment):
      line 730   normalize_segment   (inside augment_segment)
      line 790   augment_segment
      line 792   cwt_ricker

   both paths end: ... -> normalize_segment -> cwt_ricker -> (234, 9)
   [PASS] augment_segment no longer sees a scalogram (no cwt_ricker in it)
   [PASS] augment_segment still ends with normalize_segment
   [PASS] augment_training_data applies cwt_ricker AFTER augment_segment
   [PASS] augment_training_data makes no normalize_segment CALL of its own
   [PASS] augment_segment makes exactly one normalize_segment call
   [PASS] extract_beats_from_record: normalize_segment then cwt_ricker, once each

B) per-channel statistics on a real record
   record 209: X (3004, 234, 9)  X_raw (3004, 234)
   after augmentation: (5304, 234, 9)  (3004 originals + 2300 synthetic)

    ch centre Hz | orig mean  orig std |  aug mean   aug std |   d mean  std rat
     0      10.0 |    0.0078    2.6033 |    0.0035    2.6068 |   0.0044    1.001
     1      20.0 |    0.0060    2.1701 |    0.0041    2.1728 |   0.0019    1.001
     2      30.0 |    0.0030    1.4867 |    0.0022    1.4876 |   0.0008    1.001
     3      40.0 |    0.0018    0.9874 |    0.0014    0.9873 |   0.0005    1.000
     4      50.0 |    0.0012    0.6763 |    0.0010    0.6759 |   0.0003    0.999
     5      60.0 |    0.0008    0.4831 |    0.0007    0.4827 |   0.0001    0.999
     6      70.0 |    0.0006    0.3580 |    0.0006    0.3577 |   0.0001    0.999
     7      80.0 |    0.0005    0.2734 |    0.0004    0.2732 |   0.0001    0.999
     8      90.0 |    0.0004    0.2163 |    0.0004    0.2162 |   0.0000    1.000

   worst |log(std ratio)| across channels : 0.0014 (1.00x ratio would be 0.0)
   worst |mean difference| in orig-std units: 0.0017

   for contrast - the pre-fix path (perturb the scalogram, then
   global z-score over all 234x9 values):
    ch |  orig std old-aug std |     ratio
     0 |    2.5765    1.9637 |     0.762
     1 |    2.1358    1.6243 |     0.761
     2 |    1.5024    1.1422 |     0.760
     3 |    1.0162    0.7726 |     0.760
     4 |    0.7050    0.5363 |     0.761
     5 |    0.5090    0.3874 |     0.761
     6 |    0.3794    0.2891 |     0.762
     7 |    0.2904    0.2215 |     0.763
     8 |    0.2314    0.1768 |     0.764

   [PASS] every channel std ratio within 0.85 - 1.18

   [PASS] every channel mean within 0.10 orig-std

   [PASS] augmented shape matches originals

   [PASS] no NaN / inf in augmented

   [PASS] counts: S x7, V x3

OVERALL: PASS
```

**Every channel now matches to within 0.1%.** Std ratios run 0.999 to 1.001;
the worst mean difference is 0.0017 in units of the original std.

**For contrast, the pre-fix path** put augmented beats at a **uniform 0.76x**
the original std in every channel - the global z-score over 234x9 values
divided everything by roughly the same factor. That is a 24% amplitude
mismatch applied to 4,020 of 4,690 S training samples, on the one class E2
exists to improve. It is the step 1b failure mode again, and it is now gone.

---

## Verification

**py_compile** - passed, exit code 0.

**`ast.dump` against E2 as committed:**

```
changed : ['augment_segment', 'augment_training_data',
           'extract_beats_from_record', 'load_dataset']
added   : NONE
removed : NONE
unexpected: NONE
```

All four were expected: two to carry the raw waveform, two to change where
the perturbation happens.

**`augment_segment` is now AST-identical to its E1 version** - the
`axis=0` I added for the scalogram is gone because it no longer sees one. The
perturbation logic is bit-for-bit what it was before E2, which is exactly what
a clean one-variable ablation needs.

**`build_model` is byte-identical to both E2-as-committed and E1.**

**Constants** all present exactly once and unchanged: `PRE_SAMPLES = 90`,
`POST_SAMPLES = 144`, `FOCAL_ALPHA = 0.50`, `FOCAL_GAMMA = 2.0`,
`ADAM_LEARNING_RATE = 1e-4`, `RR_LOCAL_WINDOW = 10`, `RR_CLIP_MIN = 0.2`,
`RR_CLIP_MAX = 3.0`, `SAMPLING_RATE_HZ = 360.0`.

**Augmentation multipliers** `[0, 6, 2, 0]` - identical to E2 and to E1.
Confirmed empirically too: the augmented set has exactly 7x the S beats and
3x the V beats of the original, with N unchanged.

**RR features and record lists**: the `rr = [...]` block and
`RR_FEATURE_NAMES` byte-identical; `DS1` `9f20e3ac1758a312...`, `DS2`
`b8a3e6bbdeeec72a...`, `DS1_VAL` `0d9df3612a6111a1...` =
`['207','220','223']`. The wavelet target frequencies are unchanged.

---

## Noted for later: the 60 Hz channels

You are almost certainly right that this is mains. MIT-BIH was recorded at
the BIH in Boston on US mains at **60 Hz**, and channels 5 and 6 sit at 60 and
70 Hz with `|r| = 0.9968`. Two things point the same way:

- The 60 Hz channel has no local prominence in the energy profile - std falls
  monotonically 2.62 -> 0.21 from 10 to 90 Hz, so whatever 60 Hz content
  exists is riding a smoothly decaying envelope rather than standing out as
  cardiac structure.
- The MIT-BIH signals were digitised from analogue tape with a passband of
  roughly 0.1-100 Hz and no notch filter documented per record, so residual
  mains is expected rather than surprising.

**No action taken** - the 9 scales stay as they are so this remains a clean
match to Zahid et al.'s 10-90 Hz range. Recorded here as a candidate for a
later step: a 50/60/70 Hz notch before the CWT, or dropping to 6-7 scales that
skip the mains neighbourhood, would test whether those channels carry signal
or interference. Worth doing only after E2 has a number.

---

## Falsifiable predictions

Unchanged from E2, since nothing about the split, the multipliers or the
architecture moved:

1. **425 steps/epoch**; `train_distribution` N=40301 / S=670 / V=3105.
2. **First Conv1D reports 1,472 parameters**; total **239,171**.
3. `config.wavelet_centre_frequencies` = `[10, 20, ..., 90]`.

New to this fix:

4. **E2's result is now interpretable as a one-variable ablation against E1's
   argmax 0.6645.** Before the fix it was not - 85.7% of S training samples
   were mis-scaled. If E2 still underperforms E1 after this, the wavelet
   representation itself is the thing that did not help, and that conclusion
   is now sound.

---

## Commit

```
11cd348  E2 fix: identical normalization path for original and augmented beats
```

Pushed to `origin/main`.

---

## Problems

1. **`X_raw` costs memory and is computed for sets that never use it.**
   `load_dataset` builds a `(n, 234)` float32 array for validation and test
   too, which the call sites discard with `_`. That is ~46 MB for DS2 and ~6
   MB for validation - small next to the 415 MB scalogram, but genuinely
   wasted. I kept the signature uniform rather than adding a flag, because a
   `return_raw=` parameter is more surface area than the memory is worth.

2. **Augmented beats are still not perfectly identical in path to originals** -
   they go `raw -> z-score -> perturb -> z-score -> CWT`, originals go
   `raw -> z-score -> CWT`. The second z-score is inherent to how
   `augment_segment` has always worked (it renormalises after adding noise and
   scaling), and it is what the task specified. The measured result is that
   this makes no meaningful difference: per-channel stats match to 0.1%.
   Worth knowing it is a same-tail equivalence, not a literally identical call
   sequence.

3. **The channel std disparity remains** - 12.55x from the 10 Hz channel to
   the 90 Hz one. The fix aligned augmented beats with originals; it did not
   equalise the scales relative to each other. If E2 underperforms,
   per-channel normalisation is the first thing to try, and it would now be
   safe to add in one place because both paths share the tail.

4. Carried over: hard constraint 2 still violated (augmentation on); record
   114 lead swap unfixed; record 207 still a validation outlier;
   `tools/inspect_ds1.py` JSON stale; stale root `__pycache__/`.
