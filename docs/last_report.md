# Last report — E8: variable-length R-1 to R+1 segmentation with mask channel

**Status: written and verified locally. NOT YET TRAINED.** No `results/E8_*`
folder exists and no row has been added to `docs/ablation.md`; E8 is
pre-registered in the generated table's "Pre-registered, NOT YET RUN" section
instead.

## Why

E6 fires S at nearly the right frequency — 1581 predictions against 1836 true
S beats — but only 622 are correct. E7 showed the sampler is saturated
(+0.0005 recall for 2× exposure, precision 0.3934 → 0.2789). The decision
rule and class exposure are both spent; what remains is discriminability.

De Waele et al. (2026), same split, reach S sensitivity 0.9116 against our
0.3442. Their window runs from the previous R peak to the next and discards
spans over 2 s, so **window length itself encodes prematurity**. Our fixed
234-sample window destroyed that.

## What changed

`ast.dump` diff against `HEAD` — 4 functions changed, 4 added, **15
identical**:

| | |
|---|---|
| CHANGED | `extract_beats_from_record`, `load_dataset`, `make_balanced_dataset`, `augment_training_data` |
| ADDED | `empty_segmentation_stats`, `merge_segmentation_stats`, `summarize_segmentation_stats`, `gather_training_batch` |
| IDENTICAL | `build_model`, `cwt_ricker`, `ricker_wavelet`, `normalize_segment`, `tune_decision_weights`, `weights_for_ratio`, `make_callbacks`, `reset_seeds`, +7 |

- **`extract_beats_from_record`** — window is `ann_samples[i-1] … ann_samples[i+1]`.
  Reject if span > 720 (2 s), if it runs off the signal edge, or if it is
  non-positive. Z-score and CWT the **real portion only** (no padding before
  the transform — a padded step is exactly what a Ricker responds to most
  strongly), then zero-pad right to (720, 9) and append a binary mask as
  channel 9. Emits (720, 10).
- **`make_balanced_dataset`** — now streams **indices**, materialising each
  batch through `tf.numpy_function`. The E6/E7 form sliced `X_tr_aug[mask]`
  per class and handed it to `from_tensor_slices`, materialising the training
  set twice over. At (234, 9) that was 371 MB a copy; at (720, 10) it is
  1.27 GB a copy. The draw sequence is unchanged — `shuffle` and
  `sample_from_datasets` depend on element count and seed, never contents.
- **`augment_training_data`** — now raises. It is uncallable under
  variable-length beats (there is no rectangular `X_raw`), and has been
  uncalled since E6.
- **`SAMPLER_RATIO_GRID = [1.0]`** — E7 settled the ratio question, so the
  sweep machinery is kept (identical selection path, seed reset and
  metrics shape) but runs one point at E6's exact 1:1:1.

## Verification

`python -m py_compile src/train.py` — clean.

Local checks ran the **real functions**, AST-extracted from `src/train.py`
and exec'd with numpy/wfdb only (TensorFlow is not installable here), so
nothing was reimplemented. On record 101 (1251 beats), all pass:

shape (n, 720, 10) · float32 · no NaN · no inf · mask strictly binary
({0.0, 1.0}) · mask sums equal the true span lengths as a multiset · mask is
a prefix, so padding is on the right · scalogram **exactly** zero wherever
the mask is zero (max |x| in padding = 0) · non-zero where the mask is 1 ·
every span ≤ 720 · RR features finite and inside the clip · first beat
reproduces an independent `normalize_segment → cwt_ricker` bit for bit.

**Parameter count.** `build_model` is byte-identical to E6's by `ast.dump`,
so the only difference is the input shape. First Conv1D 5·9·32+32 = 1472 →
5·10·32+32 = 1632, **+160**. The same arithmetic reproduces E6's published
239,171 exactly, and gives **239,331** for 10 channels. Sequence length
234 → 720 changes no parameter count (Conv1D/BN depend on channels;
BiLSTM with `return_sequences=False` on feature dim and units; the head sees
a fixed 128+16).

**Confirmed unchanged vs E6 (`5b3b203`), by AST:** `DS1`, `DS2`, `DS1_VAL`,
`VAL_SELECTION_RULE`, `RR_FEATURE_NAMES` (5), `RR_LOCAL_WINDOW`, `RR_CLIP_*`,
`WAVELET_TARGET_FREQS_HZ`, `ADAM_LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`,
`SEED`, `LEAD_INDEX`, `SAMPLER`, `THRESHOLD_GRID`, `AAMI_MAP`, `build_model`.
Wavelet widths unchanged at 8.1028 … 0.9003 → centres 10 … 90 Hz. Sampler
weights [1/3, 1/3, 1/3]; loss plain `categorical_crossentropy`; augmentation
uncalled.

## The rejection bias — read this before comparing E8 to anything

Measured on the real records, all 44:

| split | class | annotated | accepted | > cap | edge | rej % | span mean | span std |
|---|---|---|---|---|---|---|---|---|
| DS1_TRAIN | N | 40305 | 35025 | 5280 | 0 | 13.10% | 520.4 | 95.4 |
| DS1_TRAIN | S | 670 | 637 | 33 | 0 | 4.93% | **407.3** | 143.7 |
| DS1_TRAIN | V | 3105 | 2878 | 227 | 0 | 7.31% | 474.5 | 126.6 |
| DS1_VAL | N | 5540 | 5361 | 179 | 0 | 3.23% | 587.3 | 82.4 |
| DS1_VAL | S | 273 | 273 | 0 | 0 | 0.00% | **423.7** | 95.4 |
| DS1_VAL | V | 683 | 602 | 81 | 0 | 11.86% | 500.6 | 103.3 |
| DS2 | N | 44238 | 37400 | 6838 | 0 | 15.46% | 524.0 | 95.5 |
| DS2 | S | 1836 | 1565 | 271 | 0 | 14.76% | **506.4** | 73.7 |
| DS2 | V | 3221 | 3189 | 32 | 0 | 0.99% | 472.0 | 91.1 |

Zero edge rejections and zero invalid spans anywhere. The `annotated` column
reconciles with every previous run: DS2 shows 44238/1836/3221 against the
44233/1836/3220 support all earlier runs report — the +5 N and +1 V are beats
the old fixed 234-sample window rejected at the signal edge and this one does
not. The beat population is the same; only the acceptance rule changed.

**Two consequences, both material:**

1. **DS2 shrinks 49,289 → 42,154 beats, losing 271 of 1836 true S.** E8's
   macro-F1 is computed over a different population than every row in
   `docs/ablation.md`. It is **not** a clean one-variable ablation and its
   confusion matrix cannot be differenced against E6's cell by cell. This is
   pre-registered in the ablation table, not left for the write-up.
2. **The mechanism is ~6× weaker on test than on train.** Mean span N vs S:
   DS1_TRAIN 520.4 vs 407.3 (gap 113), DS1_VAL 587.3 vs 423.7 (gap 164),
   DS2 524.0 vs 506.4 (**gap 17.6**). The feature E8 adds barely separates
   the classes in the set it will be judged on.

`tools/make_ablation.py`'s `ds2_support()` gate will fail when E8 is added —
correctly. Its message now names E8 and says to give it its own support
reference rather than relaxing the check.

## Falsifiable prediction

**Mechanical, checkable the moment `metrics.json` returns — any mismatch is a
bug, not a result:**
`config.total_parameters` = **239,331** · `config.steps_per_epoch` = **302**
(ceil(38540/128), down from E6's 345) · `train_distribution` N=35025 S=637
V=2878 · DS2 evaluated on **42,154** beats, N=37400 S=1565 V=3189 ·
`sampling_weights` [1/3, 1/3, 1/3] · `input_length` 720 · `n_input_channels` 10.

**Substantive:** validation will improve more than test. Specifically —
E8 beats E6 on **validation** macro-F1 (> 0.5540), but its **test** macro-F1
gain over E6's 0.7263 is smaller than its validation gain, narrowing the
test-minus-validation gap below E6's +0.1723. Reason: the span gap between N
and S is 164 samples in DS1_VAL and 17.6 in DS2, so the model will lean on a
feature that largely is not there at test time.

**Sharper, and the one that decides it:** test S recall stays **below 0.50**
(E6: 0.3442, on a now-different denominator).

**What would falsify the reading:** test S-F1 above **0.45** against E6's
0.3641. That would mean the representation change works despite the span
distribution shift, and the DS2 span statistics above are not the binding
constraint.

## Not done

- Not trained. Kaggle run pending; `docs/ablation.md` gets its row only when
  a `metrics.json` comes back.
- Record 114's swapped leads (known fact 2) remain unfixed — out of scope.
