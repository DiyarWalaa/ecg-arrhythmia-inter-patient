# Last report — E9: E6's window on E8's beat population

**Status: written and verified locally. NOT YET TRAINED.** No `results/E9_*`
folder and no ablation row; E9 is pre-registered in the generated table's
"Pre-registered, NOT YET RUN" section.

## Why

E8 reached S-F1 0.4752 and S recall 0.6594 against E6's 0.3641 and 0.3388 —
but changed **two** things at once: the representation *and* the beat
population. Its 2-second span cap rejected 15.5% of N, 14.8% of S and 1.0% of
V, so DS2 fell from 49,289 beats to 42,154. The gain cannot be attributed to
the representation until both are scored on the same beats.

E9 holds the representation at E6's and borrows only E8's acceptance rule.

## What changed

Built by restoring E7's `src/train.py` — whose ratio-1.0 arm provably
reproduces E6's selection (0.5540 at epoch 6) — and applying the filter,
rather than un-patching E8. `ast.dump` against that base:

| | |
|---|---|
| CHANGED (2) | `extract_beats_from_record`, `load_dataset` |
| ADDED (3) | `empty_segmentation_stats`, `merge_segmentation_stats`, `summarize_segmentation_stats` |
| IDENTICAL (17) | `build_model`, `make_balanced_dataset`, `augment_training_data`, `augment_segment`, `cwt_ricker`, `tune_decision_weights`, `weights_for_ratio`, `make_callbacks`, `normalize_segment`, +8 |

Module-level: `MAX_SPAN_SAMPLES` added, `SAMPLER_RATIO_GRID` pinned to
`[1.0]`, `metrics` extended. Nothing else.

The span rule is applied **before** the window and never reaches the model:
a beat is kept if `0 < ann[i+1] - ann[i-1] <= 720`; the model then sees the
unchanged `r-90 .. r+144`.

## Verification

- `python -b -m py_compile src/train.py` — clean.
- `build_model` **identical to both E6 and E8** by `ast.dump`. Parameter
  arithmetic at 9 channels gives **239,171**, matching E6's published count
  (the same formula gives E8's 239,331 at 10 channels).
- 25-item unchanged-vs-E6 audit — all match: `DS1`, `DS2`, `DS1_VAL`,
  `RR_FEATURE_NAMES` (5), `RR_CLIP_*`, `RR_LOCAL_WINDOW`,
  `WAVELET_TARGET_FREQS_HZ`, `ADAM_LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`,
  `SEED`, `LEAD_INDEX`, `PRE_SAMPLES` 90, `POST_SAMPLES` 144, `SAMPLER`,
  `THRESHOLD_GRID`, `AAMI_MAP`, `cwt_ricker`, `augment_training_data`.
- Wavelet widths 8.1028 … 0.9003 → centres 10 … 90 Hz. Sampler weights
  [1/3, 1/3, 1/3]; loss plain `categorical_crossentropy`; augmentation
  uncalled.
- Shapes: DS1_TRAIN **(38536, 234, 9)**, DS1_VAL (6234, 234, 9), DS2
  **(42148, 234, 9)**. All finite. Train memory 0.32 GB.

Checks ran the **real functions**, AST-extracted from `src/train.py` and
exec'd with numpy/wfdb only — nothing reimplemented.

## Population — matches E8 exactly except for 12 unreachable beats

The span-cap rejections reproduce E8's **identically**: DS1_TRAIN
5280/33/227, DS1_VAL 179/0/81, DS2 6838/271/32.

| split | E9 accepted | E8 accepted | difference |
|---|---|---|---|
| DS1_TRAIN | 35021 / 637 / 2878 | 35025 / 637 / 2878 | −4 N |
| DS1_VAL | 5359 / 273 / 602 | 5361 / 273 / 602 | −2 N |
| DS2 | **37395 / 1565 / 3188** | 37400 / 1565 / 3189 | −5 N, −1 V |

**E9 is scored on 42,148 DS2 beats against E8's 42,154.** The 6 excluded
beats are **5 N and 1 V**, every one the first annotated beat of its record,
where the R peak sits 28–88 samples in so E6's `r − 90` window starts before
sample 0 and there is no signal to read. E8's `R−1..R+1` window did not have
that problem.

**The S class is identical on both sides at 1,565 beats**, so every S metric
— recall, precision, F1 — is exactly comparable. N recall can move by at most
1.3e-4 and V recall by at most 3.1e-4, bounding macro-F1 incomparability
below **2e-4** against an E8-vs-E6 effect size of 0.08 — roughly **400×
smaller than what is being measured**. Recorded in `metrics.json` under
`population_matches_e8_except` and in the ablation note, so the comparison is
precise rather than approximate.

Rejected alternatives: re-running E8 with the fixed-window edge rule would
spend a full run to remove an uncertainty 400× below the effect;
left-padding the window at the record start would fabricate a signal pattern
the model sees nowhere else.

A run-time assertion in section 12 hard-stops if the accepted counts are not
exactly 35021/637/2878, 5359/273/602, 37395/1565/3188.

## Falsifiable prediction

**Mechanical, checkable on return — any mismatch is a bug, not a result:**
`total_parameters` = **239,171** · `steps_per_epoch` = **302**
(ceil(38536/128), same as E8's by coincidence of rounding; E6 ran 345) ·
input (n, 234, 9) · train N=35021 S=637 V=2878 · DS2 42,148 beats
37395/1565/3188 · `sampling_weights` [1/3, 1/3, 1/3] ·
`mask_channel_index` null.

**Substantive.** The E8 span cap removes the beats with long compensatory
pauses — the post-ectopic N beats that most resemble an S beat under a fixed
window. So E9 should land **between** E6 and E8 on S, and **above both on
macro-F1**, since it does not inherit E8's V collapse (V-F1 0.5661, 4,405 N
called V).

Concretely: **E9 test S-F1 lands in [0.40, 0.50]** — above E6's 0.3641,
at or below E8's 0.4752 — and **E9 macro-F1 exceeds E8's 0.6495**.

**What this decides.** If E9's S-F1 reaches ≥ 0.45, most of E8's S gain was
the easier population, not the variable-length representation, and E8's
headline should be restated. If E9's S-F1 stays ≤ 0.40, the representation is
doing the work and E8's gain is real.

**Falsifier for my reasoning:** E9 S-F1 below 0.3641 (i.e. no better than E6
despite the easier population) would mean the span cap does not remove the
confusable N beats at all, and the population story is wrong in both
directions.

## Also in this session

E8's artefact was committed and its ablation row added (17 runs). The gate
now tracks which DS2 population each row scored — `full`,
`span_capped_720` (E8), `span_capped_720_fixed_window` (E9) — and refuses any
artefact disagreeing with its declaration. Verified by negative test.

## Not done

- E9 not trained. Kaggle run pending; the ablation row follows the artefact.
- Record 114's swapped leads (known fact 2) remain unfixed — out of scope.
