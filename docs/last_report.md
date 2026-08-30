# Last report — E10: fixed-window width sweep

**Status: written and verified locally. NOT YET TRAINED.** No `results/E10_*`
folder and no ablation row; E10 is pre-registered in the generated table.
E9's artefact was recorded first (18 runs).

---

## Part 1 — E9 recorded, and it settles what E8 measured

| | E9 argmax | E9 tuned | E8 argmax | E6 argmax |
|---|---|---|---|---|
| macro-F1 | **0.7032** | 0.7046 | 0.6495 | 0.7263 |
| S F1 | 0.2976 | 0.3125 | **0.4752** | 0.3641 |
| V F1 | **0.8548** | 0.8488 | 0.5661 | 0.8503 |
| accuracy | 0.9226 | 0.9147 | 0.8410 | 0.9352 |

Mechanical predictions exact again: 239,171 params, 302 steps/epoch, train
35021/637/2878, DS2 37395/1565/3188.

**My substantive prediction was wrong, in the way I named as falsifying it.**
I predicted E9's S-F1 in [0.40, 0.50] and wrote that below 0.3641 "would mean
the population story is wrong in both directions". It came in at **0.2976**.
The span-capped population is *harder* for S, not easier — so **none** of
E8's S gain came from an easier test set.

The clean result, on identical beats (S identical at 1,565 both sides): the
wide variable window is worth **+0.1776 S-F1** and costs **−0.2887 V-F1**.
A straight S-against-V trade at a fixed parameter count.

One honest amendment: E9's tuning raised `w_S` to 2.0 — above `w_N` — and the
result did *not* get worse (+0.0014, noise). The standing threshold rule is
now three clean confirmations and one flat case, not four.

---

## Part 2 — E10

### Why

E8 and E9 differ in **two** ways: how much **context** the model sees, and
whether an explicit **length signal** exists (E8's mask channel encodes the
R−1..R+1 span). A fixed wide window has the context and no length signal, so
sweeping the fixed width separates them.

### What changed

`ast.dump` vs HEAD — **18 functions identical**, including `build_model`,
`make_balanced_dataset`, `tune_decision_weights`, `load_dataset`, `cwt_ricker`:

| | |
|---|---|
| CHANGED (4) | `extract_beats_from_record`, and the 3 `*_segmentation_stats` helpers (new rejection bucket) |
| ADDED (2) | `assert_cnn_input`, `load_ds1_for_window` |

`WINDOW_GRID = [(90,144), (140,220), (185,295), (230,370)]` → widths
**234 / 360 / 480 / 600**, each keeping E9's ≈0.385/0.615 split (measured
0.3846, 0.3889, 0.3854, 0.3833). Sampler pinned to 1:1:1 (`SAMPLER_RATIO`
replaces the retired ratio grid); no mask, no padding, no architecture or
loss change; `DS1_VAL` unchanged.

### Population control — all four arms on identical beats

Acceptance is decided **once**, by the span cap **and** the largest window in
the grid `(230, 370)`, then applied to every arm regardless of the width it
reads. Verified on all 44 records by running the real extraction code: the
four arms return **byte-identical labels and RR features**, and per-class
counts agree exactly (a mismatch raises before training).

| split | E9 | E10 | cost |
|---|---|---|---|
| DS1_TRAIN | 35021 / 637 / 2878 | 35006 / 637 / 2877 | −15 N, −1 V |
| DS1_VAL | 5359 / 273 / 602 | 5358 / 273 / 602 | −1 N |
| DS2 | 37395 / 1565 / 3188 | **37377 / 1565 / 3187** | −18 N, −1 V |

**36 beats total — 34 N, 2 V, and no S at any stage.** S has been 1,565 in
DS2 for E8, E9 and E10 alike, so the S column compares exactly across all
three. Baked in as a hard assertion; the run aborts if the counts differ.

### DS2 is not evaluated inside the sweep — structurally

Stronger than the BETA and sampler sweeps, which kept `X_test` in memory and
relied on the loop not mentioning it. Here **DS2 is not read at all** until
section 23B, after selection. Proved by AST:

- Section 22 (sweep) begins line 1940; last `SELECTED_WINDOW` assignment line
  2125; threshold tuning (validation-only) line 2302; **first DS2 read line
  2420**, in section 23B.
- References to any of `X_test`, `y_test`, `RR_test`, `y_test_encoded`,
  `y_test_cat`, `SEG_STATS_TEST` between the sweep and 23B: **0**.
- `tune_decision_weights` contains no DS2 name.

### Parameter count — identical for every arm

239,171 for all four widths. Conv1D/BatchNorm counts depend on channels, the
BiLSTM runs `return_sequences=False` so it depends on feature dim and units,
and the Dense head sees a fixed 128+16 — none depend on sequence length. The
same arithmetic reproduces E6's and E9's published 239,171. After three
`MaxPooling1D(2)` the arms carry 29 / 45 / 60 / 75 timesteps into the BiLSTM.
A runtime assert refuses to proceed if the arms' counts ever differ.

### Verification summary

`python -m py_compile src/train.py` clean · shapes `(n, W, 9)` for
W ∈ {234, 360, 480, 600} across train/val/test · all finite · wavelet widths
8.1028 … 0.9003 → 10 … 90 Hz · sampler `[1/3, 1/3, 1/3]`, plain
`categorical_crossentropy` · `DS1_VAL = ['207','220','223']` · augmentation
uncalled. Checks ran the **real functions**, AST-extracted and exec'd with
numpy/wfdb only.

### Falsifiable prediction

**Mechanical:** `total_parameters` = **239,171** for every arm ·
`steps_per_epoch` = **301** (ceil(38520/128); E9 ran 302) · train
35006/637/2877 · DS2 **42,129** beats 37377/1565/3187 ·
`parameter_counts_by_width` = {234: 239171, 360: 239171, 480: 239171,
600: 239171}.

**Substantive.** E8's mask channel is a *direct* encoding of prematurity;
a wide fixed window only makes the neighbouring beats visible and leaves the
network to infer timing from morphology it has already shown it cannot use
(E4 and E5 eliminated capacity and RR attenuation). So I expect **context
alone to recover less than half** of E8's S gain:

- The selected arm's **test S-F1 lands in [0.30, 0.38]** — above E9's 0.2976,
  well below E8's 0.4752.
- Validation **prefers a wider window than 234** (the selected width is 360,
  480 or 600), because more context helps a little.
- **V-F1 stays above 0.80**, since no arm has E8's length signal to overfire V.

**What this decides.** S-F1 ≥ 0.42 in any arm ⇒ context is the mechanism and
the mask channel is incidental. All four arms within ±0.02 of E9's 0.2976 ⇒
width is irrelevant and the explicit length signal is the whole finding.

**Falsifier for my reasoning:** the 234 control arm winning on validation
would mean extra context actively hurts, and my "context helps a little"
premise is wrong.

## Not done

- E10 not trained. Runtime note: the sweep re-extracts DS1 per arm, so
  extraction cost is ~7× E9's, plus four training runs.
- Record 114's swapped leads (known fact 2) remain unfixed — out of scope.
