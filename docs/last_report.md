# Last report

**Task:** verify the replacement `results/step4_class_alpha/metrics.json`,
then fill the pending step 4 row in `docs/ablation.md`.

**Date:** 2026-08-22

---

## Verification gate

The previous file in this folder was a mislabelled copy of step 3, so this one
was gated before any of it was used. All eight checks passed:

| check | result |
|---|---|
| `run_name == "step4_class_alpha"` | PASS - `step4_class_alpha` |
| `focal_loss_alpha` is a 3-element vector | PASS - `[0.24275019764900208, 1.8826956748962402, 0.8745541572570801]` |
| `focal_loss_alpha` is not the scalar `0.5` | PASS - type `list` |
| `oversampling is False` | PASS - `False` |
| not a duplicate of step 3 | PASS - sha256 `8c728e239e8d404a...` vs step 3's `f6e91c1dcaf43d0b...` |
| timestamp differs from step 3's | PASS - `2026-08-22T16:35:36.938210` |
| `config.focal_loss_beta` present | PASS - `0.5` |
| `config.focal_alpha_class_counts` present | PASS - `[40301, 670, 3105]` |

Two further sanity checks: the alpha vector sums to exactly 3.000000 and
orders N < V < S. This is a genuine step 4 run.

The alpha vector also **matches my step 4 prediction to every digit**
(`[0.2428, 1.8827, 0.8746]`), as do the class counts.

---

## What changed

- `docs/ablation.md` - the pending step 4 row is filled, every cell read
  programmatically from the JSON. The generator now carries a hard assertion
  on `run_name`, alpha shape and `oversampling`, so it refuses to emit a row
  from a mislabelled file rather than silently producing one.
- Added a **test-minus-validation gap** table, computed from the metrics
  files, because that gap is the clearest evidence for the step 5 rationale.
- `results/step4_class_alpha/` is now committed.

Step 4 row:

| step | description | commit | macro-F1 | S recall | S precision | S F1 | V F1 | accuracy |
|---|---|---|---|---|---|---|---|---|
| 4 | remove oversampling, per-class focal alpha | `3b67016` | 0.5599 | 0.1106 | 0.2016 | 0.1428 | 0.5895 | 0.8892 |

Notes recorded with it: best epoch 3 of 13; val macro-F1 spiked to
0.7288 and fell to 0.5791 the next epoch, a 0.1497 single-epoch
swing; the epochs 4-13 plateau averaged 0.5661 so the peak sat
0.1627 above it; first run where test scored below validation
(-0.1689) against +0.1340 and +0.1027 for steps 2 and 3; and that the step
3 stated condition is **not evaluable** from this run because selection was
confounded, so it carries forward to step 5.

---

## Step 4 predictions - scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | 345 steps/epoch, `oversampling: false`, `train_distribution` unchanged | **PASS** - `oversampling: false`, distribution N=40301 / S=670 / V=3105 = 44,076 samples = 345 steps at batch 128 |
| 2 | alpha `[0.2428..., 1.8827..., 0.8746...]` summing to 3.0, counts `[40301, 670, 3105]` | **PASS** - exact |
| 3 | S->V misclassification falls from 866 | **FAIL** - 871, essentially unchanged |
| 4 | macro-F1 lands above step 3's 0.6645 | **FAIL** - 0.5599, the worst of any run |

Predictions 3 and 4 failed, but **both are confounded** by the selection
failure: the evaluated checkpoint is a one-epoch noise spike, not the model
the change produces. I am recording them as failed rather than excusing them -
a prediction that cannot be cleanly tested is a prediction I should not have
stated without a selection-stability caveat. What can be said is that the
mechanism I predicted (V weight rising only 1.75x against N's 0.49x should
stop V absorbing S beats) shows no sign of working at this checkpoint.

**One genuinely good sign, unrelated to the predictions:** record 207 stopped
being a catastrophe. Its validation accuracy went 0.1733 (step 3) -> 0.9037
and N recall 0.0824 -> 0.9643. S recall on 207 is still 0.0000.

---

## Problems

1. **Step 4's headline number does not measure step 4.** 0.5599 is in the
   table because it is what the run produced, but the notes column says
   plainly that selection was confounded. Do not cite it as evidence that
   removing oversampling hurt.

2. **Two questions now ride on step 5**: does the swing shrink below 0.1497,
   and does macro-F1 recover above 0.6800. The second is inherited from step
   3 and has now been deferred twice.

3. **The ablation generator is the guard, and it is not in the repo.** The
   assertions that caught the mislabelled file live in a scratchpad script,
   so `docs/ablation.md` is reproducible only by me re-running it. If you want
   that protection to persist, the generator belongs in `tools/`. Say the
   word.

4. Carried over: record 114 lead swap still unfixed; `tools/inspect_ds1.py`
   still writes its JSON into `tools/`; stale root `__pycache__/` present.
