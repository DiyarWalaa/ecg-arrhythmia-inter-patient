# Ablation log

One row per training run. Every number here is read programmatically from that
run's `results/<run>/metrics.json` - never typed from memory.

Rules (see CLAUDE.md): ONE change per run, and nothing is ever tuned against
DS2. All metrics below are on DS2 (inter-patient test set).

> **Validation set changed at step 5.** Rows **baseline through step 4** used
> a 3-record validation set, `DS1_VAL = ['207','220','223']`. From **step 5**
> onward it is 5 records, `['106','118','207','220','223']`, and the training
> pool shrinks from 44,076 to 39,774 beats. Test-set metrics below are all on
> the same untouched DS2, so the columns remain meaningful - but model
> *selection* changed, and `FOCAL_ALPHA` shifts because it is derived from
> DS1_TRAIN counts. **Step 5 onward is not a like-for-like comparison with
> what precedes it.**

| step | description | commit | macro-F1 | S recall | S precision | S F1 | V F1 | accuracy |
|---|---|---|---|---|---|---|---|---|
| 0 | colleague's script, paths made configurable | `b58d6b6` | 0.6358 | 0.1171 | 0.3298 | 0.1728 | 0.7694 | 0.9294 |
| 1 | patient-wise validation split (DS1_VAL = 207,220,223) | `ae9a91c` | 0.4569 | 0.0065 | 0.2000 | 0.0127 | 0.4215 | 0.8682 |
| 1b | RR normalization fitted on training set only | `10e06f2` | 0.4742 | 0.0071 | 0.1806 | 0.0136 | 0.4641 | 0.8843 |
| 2 | select on val macro-F1 instead of val_loss | `5c1b9d6` | 0.6800 | 0.1280 | 0.4024 | 0.1942 | 0.8718 | 0.9476 |
| 3 | patient-relative RR ratio features (5 dimensionless) | `1a2509c` | 0.6645 | 0.1514 | 0.3634 | 0.2138 | 0.8004 | 0.9451 |
| 4 | remove oversampling, per-class focal alpha | `3b67016` | 0.5599 | 0.1106 | 0.2016 | 0.1428 | 0.5895 | 0.8892 |

## Notes

- **step 0** (`baseline`) - train distribution N=45839 / S=943 / V=3788.
  Baseline reproduced on Kaggle (TF 2.20, P100).

- **step 1** (`step1_patient_val`) - train distribution N=40301 / S=670 / V=3105. Ran 8 epochs; `val_loss` was minimal at epoch 1, so EarlyStopping(monitor='val_loss', restore_best_weights=True) **restored epoch 1** - the evaluated model was trained for ONE epoch. Peak val_accuracy 0.7368, below the 0.8528 an all-N prediction scores on DS1_VAL.
  Fixes the beat-level split that put the same patient in training and validation.

- **step 1b** (`step1b_rr_trainstats`) - train distribution N=40301 / S=670 / V=3105. Ran 8 epochs; `val_loss` was minimal at epoch 1, so EarlyStopping(monitor='val_loss', restore_best_weights=True) **restored epoch 1** - the evaluated model was trained for ONE epoch. Peak val_accuracy 0.7390, below the 0.8528 an all-N prediction scores on DS1_VAL.
  **Not a single-variable comparison against the baseline.** This step also changed how DS2 is preprocessed: the test set is now scaled by *training* statistics instead of its own, so baseline and step 1 test numbers were produced under different preprocessing. The change is required - fitting a scaler on DS2 is fitting a parameter on the test set - but the delta here mixes two effects.

- **step 2** (`step2_macrof1_selection`) - train distribution N=40301 / S=670 / V=3105. Ran 24 epochs; selection on `val_macro_f1` chose **epoch 14** (best val macro-F1 0.5460); early stopping fired: True.
  First run whose evaluated model trained for more than one epoch. Per-record validation exposed record **207** as a severe outlier: N recall 0.0700, S recall 0.0000, accuracy 0.1647, against 0.9780 and 0.9602 on 220 and 223. Still 1350 of 1836 test S beats (73.5%) predicted as N.

- **step 3** (`step3_rr_ratios`) - train distribution N=40301 / S=670 / V=3105. Ran 18 epochs; selection on `val_macro_f1` chose **epoch 8** (best val macro-F1 0.5618); early stopping fired: True.
  **macro-F1 REGRESSED 0.6800 -> 0.6645.** But the S->N leak roughly halved: 1350 -> 692 of 1836 test S beats. Those beats did not become correct S predictions - they moved to **V** instead (S->V 251 -> 866), and V F1 fell 0.8718 -> 0.8004. S recall did rise 0.1280 -> 0.1514 and S F1 0.1942 -> 0.2138, so the ratio features do carry usable prematurity signal; the model is simply spending it on the wrong class boundary. **Stated condition: if step 4 does not recover macro-F1 above 0.6800, the ratio features get reconsidered.**

- **step 4** (`step4_class_alpha`) - train distribution N=40301 / S=670 / V=3105. Ran 13 epochs; selection on `val_macro_f1` chose **epoch 3** (best val macro-F1 0.7288); early stopping fired: True.
  **Best epoch was 3 of 13.** Val macro-F1 spiked to 0.7288 there and fell to 0.5791 the next epoch - a **0.1497 single-epoch swing**. The plateau over epochs 4-13 averaged 0.5661, so the selected peak sat **0.1627 above it** and lasted one epoch.

  **First run where test scored BELOW validation (-0.1689).** Steps 2 and 3 both scored above (+0.1340 and +0.1027). The checkpoint was selected on noise - which is what motivated step 5.

  **The step 3 stated condition (macro-F1 must recover above 0.6800) is NOT EVALUABLE from this run**, because the selection was confounded. It carries forward to step 5.

## Test-minus-validation gap

A model selected on a trustworthy validation signal should not score wildly
differently on test. The gap is `test macro-F1 - best_val_macro_f1`:

| step | best val macro-F1 | test macro-F1 | gap |
|---|---|---|---|
| 2 | 0.5460 | 0.6800 | +0.1340 |
| 3 | 0.5618 | 0.6645 | +0.1027 |
| 4 | 0.7288 | 0.5599 | -0.1689 |

Steps 2 and 3 scored **above** validation, which is the expected direction
when validation holds only three patients and two of them are hard. Step 4 is
the first run to score **below** it, by 0.1689 - the signature of a checkpoint
picked at a noise spike rather than a genuine optimum.

## Reading these numbers

Steps 1 and 1b both **evaluated a one-epoch model**: `val_loss` rose
monotonically after epoch 1 under focal loss while `val_accuracy` stayed flat
(spread 0.03), so `restore_best_weights` kept epoch-1 weights and patience
stopped the run at epoch 8. Their macro-F1 drop to 0.4569 / 0.4742 measures
that selection failure, **not** the patient-wise split or the RR fix.

Step 2 changed the selection metric and the same pipeline reached macro-F1
0.6800 - still the best result so far. Step 3 is the first step to move the
S->N leak (73.5% -> 37.7%), at the cost of macro-F1, because the freed S
beats went to V rather than to S.

Step 4 removed duplicate oversampling and replaced the scalar focal alpha
(which rebalanced nothing) with a per-class vector. Its test macro-F1 of
0.5599 is the worst of any run, **but the number does not measure the
change**: selection landed on a one-epoch spike. Note also that the swap is
close to weight-neutral in relative terms - effective S:N emphasis moves only
7.000x -> 7.756x - while cutting mean alpha per sample to 62% and
steps/epoch from 425 to 345, so an epoch carries roughly half the loss mass
it used to.

Step 5 enlarges the validation set to cut that selection variance. It changes
no model, loss, feature or hyperparameter. Two questions carry into it: does
the swing shrink, and - inherited from step 3 - does macro-F1 recover above
0.6800.
