# Ablation log

One row per training run. Every number here is read programmatically from that
run's `results/<run>/metrics.json` - never typed from memory.

Rules (see CLAUDE.md): ONE change per run, and nothing is ever tuned against
DS2. All metrics below are on DS2 (inter-patient test set).

| step | description | commit | macro-F1 | S recall | S precision | S F1 | V F1 | accuracy |
|---|---|---|---|---|---|---|---|---|
| 0 | colleague's script, paths made configurable | `b58d6b6` | 0.6358 | 0.1171 | 0.3298 | 0.1728 | 0.7694 | 0.9294 |
| 1 | patient-wise validation split (DS1_VAL = 207,220,223) | `ae9a91c` | 0.4569 | 0.0065 | 0.2000 | 0.0127 | 0.4215 | 0.8682 |
| 1b | RR normalization fitted on training set only | `10e06f2` | 0.4742 | 0.0071 | 0.1806 | 0.0136 | 0.4641 | 0.8843 |

## Notes

- **step 0** (`baseline`) - train distribution N=45839 / S=943 / V=3788.
  Baseline reproduced on Kaggle (TF 2.20, P100).

- **step 1** (`step1_patient_val`) - train distribution N=40301 / S=670 / V=3105. Ran 8 epochs; val_loss was minimal at epoch 1, so EarlyStopping(monitor='val_loss', restore_best_weights=True) **restored epoch 1** - the evaluated model was trained for ONE epoch. Peak val_accuracy 0.7368, below the 0.8528 an all-N prediction scores on DS1_VAL.
  Fixes the beat-level split that put the same patient in training and validation.

- **step 1b** (`step1b_rr_trainstats`) - train distribution N=40301 / S=670 / V=3105. Ran 8 epochs; val_loss was minimal at epoch 1, so EarlyStopping(monitor='val_loss', restore_best_weights=True) **restored epoch 1** - the evaluated model was trained for ONE epoch. Peak val_accuracy 0.7390, below the 0.8528 an all-N prediction scores on DS1_VAL.
  **Not a single-variable comparison against the baseline.** This step also changed how DS2 is preprocessed: the test set is now scaled by *training* statistics instead of its own, so baseline and step 1 test numbers were produced under different preprocessing. The change is required - fitting a scaler on DS2 is fitting a parameter on the test set - but it means the delta here mixes two effects.

## Reading these numbers

Steps 1 and 1b both **evaluated a one-epoch model**: `val_loss` rose
monotonically after epoch 1 under focal loss while `val_accuracy` stayed flat
(spread 0.03), so `restore_best_weights` kept epoch-1 weights and patience
stopped the run at epoch 8. The macro-F1 drop from 0.6358 to 0.4569/0.4742 is
therefore **not** evidence that the patient-wise split or the RR fix hurt the
model - it is the cost of selecting on `val_loss`. Step 2 changes the
selection metric to `val_macro_f1`; the first run after that is the first one
whose test numbers are comparable to the baseline's on model quality.
