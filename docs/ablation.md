# Ablation log

One row per training run. Every number here is copied programmatically from
that run's `results/<run>/metrics.json` - never typed from memory.

Rules (see CLAUDE.md): ONE change per run, and nothing is ever tuned against
DS2. All metrics below are on DS2 (inter-patient test set).

| step | description | commit | macro-F1 | S recall | S precision | S F1 | V F1 | accuracy |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline (colleague's script, paths made configurable) | `b58d6b6` | 0.6358 | 0.1171 | 0.3298 | 0.1728 | 0.7694 | 0.9294 |

## Notes

- **step 0** - reproduced on Kaggle (TF 2.20, P100). Train distribution
  N=45839 / S=943 / V=3788; test distribution N=44233 / S=1836 / V=3220.
  Blocking problem: 1243 of 1836 true S beats are predicted as N, so S
  recall sits at 0.1171.
