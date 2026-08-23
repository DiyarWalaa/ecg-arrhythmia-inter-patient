# CLAUDE.md

## Project

Inter-patient ECG heartbeat classification into the three AAMI classes
**N / S / V**, on the MIT-BIH Arrhythmia Database. An academic paper is in
progress, so every reported number must be traceable and reproducible.

---

## HARD CONSTRAINTS — never violate

1. **Inter-patient evaluation only.** DS1 is the training pool, DS2 is the
   test set. No DS2 record may influence training, model selection, early
   stopping, checkpoint choice, or any threshold. If a validation set is
   needed, hold out **whole patients from DS1** — never a random split
   across DS2, and never beats from a DS1 patient that also appear in
   training (patient-level disjointness).
2. **No data expansion.** No synthetic beats, no augmentation, no duplicate
   oversampling, no SMOTE. Class imbalance is handled by loss/weighting
   only.
3. **DS1/DS2 record lists are from de Chazal et al. 2004.** They are fixed.
   Never edit, reorder, extend, or "fix" them — in `src/train.py`, in
   `tools/inspect_ds1.py`, or anywhere else.

### Known facts

Established, verified, and not to be re-litigated:

1. **Augmentation is ACTIVE again as of E0.** It was removed at step 4 and
   **restored at E0**, because the step 4/5/6 configuration performed worse
   and E0 exists to re-anchor on the step 3 training setup.
   `augment_training_data()` is called from section 18 and expands
   **S x7, V x3** (verified by AST call-graph, not grep).

   This **violates hard constraint 2** and is accepted deliberately, for
   now. Two known defects come with it: the RR feature vector is copied
   **unchanged** across every duplicate - which is worse than it was at
   step 3, since those are now the five patient-relative ratios rather
   than two raw intervals - and the `np.roll` shift moves the R-peak off
   its aligned position. Removing it again is a later step, and it must be
   paired with a replacement balancing mechanism.

   **Class balancing is in the data, not the loss.** `FOCAL_ALPHA` is back
   to the **scalar 0.50**, which scales the whole loss and rebalances
   nothing. The step 4 per-class alpha vector and the step 6 `BETA_GRID`
   sweep were both reverted at E0.

   **History, so old numbers are read correctly.** Runs **baseline through
   step 3** were produced *with* augmentation active: it expanded S x7 and
   V x3 while copying the RR feature vector unchanged across every copy.
   The baseline ran 449 steps/epoch at batch 128 = 57,425 samples
   = 41255 N + 849x7 S + 3409x3 V; without augmentation it would have been
   356. Treat every metric from those runs as produced under data
   expansion.

   Do not try to detect augmentation from `train_distribution` in
   `metrics.json` - that field is computed **before** the split and before
   augmentation, so it can never reveal it. This exact mistake was made once
   already. Check `config.oversampling` (present from step 4 onward) or the
   steps/epoch count instead.

2. **Record 114 has leads `['V5', 'MLII']`** - the two channels are swapped
   relative to every other DS1 record. With `LEAD_INDEX = 0` the pipeline
   reads **V5** for this record, not MLII, so record 114 feeds a different
   lead into the model than the other 21 DS1 records.
   **To be fixed in a later step.**

3. **DS1_VAL is `['207','220','223']` - enlarged at step 5, REVERTED at
   E1.** Step 5 grew it to `['106','118','207','220','223']` to cut
   selection variance. That worked on its own terms (largest single-epoch
   val macro-F1 swing fell 0.1497 -> 0.0724) but **biased model
   selection**, and E1 put it back.

   Why the 5-record set failed: 106 contributes **520 V beats and ZERO S
   beats**, 118 contributes 16 V to 96 S. Together they made validation
   85.3% N / 11.3% V / 3.4% S. Val V-F1 peaks at **epoch 2** while val
   S-F1 is still climbing at **epoch 6**, so macro-F1 tracks V and stops
   early. E0 ran training code identical to step 3 and differed only in
   this validation set: it selected epoch 2 instead of 8 and scored test
   macro-F1 **0.5591** against a required 0.6400.

   Accepted limitation of the 3-record set: 220 has zero V beats, so
   validation V-F1 rests on two records. That is cheaper than the
   selection bias the fix introduced.

   **Consequence for the ablation table: steps 5, 6 and E0 used the
   5-record validation set and are not directly comparable to anything
   else.** The rule is recorded in `metrics.json` under
   `val_selection_rule`; do not change `DS1_VAL` again without recording
   why, because it breaks comparability in both directions.

4. **Records 201 and 202 are the SAME SUBJECT** (per the PhysioNet
   documentation). 201 is in DS1, 202 is in DS2. This is a genuine subject
   leak across the train/test boundary, inherited from de Chazal et al. 2004.
   - **Do NOT change the record lists** to fix it - see hard constraint 3.
   - **Disclose it as a limitation** in the paper.
   - **Never place 201 in the validation set** - doing so would stack a
     second leak on top of the first. It is in DS1_TRAIN, and so is 209
     (which holds 383 of the 943 DS1 S beats and cannot be spared from
     training).

---

## NEVER do without asking first

- `git push --force` (or `--force-with-lease`)
- Rewriting history (`rebase`, `commit --amend`, `reset --hard` on pushed work)
- Deleting anything under `data/`
- Deleting anything under `results/`
- Changing the DS1 or DS2 record lists

---

## Current state

See `docs/ablation.md` for the full run-by-run table - it is the source of
truth and every number there is read programmatically from a
`results/<run>/metrics.json`.

Best test macro-F1 so far is **step 2** (`5c1b9d6`) at 0.6800. The baseline
(`b58d6b6`) scored 0.6358, but from a leaked validation split and the full
22-record training pool, so it is not a fair target.

| run | macro-F1 | S F1 | S recall | accuracy |
|---|---|---|---|---|
| baseline | 0.6358 | 0.1728 | 0.1171 | 0.9294 |
| step 2 | 0.6800 | 0.1942 | 0.1280 | 0.9476 |
| step 3 | 0.6645 | 0.2138 | 0.1514 | 0.9451 |
| E0 | 0.5591 | - | - | - |

**E1 restores the step 3 configuration exactly** (step 3 training code +
`DS1_VAL = ['207','220','223']`) and adds validation-tuned decision
thresholds on top. Because the pipeline is deterministic, E1's
**plain-argmax** test result should reproduce step 3's confusion matrix
`[[43233,479,521],[692,278,866],[138,8,3074]]`. Any mismatch is a bug, not
a result.

**Blocking problem:** the model **under-calls S by roughly 6x**. It
predicts S 584 times against 1836 true S beats. Crucially, our S
*precision* (0.402) is already in line with the literature - de Chazal 2004
at 0.385, Zhou 2021 at 0.415 - while our S *recall* is 0.128 against their
0.759 and 0.894. Comparable precision with an order-of-magnitude worse
recall points at the **decision rule**, not the features: the model ranks S
reasonably but almost never lets it win the argmax.

E1 addresses this directly with per-class decision weights `w`, tuned on
validation and applied to DS2 once (`prediction = argmax(w * p)`). A
secondary failure remains from step 3: freed S beats tend to land in **V**
rather than S (S->V rose 251 -> 866).

---

## Structure

```
.
├── CLAUDE.md                        this file
├── .gitignore
├── data/                            MIT-BIH database (gitignored, not in repo)
│   └── mit-bih-arrhythmia-database-1.0.0/
├── docs/                            notes, tables, reports
│   ├── ablation.md                  one row per run — the experiment log
│   ├── ds1_beat_counts.txt          console output of tools/inspect_ds1.py
│   └── last_report.md               overwritten at the end of every task
├── notebooks/                       exploratory Jupyter notebooks
├── results/                         one folder per run, holds metrics.json
│   └── baseline/
│       └── metrics.json
├── src/                             training code
│   └── train.py                     the full pipeline (needs TensorFlow)
└── tools/                           small CPU-only analysis scripts
    ├── inspect_ds1.py               per-record N/S/V beat counts for DS1
    └── ds1_beat_counts.json         machine-readable output of the above
```

`runs/` (gitignored) is where `src/train.py` writes figures, models and
metrics at execution time; the metrics.json that matters is copied back
into `results/<run_name>/`.

---

## Workflow

1. Edit code here, locally.
2. Commit and push to GitHub (`origin/main`).
3. Kaggle clones the repo and runs `src/train.py` on a P100.
4. The resulting `metrics.json` comes back into `results/<run_name>/`.
5. Add a row to `docs/ablation.md` and overwrite `docs/last_report.md`.

`src/train.py` reads three environment variables, so nothing needs editing
to change where it runs:

- `ECG_DATA_DIR` — dataset location (default: `<project root>/data/mit-bih-arrhythmia-database-1.0.0`)
- `ECG_OUT_DIR` — output root (default: `<project root>/runs`)
- `ECG_RUN_NAME` — run folder name (default: `baseline`)

`BASE_DIR` in `src/train.py` resolves to the **project root** (the parent of
`src/`), not to `src/` itself. Keep it that way.

---

## Conventions

- **ONE change per run.** Never batch two fixes into a single training run —
  a run that changes two things tells you nothing about either.
- **Commit format:** `step N: <what changed>`
- **After each run**, add a row to `docs/ablation.md`.
- **Every number in the paper or in any doc must trace to a
  `results/<run>/metrics.json`.** No remembered, estimated, or hand-copied
  numbers.
- **Never tune anything against DS2** — not hyperparameters, not thresholds,
  not the stopping epoch. DS2 is read exactly once per run, at the end.
- **At the end of EVERY task, overwrite `docs/last_report.md`.**

---

## Environment

- **No TensorFlow locally.** `src/train.py` cannot be run on this machine.
  The only local check available is:

  ```
  python -m py_compile src/train.py
  ```

- `tools/` scripts need only `wfdb` and `numpy` and do run locally:

  ```
  python tools/inspect_ds1.py
  ```

- Training happens on Kaggle (TF 2.20, P100).
