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

1. **Augmentation was REMOVED in step 4.** It is no longer active.
   `augment_training_data()` and `augment_segment()` are still defined in
   `src/train.py` sections 9 and 10, but they are **dead code** - marked
   `[DEPRECATED]`, with zero call sites (verified by AST call-graph, not
   grep). Section 18 passes the training arrays straight through. Do not
   re-enable them: synthetic beats violate hard constraint 2.

   Class balancing now happens **in the loss**, via a per-class
   `FOCAL_ALPHA` vector derived at runtime from the DS1_TRAIN counts as
   `(1 / count_c) ** FOCAL_BETA`, rescaled to sum to `NUM_CLASSES`.

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

3. **DS1_VAL changed at step 5**, from 3 records to 5:
   `['207','220','223']` -> `['106','118','207','220','223']`.
   With 3 records the selection signal was too noisy to trust - val
   macro-F1 swung 0.7288 -> 0.5791 between consecutive epochs and the
   selected peak sat 0.1627 above the surrounding plateau for exactly one
   epoch. Record 220 also has zero V beats, so validation V-F1 rested on
   only two records.
   **Consequence for the ablation table: rows baseline through step 4 used
   the 3-record validation set and are NOT directly comparable to step 5
   onward.** Training also shrank from 44,076 to 39,774 beats, so
   `FOCAL_ALPHA` shifts slightly - it is derived from DS1_TRAIN counts.
   The selection rule is recorded in `metrics.json` under
   `val_selection_rule` and must not be re-derived casually: changing
   `DS1_VAL` again breaks comparability with everything before it.

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

**Blocking problem:** S beats are still misclassified, but the failure
mode moved at step 3. It is no longer purely S->N: the patient-relative RR
features cut S->N from 1350 to 692 of 1836, but those beats went to **V**
(S->V 251 -> 866) rather than to S. Fixing S sensitivity without collapsing
N precision or V precision is the current research objective.

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
