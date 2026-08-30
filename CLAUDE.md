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

1. **Augmentation is GONE as of E6, and hard constraint 2 now holds.**
   It was removed at step 4, **restored at E0** to re-anchor on the step 3
   training setup, and **removed again at E6** - this time paired with a
   replacement balancing mechanism, which is what the earlier removal
   lacked. E6 balances by **sampling** rather than by duplicating: a
   `balanced_batch` sampler draws N/S/V at `[1/3, 1/3, 1/3]`, and the loss
   is plain `categorical_crossentropy`. Verified in
   `results/E6_balanced_sampling/metrics.json`: `config.oversampling` is
   `False`, `config.focal_loss_used` is `False`, `config.sampler` is
   `balanced_batch`. E7 is an E6 derivative and used the same sampler by
   construction, but its artefact was lost, so that is not independently
   confirmable.

   **E0 through E5 violated hard constraint 2** (`config.oversampling` is
   `True` in all six artefacts), deliberately and with two known defects:
   `augment_training_data()` expanded **S x7, V x3** while copying the RR
   feature vector **unchanged** across every duplicate - worse than at step
   3, since those are the five patient-relative ratios rather than two raw
   intervals - and the `np.roll` shift moved the R-peak off its aligned
   position. **Read every number from baseline through E5 as produced under
   data expansion.** E6 is the first clean run since step 5, and it is also
   the best run, so nothing is being given up by keeping the constraint.

   **Class balancing moved from the data to the sampler, never to the
   loss.** Through E5, `FOCAL_ALPHA` was the **scalar 0.50**, which scales
   the whole loss and rebalances nothing; the step 4 per-class alpha vector
   and the step 6 `BETA_GRID` sweep were both reverted at E0. Step 6 later
   swept `BETA` properly and found it spans only 0.0606 across the entire
   grid - see standing finding 2 under Current state. **Do not reach for a
   loss-reweighting scheme again.**

   **History, so old numbers are read correctly.** The baseline ran 449
   steps/epoch at batch 128 = 57,425 samples = 41255 N + 849x7 S + 3409x3 V;
   without augmentation it would have been 356.

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
truth. **Check its `source` column before quoting a number.** Fourteen rows
are read programmatically from a `results/<run>/metrics.json`; two - **step
6** and **E7** - are marked `console log`, because those Kaggle sessions
expired before the artefact was written. Console rows are reconciled against
their own printed confusion matrix and against DS2, but they cannot be
re-derived from `results/` and must be re-run before anything from them goes
in the paper.

Best test macro-F1 so far is **E6** (`5b3b203`) - balanced 1:1:1 batch
sampling with plain cross-entropy, at 239,171 parameters, `DS1_VAL =
['207','220','223']`, `best_epoch` 6.

| run | macro-F1 | S F1 | S recall | S precision | V F1 | accuracy |
|---|---|---|---|---|---|---|
| E6 tuned | **0.7372** | 0.3653 | 0.3442 | 0.3892 | 0.8792 | 0.9395 |
| E6 argmax | 0.7263 | 0.3641 | 0.3388 | 0.3934 | 0.8503 | 0.9352 |
| E2 argmax | 0.7178 | 0.2686 | 0.1972 | 0.4214 | 0.9111 | 0.9503 |
| E7 argmax | 0.7114 | 0.3061 | 0.3393 | 0.2789 | 0.8685 | 0.9261 |
| step 2 | 0.6800 | 0.1942 | 0.1280 | 0.4024 | 0.8718 | 0.9476 |

E6's **tuned** row is the headline number; its argmax row is the one to
compare against any run that does not tune thresholds. Step 2 (0.6800) was
the best for a long stretch and is kept above as the pre-wavelet reference.

**Blocking problem, restated after E6.** S recall is **0.3442** against
0.759 (de Chazal 2004) and 0.894 (Zhou 2021), while S precision **0.3892**
remains in line with their 0.385 and 0.415. What changed is *why*. Through
E1 the model barely called S at all - 584 S predictions against 1836 true S
beats - so the decision rule was the obvious suspect. E6 predicts S **1581**
times against those same 1836, i.e. at very nearly the right frequency, and
still only 622 of them are correct. The model is no longer refusing to call
S; **it cannot tell S from N.** That is a representation problem, not a
threshold problem, and the four findings below say which levers are already
spent.

### Standing findings - established, do not re-run

1. **The sampler is saturated (E7, console row).** Sweeping the S:N drawing
   ratio over 1, 2, 3, 4 moved S recall by **+0.0005** (0.3388 -> 0.3393)
   while S precision fell **0.3934 -> 0.2789** and N-called-S rose 938 ->
   1526. Validation macro-F1 spanned only 0.0178 across the whole grid, so
   the selection was inside the noise. More S exposure buys nothing; it only
   trades precision away. E6's 1:1:1 is the right setting.

2. **The focal-alpha BETA exponent is not a lever (step 6, console row).**
   The grid `[0.0, 0.25, 0.41, 0.50]` spans **0.0606** in validation
   macro-F1 end to end, and the flat `[1,1,1]` control lands within 0.0603
   of the winner. Reweighting the loss by class frequency does not move this
   problem. E6 dropped focal loss entirely for plain cross-entropy and beat
   every focal run.

3. **Capacity is not the constraint (E4).** Cutting the network from 239,171
   to 16,283 parameters destroyed the minority class outright: the model
   predicted S **11 times in total** against 1836 true S beats, S-F1 0.0000,
   S recall 0.0000. Combined with E5 (a direct RR skip to the output layer
   changed nothing), S is limited by neither overfitting nor RR attenuation.

4. **Threshold tuning transfers only when S is left at parity with N.**
   Coordinate ascent on validation has been run four times, and the sign of
   the test-set change tracks one thing - whether `w_S` was raised above
   `w_N`:

   | run | w (N, S, V) | val change | **test change** |
   |---|---|---|---|
   | E1 | 1.0, **4.0**, 4.0 | +0.0588 | **-0.0495** |
   | E2 | 1.0, **1.4142**, 0.5 | +0.0265 | **-0.0846** |
   | E6 | 1.0, **1.0**, 0.3536 | +0.0017 | **+0.0108** |
   | E7 | 1.0, **1.4142**, 4.0 | - | **-0.0170** |

   The only vector that transferred is E6's, which **does not upweight S at
   all** and downweights V to 0.354. Every run that raised `w_S` lost on
   test, including E2 at a mild 1.41x - which lost the most of any of them.
   Note also that the bigger the validation gain, the worse the test result:
   tuning two minority weights against **273 validation S beats** overfits
   the validation set. Do not tune `w_S` upward again without a larger
   validation S pool.

**What is left.** The four spent levers are all *rebalancing* levers - loss
weights, sampling ratios, decision thresholds, capacity. None of them
addresses the fact that the representation does not separate S from N. The
remaining directions are the input representation and the RR context window,
not another rebalancing knob.

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
  Run BOTH local checks before every push:

  ```
  python -m py_compile src/train.py        # syntax
  python tools/check_dangling.py           # names that will not resolve
  ```

  **`py_compile` alone is not enough, and neither is an `ast.dump` diff.**
  E10 crashed 15 minutes into a Kaggle run on
  `SAMPLING_WEIGHTS = weights_for_ratio(SAMPLER_RATIO)` - a call that a
  restructure had moved ABOVE the definition it calls. `py_compile` passed
  (valid syntax); the `ast.dump` function diff passed and reported
  `weights_for_ratio` IDENTICAL, correctly, because the function was never
  touched. Only the call site had moved.

  `tools/check_dangling.py` reports three things and exits non-zero on the
  first two:

  - **UNRESOLVED** - a called name that is not a module binding, a local, an
    import or a builtin.
  - **TOO EARLY** - a call in MODULE scope whose target is bound further
    down the file. `src/train.py` executes top to bottom, so for module
    scope "defined later" fails exactly like "not defined". This is the
    E10 bug, and a resolve-only checker would have passed it.
  - **LOAD ORDER** (advisory) - a module-scope read of a name bound later.

  A call inside a function to a function defined later is legal and is not
  reported. The checker has negative tests: it must flag all three
  categories on a deliberately broken file and stay silent on a clean one.

- `tools/` scripts need only `wfdb` and `numpy` and do run locally:

  ```
  python tools/inspect_ds1.py
  ```

- Training happens on Kaggle (TF 2.20, P100).
