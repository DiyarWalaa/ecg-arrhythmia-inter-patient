# Last report — console-sourced rows for step 6 and E7, and a CLAUDE.md refresh

Two pieces of work, landing in one commit: recording two runs whose artefacts
were lost, then bringing `CLAUDE.md` back in line with the regenerated table.

---

## Part 1 — step 6 and E7 recorded from console logs

Two Kaggle sessions expired before `metrics.json` could be written.
`tools/make_ablation.py` learned a second, explicitly lower-provenance kind of
row, and `docs/ablation.md` was regenerated. No existing row moved and no
existing number changed.

| step | run | commit | macro-F1 | S F1 | accuracy | source |
|---|---|---|---|---|---|---|
| 6 | `step6_beta_sweep` | `25229d9` | 0.5408 | 0.0963 | 0.8919 | console log |
| E7 | `E7_sampler_ratio` | `1339261` | 0.7114 | 0.3061 | 0.9261 | console log |

The table gained a **`source`** column: 14 rows read `` `metrics.json` ``,
these two read **`console log`**. The header opens by telling the reader to
check that column and states that these rows cannot be re-derived from
`results/` and should not be quoted in the paper without re-running them.

**Commit hashes corrected.** The hashes supplied with the logs, `d7f6e02` and
`c3042ad`, are the `docs: last_report` commits. Every other row carries the
**code** commit, so the rows use `25229d9` and `1339261`; the supplied hash is
recorded in each note as the docs commit that followed.

### Console rows are gated, not trusted

There is no artefact to hash, so the duplicate-copy gate cannot protect these
rows. `check_console_run()` substitutes four checks, each confirmed to fire on
a deliberately corrupted spec:

1. **Self-consistency.** `report_from_cm()` recomputes the full per-class
   report from the printed confusion matrix; every scalar the console *also*
   printed must agree to within 5e-5. Ten scalars checked, all agree.
2. **Same test set.** Per-class support must equal the DS2 the archived runs
   score — N=44233 / S=1836 / V=3220, derived by `ds2_support()` and required
   consistent across artefacts first.
3. **Reproduction claims verified, not asserted.** Step 6's matrix is checked
   equal to `results/step5_bigger_val/metrics.json`; E7's `ratio 1.0` arm is
   checked equal to E6's selection (0.5540 at epoch 6).
4. **Sweep integrity.** Exactly one setting selected, and its validation score
   must equal the row's `best_val_macro_f1`.

### Findings recorded

- **Step 6** — the BETA grid spans only 0.0606 end to end and the flat
  `[1,1,1]` control lands within 0.0603 of the winner. The test row reproduces
  step 5's confusion matrix exactly (same BETA, same seed), so it is a
  determinism confirmation, not an independent measurement.
- **E7** — failed its criterion. 2x S exposure moved S recall +0.0005 while S
  precision fell 0.3934 → 0.2789. Validation spanned 0.0178 across the grid.

---

## Part 2 — `CLAUDE.md` "Current state" refreshed

Was stale: it named step 2 (0.6800) as best and its table stopped at E0.

**E6 (`5b3b203`) is now recorded as the best run** — balanced 1:1:1 batch
sampling, plain cross-entropy, 239,171 parameters, `best_epoch` 6:
macro-F1 **0.7372 tuned** / 0.7263 argmax, S-F1 0.3653, S recall 0.3442,
S precision 0.3892, V-F1 0.8792, accuracy 0.9395. All verified against
`results/E6_balanced_sampling/metrics.json` — they are the **tuned** column.

**The blocking problem was restated.** Through E1 the model barely called S at
all (584 predictions against 1836 true), so the decision rule was the suspect.
E6 predicts S **1581** times — very nearly the right frequency — and still
only 622 are correct. The model is no longer refusing to call S; it cannot
tell S from N. That is a representation problem, not a threshold problem.

### One standing finding was corrected before recording

The finding as given — *"threshold tuning transfers only with MILD weights
(S ×1.41, V unchanged); aggressive weights hurt in E1 and E2"* — does not
survive the artefacts. `S ×1.41` is close to the vector that failed **worst**,
and no transferring run left V unchanged:

| run | w (N, S, V) | val change | **test change** |
|---|---|---|---|
| E1 | 1.0, **4.0**, 4.0 | +0.0588 | **-0.0495** |
| E2 | 1.0, **1.4142**, 0.5 | +0.0265 | **-0.0846** |
| E6 | 1.0, **1.0**, 0.3536 | +0.0017 | **+0.0108** |
| E7 | 1.0, **1.4142**, 4.0 | - | **-0.0170** |

The pattern that fits all four is **whether `w_S` was raised above `w_N`**.
The only vector that transferred is E6's, which does not upweight S at all and
downweights V to 0.354. E2 upweighted S by exactly 1.41x and lost the most of
any run. Secondary signal: the larger the validation gain, the worse the test
result — tuning two minority weights against 273 validation S beats overfits
the validation set. Recorded in that form.

The other three findings (sampler saturated, BETA not a lever, capacity not
the constraint) matched the artefacts and were recorded as given, with the
supporting numbers filled in from `results/`.

### Known fact 1 also refreshed

It declared augmentation active and hard constraint 2 violated, and said
`FOCAL_ALPHA` was the scalar 0.50 — all true through E5, all false at E6, and
directly contradicting the section above. Verified across artefacts:
`config.oversampling` is `True` for E0–E5 and `False` at E6, where
`focal_loss_used` is `False`, `loss` is `categorical_crossentropy` and
`sampler` is `balanced_batch`. **Hard constraint 2 holds again as of E6**, and
the run that restored it is also the best run. The reading guidance for older
numbers (treat baseline→E5 as produced under data expansion; never infer
augmentation from `train_distribution`) was kept.

---

## Verification

- `python -m py_compile tools/make_ablation.py` — clean.
- `python tools/make_ablation.py` — "wrote docs/ablation.md (16 runs, gate passed)".
- `python tools/make_ablation.py --check` — up to date.
- Seven negative tests on the console gates all fire; both clean specs pass.
- Every number written into `CLAUDE.md` was read back from a
  `results/<run>/metrics.json`, except the two console rows and E4's 16,283
  parameter count, which is not in E4's artefact and comes from the E4 row
  description already in the ablation log.

## Open

- **Step 6 and E7 have no archived artefact.** If either result is going in
  the paper, re-run it.
- E7's sampler configuration cannot be confirmed from `results/` for the same
  reason; it is an E6 derivative by construction.
