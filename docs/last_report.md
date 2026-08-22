# Last report

**Task:** reorganize the research repo, fix paths broken by the move, add
project context (CLAUDE.md), run the DS1 inspection, and start the ablation
log.

**Date:** 2026-08-22

---

## What changed

- Created the folder structure: `src/`, `tools/`, `notebooks/`, `docs/`,
  `results/`.
- Moved `train.py` -> `src/train.py`, `inspect_ds1.py` -> `tools/inspect_ds1.py`,
  and `metrics.json` -> `results/baseline/metrics.json`, all with `git mv`
  so history is preserved. `git log --follow src/train.py` still reaches the
  original `e6fb58b baseline: colleague's original script, unmodified`.
- Fixed `BASE_DIR` in `src/train.py`: it now resolves the **project root**
  (parent of `src/`) instead of `src/` itself, so the `DATA_DIR` and
  `ECG_OUT_DIR` fallbacks keep working after the move. This is the only
  change to that file - one line, plus one comment line. Nothing else was
  touched: no model, loss, hyperparameter, or data-loading change.
- `tools/inspect_ds1.py` needed **no** path fix. It already computed
  `dirname(dirname(abspath(__file__)))`, which was wrong at the repo root but
  is exactly right now that the file lives in `tools/`. Confirmed by running
  it: `DATA_DIR` resolved to the correct dataset folder.
- `.gitignore`: `*.png` -> `runs/**/*.png`, so figures under `results/` can be
  committed later. `data/` and `runs/` remain ignored (verified with
  `git check-ignore`).
- Added `CLAUDE.md` with the project description, the three hard constraints,
  the never-without-asking list, current baseline state, the folder tree,
  the Kaggle workflow, conventions, and the local environment limits.
- Ran `python tools/inspect_ds1.py`; full console output saved to
  `docs/ds1_beat_counts.txt`.
- Added `docs/ablation.md` with the header row and one filled row for the
  baseline. The numbers were read programmatically out of
  `results/baseline/metrics.json`, not typed by hand.
- Added `notebooks/.gitkeep` so the empty folder actually exists in git.

---

## Files touched

Moved (history preserved):

- `train.py` -> `src/train.py`
- `inspect_ds1.py` -> `tools/inspect_ds1.py`
- `metrics.json` -> `results/baseline/metrics.json`

Modified:

- `src/train.py` (one line: `BASE_DIR`)
- `.gitignore`

Created:

- `CLAUDE.md`
- `docs/ablation.md`
- `docs/ds1_beat_counts.txt`
- `docs/last_report.md`
- `notebooks/.gitkeep`

Created as a side effect of running the inspection:

- `tools/ds1_beat_counts.json` (written by `inspect_ds1.py` itself)

---

## Verification

**py_compile**

```
$ python -m py_compile src/train.py
$ echo $?
0
```

Passed, no output, exit code 0.

**The 7 required constants, all still present in `src/train.py`**

| constant | line | status |
|---|---|---|
| `alpha=0.50` | 506 | present |
| `gamma=2.0` | 507 | present |
| `learning_rate=1e-4` | 838 | present |
| `test_size=0.1` | 661 | present |
| `multiplier = 6` | 442 | present |
| `PRE_SAMPLES = 90` | 161 | present |
| `POST_SAMPLES = 144` | 162 | present |

All 7 confirmed by `grep -F` against `src/train.py` after the move.

**Diff scope check**

`git show --stat HEAD` reports `train.py => src/train.py | 3 +-` - a detected
rename with two insertions and one deletion, which is exactly the `BASE_DIR`
line plus its comment. No other line in the training script changed.

**Independent reproducibility cross-check**

`tools/inspect_ds1.py` counts DS1 totals of **N=45839, S=943, V=3788**.
`results/baseline/metrics.json` records a `train_distribution` of
**N=45839, S=943, V=3788**. Exact match, which independently confirms the
baseline run used the intended DS1 records and the intended beat-window
rejection rule.

**.gitignore behaviour**

```
$ git check-ignore -v data/x runs/y results/baseline/fig.png
.gitignore:1:data/	data/x
.gitignore:2:runs/	runs/y
```

`data/` and `runs/` still ignored; a PNG under `results/` is no longer ignored.

---

## Commit

```
8ccf5c6  chore: reorganize repo, add CLAUDE.md and ablation table
```

Pushed to `origin/main`:

```
To https://github.com/DiyarWalaa/ecg-arrhythmia-inter-patient.git
   b58d6b6..8ccf5c6  main -> main
```

`git status -sb` afterwards reports `## main...origin/main` with no divergence.

Note: this report file itself is written after that commit, so it lands in a
small follow-up commit.

---

## Tree

```
.
├── CLAUDE.md
├── .gitignore
├── data/                                   (gitignored, not in repo)
│   └── mit-bih-arrhythmia-database-1.0.0/
├── docs/
│   ├── ablation.md
│   ├── ds1_beat_counts.txt
│   └── last_report.md
├── notebooks/
│   └── .gitkeep
├── results/
│   └── baseline/
│       └── metrics.json
├── src/
│   └── train.py
└── tools/
    ├── ds1_beat_counts.json
    └── inspect_ds1.py
```

---

## inspect_ds1 output

```
DATA_DIR: C:\Users\athar\OneDrive\Documents\research\time_series\data\mit-bih-arrhythmia-database-1.0.0

 record       N       S       V   total    S share  leads
------------------------------------------------------------------------
    101    1858       3       0    1861     0.16%  ['MLII', 'V1']
    106    1506       0     520    2026     0.00%  ['MLII', 'V1']
    108    1738       4      17    1759     0.23%  ['MLII', 'V1']
    109    2491       0      38    2529     0.00%  ['MLII', 'V1']
    112    2536       2       0    2538     0.08%  ['MLII', 'V1']
    114    1819      12      43    1874     0.64%  ['V5', 'MLII']
    115    1952       0       0    1952     0.00%  ['MLII', 'V1']
    116    2301       1     109    2411     0.04%  ['MLII', 'V1']
    118    2164      96      16    2276     4.22%  ['MLII', 'V1']
    119    1542       0     444    1986     0.00%  ['MLII', 'V1']
    122    2475       0       0    2475     0.00%  ['MLII', 'V1']
    124    1535      31      47    1613     1.92%  ['MLII', 'V4']
    201    1634     128     198    1960     6.53%  ['MLII', 'V1']
    203    2528       2     444    2974     0.07%  ['MLII', 'V1']
    205    2570       3      71    2644     0.11%  ['MLII', 'V1']
    207    1542     106     210    1858     5.71%  ['MLII', 'V1']
    208    1585       2     992    2579     0.08%  ['MLII', 'V1']
    209    2620     383       1    3004    12.75%  ['MLII', 'V1']
    215    3194       3     164    3361     0.09%  ['MLII', 'V1']
    220    1952      94       0    2046     4.59%  ['MLII', 'V1']
    223    2044      73     473    2590     2.82%  ['MLII', 'V1']
    230    2253       0       1    2254     0.00%  ['MLII', 'V1']
------------------------------------------------------------------------
  TOTAL   45839     943    3788   50570

Lead check:
  record 114: channel 0 is V5, not MLII  -> leads = ['V5', 'MLII']

DS1 records ranked by S-beat count:
  209: 383 S beats  (3004 total)
  201: 128 S beats  (1960 total)
  207: 106 S beats  (1858 total)
  118: 96 S beats  (2276 total)
  220: 94 S beats  (2046 total)
  223: 73 S beats  (2590 total)
  124: 31 S beats  (1613 total)
  114: 12 S beats  (1874 total)

Saved: C:\Users\athar\OneDrive\Documents\research\time_series\tools\ds1_beat_counts.json
```

---

## Problems

Three things worth flagging. None of them blocked the task.

1. **`src/train.py` contains dead augmentation code that would violate hard
   constraint 2.** `augment_training_data()` (line ~422) expands S beats 6x
   and V beats 2x with synthetic augmentation, and `augment_segment()` (line
   ~373) generates the synthetic samples. **The function is never called** -
   I grepped for every call site and there is none, and the baseline
   `train_distribution` matching the raw DS1 counts exactly confirms it did
   not run. It is left in place because you asked for no changes to
   `train.py` beyond the path fix, but it is a live trap: wiring it up would
   silently break the no-data-expansion constraint. Flagged in CLAUDE.md
   under "Landmine". Say the word and I will remove it in its own commit.

   Related: the `multiplier = 6` you asked me to confirm is *inside* that dead
   function. It is present as required, but it is not currently affecting any
   training.

2. **`tools/inspect_ds1.py` writes `ds1_beat_counts.json` next to itself, into
   `tools/`, rather than into `docs/` or `results/`.** That is the script's own
   hardcoded behaviour and I did not change it, since the path is not broken -
   just untidy for a folder meant to hold code. I committed the JSON so the
   counts stay traceable. Worth a one-line fix to redirect it to `docs/` if
   you want.

3. **A stale `__pycache__/` from before the move still sits at the repo root**,
   holding bytecode for the old `train.py` location. It is gitignored and
   harmless, so I left the filesystem alone rather than deleting anything
   unasked.

Also, minor and non-blocking: `runs/**/*.png` is technically redundant, since
the existing `runs/` rule already ignores everything beneath it. I used the
exact pattern you specified.
