# Last report

**Task:** step 2 - validation instrumentation and model selection on macro-F1.
Also: added the step 1 and step 1b runs to `docs/ablation.md`.

**Date:** 2026-08-22

---

## What changed

**Validation instrumentation**

- New `ValidationMetrics` Keras callback (section 21). At each epoch end it
  predicts on the validation set and computes macro-F1, per-class F1, recall,
  precision and support, and the validation confusion matrix.
- It writes `val_macro_f1` and `val_f1_N` / `val_f1_S` / `val_f1_V` into the
  `logs` dict, so EarlyStopping, ReduceLROnPlateau and `History` all see them.
- It prints a compact per-epoch line:
  `epoch N  val macro-F1 x.xxxx   N-F1 ...  S-F1 ...  V-F1 ...`
- The callback is **first** in the callbacks list. Keras hands one shared
  `logs` dict to every callback in `on_epoch_end` in list order, so the
  metric must be written before the two callbacks that read it.

**Model selection**

- `EarlyStopping`: `monitor='val_macro_f1'`, `mode='max'`, `patience=10`,
  `restore_best_weights=True`.
- `ReduceLROnPlateau`: `monitor='val_macro_f1'`, `mode='max'`, `factor=0.5`,
  `patience=5`, `min_lr=1e-6`.

**Diagnostics after training** (new section 22B)

- `best_epoch` and `best_val_macro_f1`, taken as the argmax over the
  callback's own per-epoch records.
- Per-record breakdown over 207 / 220 / 223 separately: beat count, accuracy,
  macro-F1, and per-class support / recall / precision / F1 plus a confusion
  matrix. Printed, and written to `metrics.json` under `val_per_record`.
- To make that possible, DS1_VAL is now loaded **one record at a time** and
  concatenated, keeping a `val_record_ids` array. Same records, same order,
  same beats - verified byte-identical to the single call (see Verification).

**Persistence**

- `metrics.json` gains `best_epoch`, `best_val_macro_f1`,
  `early_stopping_fired` and `val_per_record`.
- `history.json` gains `val_metrics_per_epoch` with the full per-epoch
  validation record. The scalar `val_macro_f1` and `val_f1_*` series arrive
  automatically through Keras `History` because the callback wrote them into
  `logs`.
- Added `precision_recall_fscore_support` to the existing `sklearn.metrics`
  import.

**Ablation table** - `docs/ablation.md` now carries step 1 and step 1b, read
programmatically from their `metrics.json`. Both rows record that
EarlyStopping restored epoch 1; the step 1b note records that DS2
preprocessing also changed, so it is not a single-variable comparison against
the baseline.

Nothing else changed: no model architecture, loss, learning rate,
augmentation, RR feature, or split change. `DS1_VAL` is still
`['207','220','223']`.

---

## Files touched

- `src/train.py` - sklearn import, section 12 (val loading with provenance),
  section 21 (callback + monitors), new section 22B, metrics block, history
  block
- `docs/ablation.md` - regenerated with three rows
- `results/step1_patient_val/`, `results/step1b_rr_trainstats/` - committed
- `docs/last_report.md` - this file

---

## Step 1b predictions - scorecard

I made three falsifiable predictions last step. The runs are now in, so:

| # | prediction | outcome |
|---|---|---|
| 1 | steps/epoch stays 425 | **PASS** (indirect) - `train_distribution` is exactly N=40301 / S=670 / V=3105 as predicted, which fixes the post-augmentation count at 54,306 = 425 steps |
| 2 | prints mean `[274.881103515625, 275.1480712890625]`, std `[77.20227813720703, 75.38490295410156]` | **PASS** - `config.rr_norm_mean` and `config.rr_norm_std` match to all 16 digits |
| 3 | val_accuracy clears 0.8528 | **FAIL** - peaked at 0.7390, up only 0.0022 from step 1's 0.7368 |

**Prediction 3 failing matters more than the other two passing.** I argued the
RR scale mismatch was why validation sat below trivial. It was a real defect
and fixing it was correct, but it was **not** the cause: removing a 20.5%
scale error moved peak val_accuracy by 0.002. The below-trivial validation
performance has a different source, and step 2's per-record and per-class
instrumentation exists precisely to find it. Current suspects, in order:
the model over-predicting V (augmentation gives V x3 and focal loss pushes
minority recall), and the record 114 lead swap.

---

## Verification

**py_compile**

```
$ python -m py_compile src/train.py
$ echo $?
0
```

**No changed line inside a protected function** (AST line-span method, both
sides of the diff - 262 added, 20 deleted):

| function | lines (new) | added-side | deleted-side |
|---|---|---|---|
| `build_model` | 796-941 | CLEAN | CLEAN |
| `categorical_focal_loss` | 537-571 | CLEAN | CLEAN |
| `augment_segment` | 405-447 | CLEAN | CLEAN |
| `augment_training_data` | 454-530 | CLEAN | CLEAN |
| `extract_beats_from_record` | 277-349 | CLEAN | CLEAN |
| `fit_rr_norm` | 242-258 | CLEAN | CLEAN |
| `apply_rr_norm` | 261-270 | CLEAN | CLEAN |

Stronger check, since line spans alone can miss a same-line edit: every
function present in both versions was compared by `ast.dump`. **Zero
pre-existing functions changed.** The only functions added are `__init__` and
`on_epoch_end`, the two methods of the new callback class.

**Constants unchanged**

| constant | occurrences |
|---|---|
| `alpha=0.50` | 2 |
| `gamma=2.0` | 2 |
| `learning_rate=1e-4` | 1 |
| `multiplier = 6` | 1 |
| `PRE_SAMPLES = 90` | 1 |
| `POST_SAMPLES = 144` | 1 |

**Record literals unchanged**

```
DS1      IDENTICAL  sha256 9f20e3ac1758a312...
DS2      IDENTICAL  sha256 b8a3e6bbdeeec72a...
DS1_VAL  IDENTICAL  sha256 0d9df3612a6111a1...
```

Same hashes as steps 1 and 1b.

**Behavioural test without TensorFlow.** `load_dataset`,
`extract_beats_from_record`, `normalize_segment` and the `ValidationMetrics`
class were lifted out of the edited `src/train.py` by AST - so the test
exercises the shipped code - and run against the real MIT-BIH records with a
stubbed `tf.keras.callbacks.Callback` and a fake model.

`scikit-learn` is not installed on this machine (CLAUDE.md: tools need only
wfdb + numpy), so `precision_recall_fscore_support` and `confusion_matrix`
were replaced with numpy reference implementations matching sklearn's
`average=None` semantics. The test therefore validates **my** aggregation and
plumbing, not sklearn's arithmetic. I did not install packages into your
environment to get around this.

```
A) per-record load vs single load_dataset(DS1_VAL)
Loading record 207 ...
Loading record 220 ...
Loading record 223 ...
Loading record 207 ...
Loading record 220 ...
Loading record 223 ...
   shapes single (6494, 234)  per-record (6494, 234)
   arrays byte-identical: True -> PASS
   record ids length 6494 == beats 6494 : True
     207 -> 1858 beats
     220 -> 2046 beats
     223 -> 2590 beats

B) ValidationMetrics callback
   class lifted from src/train.py OK
  epoch   1  val macro-F1 0.2367   N-F1 0.4912  S-F1 0.0680  V-F1 0.1510
  epoch   1  val macro-F1 0.3068   N-F1 0.9205  S-F1 0.0000  V-F1 0.0000
   all-N model: macro-F1 0.3068 vs accuracy 0.8528
   [PASS] val_macro_f1 in logs
   [PASS] val_macro_f1 correct
   [PASS] macro-F1 == mean of 3 per-class F1
   [PASS] val_f1_N in logs & correct
   [PASS] val_f1_S in logs & correct
   [PASS] val_f1_V in logs & correct
   [PASS] pre-existing log keys preserved
   [PASS] records has 1 entry, epoch 1-based
   [PASS] record key val_macro_f1
   [PASS] record key val_f1
   [PASS] record key val_recall
   [PASS] record key val_precision
   [PASS] record key val_support
   [PASS] record key val_confusion_matrix
   [PASS] recall matches sklearn
   [PASS] precision matches sklearn
   [PASS] support sums to n beats
   [PASS] confusion matrix 3x3 and sums to n
   [PASS] all-N degenerate: no crash, S/V F1 == 0
   [PASS] all-N macro-F1 < all-N accuracy (the whole point)

C) per-record breakdown logic
   record 207:  1858 beats  support {'N': 1542, 'S': 106, 'V': 210}
   record 220:  2046 beats  support {'N': 1952, 'S': 94, 'V': 0}
   record 223:  2590 beats  support {'N': 2044, 'S': 73, 'V': 473}
   masks partition the val set exactly: True -> PASS
   per-record supports sum to val totals {'N': 5538, 'S': 273, 'V': 683} == {'V': 683, 'N': 5538, 'S': 273} : True -> PASS

OVERALL: PASS
```

Three things worth pulling out of that output:

- **Per-record loading is byte-identical** to `load_dataset(DS1_VAL, ...)` for
  X, y and RR. The provenance tracking changes nothing about the data.
- **Per-record supports match `docs/ds1_beat_counts.txt` exactly** (207:
  1542/106/210, 220: 1952/94/0, 223: 2044/73/473), an independent
  cross-check that the masks are correct.
- **On an all-N model the new metric reads macro-F1 0.3068 while accuracy
  reads 0.8528.** That gap is the entire justification for the change: the
  old signal rewards ignoring S, the new one does not.

**Falsifiable predictions for the next Kaggle run**

1. **The run trains past epoch 8.** Under `val_macro_f1` with `patience=10`,
   epoch 1 is no longer the best epoch - epoch 1's val macro-F1 is low because
   the model is near all-N at that point. I expect `best_epoch >= 5` and
   `early_stopping_fired` either false (ran all 40) or with
   `stopped_epoch > 15`. **If `best_epoch` comes back as 1 again, selecting on
   macro-F1 did not change the outcome and the problem is upstream of model
   selection.**
2. **`val_per_record` will show record 220 with `support.V = 0`** and
   therefore `recall.V = 0.0`, because 220 genuinely contains no V beats.
   That is a display artefact, not a model failure - see Problems.
3. **Test macro-F1 will exceed step 1b's 0.4742**, because the evaluated model
   will have trained for more than one epoch. I am *not* predicting it beats
   the baseline's 0.6358: the baseline had a leaked validation split and the
   full 22-record training pool, so it is not a fair target.

Prediction 1 is the real test of step 2.

---

## Commit

```
5c1b9d6  step 2: validation instrumentation, select on macro-F1
```

Pushed to `origin/main`. This report lands in a small follow-up commit.

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
│   ├── baseline/
│   │   └── metrics.json
│   ├── step1_patient_val/
│   │   ├── history.json
│   │   └── metrics.json
│   └── step1b_rr_trainstats/
│       ├── history.json
│       └── metrics.json
├── src/
│   └── train.py
└── tools/
    ├── ds1_beat_counts.json
    └── inspect_ds1.py
```

---

## Problems

1. **Record 220 has zero V beats**, so its per-record V recall will always
   print `0.0000` with `support 0`. That is `zero_division=0` doing its job,
   not the model failing on 220. Read `support` before reading `recall` in
   `val_per_record`. Worth knowing before this table gets read as evidence.

2. **`restore_best_weights` does not always restore.** In Keras, EarlyStopping
   restores the best weights only when it actually fires; if training reaches
   the final epoch the last epoch's weights are kept. With `patience=10` and
   `EPOCHS=40` that is now a live possibility. I did not change the behaviour,
   but I added `early_stopping_fired` to `metrics.json` and a printed warning
   so `best_epoch` is never silently mistaken for "the epoch that was
   evaluated". This is one field beyond your list of seven - flagging it as an
   addition rather than burying it.

3. **The callback adds a full validation forward pass per epoch.** DS1_VAL is
   6,494 beats against 54,306 training samples, so roughly 12% more inference
   work per epoch on top of the validation pass Keras already runs. Expect
   epochs to be modestly slower. Not a correctness issue, but it is real cost.

4. **`best_epoch` is computed from the callback's records, not read back out
   of EarlyStopping.** They should agree, since both take the argmax of the
   same `val_macro_f1` series. If a future run shows them disagreeing, trust
   EarlyStopping and treat it as a bug in section 22B.

5. **Steps 1 and 1b are not clean comparisons and the table now says so.**
   Both evaluated one-epoch models, so their macro-F1 drop measures the
   `val_loss` selection failure, not the split or the RR fix. Step 1b
   additionally changed DS2 preprocessing. The first genuinely comparable
   number will come from the next run.

6. Carried over, unchanged: augmentation still active (S x7, V x3, still
   violating hard constraint 2); record 114 lead swap still unfixed and now
   the leading suspect for the below-trivial validation;
   `tools/inspect_ds1.py` still writes its JSON into `tools/`; stale root
   `__pycache__/` still present.
