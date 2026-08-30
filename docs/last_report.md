# Last report — fix: restore sampler weights call broken by the E10 restructure

E10 crashed 15 minutes into a Kaggle run. Fixed, root-caused, and the class
of bug is now checked automatically.

## The bug was not what the traceback suggested

`weights_for_ratio` **was never removed.** It is still defined, at line 1590.
The E10 restructure moved the *call* to line 1585 — **five lines above its
own definition**. `src/train.py` is a script: module scope executes top to
bottom, so a call above its `def` raises `NameError` exactly as if the
function did not exist.

This distinction decides the fix and the checker. A check that asks only
"does this name resolve to a module-level def?" — the obvious reading of the
request — would have **passed** line 1585, because `weights_for_ratio` does
resolve to a module-level def. The audit has to enforce
definition-*before*-use for module-scope calls as a separate category. It
does, and that is the category that fired.

## The fix: moved the call, kept the derivation

Neither of the two options offered, strictly. Since the helper was never
removed, restoring it was unnecessary; and I did not inline
`[1/3, 1/3, 1/3]`.

`SAMPLER_RATIO = 1.0` stays where it is as configuration; the derived line
`SAMPLING_WEIGHTS = weights_for_ratio(SAMPLER_RATIO)` moved to immediately
after the definition.

**Why derived rather than the literal:** `weights_for_ratio(1.0)` returns
`[0.3333333333333333] * 3`, byte-for-byte what E9 recorded in
`config.sampling_weights`, so the sampler behaviour is provably identical to
E9's — it is computed by the same function every run since E6 has used. A
hand-typed literal would be numerically identical today but is a second
source of truth that can drift, which the project convention forbids.

Two hunks, both inside section 19B: 14 insertions, 1 deletion.

## Dangling-reference audit — 1585 was the only one

`tools/check_dangling.py` walks every `ast.Call` whose func is a plain
`Name` and reports three categories:

- **UNRESOLVED** — not a module binding, local, import or builtin.
- **TOO EARLY** — a module-scope call bound further down the file.
- **LOAD ORDER** (advisory) — a module-scope *read* of a later-bound name.

Calls inside a function to a function defined later are legal and are not
reported. Locals are over-approximated inside functions, biasing toward
silence there and keeping the module-scope result trustworthy.

**On the broken file:**

```
UNRESOLVED  0
TOO EARLY   1    line 1585  weights_for_ratio(...) is defined at line 1590
LOAD ORDER  0
```

**After the fix, across the whole repo:**

| file | unresolved | too early | load order |
|---|---|---|---|
| `src/train.py` | 0 | 0 | 0 |
| `tools/make_ablation.py` | 0 | 0 | 0 |
| `tools/inspect_ds1.py` | 0 | 0 | 0 |
| `tools/check_dangling.py` | 0 | 0 | 0 |

**Line 1585 was the only dangling reference in `src/train.py`.** Nothing
further down will fail this way — including everything after the sweep, in
section 23B and the DS2 evaluation, which is where another 15 minutes would
have been lost.

The checker has its own negative test: on a deliberately broken file it must
flag one of each category (it does — unresolved `removed_helper`, too-early
`helper`, load-order `CONST`) and exit 1, while staying silent and exiting 0
on a clean file exercising comprehensions, module-level loops, lambdas,
builtins, `try/except ImportError`, and a function calling a later-defined
function.

## Nothing before the crash changed

Verified by AST against `HEAD`, 26 items, all identical: `WINDOW_GRID`,
`ACCEPT_PRE/POST`, `WINDOW_WIDTHS`, `MAX_SPAN_SAMPLES`, `E10_EXPECTED`,
`E9_ACCEPTED`, `DS1`, `DS2`, `DS1_VAL`, `PRE_SAMPLES`, `POST_SAMPLES`,
`WAVELET_TARGET_FREQS_HZ`, `RR_FEATURE_NAMES`, `SAMPLER`, `SAMPLER_RATIO`,
`BATCH_SIZE`, and the definitions of `extract_beats_from_record`,
`load_dataset`, `build_model`, `make_balanced_dataset`, `weights_for_ratio`,
`assert_cnn_input`, `load_ds1_for_window`, `cwt_ricker`,
`summarize_segmentation_stats`.

So the population (35006/637/2877 and 5358/273/602, DS2 37377/1565/3187),
the acceptance rule, the window grid and the input shape (38520, 234, 9) are
all untouched.

## Standing change

`CLAUDE.md` now requires **both** local checks before every push, with the
reason recorded — that `py_compile` checks only syntax and that an
`ast.dump` function diff will report a moved call site as IDENTICAL, because
the function itself is unchanged. That combination is what let this reach
Kaggle.

## Predictions for E10 stand unchanged

239,171 parameters for every arm · `steps_per_epoch` 301 · train
35006/637/2877 · DS2 42,129 beats · selected arm's test S-F1 in [0.30, 0.38]
· a wider-than-234 arm wins on validation · V-F1 above 0.80.
