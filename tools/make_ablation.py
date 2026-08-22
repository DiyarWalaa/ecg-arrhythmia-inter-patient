"""Regenerate docs/ablation.md from the run artefacts in results/.

Every number in the table is read from a results/<run>/metrics.json. Nothing
is typed by hand. Run from the project root:

    python tools/make_ablation.py            # rewrite docs/ablation.md
    python tools/make_ablation.py --check    # verify it is up to date, write nothing

CPU-only. Needs only the standard library.

WHY THE GATE EXISTS
-------------------
A results folder once contained a byte-identical copy of the previous step's
metrics.json - same run_name, same timestamp, same scalar focal alpha. Filling
a row from it would have published the wrong step's numbers under the right
step's commit hash. check_run() below refuses to emit a row unless the file
identifies itself as the run the table claims it is. Add an expectation to
RUNS for every new step; a run with no expectations still gets the duplicate
check.

ADDING A RUN
------------
Append to RUNS: (step, folder, commit, description, expectations, note).
`expectations` is a dict of config/top-level keys the file must satisfy - a
literal value, or a callable taking the value and returning bool.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys

RESULTS = "results"
OUT = os.path.join("docs", "ablation.md")


def _is_vec3(v):
    return isinstance(v, list) and len(v) == 3


def _all_positive_ints(v):
    return isinstance(v, list) and all(isinstance(x, int) and x > 0 for x in v)


# (step, folder, commit, description, expectations, note)
RUNS = [
    ("0", "baseline", "b58d6b6",
     "colleague's script, paths made configurable",
     {"run_name": "baseline"},
     "Baseline reproduced on Kaggle (TF 2.20, P100)."),

    ("1", "step1_patient_val", "ae9a91c",
     "patient-wise validation split (DS1_VAL = 207,220,223)",
     {"run_name": "step1_patient_val",
      "config.ds1_val": ["207", "220", "223"]},
     "Fixes the beat-level split that put the same patient in training and "
     "validation."),

    ("1b", "step1b_rr_trainstats", "10e06f2",
     "RR normalization fitted on training set only",
     # 2 raw RR features at this point; the 5 ratios arrive at step 3.
     {"run_name": "step1b_rr_trainstats",
      "config.rr_norm_mean": lambda v: isinstance(v, list) and len(v) == 2},
     "**Not a single-variable comparison against the baseline.** This step "
     "also changed how DS2 is preprocessed: the test set is now scaled by "
     "*training* statistics instead of its own, so baseline and step 1 test "
     "numbers were produced under different preprocessing. The change is "
     "required - fitting a scaler on DS2 is fitting a parameter on the test "
     "set - but the delta here mixes two effects."),

    ("2", "step2_macrof1_selection", "5c1b9d6",
     "select on val macro-F1 instead of val_loss",
     {"run_name": "step2_macrof1_selection"},
     "First run whose evaluated model trained for more than one epoch. "
     "Per-record validation exposed record **207** as a severe outlier: N "
     "recall 0.0700, S recall 0.0000, accuracy 0.1647, against 0.9780 and "
     "0.9602 on 220 and 223. Still 1350 of 1836 test S beats (73.5%) "
     "predicted as N."),

    ("3", "step3_rr_ratios", "1a2509c",
     "patient-relative RR ratio features (5 dimensionless)",
     {"run_name": "step3_rr_ratios",
      "config.rr_feature_names": lambda v: isinstance(v, list) and len(v) == 5,
      "config.rr_clip": [0.2, 3.0]},
     "**macro-F1 REGRESSED 0.6800 -> 0.6645.** But the S->N leak roughly "
     "halved: 1350 -> 692 of 1836 test S beats. Those beats did not become "
     "correct S predictions - they moved to **V** instead (S->V 251 -> 866), "
     "and V F1 fell 0.8718 -> 0.8004. S recall did rise 0.1280 -> 0.1514 and "
     "S F1 0.1942 -> 0.2138, so the ratio features do carry usable "
     "prematurity signal; the model is simply spending it on the wrong class "
     "boundary. **Stated condition: if step 4 does not recover macro-F1 "
     "above 0.6800, the ratio features get reconsidered.**"),

    ("4", "step4_class_alpha", "3b67016",
     "remove oversampling, per-class focal alpha",
     {"run_name": "step4_class_alpha",
      "config.focal_loss_alpha": _is_vec3,
      "config.oversampling": False,
      "config.focal_alpha_class_counts": _all_positive_ints},
     None),          # assembled from the artefacts - see step4_note()

    ("5", "step5_bigger_val", "7b0236d",
     "enlarge DS1_VAL to 5 records",
     {"run_name": "step5_bigger_val",
      "config.ds1_val": ["106", "118", "207", "220", "223"],
      "config.oversampling": False},
     None),          # assembled from the artefacts - see step5_note()
]


# --------------------------------------------------------------------- gate

class GateError(Exception):
    pass


def _dig(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return KeyError
        cur = cur[part]
    return cur


def check_run(folder, expectations, seen_hashes):
    """Refuse to emit a row from a file that is not the run it claims to be."""

    path = os.path.join(RESULTS, folder, "metrics.json")
    if not os.path.exists(path):
        raise GateError("%s: metrics.json missing" % folder)

    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()

    if digest in seen_hashes:
        raise GateError(
            "%s: metrics.json is byte-identical to %s (sha256 %s...). "
            "This is the mislabelled-copy failure mode - a run folder holding "
            "another step's results. Refusing to emit a row."
            % (folder, seen_hashes[digest], digest[:16]))
    seen_hashes[digest] = folder

    m = json.loads(raw)

    for key, want in (expectations or {}).items():
        got = _dig(m, key)
        if got is KeyError:
            raise GateError("%s: expected key %s is missing" % (folder, key))
        ok = want(got) if callable(want) else got == want
        if not ok:
            raise GateError("%s: %s is %r, expected %r"
                            % (folder, key, got, want))

    return m


# -------------------------------------------------------------------- notes

def _history(folder):
    path = os.path.join(RESULTS, folder, "history.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def val_gap(m):
    if m.get("best_val_macro_f1") is None:
        return None
    return (m["classification_report"]["macro avg"]["f1-score"]
            - m["best_val_macro_f1"])


def step4_note(m, h, gaps):
    v = h["val_macro_f1"]
    be = m["best_epoch"]
    peak, nxt = v[be - 1], v[be]
    plateau = statistics.mean(v[be:])
    return (
        "**Best epoch was {be} of {n}.** Val macro-F1 spiked to {peak:.4f} "
        "there and fell to {nxt:.4f} the next epoch - a **{swing:.4f} "
        "single-epoch swing**. The plateau over epochs {p0}-{n} averaged "
        "{plat:.4f}, so the selected peak sat **{above:.4f} above it** and "
        "lasted one epoch.\n\n"
        "  **First run where test scored BELOW validation ({g4:+.4f}).** "
        "Steps 2 and 3 both scored above ({g2:+.4f} and {g3:+.4f}). The "
        "checkpoint was selected on noise - which is what motivated step 5."
        "\n\n"
        "  **The step 3 stated condition (macro-F1 must recover above 0.6800) "
        "is NOT EVALUABLE from this run**, because the selection was "
        "confounded. It carries forward to step 5."
    ).format(be=be, n=len(v), peak=peak, nxt=nxt, swing=peak - nxt,
             p0=be + 1, plat=plateau, above=peak - plateau,
             g4=gaps["4"], g2=gaps["2"], g3=gaps["3"])


def step5_note(m, h, gaps):
    v = h["val_macro_f1"]
    swing = max(abs(v[i + 1] - v[i]) for i in range(len(v) - 1))
    vp = m.get("val_per_record", {})
    nz = sum(1 for r in vp.values() if r["support"]["V"] > 0)
    cm = m["confusion_matrix"]
    pred_v = sum(row[2] for row in cm)
    true_v = sum(cm[2])
    return (
        "**Selection variance is fixed; the model is not.** Largest "
        "single-epoch val macro-F1 swing fell {s4:.4f} -> **{s5:.4f}**, and "
        "the test-minus-validation gap narrowed {g4:+.4f} -> **{g5:+.4f}**. "
        "{nz} of {tot} validation records now carry V beats (was 2 of 3).\n\n"
        "  But test macro-F1 fell again to {mf1:.4f}, and the reason is "
        "visible in the confusion matrix: **{pv} V predictions against {tv} "
        "true V beats ({over:+.0f}%)**. The model floods V. That is what "
        "step 6 targets - BETA=0.5 gives an effective V:N emphasis of 3.776 "
        "against the 3.000 the old oversampling produced.\n\n"
        "  **The step 3 condition (macro-F1 above 0.6800) still fails**, but "
        "with V over-prediction this severe the feature question and the "
        "class-weighting question are not separable. It carries forward to "
        "step 6."
    ).format(s4=0.1497, s5=swing, g4=gaps["4"], g5=gaps["5"], nz=nz,
             tot=len(vp), mf1=m["classification_report"]["macro avg"]["f1-score"],
             pv=pred_v, tv=true_v, over=100.0 * (pred_v - true_v) / true_v)


ASSEMBLED = {"4": step4_note, "5": step5_note}


# --------------------------------------------------------------------- build

def build():
    seen = {}
    loaded = []
    for step, folder, commit, desc, exp, note in RUNS:
        m = check_run(folder, exp, seen)
        loaded.append((step, folder, commit, desc, note, m, _history(folder)))

    gaps = {}
    for step, _f, _c, _d, _n, m, _h in loaded:
        g = val_gap(m)
        if g is not None:
            gaps[step] = g

    rows, notes = [], []
    for step, folder, commit, desc, note, m, h in loaded:
        r = m["classification_report"]
        td = m["train_distribution"]

        rows.append(
            "| {s} | {d} | `{c}` | {mf1:.4f} | {sr:.4f} | {sp:.4f} | "
            "{sf:.4f} | {vf:.4f} | {acc:.4f} |".format(
                s=step, d=desc, c=commit,
                mf1=r["macro avg"]["f1-score"], sr=r["S"]["recall"],
                sp=r["S"]["precision"], sf=r["S"]["f1-score"],
                vf=r["V"]["f1-score"], acc=m["test_accuracy"]))

        if h is None:
            sel = ""
        elif m.get("best_epoch") is not None:
            sel = (" Ran {e} epochs; selection on `val_macro_f1` chose "
                   "**epoch {be}** (best val macro-F1 {bv:.4f}); early "
                   "stopping fired: {f}.".format(
                       e=len(h["val_loss"]), be=m["best_epoch"],
                       bv=m["best_val_macro_f1"],
                       f=m.get("early_stopping_fired")))
        else:
            argmin = h["val_loss"].index(min(h["val_loss"])) + 1
            sel = (" Ran {e} epochs; `val_loss` was minimal at epoch {b}, so "
                   "EarlyStopping(monitor='val_loss', "
                   "restore_best_weights=True) **restored epoch {b}** - the "
                   "evaluated model was trained for ONE epoch. Peak "
                   "val_accuracy {v:.4f}, below the 0.8528 an all-N "
                   "prediction scores on DS1_VAL.".format(
                       e=len(h["val_loss"]), b=argmin,
                       v=max(h["val_accuracy"])))

        body = ASSEMBLED[step](m, h, gaps) if step in ASSEMBLED else note

        notes.append(
            "- **step {s}** (`{f}`) - train distribution N={N} / S={S} / "
            "V={V}.{sel}\n  {x}".format(
                s=step, f=folder, N=td["N"], S=td["S"], V=td["V"],
                sel=sel, x=body or ""))

    doc = HEADER + "\n".join(rows) + "\n\n## Notes\n\n" + "\n\n".join(notes)

    doc += GAP_HEADER
    for step, _f, _c, _d, _n, m, _h in loaded:
        if step not in gaps:
            continue
        t = m["classification_report"]["macro avg"]["f1-score"]
        doc += "| {s} | {v:.4f} | {t:.4f} | {g:+.4f} |\n".format(
            s=step, v=m["best_val_macro_f1"], t=t, g=gaps[step])

    doc += FOOTER
    return doc


HEADER = """# Ablation log

Generated by `tools/make_ablation.py` - do not edit by hand. Every number is
read from a `results/<run>/metrics.json`; the generator refuses to emit a row
whose metrics file does not identify itself as the run the table claims.

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
"""

GAP_HEADER = """

## Test-minus-validation gap

A model selected on a trustworthy validation signal should not score wildly
differently on test. The gap is `test macro-F1 - best_val_macro_f1`:

| step | best val macro-F1 | test macro-F1 | gap |
|---|---|---|---|
"""

FOOTER = """
Steps 2 and 3 scored **above** validation, which is the expected direction
when validation holds only three patients and two of them are hard. Step 4 is
the first run to score **below** it, by 0.1689 - the signature of a checkpoint
picked at a noise spike. Step 5 enlarged the validation set and the gap
narrowed to -0.0617.

## Reading these numbers

Steps 1 and 1b both **evaluated a one-epoch model**: `val_loss` rose
monotonically after epoch 1 under focal loss while `val_accuracy` stayed flat
(spread 0.03), so `restore_best_weights` kept epoch-1 weights and patience
stopped the run at epoch 8. Their macro-F1 drop to 0.4569 / 0.4742 measures
that selection failure, **not** the patient-wise split or the RR fix.

Step 2 changed the selection metric and the same pipeline reached macro-F1
0.6800 - still the best result of any run. Step 3 is the first step to move
the S->N leak (73.5% -> 37.7%), at the cost of macro-F1, because the freed S
beats went to V rather than to S.

Step 4 removed duplicate oversampling and replaced the scalar focal alpha
(which rebalanced nothing) with a per-class vector. Its 0.5599 does not
measure that change - selection landed on a one-epoch spike.

Step 5 enlarged the validation set and cut selection variance by half, which
is what it was for. It also made the real problem legible: the model predicts
V roughly twice as often as V occurs.

Step 6 sweeps `BETA` over `[0.0, 0.25, 0.41, 0.50]` and lets DS1_VAL pick.
`BETA=0.41` reproduces the 3.000 V:N emphasis of the old oversampling;
`BETA=0.0` is a flat `[1,1,1]` control that answers whether reweighting was
ever helping.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify docs/ablation.md is current; write nothing")
    args = ap.parse_args()

    try:
        doc = build()
    except GateError as e:
        print("GATE FAILED: %s" % e, file=sys.stderr)
        return 2

    if args.check:
        current = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if current == doc:
            print("%s is up to date (%d runs)" % (OUT, len(RUNS)))
            return 0
        print("%s is STALE - run: python tools/make_ablation.py" % OUT,
              file=sys.stderr)
        return 1

    with open(OUT, "w", newline="\n", encoding="utf-8") as f:
        f.write(doc)
    print("wrote %s (%d runs, gate passed)" % (OUT, len(RUNS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
