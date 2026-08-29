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
Append to RUNS: (step, folder, commit, description, spec, note).

For an ordinary run, `folder` is the results/ directory and `spec` is a dict
of config/top-level keys the metrics.json must satisfy - a literal value, or
a callable taking the value and returning bool.

CONSOLE-LOG ROWS
----------------
Two Kaggle sessions expired before metrics.json was written, so step 6 and E7
have no artefact to read. Those rows set `folder = None` and put the run's
console output in `spec` (see CONSOLE_SPEC below). They are marked
**console log** in the source column of the table and are lower-provenance
than every other row - do not use them as a reference for a later run.

They are NOT unchecked. check_console_run() recomputes the full per-class
report from the confusion matrix the console printed and refuses the row
unless every scalar the console *also* printed agrees to within 5e-5, and
unless the per-class support matches DS2 as measured by the archived runs.
Nothing in a console row is typed twice and left unreconciled.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys

RESULTS = "results"
OUT = os.path.join("docs", "ablation.md")

# Provenance labels for the table's source column.
ARTEFACT_SOURCE = "`metrics.json`"
CONSOLE_SOURCE = "**console log**"


def _is_vec3(v):
    return isinstance(v, list) and len(v) == 3


def _all_positive_ints(v):
    return isinstance(v, list) and all(isinstance(x, int) and x > 0 for x in v)


CLASSES = ("N", "S", "V")

# Tolerance for reconciling a console-printed scalar against the value
# recomputed from the console-printed confusion matrix. The console prints
# 4 decimal places, so correct rounding cannot be off by more than 5e-5.
CONSOLE_TOL = 5e-5


def report_from_cm(cm):
    """Rebuild a classification report from a confusion matrix.

    Rows are true classes, columns predicted, both in CLASSES order. Returns
    (report, accuracy) shaped like the sklearn dict the artefacts carry.
    """
    total = sum(sum(row) for row in cm)
    rep = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(cm[r][i] for r in range(len(CLASSES)))
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rep[c] = {"precision": prec, "recall": rec, "f1-score": f1,
                  "support": support}
    rep["macro avg"] = {
        k: sum(rep[c][k] for c in CLASSES) / len(CLASSES)
        for k in ("precision", "recall", "f1-score")}
    rep["macro avg"]["support"] = total
    accuracy = sum(cm[i][i] for i in range(len(CLASSES))) / total
    return rep, accuracy


# CONSOLE_SPEC, the `spec` slot of a console row:
#   run_name            the folder the run WOULD have written
#   docs_commit         hash supplied with the console log (the last_report
#                       commit, not the code commit in the table)
#   best_val_macro_f1   validation score of the selected configuration
#   sweep               (heading, [(label, val_macro_f1, epoch, selected)])
#   argmax / tuned      {"confusion_matrix": [...], "printed": {...}}
#                       `printed` holds the scalars the console stated; every
#                       one is reconciled against the matrix.
#   weights             tuned decision weights, if the run tuned any

# (step, folder, commit, description, spec, note)
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

    # CONSOLE-LOG ROW - the Kaggle session expired before metrics.json was
    # written. Everything below is transcribed from the run's console output;
    # check_console_run() reconciles it. Commit is the code commit 25229d9;
    # the hash carried with the log, d7f6e02, is the docs commit after it.
    ("6", None, "25229d9",
     "validation-selected focal BETA sweep [0.0, 0.25, 0.41, 0.50]",
     {"run_name": "step6_beta_sweep",
      "docs_commit": "d7f6e02",
      "best_val_macro_f1": 0.6025,
      "sweep": ("BETA", [("0.00", 0.5422, 2, False),
                         ("0.25", 0.5419, 5, False),
                         ("0.41", 0.5885, 6, False),
                         ("0.50", 0.6025, 2, True)]),
      "argmax": {
          "confusion_matrix": [[41102, 710, 2421],
                               [581, 130, 1125],
                               [470, 23, 2727]],
          "printed": {"accuracy": 0.8919, "macro avg.f1-score": 0.5408,
                      "S.f1-score": 0.0963, "V.f1-score": 0.5745}},
      # Step 5 ran the selected BETA at the same seed, so an identical
      # result is the expected outcome and is verified against that artefact.
      "reproduces": "step5_bigger_val"},
     None),          # assembled from the console log - see step6_note()

    ("E0", "E0_reanchor", "042597b",
     "re-anchor: step 3 training config, 5-record validation",
     {"run_name": "E0_reanchor",
      "config.ds1_val": ["106", "118", "207", "220", "223"],
      "config.oversampling": True,
      "config.focal_loss_alpha": 0.5},
     "**Failed its 0.6400 criterion at 0.5591.** Training code identical to "
     "step 3; the ONLY difference was the 5-record validation set, and it "
     "selected epoch 2 instead of step 3's epoch 8. Cause, visible in this "
     "run's own history: `val_f1_V` peaks at epoch 2 (0.6617) while "
     "`val_f1_S` is still climbing at epoch 6 (0.2869), so macro-F1 tracks "
     "V and stops early. Record 106 contributes 520 V beats and ZERO S. "
     "This is what motivated the E1 revert."),

    ("E1", "E1_threshold_tuning", "3d3494b",
     "revert DS1_VAL to 3 records + validation-tuned thresholds",
     {"run_name": "E1_threshold_tuning",
      "config.ds1_val": ["207", "220", "223"],
      "config.oversampling": True,
      "config.focal_loss_alpha": 0.5,
      "threshold_weights": lambda v: isinstance(v, list) and len(v) == 3},
     None),          # assembled from the artefacts - see e1_note()

    ("E2", "E2_wavelet_input", "11cd348",
     "9-scale wavelet scalogram input (linear 10-90 Hz)",
     {"run_name": "E2_wavelet_input",
      "config.ds1_val": ["207", "220", "223"],
      "config.wavelet_centre_frequencies":
          lambda v: isinstance(v, list) and len(v) == 9
          and abs(v[0] - 10.0) < 1e-6 and abs(v[-1] - 90.0) < 1e-6},
     None),          # assembled from the artefacts - see e2_note()

    ("E3", "E3_log_wavelet", "fe14c34",
     "log-spaced wavelet scales 3-90 Hz",
     {"run_name": "E3_log_wavelet",
      "config.ds1_val": ["207", "220", "223"],
      "config.wavelet_centre_frequencies":
          lambda v: isinstance(v, list) and len(v) == 9
          and abs(v[0] - 3.0) < 1e-6 and abs(v[-1] - 90.0) < 1e-6},
     None),          # assembled from the artefacts - see e3_note()

    ("E4", "E4_small_model", "cb9224e",
     "revert to E2 scales, capacity 239,171 -> 16,283",
     {"run_name": "E4_small_model",
      "config.ds1_val": ["207", "220", "223"],
      "config.wavelet_centre_frequencies":
          lambda v: isinstance(v, list) and len(v) == 9
          and abs(v[0] - 10.0) < 1e-6},
     None),          # assembled from the artefacts - see e4_note()

    ("E5", "E5_rr_skip", "986dc29",
     "E2 architecture + direct RR skip to the output layer",
     {"run_name": "E5_rr_skip",
      "config.ds1_val": ["207", "220", "223"],
      "config.wavelet_centre_frequencies":
          lambda v: isinstance(v, list) and len(v) == 9
          and abs(v[0] - 10.0) < 1e-6},
     None),          # assembled from the artefacts - see e5_note()

    ("E6", "E6_balanced_sampling", "5b3b203",
     "balanced batch sampling (1:1:1) + plain cross-entropy",
     {"run_name": "E6_balanced_sampling",
      "config.ds1_val": ["207", "220", "223"],
      "config.sampler": "balanced_batch",
      "config.loss": "categorical_crossentropy",
      "config.focal_loss_used": False,
      "config.oversampling": False,
      "config.total_parameters": 239171},
     None),          # assembled from the artefacts - see e6_note()

    # CONSOLE-LOG ROW - see the step 6 row above. Commit is the code commit
    # 1339261; the hash carried with the log, c3042ad, is the docs commit.
    ("E7", None, "1339261",
     "sampler S:N ratio sweep [1, 2, 3, 4]",
     {"run_name": "E7_sampler_ratio",
      "docs_commit": "c3042ad",
      "best_val_macro_f1": 0.5620,
      # Ratio 1 is E6's [1/3, 1/3, 1/3] sampler, so this sweep entry must
      # reproduce E6's selection exactly; the gate checks it against E6's
      # metrics.json rather than taking the console log's word for it.
      "sweep_reproduces": ("1.0", "E6_balanced_sampling"),
      "sweep": ("S:N ratio", [("1.0", 0.5540, 6, False),
                              ("2.0", 0.5620, 3, True),
                              ("3.0", 0.5563, 3, False),
                              ("4.0", 0.5442, 2, False)]),
      "argmax": {
          "confusion_matrix": [[41937, 1526, 770],
                               [1180, 623, 33],
                               [47, 85, 3088]],
          "printed": {"accuracy": 0.9261, "macro avg.f1-score": 0.7114,
                      "S.recall": 0.3393, "S.precision": 0.2789,
                      "S.f1-score": 0.3061, "V.f1-score": 0.8685}},
      "tuned": {
          "confusion_matrix": [[41437, 1555, 1241],
                               [1147, 636, 53],
                               [28, 67, 3125]],
          "printed": {"accuracy": 0.9170, "macro avg.f1-score": 0.6944,
                      "S.f1-score": 0.3107, "V.f1-score": 0.8182}},
      "weights": [1.0, 1.4142, 4.0]},
     None),          # assembled from the console log - see e7_note()
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


def check_console_run(spec, ds2_support):
    """Reconcile a console-log row against itself, then against DS2.

    A console row has no artefact to hash, so the duplicate check cannot
    protect it. What can be checked is that the confusion matrix the console
    printed actually produces the scalars the console printed beside it, and
    that it scores the same DS2 every archived run scored. Both are enforced;
    a row that fails either is refused rather than published.

    Returns the spec with a recomputed `report` and `accuracy` attached to
    each decision rule, so nothing downstream re-derives a number by hand.
    """
    name = spec["run_name"]
    out = dict(spec)

    for rule in ("argmax", "tuned"):
        if rule not in spec:
            continue
        block = dict(spec[rule])
        cm = block["confusion_matrix"]

        if len(cm) != len(CLASSES) or any(len(r) != len(CLASSES) for r in cm):
            raise GateError("%s/%s: confusion matrix is not %dx%d"
                            % (name, rule, len(CLASSES), len(CLASSES)))

        report, accuracy = report_from_cm(cm)

        got_support = {c: report[c]["support"] for c in CLASSES}
        if got_support != ds2_support:
            raise GateError(
                "%s/%s: per-class support %r does not match DS2 %r as "
                "measured by the archived runs. The console matrix is not "
                "scoring the same test set."
                % (name, rule, got_support, ds2_support))

        for key, printed in block["printed"].items():
            cls, _, field = key.partition(".")
            recomputed = accuracy if key == "accuracy" else report[cls][field]
            if abs(recomputed - printed) > CONSOLE_TOL:
                raise GateError(
                    "%s/%s: console printed %s = %.4f but its own confusion "
                    "matrix gives %.6f (tolerance %g). The transcription is "
                    "wrong somewhere. Refusing to emit a row."
                    % (name, rule, key, printed, recomputed, CONSOLE_TOL))

        block["report"] = report
        block["accuracy"] = accuracy
        out[rule] = block

    if "reproduces" in spec:
        ref = json.load(open(os.path.join(RESULTS, spec["reproduces"],
                                          "metrics.json")))
        ref_cm = ref.get("confusion_matrix") or ref["test_argmax"][
            "confusion_matrix"]
        if ref_cm != spec["argmax"]["confusion_matrix"]:
            raise GateError(
                "%s: claims to reproduce %s but the confusion matrices "
                "differ (%r vs %r)."
                % (name, spec["reproduces"], spec["argmax"]["confusion_matrix"],
                   ref_cm))

    if "sweep_reproduces" in spec:
        label, ref_run = spec["sweep_reproduces"]
        entry = [e for e in spec["sweep"][1] if e[0] == label]
        if len(entry) != 1:
            raise GateError("%s: sweep has no entry %r to reproduce %s"
                            % (name, label, ref_run))
        _lab, f1, epoch, _sel = entry[0]
        ref = json.load(open(os.path.join(RESULTS, ref_run, "metrics.json")))
        if (abs(f1 - ref["best_val_macro_f1"]) > CONSOLE_TOL
                or epoch != ref["best_epoch"]):
            raise GateError(
                "%s: sweep entry %s = %.4f at epoch %d claims to reproduce "
                "%s, which selected %.4f at epoch %d."
                % (name, label, f1, epoch, ref_run,
                   ref["best_val_macro_f1"], ref["best_epoch"]))

    sel = [s for s in spec["sweep"][1] if s[3]]
    if len(sel) != 1:
        raise GateError("%s: sweep must mark exactly one selected setting, "
                        "found %d" % (name, len(sel)))
    if abs(sel[0][1] - spec["best_val_macro_f1"]) > CONSOLE_TOL:
        raise GateError(
            "%s: best_val_macro_f1 %.4f does not match the selected sweep "
            "entry %s = %.4f" % (name, spec["best_val_macro_f1"],
                                 sel[0][0], sel[0][1]))

    return out


def ds2_support(loaded):
    """Per-class DS2 support, taken from the archived runs and required equal.

    Every run scores the same untouched DS2. If the artefacts disagree about
    how many N/S/V beats that is, the console rows have nothing to be checked
    against and the table is not internally consistent either.
    """
    seen = {}
    for step, folder, _c, _d, _n, m, _h in loaded:
        if folder is None:
            continue
        cm = m.get("confusion_matrix") or m["test_argmax"]["confusion_matrix"]
        sup = tuple(sum(row) for row in cm)
        seen.setdefault(sup, []).append(step)
    if len(seen) != 1:
        raise GateError("archived runs disagree on DS2 support: %r" % (seen,))
    sup = next(iter(seen))
    return dict(zip(CLASSES, sup))


# -------------------------------------------------------------------- notes

def _history(folder):
    if folder is None:          # console-log row: no artefacts at all
        return None
    path = os.path.join(RESULTS, folder, "history.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def row_view(folder, m):
    """The four table-level facts, for an artefact run or a console row.

    Returns (report, accuracy, best_val_macro_f1, source). Both kinds of row
    go through this, so the table cannot end up reading one kind by hand.
    """
    if folder is None:
        return (m["argmax"]["report"], m["argmax"]["accuracy"],
                m["best_val_macro_f1"], CONSOLE_SOURCE)
    return (m["classification_report"], m["test_accuracy"],
            m.get("best_val_macro_f1"), ARTEFACT_SOURCE)


def val_gap(folder, m):
    report, _acc, best_val, _s = row_view(folder, m)
    if best_val is None:
        return None
    return report["macro avg"]["f1-score"] - best_val


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


def e1_note(m, h, gaps):
    """E1 carries two decision rules, so its note reports both."""
    a = m["test_argmax"]["classification_report"]
    t = m["test_tuned"]["classification_report"]
    w = m["threshold_weights"]
    s3 = json.load(open(os.path.join(RESULTS, "step3_rr_ratios",
                                     "metrics.json")))
    reproduced = (m["test_argmax"]["confusion_matrix"]
                  == s3["confusion_matrix"])
    return (
        "**The argmax column above is E1's plain-argmax result, and it "
        "reproduces step 3 EXACTLY** - confusion matrix identical "
        "(`{rep}`), same `best_epoch` {be}, same macro-F1 {mf1:.4f}. That "
        "confirms the E1 revert restored the step 3 configuration bit for "
        "bit and that the pipeline is deterministic.\n\n"
        "  **Threshold tuning did not transfer.** Coordinate ascent on "
        "validation chose w = {w} (N, S, V), lifting VALIDATION macro-F1 "
        "{v0:.4f} -> {v1:.4f}. On DS2 the same weights made things worse: "
        "macro-F1 {ta:.4f} -> {tt:.4f}, accuracy {aa:.4f} -> {at:.4f}.\n\n"
        "  It did exactly what it was asked to: **S recall {ra:.4f} -> "
        "{rt:.4f}**, roughly double. But S precision collapsed {pa:.4f} -> "
        "{pt:.4f}, because upweighting S by 4x moved {fp} true N beats into "
        "the S column. The search also raised w_V to 4.0, so N F1 fell "
        "{na:.4f} -> {nt:.4f} as well. Tuning both minority classes on 273 "
        "validation S beats overfitted the validation set."
    ).format(rep=reproduced, be=m["best_epoch"],
             mf1=a["macro avg"]["f1-score"], w=[round(x, 4) for x in w],
             v0=m["threshold_val_macro_f1_argmax"],
             v1=m["threshold_val_macro_f1"],
             ta=a["macro avg"]["f1-score"], tt=t["macro avg"]["f1-score"],
             aa=m["test_argmax"]["accuracy"], at=m["test_tuned"]["accuracy"],
             ra=a["S"]["recall"], rt=t["S"]["recall"],
             pa=a["S"]["precision"], pt=t["S"]["precision"],
             fp=m["test_tuned"]["confusion_matrix"][0][1],
             na=a["N"]["f1-score"], nt=t["N"]["f1-score"])


def e2_note(m, h, gaps):
    """E2's story is in the confusion matrix, not the headline number."""
    a = m["test_argmax"]["classification_report"]
    t = m["test_tuned"]["classification_report"]
    cm = m["test_argmax"]["confusion_matrix"]
    e1 = json.load(open(os.path.join(RESULTS, "E1_threshold_tuning",
                                     "metrics.json")))
    e1a = e1["test_argmax"]["classification_report"]
    e1cm = e1["test_argmax"]["confusion_matrix"]
    return (
        "**Best result of any run: macro-F1 {mf1:.4f}** (argmax), against "
        "E1's {p:.4f}. The gain is almost entirely V: **V-F1 {v0:.4f} -> "
        "{v1:.4f}, V precision {vp0:.4f} -> {vp1:.4f}**. Confusion with V "
        "collapsed - N-called-V {a0} -> {a1}, S-called-V {b0} -> {b1}.\n\n"
        "  **S barely moved: S-F1 {s0:.4f} -> {s1:.4f}.** A V beat is "
        "identified by a wide QRS, which a 10-90 Hz view captures well; an "
        "S beat is identified by an abnormal or absent P-wave plus early "
        "timing, and P-wave energy lies largely BELOW 10 Hz. This scale set "
        "has no channel under 10 Hz. That motivated E3.\n\n"
        "  Two caveats. `best_epoch` is **{be}** - the same one-epoch "
        "selection that confounded step 4, though here it produced the best "
        "test score rather than the worst. And threshold tuning hurt again: "
        "validation {tv0:.4f} -> {tv1:.4f} but test {mf1:.4f} -> {tt:.4f}, "
        "the second run in a row where tuned weights failed to transfer.\n\n"
        "  This run used the corrected normalization path (commit "
        "`11cd348`): its timestamp is 21:43 UTC, 12 minutes after that fix "
        "was committed at 21:31 UTC."
    ).format(mf1=a["macro avg"]["f1-score"], p=e1a["macro avg"]["f1-score"],
             v0=e1a["V"]["f1-score"], v1=a["V"]["f1-score"],
             vp0=e1a["V"]["precision"], vp1=a["V"]["precision"],
             a0=e1cm[0][2], a1=cm[0][2], b0=e1cm[1][2], b1=cm[1][2],
             s0=e1a["S"]["f1-score"], s1=a["S"]["f1-score"],
             be=m["best_epoch"],
             tv0=m["threshold_val_macro_f1_argmax"],
             tv1=m["threshold_val_macro_f1"],
             tt=t["macro avg"]["f1-score"])


def e3_note(m, h, gaps):
    """E3 tested the sub-10 Hz hypothesis and refuted it."""
    a = m["test_argmax"]["classification_report"]
    e2 = json.load(open(os.path.join(RESULTS, "E2_wavelet_input",
                                     "metrics.json")))
    e2a = e2["test_argmax"]["classification_report"]
    acc = h["accuracy"]
    v = h["val_macro_f1"]
    return (
        "**Refuted the sub-10 Hz hypothesis: macro-F1 {m1:.4f} against E2's "
        "{m0:.4f}, worse on every class** - N {n0:.4f} -> {n1:.4f}, "
        "S {s0:.4f} -> {s1:.4f}, V {v0:.4f} -> {v1:.4f}. Four channels at "
        "or below 10.74 Hz did not help S; whatever limits S here, it is "
        "not the absence of P-wave frequency coverage. E4 reverts the "
        "scales.\n\n"
        "  **The run also made the real problem legible.** Train accuracy "
        "climbs {a0:.4f} -> {a1:.4f} across {ne} epochs while validation "
        "macro-F1 falls monotonically from its epoch-1 peak of {v1e:.4f} to "
        "{vend:.4f}. `best_epoch` is **1**, as it was for E2. A "
        "239,171-parameter network is memorising 670 unique S beats. That "
        "is what E4 addresses by cutting capacity ~15x."
    ).format(m1=a["macro avg"]["f1-score"], m0=e2a["macro avg"]["f1-score"],
             n0=e2a["N"]["f1-score"], n1=a["N"]["f1-score"],
             s0=e2a["S"]["f1-score"], s1=a["S"]["f1-score"],
             v0=e2a["V"]["f1-score"], v1=a["V"]["f1-score"],
             a0=acc[0], a1=acc[-1], ne=len(acc),
             v1e=v[0], vend=v[-1])


def e4_note(m, h, gaps):
    """E4 answered its question and failed anyway - both worth recording."""
    a = m["test_argmax"]["classification_report"]
    cm = m["test_argmax"]["confusion_matrix"]
    s_pred = sum(row[1] for row in cm)
    s_true = sum(cm[1])
    acc = h["accuracy"]
    return (
        "**The capacity question was answered, and the model still failed.** "
        "`best_epoch` moved from 1 (E2, E3) to **{be}**, and final train "
        "accuracy fell {a1:.4f} against E3's 0.9717 - so the epoch-1 peak WAS "
        "a capacity artefact, exactly as predicted.\n\n"
        "  **But S collapsed to nothing: S-F1 {s1:.4f}, S recall {sr:.4f}.** "
        "The network predicts S **{sp} times in total** against {st} true S "
        "beats. macro-F1 {m1:.4f} is the second-worst of any run. 16,283 "
        "parameters cannot represent the S class at all.\n\n"
        "  Both facts matter: reducing capacity fixed the early peak and "
        "destroyed the minority class, so overfitting was never the binding "
        "constraint on S. E5 returns to E2's capacity and attacks the "
        "attenuation of the RR signal instead."
    ).format(be=m["best_epoch"], a1=acc[-1], s1=a["S"]["f1-score"],
             sr=a["S"]["recall"], sp=s_pred, st=s_true,
             m1=a["macro avg"]["f1-score"])


def e5_note(m, h, gaps):
    """E5 tested the RR-attenuation hypothesis with 15 extra parameters."""
    a = m["test_argmax"]["classification_report"]
    e2 = json.load(open(os.path.join(RESULTS, "E2_wavelet_input",
                                     "metrics.json")))
    e2a = e2["test_argmax"]["classification_report"]
    return (
        "**The RR skip did not help: macro-F1 {m1:.4f} against E2's "
        "{m0:.4f}, S-F1 {s1:.4f} against {s0:.4f}.** The only change was 15 "
        "parameters - the raw 5-feature `rr_input` concatenated onto the "
        "last dropout so the output Dense saw 69 inputs instead of 64. "
        "Everything else was byte-identical to E2.\n\n"
        "  So the RR signal is **not** being attenuated by network depth. "
        "Giving the output layer an un-mediated view of the five ratios "
        "changed nothing useful; S recall moved {r0:.4f} -> {r1:.4f} while S "
        "precision rose {p0:.4f} -> {p1:.4f}, i.e. the model became slightly "
        "more conservative, not more sensitive. `best_epoch` was **1** "
        "again.\n\n"
        "  Read together with E4, two hypotheses are now eliminated: S is "
        "not limited by overfitting (E4) and not by RR attenuation (E5). "
        "What remains is that the model is never **asked** to predict S - "
        "N outnumbers S 60:1 in every minibatch. That is what E6 changes."
    ).format(m1=a["macro avg"]["f1-score"], m0=e2a["macro avg"]["f1-score"],
             s1=a["S"]["f1-score"], s0=e2a["S"]["f1-score"],
             r0=e2a["S"]["recall"], r1=a["S"]["recall"],
             p0=e2a["S"]["precision"], p1=a["S"]["precision"])


def e6_note(m, h, gaps):
    """E6 is the best run so far and the first where tuning transferred."""
    a = m["test_argmax"]["classification_report"]
    t = m["test_tuned"]["classification_report"]
    e2 = json.load(open(os.path.join(RESULTS, "E2_wavelet_input",
                                     "metrics.json")))
    e2a = e2["test_argmax"]["classification_report"]
    return (
        "**Best run so far: macro-F1 {m1:.4f} argmax, {mt:.4f} tuned**, "
        "against E2's {m0:.4f}. S-F1 nearly doubled, {s0:.4f} -> {s1:.4f}, "
        "driven by recall {r0:.4f} -> {r1:.4f} at a modest precision cost "
        "{p0:.4f} -> {p1:.4f}. Class balance moved from duplicating data to "
        "SAMPLING it, so hard constraint 2 holds for the first time since "
        "E0.\n\n"
        "  **`best_epoch` is {be}, not 1** - the first run since step 3 to "
        "train past the first epoch and the first where **threshold tuning "
        "transferred** ({mt:+.4f} on test after failing by -0.0495 at E1 and "
        "-0.0846 at E2).\n\n"
        "  **But the sampler is under-firing.** de Waele et al. reach S "
        "recall 0.9116 at precision 0.3327 with this mechanism; we reached "
        "{r1:.4f} at {p1:.4f} - HIGHER precision than theirs, well under half "
        "the recall. N-F1 only fell {n0:.4f} -> {n1:.4f}, which says the "
        "1:1:1 ratio is not pushing S hard enough. E7 sweeps the ratio."
    ).format(m1=a["macro avg"]["f1-score"], mt=t["macro avg"]["f1-score"],
             m0=e2a["macro avg"]["f1-score"],
             s0=e2a["S"]["f1-score"], s1=a["S"]["f1-score"],
             r0=e2a["S"]["recall"], r1=a["S"]["recall"],
             p0=e2a["S"]["precision"], p1=a["S"]["precision"],
             n0=e2a["N"]["f1-score"], n1=a["N"]["f1-score"],
             be=m["best_epoch"])


# --------------------------------------------------- console-log run notes

CONSOLE_PROVENANCE = (
    "**FROM THE CONSOLE LOG - the artefact was not archived.** The Kaggle "
    "session expired before `metrics.json` was written, so there is no "
    "`results/{run}/` folder behind this row and nothing in it was read "
    "programmatically. The confusion {mats} below {is_are} what the run's "
    "console printed; every other number in the row is recomputed from "
    "{it_them} by `report_from_cm()` and reconciled against the scalars the "
    "console printed alongside (agreement to within 5e-5 is required), with "
    "the per-class support checked equal to the DS2 the archived runs score. "
    "**Treat this row as lower-provenance than every other row in the table, "
    "and do not use it as the reference point for a later run.** The commit "
    "above is the code commit; `{docs}` is the `docs: last_report` commit "
    "that followed it and was the hash carried with the log."
)


def _sweep_table(spec):
    label, entries = spec["sweep"]
    out = ["  | %s | best val macro-F1 | at epoch | selected |" % label,
           "  |---|---|---|---|"]
    for name, f1, epoch, selected in entries:
        out.append("  | %s | %.4f | %d | %s |"
                   % (name, f1, epoch, "**selected**" if selected else ""))
    return "\n".join(out)


def _provenance(spec):
    n = sum(1 for r in ("argmax", "tuned") if r in spec)
    return CONSOLE_PROVENANCE.format(
        run=spec["run_name"], docs=spec["docs_commit"],
        mats="matrix" if n == 1 else "matrices",
        is_are="is" if n == 1 else "are",
        it_them="it" if n == 1 else "them")


def step6_note(spec, h, gaps):
    """Step 6 swept BETA and reproduced step 5 exactly."""
    a = spec["argmax"]["report"]
    vals = [e[1] for e in spec["sweep"][1]]
    return (
        "{prov}\n\n"
        "{sweep}\n\n"
        "  **The entire grid spans {span:.4f} in validation macro-F1** - "
        "from `BETA=0.0`, which is no reweighting at all, to the most "
        "aggressive setting swept. Class reweighting through the focal alpha "
        "exponent is not an effective lever on this problem: the flat "
        "`[1,1,1]` control lands within {ctrl:.4f} of the winner.\n\n"
        "  **The test result is byte-identical to step 5**: confusion matrix "
        "`{cm}`, verified equal to `results/{ref}/metrics.json` by the "
        "generator. Step 5 already trained at the selected BETA from the "
        "same seed, so this is a **determinism confirmation, not an "
        "independent measurement** - macro-F1 {m:.4f}, S-F1 {s:.4f} and V-F1 "
        "{v:.4f} all repeat step 5's exactly. Two identical rows are also "
        "the precise signature the duplicate-hash gate exists to catch; here "
        "it is the expected outcome, and it is checked against the step 5 "
        "artefact rather than assumed."
    ).format(prov=_provenance(spec), sweep=_sweep_table(spec),
             span=max(vals) - min(vals),
             ctrl=spec["best_val_macro_f1"] - vals[0],
             cm=spec["argmax"]["confusion_matrix"], ref=spec["reproduces"],
             m=a["macro avg"]["f1-score"], s=a["S"]["f1-score"],
             v=a["V"]["f1-score"])


def e7_note(spec, h, gaps):
    """E7 swept the sampler ratio, failed its criterion, and found saturation."""
    a = spec["argmax"]["report"]
    t = spec["tuned"]["report"]
    e6 = json.load(open(os.path.join(RESULTS, "E6_balanced_sampling",
                                     "metrics.json")))
    e6a = e6["test_argmax"]["classification_report"]
    e6t = e6["test_tuned"]["classification_report"]
    vals = [e[1] for e in spec["sweep"][1]]
    return (
        "{prov}\n\n"
        "  **FAILED its stated criterion** (S-F1 > 0.40 and macro-F1 >= "
        "{crit:.4f}, E6's argmax score): it scored macro-F1 {m:.4f} and "
        "S-F1 {s:.4f}.\n\n"
        "{sweep}\n\n"
        "  **The sampler is SATURATED.** Doubling S exposure moved S recall "
        "by **{dr:+.4f}** ({r0:.4f} -> {r1:.4f}) while S precision fell "
        "{p0:.4f} -> {p1:.4f} and N-called-S rose {n0} -> {n1}. Drawing S "
        "more often does not make the model find more S - it only makes it "
        "guess S more often on N. S recall is pinned near 0.34 across the "
        "whole grid.\n\n"
        "  The `ratio 1.0` arm is E6's sampler, and it reproduces E6's "
        "selection exactly - {r1f1:.4f} at epoch {r1e}, verified against "
        "`results/E6_balanced_sampling/metrics.json` - so the sweep was "
        "correctly anchored even though its own artefact was lost.\n\n"
        "  **The selection was within noise.** Validation macro-F1 spans "
        "only {span:.4f} across all four ratios, narrower than the "
        "single-epoch validation swings already seen at step 4 (0.1497) and "
        "step 5 (0.0724), so which ratio won is not a meaningful "
        "result.\n\n"
        "  Threshold tuning did not transfer either: w = {w} took test "
        "macro-F1 {m:.4f} -> {mt:.4f} and accuracy {aa:.4f} -> {at:.4f}, "
        "after it had transferred at E6. **E6 remains the best run**, at "
        "macro-F1 {e6m:.4f} argmax / {e6t:.4f} tuned."
    ).format(prov=_provenance(spec), sweep=_sweep_table(spec),
             crit=e6a["macro avg"]["f1-score"],
             m=a["macro avg"]["f1-score"], s=a["S"]["f1-score"],
             dr=a["S"]["recall"] - e6a["S"]["recall"],
             r0=e6a["S"]["recall"], r1=a["S"]["recall"],
             p0=e6a["S"]["precision"], p1=a["S"]["precision"],
             n0=e6["test_argmax"]["confusion_matrix"][0][1],
             n1=spec["argmax"]["confusion_matrix"][0][1],
             span=max(vals) - min(vals),
             w=[round(x, 4) for x in spec["weights"]],
             mt=t["macro avg"]["f1-score"],
             aa=spec["argmax"]["accuracy"], at=spec["tuned"]["accuracy"],
             r1f1=e6["best_val_macro_f1"], r1e=e6["best_epoch"],
             e6m=e6a["macro avg"]["f1-score"],
             e6t=e6t["macro avg"]["f1-score"])


ASSEMBLED = {"4": step4_note, "5": step5_note, "6": step6_note,
             "E1": e1_note, "E2": e2_note, "E3": e3_note, "E4": e4_note,
             "E5": e5_note, "E6": e6_note, "E7": e7_note}


# --------------------------------------------------------------------- build

def build():
    seen = {}
    loaded = []
    for step, folder, commit, desc, spec, note in RUNS:
        m = spec if folder is None else check_run(folder, spec, seen)
        loaded.append((step, folder, commit, desc, note, m, _history(folder)))

    # DS2 is measured from the archived runs, then every console row's
    # confusion matrix is required to score that same DS2.
    support = ds2_support(loaded)
    loaded = [(step, folder, commit, desc, note,
               check_console_run(m, support) if folder is None else m, h)
              for step, folder, commit, desc, note, m, h in loaded]

    gaps = {}
    for step, folder, _c, _d, _n, m, _h in loaded:
        g = val_gap(folder, m)
        if g is not None:
            gaps[step] = g

    rows, notes = [], []
    for step, folder, commit, desc, note, m, h in loaded:
        r, acc, _bv, source = row_view(folder, m)

        rows.append(
            "| {s} | {d} | `{c}` | {mf1:.4f} | {sr:.4f} | {sp:.4f} | "
            "{sf:.4f} | {vf:.4f} | {acc:.4f} | {src} |".format(
                s=step, d=desc, c=commit,
                mf1=r["macro avg"]["f1-score"], sr=r["S"]["recall"],
                sp=r["S"]["precision"], sf=r["S"]["f1-score"],
                vf=r["V"]["f1-score"], acc=acc, src=source))

        if folder is None:
            label, entries = m["sweep"]
            chosen = [e for e in entries if e[3]][0]
            sel = (" Swept {lab} over {n} settings; selection on "
                   "`val_macro_f1` chose **{lab} = {v}** (best val macro-F1 "
                   "{bv:.4f}, at epoch {ep}).".format(
                       lab=label, n=len(entries), v=chosen[0],
                       bv=chosen[1], ep=chosen[2]))
        elif h is None:
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

        if folder is None:
            lead = ("- **step {s}** (`{f}` - **no archived artefact**) - "
                    "train distribution not recoverable; the console log "
                    "did not print it.{sel}\n  {x}").format(
                        s=step, f=m["run_name"], sel=sel, x=body or "")
        else:
            td = m["train_distribution"]
            lead = ("- **step {s}** (`{f}`) - train distribution N={N} / "
                    "S={S} / V={V}.{sel}\n  {x}").format(
                        s=step, f=folder, N=td["N"], S=td["S"], V=td["V"],
                        sel=sel, x=body or "")
        notes.append(lead)

    doc = HEADER + "\n".join(rows) + "\n\n## Notes\n\n" + "\n\n".join(notes)

    doc += GAP_HEADER
    for step, folder, _c, _d, _n, m, _h in loaded:
        if step not in gaps:
            continue
        r, _acc, bv, source = row_view(folder, m)
        doc += "| {s} | {v:.4f} | {t:.4f} | {g:+.4f} | {src} |\n".format(
            s=step, v=bv, t=r["macro avg"]["f1-score"], g=gaps[step],
            src=source)

    doc += FOOTER
    return doc


HEADER = """# Ablation log

Generated by `tools/make_ablation.py` - do not edit by hand.

**Check the `source` column before you quote a number.** Most rows are read
programmatically from a `results/<run>/metrics.json`, and the generator
refuses to emit one whose metrics file does not identify itself as the run
the table claims. Two rows - **step 6** and **E7** - are marked
`console log`: their Kaggle sessions expired before `metrics.json` could be
written, so those numbers are transcribed from the run's console output and
no artefact exists behind them. They are reconciled (every console-printed
scalar must match the confusion matrix the console printed beside it, to
within 5e-5, and that matrix must score the same DS2 as the archived runs)
but they are **not** artefact-backed, they cannot be re-derived from
`results/`, and they should not be used as the reference point for a later
run or quoted in the paper without re-running them.

Rules (see CLAUDE.md): ONE change per run, and nothing is ever tuned against
DS2. All metrics below are on DS2 (inter-patient test set).

> **Validation set changed at step 5 and changed back at E1.** Rows
> **baseline through step 4** used a 3-record validation set,
> `DS1_VAL = ['207','220','223']`. **Steps 5, 6 and E0** used the 5-record
> set `['106','118','207','220','223']`, with the training pool shrinking
> from 44,076 to 39,774 beats; **E1 onward** is back to the 3-record set.
> Test-set metrics below are all on the same untouched DS2, so the columns
> remain meaningful - but model *selection* differs, and `FOCAL_ALPHA`
> shifts because it is derived from DS1_TRAIN counts. **Steps 5, 6 and E0
> are not like-for-like comparisons with anything outside that window.**

| step | description | commit | macro-F1 | S recall | S precision | S F1 | V F1 | accuracy | source |
|---|---|---|---|---|---|---|---|---|---|
"""

GAP_HEADER = """

## Test-minus-validation gap

A model selected on a trustworthy validation signal should not score wildly
differently on test. The gap is `test macro-F1 - best_val_macro_f1`:

| step | best val macro-F1 | test macro-F1 | gap | source |
|---|---|---|---|---|
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
0.6800. Step 3 is the first step to move the S->N leak (73.5% -> 37.7%), at
the cost of macro-F1, because the freed S beats went to V rather than to S.

Step 4 removed duplicate oversampling and replaced the scalar focal alpha
(which rebalanced nothing) with a per-class vector. Its 0.5599 does not
measure that change - selection landed on a one-epoch spike.

Step 5 enlarged the validation set and cut selection variance by half, which
is what it was for. It also made the real problem legible: the model predicts
V roughly twice as often as V occurs.

Step 6 swept `BETA` over `[0.0, 0.25, 0.41, 0.50]` and let DS1_VAL pick.
`BETA=0.41` reproduces the 3.000 V:N emphasis of the old oversampling and
`BETA=0.0` is a flat `[1,1,1]` control. The control came within 0.0603 of the
winner and the whole grid spanned 0.0606, so the answer is that **the focal
alpha exponent is not a lever on this problem at all**. The winning setting
was step 5's, from the same seed, and reproduced step 5's confusion matrix
exactly.

The wavelet arc (E2-E5) then settled two questions about V and two hypotheses
about S: the 10-90 Hz scalogram fixed V (V-F1 0.8004 -> 0.9111) and S was
shown to be limited by neither overfitting (E4) nor RR attenuation (E5).

E6 attacked S where it actually lives - class exposure in the minibatch -
and is the best run in the table, macro-F1 0.7263 argmax and 0.7372 tuned,
with S-F1 nearly doubled. **E7 then swept the sampler ratio and found it
saturated**: across S:N ratios 1 through 4, S recall never left the
neighbourhood of 0.34 while S precision fell away, and validation macro-F1
spanned only 0.0178 across the whole grid. More S exposure is not the
remaining lever either.

Taken together, step 6 and E7 rule out the two obvious knobs - reweighting
the loss and reweighting the sampler. Both are dead ends on S recall, and
both were established at a **cost**: they are the two rows in this table with
no archived artefact.
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
