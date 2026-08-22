"""
Count N / S / V beats per DS1 record.

CPU-only. Needs wfdb and numpy, NOT tensorflow.
Run from the project root:

    python tools/inspect_ds1.py

Purpose: we need to hold out whole patients from DS1 as a validation set.
That choice must be made from the actual beat counts, not guessed - a
validation set with almost no S beats gives a val S-F1 too noisy to
select a checkpoint on.

NOTE: AAMI_MAP, DS1 and the beat-window rejection rule are duplicated
from train.py on purpose, so this tool runs without importing
tensorflow. If any of them change in train.py, change them here too.
"""

import os
import json
from collections import Counter

import numpy as np
import wfdb

# --- kept identical to train.py -------------------------------------
DATA_DIR = os.environ.get(
    "ECG_DATA_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "mit-bih-arrhythmia-database-1.0.0",
    ),
)

DS1 = ['101', '106', '108', '109', '112', '114', '115', '116',
       '118', '119', '122', '124', '201', '203', '205', '207',
       '208', '209', '215', '220', '223', '230']

AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
}

PRE_SAMPLES = 90
POST_SAMPLES = 144
SEGMENT_LENGTH = PRE_SAMPLES + POST_SAMPLES
LEAD_INDEX = 0
# --------------------------------------------------------------------


def count_record(rec):
    """Count usable beats per class, applying train.py's rejection rules."""
    path = os.path.join(DATA_DIR, rec)
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, 'atr')

    signal = record.p_signal[:, LEAD_INDEX]
    samples = ann.sample
    symbols = ann.symbol

    counts = Counter()
    skipped_edge = 0

    # i from 1 to len-2: train.py needs a previous and a next R-peak for RR
    for i in range(1, len(samples) - 1):
        sym = symbols[i]
        if sym not in AAMI_MAP:
            continue

        start = samples[i] - PRE_SAMPLES
        end = samples[i] + POST_SAMPLES
        if start < 0 or end > len(signal) or (end - start) != SEGMENT_LENGTH:
            skipped_edge += 1
            continue

        counts[AAMI_MAP[sym]] += 1

    return counts, skipped_edge, record.sig_name


def main():
    print(f"DATA_DIR: {DATA_DIR}\n")

    rows = []
    totals = Counter()

    print(f"{'record':>7} {'N':>7} {'S':>7} {'V':>7} {'total':>7}   {'S share':>8}  leads")
    print("-" * 72)

    for rec in DS1:
        counts, skipped, sig_name = count_record(rec)
        total = sum(counts.values())
        s_share = counts['S'] / total if total else 0.0

        rows.append({
            "record": rec,
            "N": counts['N'],
            "S": counts['S'],
            "V": counts['V'],
            "total": total,
            "leads": sig_name,
            "skipped_edge": skipped,
        })
        totals.update(counts)

        print(f"{rec:>7} {counts['N']:>7} {counts['S']:>7} {counts['V']:>7} "
              f"{total:>7}   {s_share:>7.2%}  {sig_name}")

    grand = sum(totals.values())
    print("-" * 72)
    print(f"{'TOTAL':>7} {totals['N']:>7} {totals['S']:>7} {totals['V']:>7} {grand:>7}")

    # --- lead-order check (Phase E step 8) --------------------------
    odd = [r for r in rows if r["leads"][LEAD_INDEX] != "MLII"]
    print("\nLead check:")
    if odd:
        for r in odd:
            print(f"  record {r['record']}: channel {LEAD_INDEX} is "
                  f"{r['leads'][LEAD_INDEX]}, not MLII  -> leads = {r['leads']}")
    else:
        print(f"  all DS1 records have MLII at channel {LEAD_INDEX}")

    # --- records ranked by S count ----------------------------------
    print("\nDS1 records ranked by S-beat count:")
    for r in sorted(rows, key=lambda x: -x["S"])[:8]:
        print(f"  {r['record']}: {r['S']} S beats  ({r['total']} total)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ds1_beat_counts.json")
    with open(out, "w") as f:
        json.dump({"per_record": rows, "totals": dict(totals)}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
