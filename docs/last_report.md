# Last report — E8 recorded; E9 blocked on a 12-beat population mismatch

Two things happened: E8's returned artefact was recorded in the ablation
table, and **E9 was stopped before implementation** because its stated
acceptance test cannot be met.

---

## Part 1 — E8 recorded

`results/E8_variable_segmentation/` was untracked. It is now committed and
`docs/ablation.md` has its row (17 runs, gate passed).

| | argmax | tuned | E6 argmax |
|---|---|---|---|
| macro-F1 | 0.6495 | 0.6706 | 0.7263 |
| S recall | **0.6594** | 0.4505 | 0.3388 |
| S F1 | **0.4752** | 0.4565 | 0.3641 |
| V F1 | 0.5661 | 0.6236 | 0.8503 |
| accuracy | 0.8410 | 0.8785 | 0.9352 |

**Every mechanical prediction from last session was exact:**
`total_parameters` 239,331 · `steps_per_epoch` 302 · train N=35025 S=637
V=2878 · DS2 37400/1565/3189 · weights [1/3,1/3,1/3] · input (720, 10).

**The substantive predictions were wrong, and worth recording as wrong.**
I predicted test S recall would stay below 0.50 — it reached **0.6594**. I
predicted validation macro-F1 would beat E6's 0.5540 — it **fell to 0.5090**,
selecting epoch 5. My own stated falsifier was "test S-F1 above 0.45"; it
came in at 0.4752. So the reading that the DS2 span-gap collapse would cap
the S gain is falsified on S.

What I did not predict: **V collapsed.** V-F1 0.8503 → 0.5661, with 4,405 N
beats called V. That, not S, is why macro-F1 fell to 0.6495 — E8 is the
best run on S and one of the worse runs overall.

Also notable: threshold tuning transferred again (+0.0211 on test) and the
selected vector **down-weighted S** to w = [1.0, 0.25, 0.3536]. That is the
second confirmation of the standing rule — the only two vectors that ever
transferred, E6's and E8's, both leave `w_S` at or below `w_N`.

### The gate now tracks which DS2 a row was scored on

E8 scores 42,154 DS2 beats against the 49,289 every earlier row used. Rather
than relax the support check, each run now declares a **population** and the
gate refuses any artefact that disagrees with its declaration. The table has
a new `DS2` column: every row reads `full` except E8's `span_capped_720`.
Verified: re-declaring E8 as `full` is refused.

---

## Part 2 — E9 stopped, as instructed

> "If any count differs, the filter is not reproducing E8's population —
> stop and report rather than proceeding."

The counts differ, by **12 beats**. This is not a bug in the filter; the two
runs use different *edge* rules and the difference is unreachable under a
fixed window.

E8 rejects a beat when `R[i-1]..R[i+1]` leaves the signal. E6 rejects when
`r-90 .. r+144` leaves it. Twelve beats pass the first and fail the second:

| split | class | E8 rule | E9 = E6 window ∧ span ≤ 720 | short by |
|---|---|---|---|---|
| DS1_TRAIN | N | 35025 | 35021 | 4 |
| DS1_VAL | N | 5361 | 5359 | 2 |
| DS2 | N | 37400 | 37395 | 5 |
| DS2 | V | 3189 | 3188 | 1 |

**S matches exactly in all three splits** (637 / 273 / 1565), as do
DS1_TRAIN V and DS1_VAL V.

Every one of the 12 is the **first annotated beat of a record**, where the R
peak sits 28–88 samples in, so `r - 90 < 0`:

```
101 r=83  window [-7,227)      207 r=50  window [-40,194)
108 r=88  window [-2,232)      220 r=28  window [-62,172)
118 r=68  window [-22,212)     100 r=77  window [-13,221)
230 r=75  window [-15,219)     123 r=70  window [-20,214)
210 r=57  window [-33,201)     214 r=58  window [-32,202)
222 r=81  window [-9,225)      233 r=42  window [-48,186)  <- the V
```

There is no signal before sample 0, so a fixed 234-sample window cannot
include them. Reproducing E8's population exactly would require padding the
window at the record start, which changes E6's representation and violates
"CHANGE NOTHING ELSE".

### What this costs, quantified

The DS2 difference is 6 beats out of 42,154 — **0.014%**, of which **zero are
S**. Bounds on the resulting incomparability:

- Every S metric — recall, precision, F1 — is computed on the identical 1,565
  beats and is **exactly** comparable.
- N support 37400 → 37395, so N recall moves by at most 1.3e-4.
- V support 3189 → 3188, so V recall moves by at most 3.1e-4.
- macro-F1 is therefore uncertain by **< 2e-4**, against an E8-vs-E6 effect
  size of 0.08 — three orders of magnitude larger.

Also worth weighing: baseline through E7 all excluded these same 12 beats,
because they all used the fixed window. E9 excluding them keeps it consistent
with the entire table; **E8 is the outlier that uniquely gained them.**

### Recommendation

Proceed with E9 on 42,148 DS2 beats, documenting the 12 exclusions and
declaring a third DS2 population so the gate keeps the distinction visible.
The S comparison — the entire point of E9 — is exact either way.

Awaiting a decision rather than assuming one.

## State

- Committed and pushed: E8 artefact + ablation row + population gate.
- `src/train.py` is **untouched** — still E8 as trained. No E9 code written.
