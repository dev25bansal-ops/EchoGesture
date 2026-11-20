# The acceptance gate: what it really buys (and why it was NOT raised)

Date: 2026-11-19
Status: **analysis complete. A 0.40 threshold was tried, built, measured, and
REVERTED.** `MIN_MARGIN` ships at 0.15. Two of the reasons originally given for
that revert have since been **shown to be false** and are corrected below.

## Correction, first

An earlier version of this document recommended raising `MIN_MARGIN` from 0.15
to 0.40, on the strength of the aggregate grid below. That recommendation was
reverted, and the two pieces of evidence then cited were:

- *"The user's own correctly-signed ELEVATOR clip scores ELEVATOR1 at 49.9%
  with a margin of 0.216, so a 0.40 cut would have declined a correct answer."*
- *"The CHEEK → RAZOR2 error came in at 80.7% top-1, so a 0.40 cut would still
  have accepted it."*

**Both are wrong, and the second describes an error that does not happen.**
Re-measured on 2026-11-19 through the shipped pipeline with the app's
`SignSegmenter` ported faithfully to Python:

| | measured |
|---|---|
| the ELEVATOR recording | 15.86 s, 485 frames, hand coverage 0.181 |
| segmenter kept | frames [175,358) = 5.99 s (37.7%) |
| **prediction** | **FLIP at 0.3046** |
| **ELEVATOR1** | **ranked 83rd of 100, probability 0.0015** |
| margin | 0.2506 |
| per-member argmax | **8 of 9 members say FLIP** |

The clip is not a correct answer at a thin margin — it is a **wrong** answer,
and the shipped 0.15 gate **already accepts it**. The 49.9%/0.216 figure came
from the 2×-speed replay that `docs/WINDOW_PLACEMENT_2026-11-19.md` retracted;
the retraction was written but this document and the source comment were never
corrected.

The CHEEK→RAZOR2 claim is contradicted by
`docs/WHY_CHEEK_BECOMES_RAZOR2.md`: **RAZOR2 is predicted 0 times in 32 CHEEK
clips**. The real confusions are PATIENT2, APPLE, MYSELF and COMB2.

So the revert stands, but for a different and weaker reason — see
§Recommendation. It is a product trade, not a proof that 0.40 is wrong.

The grid below is retained because it is correct and is the reason the wrong
conclusion was tempting.

## What ResultPolicy claims vs what it does

`ResultPolicy.kt` says:

> Measured coverage on that population: a 0.15 margin keeps 61% of captures at
> 94% accuracy.

That was measured on 82 validation clips. At the native geometry the app
actually runs, on the full 1,251-clip test split:

| threshold | coverage | accuracy | wrong readings shown |
|---|---|---|---|
| 0.15 (shipped) | 86% | 92.1% | 85 |
| 0.20 | 83% | 93.4% | 68 |
| 0.25 | 80% | 94.9% | 51 |
| 0.30 | 76% | 96.1% | 37 |
| 0.35 | 73% | 96.7% | 30 |
| 0.40 (tried, reverted) | 70% | 97.4% | 23 |
| 0.50 | 64% | 98.4% | 13 |

The aggregate trade is real. It is also the wrong instrument, for the reason
given above: the errors that matter arrive with *large* margins.

## Why per-sign thresholds did not work

Attempted first, and rejected on evidence. Calibrating one threshold per gloss
on the val split (`python .scratch/calibrate_policy.py`) moved test accuracy
from 92.1% to **92.3%** — 2 clips out of 1,251. The per-sign arms do not help
because:

- The val split holds **346 clips over 100 signs ≈ 3.5 clips per sign.** A
  threshold fitted on 3–5 clips is noise.
- The confidently-wrong readings are not marginal. `DOWNSIZE1 → VLOG` is 91.3%
  top-1 with a **0.884** margin. No threshold below 0.88 could catch it, and a
  threshold that high would decline most of the vocabulary.

Per-sign calibration is a dead end here and **should not be shipped.** The
calibration script and its output remain in `.scratch/` so this is checkable.

## What does work: raise the global threshold

Raising the single global threshold helps the weak signs far more than it costs
overall. On the eight weakest signs (CHEEK, APPLE, SHAVE1, RAZOR2, SINK, SCARF2,
CHEESEGRATER, PATIENT2), 94 test clips:

| threshold | accepted | accuracy on weak signs | wrong shown |
|---|---|---|---|
| 0.15 (current) | 72 | 75.0% | **18** |
| 0.25 | 56 | 83.9% | 9 |
| 0.30 | 53 | 86.8% | 7 |
| 0.35 | 49 | 91.8% | 4 |
| **0.40** | **44** | **93.2%** | **3** |
| 0.50 | 37 | 94.6% | 2 |

At 0.40 the weak signs go from **75% → 93%** accurate and wrong readings drop
from 18 to 3, while overall coverage falls only from 86% to 70% and overall
accuracy rises from 92.1% to 97.4%.

## Recommendation

**Leave `MIN_MARGIN` at 0.15.** A 0.40 cut was built and measured, and it is a
genuine trade rather than a clear win:

| | 0.15 | 0.40 |
|---|---|---|
| coverage | 86% | 70% |
| accuracy of accepted readings | 92.1% | 97.4% |
| wrong readings shown | **85** | **23** |
| correct answers refused (cumulative) | 74 | 217 |

Those refusal counts are cumulative from 0, not the cost of the change. Moving
the gate from 0.15 to 0.40 **additionally refuses 143 correct readings while
catching 62 wrong ones — a marginal ratio of 2.31:1** (recomputed from
`.scratch/thresh/probs_test.npz` on 2026-11-19). An earlier version of this
table presented the 217 cumulative figure next to the 62 marginal one, which
overstated the cost by conflating two different quantities.

**0.40 is the worst point on the curve, so it is the wrong end to argue from.**
The per-step frontier is:

| step | newly refused correct | newly caught wrong | ratio |
|---|---|---|---|
| 0.15 → 0.20 | 28 | 17 | 1.65:1 |
| 0.20 → 0.25 | 20 | 17 | **1.18:1** |
| 0.25 → 0.30 | 34 | 14 | 2.43:1 |
| 0.30 → 0.35 | 27 | 7 | 3.86:1 |
| 0.35 → 0.40 | 34 | 7 | 4.86:1 |

**0.25 is the real candidate**: from 0.15 it refuses 48 correct to catch 34
wrong (1.41:1) at 79.5% coverage and 94.89% accepted-accuracy — roughly 40% of
the wrong readings for less than half the cost ratio of 0.40. The decision is
properly "0.15 or 0.25", not "0.15 or 0.40".

One thing does argue for raising it: the accepting behaviour is what the demo
shows. A 30.5%-confidence wrong reading presented as "best match" is the failure
the UI was written to prevent, and at 0.15 the app does exactly that. If a live
take lands under 0.40, **re-record rather than ship the reading** — that is the
practical answer and costs nothing.

The lever that most improves the user's experience is still **how the clip is
recorded**, not where the gate sits.

## What would work instead, and is not yet built

- **Record within the capture window.** The recording that failed was 15.9 s for
  a window designed around 3.7 s, with the hand in frame 18% of the time. On 15
  clean test clips ELEVATOR1 is recognised at **73.3% recall, median confidence
  0.847, median margin 0.801**. The model can do this sign; that recording did
  not give it a chance.
- **Cross-member agreement as a second signal.** Tested on 2026-11-19: at 86%
  coverage, requiring 9/9 members to agree gives 82 wrong readings vs margin's
  85 — and it loses as the gate tightens. Not worth new code this close to the
  deadline. Full numbers in `docs/PER_CLASS_THRESHOLDS_2026-11-19.md`.
- **A "not in my vocabulary" class.** The model has no way to say "none of
  these". Out-of-distribution input becomes some gloss with a number attached.
  This needs training data the project does not have, so it is out of scope for
  the deadline.
- **Many more clips per sign.** CHEEK is 75% because the corpus holds 32 clips
  of it. This is the slow fix.

## Reproduction

```
C:/Python314/python.exe .scratch/gate_frontier.py         # the threshold grid, all 100 signs
C:/Python314/python.exe .scratch/per_class_thresholds2.py # per-class, three ways
C:/Python314/python.exe .scratch/score_phone_clip.py "<video>" --expect ELEVATOR1
C:/Python314/python.exe .scratch/score_probs.py val test   # rebuilds the probability matrix
```

Test is never used to choose a threshold. **The third script is the one that
overturned the original recommendation**, and should be run before anyone
proposes changing `MIN_MARGIN` again — in either direction.
