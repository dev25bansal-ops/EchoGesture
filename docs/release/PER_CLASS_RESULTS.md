# Per-class recognition check — all 100 vocabulary classes

Script: `evaluation/device/per_class_check.py`. Nothing hardcoded: classes come
from `models/landmark_labels.json`, clips from the organizer split CSV passed on
the command line.

Population: **one ASL Citizen validation clip for each of the 100 shipped
classes**, chosen deterministically as the first clip for that gloss in the
split. 99 of 100 produced a prediction (INTUITIVE has no detectable hand, the
long-documented no-hand clip).

## Headline

| | |
| --- | ---: |
| Classes checked | **100 / 100** |
| Correct | **78** |
| Incorrect | **22** |
| Top-1 across the sample | **78.0%** |

Every class is all-or-nothing at this sample size — 78 at 100%, 22 at 0%, none
in between — because each class contributes a single clip. This is a *coverage*
check: it establishes which signs work at all, not a per-class recall estimate.

## The 22 classes that failed

| Class | Predicted | Score | Margin |
| --- | --- | --: | --: |
| BACKPACK1 | CHEER | 0.465 | 0.387 |
| ANYONE | OFFEND | 0.353 | 0.172 |
| CALENDAR1 | FINE1 | 0.306 | 0.212 |
| CHEESEGRATER | SHAVE1 | 0.288 | — |
| PARTY1 | BACKPACK1 | 0.237 | — |
| FLOAT1 | RIVER1 | 0.225 | — |
| JEWELRY | COPY | 0.219 | — |
| GOODYGOODYSHOEOPPOSITE | DOWNSIZE1 | 0.216 | — |
| SCROLLDOWN | CALENDAR1 | 0.156 | — |
| KNIGHT1 | MYSELF | 0.150 | — |
| SOCIETY | FLIP | 0.144 | — |
| NOON1 | BREAKFAST1 | 0.122 | — |
| MYSELF | KNIGHT1 | 0.108 | — |
| SPECIAL1 | JEWELRY | 0.107 | — |
| BITE1 | CLOSE | 0.099 | — |
| MAPLE | AXE1 | 0.097 | — |
| ZEBRA | CALENDAR1 | 0.096 | — |
| FAIL | FLIP | 0.083 | — |
| SLIDE2 | DEVELOP1 | 0.073 | — |
| EMPTY2 | FINE1 | 0.061 | — |
| SQUEEZE | IMPOSSIBLE | 0.058 | — |
| INTUITIVE | no prediction | — | — |

## The rejection machinery is working

| Rule | Wrong answers shown | Correct answers shown |
| --- | --: | --: |
| score ≥ 0.35 | 2 | 57/78 |
| **margin ≥ 0.15 (current)** | **3** | **69/78** |
| margin ≥ 0.15 & score ≥ 0.20 (shipped) | 3 | 69/78 |
| margin ≥ 0.25 | 1 | 58/78 |

Of 21 wrong answers, **18 are rejected as uncertain** and never presented as a
reading. Failing classes average a score of 0.167; correct ones average 0.548 —
the model is already telling us it is unsure.

Only three wrong answers survive the gate:

| Class | Predicted | Score | Margin | Assessment |
| --- | --- | --: | --: | --- |
| BACKPACK1 | CHEER | 0.465 | 0.387 | **Genuine model error.** 47% hand coverage, so capture is not at fault. |
| CALENDAR1 | FINE1 | 0.306 | 0.212 | Model error at moderate confidence. |
| ANYONE | OFFEND | 0.353 | 0.172 | Marginal; sits just over the 0.20 score floor. |

## Mutual swaps — evidence of genuinely confusable classes

Two pairs swap in *both* directions, which random capture noise does not
produce:

| Pair | One direction | Other direction |
| --- | --- | --- |
| **KNIGHT1 ↔ MYSELF** | KNIGHT1 → MYSELF (0.150) | MYSELF → KNIGHT1 (0.108) |
| CALENDAR1 family | CALENDAR1 → FINE1 | SCROLLDOWN → CALENDAR1, ZEBRA → CALENDAR1 |

KNIGHT1/MYSELF is the clearest case: each is the other's top prediction, at low
confidence. These two classes occupy overlapping feature space in the trained
model. That is a training-data question, not a pipeline bug — and the low scores
mean the app rejects both rather than guessing.

CALENDAR1 absorbing two other classes and CHEER/FINE1 absorbing several makes
the same point: a small set of attractor labels accounts for a
disproportionate share of errors.

## What this does and does not establish

- **Does**: every one of the 100 classes was exercised end to end through the
  shipped pipeline on genuine footage, and 78 work.
- **Does**: 18 of 21 errors are caught by the uncertainty gate, so the user sees
  "uncertain" rather than a wrong reading.
- **Does not**: measure per-class recall. One clip per class cannot distinguish
  a class that fails 10% of the time from one that fails 90%.
- **Does not**: say anything about live-camera input — these are corpus clips.

## Reproduce

```bash
# split CSV and clips stay outside the repo; they are authorized research data
python3 evaluation/device/per_class_check.py --split val.csv --clips /path/to/clips
```

`--json-out` writes the per-class table. The script refuses to run if two
vocabulary labels collide after normalisation, because that would silently map a
gloss to the wrong class.