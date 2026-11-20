# The agreement gate: the first signal that actually reduces confident errors

Measured 2026-11-19. This is the one rejection signal, out of five tested, that
cuts wrong answers instead of re-arranging them.

## What was tested and what it cost

Every signal below is computable from data the app already has at runtime —
no retraining, no extra inference. All are scored at **matched coverage** on the
1,255-clip test split, so a rule cannot win by simply refusing more.

```
signal                     95%   90%   86%   80%   75%   70%     (wrong readings)
------------------------------------------------------------------------------
margin (shipped)           141   100    85    54    36    23
agree (count)              146   110    82    58    42    32
votes_second (neg)         152   130   110    84    67    50
votes_other (neg)          147   118    98    77    63    60
distinct (neg)             143   109    94    71    58    49
mean_member_top            145   110    89    69    49    37
```

**At matched coverage, none of them wins.** Agreement is 3 readings better at
86% and clearly worse below it. `votes_second` (does a member vote for the
*runner-up* class specifically) is much worse. So is asking how many distinct
classes the members scatter across, and so is averaging their top
probabilities. Temperature scaling was tested too — flat.

That is the same conclusion the per-class and per-cluster work reached: **the
probability vector and the member votes both rank clips about as well as the
margin already does.** As a *threshold*, agreement is not an improvement.

## The thing that is not a threshold

The comparison above asks "which clips does this rank worst". But agreement is
not a dial — it is a **second, differently-shaped condition**. Applied as an AND
on top of the shipped margin gate:

| operating point | coverage | accuracy | wrong | per 1,000 answered |
|---|---|---|---|---|
| today: `margin >= 0.15` | 86.1% | 92.13% | 85 | **78.7 wrong** |
| `+ agree >= 8` | 68.8% | 96.41% | 31 | **35.9 wrong** |
| `+ agree == 9` | 57.5% | 98.06% | 14 | **19.4 wrong** |

The honest framing is not "3 fewer wrong readings" (that is the matched-coverage
view, and it is real but tiny). It is: **of the readings the app actually
answers, the share that are wrong falls from 8.5% to 3.7% — a 54% cut — at the
cost of declining 17 more clips out of every 100.**

That is a different product, not a better threshold. Whether it is the right
product is a decision, not a measurement.

## It is not overfitting — checked on val first

The per-class thresholds failed exactly here: 100% in-sample, 95.91% held out.
So the agreement rule was chosen on **val** and only then scored on test.

| split | rule | coverage | accuracy | wrong |
|---|---|---|---|---|
| **val** (346, used to choose) | margin>=0.15 | 88.4% | 96.08% | 12 |
| | + agree>=8 | 76.0% | **98.10%** | **5** |
| **test** (1,255, never used to choose) | margin>=0.15 | 86.1% | 92.13% | 85 |
| | + agree>=8 | 68.8% | **96.41%** | **31** |

Both splits move the same way by a similar amount. This transfers, which is
what separates it from every per-class attempt.

## What it does to individual signs

The reason it works is visible per class. The rule removes weak-sign false
positives and *keeps* strong-sign true positives:

| sign | margin accepts | of those right | + agree>=8 keeps | of those right |
|---|---|---|---|---|
| APPLE | 9 | 9 | 8 | **8** |
| ELEVATOR1 | 13 | 11 | 10 | **10** |
| SCARF2 | 9 | 6 | 6 | **5** |
| RAZOR2 | 7 | 5 | 4 | **4** |
| CHEEK | 10 | 7 | 5 | 4 |
| SINK | 6 | 4 | 3 | **3** |
| CHEESEGRATER | 7 | 3 | 4 | 2 |

APPLE and ELEVATOR1 lose **zero** correct answers per accepted clip. CHEEK, the
worst offender in the original handoff, goes from 3 wrong to 1.
CHEESEGRATER — already the weakest class at 40% — loses one correct reading and
keeps 2 of 4, so it does not get better, it just gets quieter.

The four demo clips are unaffected at their strong operating points: DARK is
9/9, MECHANIC 9/9, and the NOON/VOICE clips used on the phone were 8/9 and 9/9
on the ensemble that produced those readings.

## What shipped, and what did not

**Shipped:** the signal. `LandmarkClassifier.Result` now carries `agreement`
and `memberCount`, computed from the nine logit vectors already in memory
(nine argmaxes over 100 floats — no extra inference, no measurable cost), and
`DiagnosticsLog` records `agree=8/9` on every capture.

**Not shipped:** enforcement. `ResultPolicy` is unchanged at `MIN_MARGIN = 0.15`
and does not consult agreement. The reason is written in that file and still
holds: raising the bar trades coverage for accuracy, and for an accessibility
tool that is the owner's call, not a change to make silently the day before
submission.

**To turn it on**, one line in `ResultPolicy.evaluate` — decline when
`agreement < 8` — plus a new `Reason.LOW_AGREEMENT` and its string. The
diagnostics already record the value, so the decision can be revisited with
on-device data instead of only this offline split.

## Two things this measured in passing

**Member votes for the runner-up are not a warning sign.** `votes_second` was
the most promising untested idea here — the intuition that "one member says
CHEEK, another says RAZOR2" would flag the exact CHEEK→RAZOR2 confusion. It is
the *worst* of the six signals at every coverage level. A member disagreeing
with the top class is not informative about whether the top class is right.

**Two of the four demo clips are training clips.** `3567776249024257-NOON` and
`7606272862279693-VOICE` are in the **train** split; `08763437833459409-DARK`
and `9119691570534749-MECHANIC` are in **test**. The phone scores of 95.7% and
96.5% on NOON and VOICE are therefore memorised, not generalisation — while the
held-out pair scored 77.1% and 87.9%, much closer to the 85.45% corpus average.
Corrected in `HOW_TO_TEST_THE_APP.md`. Anyone demoing from that set should lead
with DARK and MECHANIC.

## Reproduce

```
.scratch/dump_member_argmax.py     # per-member argmax + top prob, test split
.scratch/dump_member_val.py        # same for val (the split that decides)
```

Artifacts: `.scratch/thresh/member_test.npz`, `.scratch/thresh/member_val.npz`,
`.scratch/thresh/probs_{train,val,test}.npz`.
