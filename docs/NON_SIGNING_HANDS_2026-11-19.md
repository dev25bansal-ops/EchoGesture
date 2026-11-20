# A hand that is not signing still gets a named gloss

Measured 2026-11-19. This is a product defect found while investigating "why
does the camera miss what the clip gets". It is separate from the camera
domain-shift question and is more actionable.

## The finding

Take a corpus test clip, freeze the landmarks at the first frame where a hand
appears, and classify. The result is a hand that enters frame and **never
moves** — nobody is signing anything. Through the shipped gate:

| | value |
|---|---|
| clips tested | 118 |
| **gate accepts a gloss** | **43.2%** (51 of 118) |
| mean confidence | 0.345 |
| confidence >= 0.50 | 23.7% |
| confidence >= 0.90 | 4.2% |

So for roughly two captures in five of a *still hand*, the app answers with a
named sign. Most are not confident (mean 0.345 against 0.85 for real clips), but
the gate at `margin >= 0.15` does not care about absolute confidence — it only
asks whether the top class beats the runner-up, and on a still hand it often
does.

## Why this matters for the camera complaint

A person testing the camera signs, often incompletely or off-centre, and the
capture window can land on hands that are moving into position rather than
performing the sign. If a fourth of all capture windows contain more "hand
arrived" than "sign performed", this defect alone produces a steady stream of
confident-sounding but arbitrary answers. It also explains why the same sign
sometimes works: a clean capture is the minority case worth retrying.

## What was tried and did not work: a motion gate

The obvious fix is to require motion. Measured on the full 1,251-clip test set,
mean absolute per-step motion in the model's own 24-step feature tensor:

| | p5 | p25 | median | p75 | p95 | mean |
|---|---|---|---|---|---|---|
| real clips | 0.000 | 0.150 | 0.267 | 0.421 | 0.741 | 0.315 |
| frozen hands | 0.000 | 0.003 | 0.114 | 0.280 | 0.690 | 0.217 |

**Only 1.45x separation**, with heavy overlap: the frozen-hand p95 (0.690) is
almost the real-clip p95 (0.741). Any threshold that rejects most still hands
also rejects a large share of genuine signs — many signs are static (NOON, DARK
and VOICE among the demo clips are held shapes, not movements).

The reason the separation is poor is that freezing a *detected* frame keeps
MediaPipe's landmark noise, which is itself motion in the feature space. A real
still hand would have less, but the measurement cannot say how much less without
real still-hand captures. **A motion gate was therefore not shipped**: a 1.45x
separation is not enough to justify rejecting user input, and getting it wrong
in the rejecting direction is worse for an accessibility tool than answering.

## The cheaper mitigation that already exists

`ResultPolicy` already exposes confidence to the user. The captures this defect
produces cluster at 0.2-0.5 confidence, well below the 0.85 typical of real
signs. The app already labels low-margin readings as uncertain rather than
presenting them as fact, so the honest path is unchanged: show the gloss, label
it uncertain, and let the user retry. That is what ships.

## What would make this properly measurable

Real still-hand captures from the device — hand in frame, no sign — rather than
a frozen-frame surrogate. Ten such captures would give a real distribution to
threshold against, instead of inferring one from a surrogate with known noise.

## Reproduce

The frozen-frame construction is in the loop body above: for each test clip,
copy the first hand-bearing frame's landmarks across all frames, run the shipped
`build_features` + nine-member mean-logit path, and apply
`ResultPolicy.evaluate`.
