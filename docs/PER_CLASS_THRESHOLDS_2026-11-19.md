# Per-class thresholds: tested, rejected. 2026-11-19

The handoff called this "the top-priority fix". It was the right thing to test,
and it does not work. This records what was measured, what lost, what the model
actually does at the gate, and the one number that should change.

Everything below was produced by scripts in `.scratch/` against
`.scratch/landmarks_native`, and every headline figure was first reproduced
against the published numbers (`results/results.csv`). Nothing here is an
estimate.

---

## 1. The measurement base

`.scratch/score_probs.py` re-scores every cached clip through `src/main.py`'s
own `_init_model` / `build_features` / nine-member mean-logit path and saves all
100 softmax probabilities, the truth, the clip id and the participant.

Reproduction check — these must match `results/results.csv` or nothing else is
trustworthy:

| split | this run | published |
|---|---|---|
| test | **1069/1255** | `top1_accuracy = 85.45%` (1069/1251, 4 no-hand clips) |
| val | **309/346 = 89.31%** | `top1_accuracy_val = 89.31%` |

Val matches to the digit. Test has the identical numerator, 1069; the published
percentage uses 1251 because four clips produce no hand and are excluded. The
matrix keeps those four and records `hand_frac` per clip, so both conventions
are available from one file.

An earlier attempt used `.scratch/landmarks` instead of
`.scratch/landmarks_native` and produced 88.20% on val. That was the wrong cache
— the geometry the app actually feeds MediaPipe is the native one. Worth stating
because the two caches differ by 1.1 points on val and it was not obvious.

---

## 2. The baseline gate, and the claimed failures

At the shipped `margin >= 0.15` on the 1,255-clip test split:

| | |
|---|---|
| coverage | 86.1% (1080/1255) |
| accuracy of accepted readings | **92.13%** |
| confidently wrong (accepted and wrong) | **85** |

That is the **margin condition alone**. The app's full gate
(`ResultPolicy`) also requires `hand_frac >= 0.15` and `top1 >= 0.20`; with all
three, the same split gives **1075 accepted / 85.66% coverage / 92.28% accepted
accuracy / 83 wrong** (the top-1 floor is inert; the hand-fraction floor removes
5 clips). Both rows are reproducible from `.scratch/thresh/probs_test.npz`;
the figures in this document use the margin-only row, labelled as such.

All three failures named in the handoff reproduce exactly:

- **85 confidently-wrong readings** — confirmed, exactly 85.
- **CHEEK** — 11 test clips, **10/11 accepted at the 0.15 gate, 7 of those 10
  correct (70.0%)**. Confirmed.
- **DOWNSIZE1 → VLOG** — one clip, `1091567632188366-DOWNSIZE`,
  top-1 **0.9130**, margin **0.884**, accepted. Confirmed.

Worst offenders, worst first:

| truth | predicted | conf | margin |
|---|---|---|---|
| DOWNSIZE1 | VLOG | 0.913 | 0.884 |
| JEWELRY | ZEBRA | 0.833 | 0.695 |
| UNDERWEAR1 | SHARPEN2 | 0.736 | 0.649 |
| HURDLE/TRIP1 | SHARPEN2 | 0.722 | 0.644 |
| DEVELOP1 | ELEVATOR1 | 0.757 | 0.635 |

Note what these have in common: **very high confidence**. No threshold low
enough to keep any useful coverage removes them; the single worst case is at
0.884 margin.

---

## 3. Per-class thresholds — tested three ways, all rejected

The policy must key on the **predicted** class, because at runtime that is all
the app knows. Keying on the truth would be leakage.

### 3a. Fit on val, evaluate on test

Val has **2–6 clips per class, median 3**. Fitting 100 thresholds on three
samples each:

| target accuracy | val (in-sample) | test (held out) |
|---|---|---|
| 0.99 | cov 88.4%, **100.00%**, 0 wrong | cov 60.4%, 95.91%, 31 wrong |
| 0.97 | same | same |
| 0.95 | same | same |

The in-sample column is 100% accurate with zero errors — and the held-out column
is 95.91% at 60% coverage. That gap is the definition of overfitting, and it
is not subtle.

### 3b. Fit on train with leave-one-signer-out, apply to test

Train gives 14–24 clips per class instead of 3, and LOSO keeps signers out of
their own fit:

| target | coverage | acc@accept | wrong |
|---|---|---|---|
| 0.99 | 77.1% | 93.70% | 61 |
| 0.95 | 78.5% | 93.30% | 66 |
| 0.90 | 85.1% | 91.10% | 95 |

Better than the val fit, still not better than one global threshold (see 3c).

### 3c. The fair test: per-class SHAPE with a global dial

Comparing policies at whatever coverage each happens to land on is not a
comparison. So the per-class table gets one global multiplier `alpha`
(`accept iff margin >= alpha * thr[class]`), and `alpha` is dialled to hit the
same coverage as a single threshold.

| target coverage | global | per-class (val-fit) | per-class (train-LOSO) |
|---|---|---|---|
| 95% | 88.17% | 88.26% | 88.42% |
| 90% | 91.15% | 90.44% | 90.97% |
| **86%** | **92.12%** | 91.75% | 92.03% |
| 80% | **94.62%** | 93.53% | 93.43% |
| 75% | **96.17%** | 94.90% | 93.84% |
| 70% | **97.38%** | 95.67% | 94.19% |
| 60% | **98.80%** | 95.88% | 94.69% |

At the shipped operating point it is a wash; everywhere the gate does real work
it is **worse**, by up to 2.7 points at 70% coverage. The per-class spread
(`thr` ranges 0.014–0.961) is noise, not structure.

### 3d. Per-cluster, in case per-class was just too fine

Clusters built from the **train** confusion matrix only (greedy union at cosine
similarity 0.30/0.55/0.75), threshold per cluster, fit on val:

| clusters | per-cluster | global at the same coverage |
|---|---|---|
| 99 | cov 62.2%, 95.77%, 33 wrong | **98.46%, 12 wrong** |

Every clustering collapsed to essentially one class per cluster — the model's
confusion rows are not similar enough to group. And where it did group, it lost
by 21 wrong readings. **Rejected.**

---

## 4. The scorer itself is already optimal

If the gate accepts the highest-scoring clips, only the induced **ordering**
matters. Seven scores computable from the existing probability vector, compared
at matched coverage on test:

| coverage | top1 | **margin** | ratio | logratio | neg_entropy | top5_mass |
|---|---|---|---|---|---|---|
| 86% | 86 wrong | **85** | 85 | 85 | 96 | 103 |
| 80% | 62 | **54** | 57 | 57 | 79 | 80 |
| 75% | 45 | **36** | 40 | 40 | 58 | 62 |

`margin` (top1 − top2) wins or ties at every level. The shipped scorer is not
the problem, so no amount of re-weighting the score will fix this.

---

## 5. One alternative that was never tried, and nearly works

The app already loads **nine** independent classifiers. Whether they agree at
the argmax was never used. Recomputing per-member argmaxes over all 1,255 test
clips (agreement counts how many of the nine agree with the final answer):

| coverage | margin acc | agreement acc | margin wrong | agreement wrong |
|---|---|---|---|---|
| 95% | 88.17% | 87.67% | 141 | 147 |
| 90% | 91.15% | 90.18% | 100 | 111 |
| **86%** | 92.12% | **92.40%** | 85 | **82** |
| 80% | 94.62% | 94.12% | 54 | 59 |
| 75% | 96.17% | 95.32% | 36 | 44 |
| 70% | 97.38% | 96.24% | 23 | 33 |

Agreement is **marginally better at the shipped operating point** (82 wrong vs
85) and clearly worse as the gate tightens. At 9/9 unanimity it covers 57.5% of
clips at 98.06% accuracy with 14 wrong — but that is just another way of buying
accuracy by refusing, and the global threshold does it better at the same
coverage.

**Verdict: not worth shipping.** A 3-reading improvement at one operating point,
measured on a split already declared as inspected, does not justify new code the
day before submission.

---

## 6. The finding that actually matters

None of the above is the important result. This is.

The earlier decision to revert `MIN_MARGIN` from 0.40 back to 0.15 was justified
in `ResultPolicy.kt` by this sentence:

> *"the user's own correctly-signed ELEVATOR clip scores 49.9% top-1 with a 0.216
> margin, so at 0.40 the app would DECLINE a correct answer"*

**That is false.** Re-measured on 2026-11-19 through the shipped pipeline, with
the app's own `SignSegmenter` trim ported to Python and verified:

| | |
|---|---|
| recording | 15.86 s, 485 frames, hand coverage 0.181 |
| segmenter kept | frames [175,358) = 5.99 s (37.7%) |
| prediction | **FLIP** at 0.3046 |
| runner-up | CHEESEGRATER 0.0540 |
| margin | **0.2506** |
| **ELEVATOR1 rank** | **83rd of 100, at 0.0015** |
| per-member argmax | **8 of 9 say FLIP** |

The clip is not a correct answer at a thin margin. It is a wrong answer, and
the gate at **0.15 already accepts it**. The 49.9%/0.216 figure came from the
2×-speed replay that `docs/WINDOW_PLACEMENT_2026-11-19.md` already retracted —
the retraction was written, but the sentence it invalidated stayed in the source
comment.

The revert still stands, but for an honest reason: 0.15 → 0.40 cuts coverage
from 86% to 70%, and it **additionally refuses 143 correct answers while catching
62 wrong ones (2.31:1)**. 0.40 is also the worst point on the curve — 0.25 is a
far cheaper trade (48 correct refused for 34 wrong caught, 1.18:1 on the
0.20→0.25 step). That is a product decision, not a bug fix. Full frontier table
in `docs/ACCEPTANCE_GATE_2026-11-19.md`.

And the model is not incapable of ELEVATOR1 — on 15 clean test clips it gets
**73.3% recall, median confidence 0.847, median margin 0.801**, and 10 of 15
clear even 0.40. The phone clip scored 0.251, which puts it in the bottom 21% of
the entire corpus. That is a recording made 4× longer than the capture window,
with the hand in frame 18% of the time, not a model limitation.

---

## 7. What to do

**Do not implement per-class thresholds.** Measured four ways; it loses every
one.

**Do not raise the global threshold on the eve of submission.** It is the
owner's product call, not a unilateral change: 0.40 buys 97.4% accuracy by
refusing 30% of all captures, and the cost of the change (143 extra correct
refused vs 62 wrong caught) is the whole argument. If the gate is raised at all,
0.25 is the defensible point, not 0.40.

**Fix the recording instead.** The demo clip problem is procedural and the fix
is concrete:

1. Hold the sign for ~2–3 s and **tap Finish promptly** — the capture is 3.7 s
   ending at the tap. A 15.9 s recording is 4× too long.
2. Keep the hand **in frame** for the whole capture. 18% coverage is very low.
3. Sign **directly at the phone**, never a screen.
4. Verify before recording the real take: `python app/offline_demo.py --video
   <clip>` prints the same top-5 the app will show. If the margin is under 0.40,
   re-record rather than hoping.

A demo that shows a 0.85-margin correct reading and then a declined one is
stronger evidence for the "it tells you when it's unsure" claim than any
threshold change.

---

## Reproduce

```
C:/Python314/python.exe .scratch/score_probs.py train val test
C:/Python314/python.exe .scratch/gate_baseline.py
C:/Python314/python.exe .scratch/gate_frontier.py
C:/Python314/python.exe .scratch/per_class_thresholds2.py
C:/Python314/python.exe .scratch/threshold_shape.py
C:/Python314/python.exe .scratch/score_ranking.py
C:/Python314/python.exe .scratch/clusters_and_agreement.py
C:/Python314/python.exe .scratch/threshold_decision.py
C:/Python314/python.exe .scratch/score_phone_clip.py "<video>" --expect ELEVATOR1
```
