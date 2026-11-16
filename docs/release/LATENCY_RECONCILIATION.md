# Latency: what is actually measured, and the contradiction to fix before presenting

Written 2026-11-16 while auditing Dev's branch `hackathon/dev-model` @ `465166f`.
**This document exists because the branch contradicts itself, and a judge who
reads two files will find it.**

---

## The contradiction, stated plainly

`465166f` reports two per-frame detector numbers, in the same commit:

| Artifact | Per-frame detector | Verdict |
| --- | --- | --- |
| `results/latency_per_frame_distribution.json` | **15.34 ms p50 / 22.95 ms p95** | PASS under 50 ms |
| `results/results.csv` (`latency_per_frame_p95`) | **22.95 ms p95** | pass |
| `docs/hackathon/CLAIMS_AND_LIMITS.md` | **15.3 / 23.0 ms** | inside budget |
| `docs/hackathon/LATENCY_DECOMPOSITION.md` | **135 ms/frame** (4,300–4,400 ms ÷ 32 frames) | budget FAILS |
| `src/landmark_frontend.py` (module docstring) | **133 ms p50, min 69** | "criterion FAILS" |
| `results.csv` (`latency_end_to_end_p50`) | **4,650 ms** total | — |

### Why these cannot both be true

`detect_hands()` processes frames **in a Python loop, one at a time**
(`src/landmark_frontend.py:184-196`). Cost per frame therefore *multiplies* by
frame count. For a 32-frame clip:

- At **15.34 ms/frame** → detection ≈ **491 ms**, and end-to-end would be well
  under 1 s.
- At **135 ms/frame** → detection ≈ **4,320 ms**, matching the measured
  **4,650 ms** end-to-end.

The 4.65 s end-to-end figure is a direct timing of `predict()`, and its
decomposition (decode 45–312 ms + detection 4,300–4,400 ms + features 6–8 ms +
classifier ~7 ms) sums to the measured total. **Only ~135 ms/frame is consistent
with it.** The 15 ms figure cannot be reconciled with the same session's own
end-to-end measurement.

### Which artifact is defensible

`LATENCY_DECOMPOSITION.md` also pre-empts the contention explanation directly:

> "A `distllm` model server was competing for CPU during these runs, but **pinning
> to an idle core gave p50 163 ms**, so contention is not the explanation."

That directly contradicts the note in `latency_per_frame_distribution.json`, which
attributes the 133/199 ms figures entirely to a `distllm` server and reports
15 ms "on an unloaded machine".

**Conclusion: treat the 15.34 ms PASS as unverified.** The end-to-end
measurement is the one that is internally consistent, and it implies failure of
the per-frame budget on desktop. The `latency_per_frame_distribution.json`
numbers appear to be measuring something narrower than a real detection pass —
most plausibly a warm call that is not re-running full detection — and that has
not been demonstrated.

**Do not put 15.3 ms on a slide until this is re-run and reconciled.** A judge
who divides 4,650 by 32 gets 145 ms and will ask.

---

## The figures that are not in dispute

These are separate measurements and must never be combined:

| Measurement | Value | What it is | Source |
| --- | --: | --- | --- |
| Nine-member classifier | **4.20 ms p95** (3.1 p50) | desktop x86, warm process, all nine `.tflite` interpreters, classifier only | `results.csv` `latency_model_only_p95`; `eval_summary_test.json` |
| Video decode + resize | 45–312 ms | per clip | `LATENCY_DECOMPOSITION.md` |
| Feature build | 6–8 ms | per clip, both widths | same |
| Full `predict()` clip→JSON | **4,650 ms p50** | desktop x86, warm process, 32 frames | `results.csv` |
| Cold start | **159,000 ms** | one-time per process; 134.8 s is MediaPipe lazy graph build | `results.csv` |
| Android on-device | **NOT MEASURED** | — | — |
| Physical phone clip→result | **NOT MEASURED** | — | — |

The classifier — the part this project spent days improving — is **0.15%** of
end-to-end time. Presenting "4 ms" or "15 ms" as the system's speed is misleading
by two orders of magnitude.

---

## Is the 50 ms budget even the per-frame criterion?

**UNKNOWN — unverified.** The brief states a ≤50 ms latency target. It is not
established from the official rubric that the scored quantity is *per-frame
detector* time rather than *clip-to-result*. Treating per-frame as the scored
metric is a **choice of interpretation, not a verified reading**.

Until the rubric is quoted verbatim:

- Do not claim "latency criterion PASS".
- Do not claim "within the 50 ms budget".
- Say instead: *"Classifier inference is ~4 ms. The full pipeline on a laptop is
  ~4.65 s, dominated by the MediaPipe detector. On-device latency is unmeasured."*

---

## What to say in the demo

> "The classifier is about four milliseconds. The whole pipeline is not — it is
> about four and a half seconds on a laptop, and almost all of that is MediaPipe
> detecting hands frame by frame. We have not measured it on a phone. I'm not
> going to quote a number I haven't measured on the device it runs on."

That is honest, checkable, and it does not depend on resolving the 15-vs-135
question. Do not quote a per-frame PASS.

---

## Action for Dev

1. Re-run `scripts/ml/latency_per_frame.py` and `predict()` **back to back in the
   same process, same machine, same session**, and publish both.
2. If per-frame is genuinely ~15 ms, the end-to-end must fall to ~0.7 s. Measure
   it and show it. If it does not, the 15 ms is not measuring detection.
3. Update the stale `src/landmark_frontend.py` docstring, which still says
   "FAILS" while the results files say PASS. It is the first thing a reader of
   that module trusts.
4. Fix the `n=192` vs `n=160` inconsistency in the `latency_per_frame_p95` row of
   `results.csv` — 192 was the old contended run.

None of this requires retraining or touching the weights.
