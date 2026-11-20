# What we can claim, and what we cannot

Written 16 Nov 2026 by Dev, for the submission. Every row states the evidence
and how to reproduce it. Anything marked **cannot claim** should not appear in a
deck, a demo script or a README without being qualified the same way here.

## Can claim, with evidence

| Claim | Number | Evidence | Reproduce |
| --- | --- | --- | --- |
| Top-1 on unseen signers | **85.45%** | 9-model FP16 ensemble, 1,251 clips, 11 unseen participants, 100 classes, chance 1.00% | `python scripts/ml/eval_tflite_cache.py --cache .scratch/landmarks_native` |
| Top-5 | **96.64%** | same run | same |
| Val top-1 | **89.31%** | 346 clips, 5 unseen signers | `... --split val` |
| Beats chance by | **85×** | 85.45 vs 1.00% chance | arithmetic |
| Quantization cost | **0.00 pts** | FP16 vs FP32, half the size | `results/QUANTIZATION.md` |
| Model size | **16.33 MB** | 9 FP16 members 8.51 MB + MediaPipe detector 7.82 MB — the weight payload a judge finds in the APK. Classifiers alone are 8.51 MB | `ls -l models/*_fp16.tflite models/hand_landmarker.task` |
| Pixel robustness | **+4.48 PASS** | full split, 4 corruptions, all under the 5-pt budget | `scripts/ml/robustness_pixel_parallel.py --cache .scratch/landmarks_native` |
| Classifier latency | **~4 ms/clip p95** | nine members, desktop x86, warm; per-clip p50 3.3–3.7 / p95 3.9–4.1 ms across two clean runs of the cited script (`results.csv` row `latency_model_only_p95`) | `scripts/ml/eval_tflite_cache.py` |
| **Per-frame detector latency** | **p50 15.3 ms / p95 23.0 ms** | desktop x86, 160 frames × 3 clean runs (the older n=192 sample belonged to the retracted CPU-contended run) — **inside the 50 ms budget** | `python scripts/ml/latency_per_frame.py` |
| Confidence calibration | **99.7% at >0.8** | 594 of 1,251 clips; 64.9% of all errors sit below 0.4 (118/182, matches the band table below) | `python scripts/ml/confidence_calibration.py` |
| Abstain-and-reask | **95.93% at 72.6% coverage** | answering only clips above 0.5 confidence | same |

## Can claim, with a caveat that must travel with it

| Claim | Number | The caveat |
| --- | --- | --- |
| Landmark robustness | +5.76 raw FAIL / **+2.39 gated PASS** | Unconditional per-clip scoring fails on `drop_hand_25pct` (+5.76). At the app's own margin ≥ 0.15 acceptance gate the same corruption costs +2.39 at 86%/80% coverage (92.13% clean / 89.74% corrupted). Corrected 2026-11-18: the previous +5.52 / +1.24 pair used the coords-kept diagnostic for its corrupted column and matched no committed script; re-measured by `scripts/ml/robustness_gated_recheck.py`, clean column reproduced exactly. Both rows ship in `results.csv`. A harder-hand-dropout retrain was also tried and lost more clean accuracy than it recovered. See [DROP_HAND_ANALYSIS.md](DROP_HAND_ANALYSIS.md) |
| Cold start | **15–63 s** (machine-state dependent) | One-time per process, dominated by MediaPipe's lazy graph build on first `detect()`. Measured 62.6 s on 17 Nov (import 0.9 s + graph 60.9 s + first predict 0.9 s) and 14.7 s on an 18 Nov clean re-run of the same script — the spread is real and any single figure overstates its precision. Paid inside `_init_model()`, so it never lands in a scored prediction. The earlier 159 s was measured under CPU contention. Pinning to 8 cores made it *worse* (94 s) — MediaPipe wants more cores |
| End-to-end per clip | **~720 ms** | Warm `predict()` on a 32-frame clip, desktop x86. The detector dominates: an isolated detector-only pass is ~600 ms (20.3 ms/frame over 160 frames); the stage figure of 1,136 ms in the committed JSON was a separate clip including video decode, and results.csv's old '4.2 ms classifier = 0.6%' pairing mixed a mean into a single-pass stage split. The classifier is ~4 ms mean / ~8 ms p95. This does not come down without a cheaper detector model. The earlier 4,650 ms was CPU-contended |

### Confidence is the strongest demo claim we have

Re-measured on the native cache after the geometry fix:

| confidence | clips | accuracy | share of all errors |
| --- | --: | --: | --: |
| 0.0–0.2 | 67 | 28.4% | 26.4% |
| 0.2–0.4 | 172 | 59.3% | 38.5% |
| 0.4–0.6 | 189 | 75.1% | 25.8% |
| 0.6–0.8 | 229 | 93.4% | 8.2% |
| **0.8–1.0** | **594** | **99.7%** | **1.1%** |

**64.9% of all mistakes are below 0.4 confidence** (118 of 182 errors; the band
table above sums to 26.4 + 38.5). Above 0.8 confidence the model
is right 99.7% of the time over 594 clips.

Two operating points that make a good live demo:

- **abstain below 0.5**: answers 72.6% of clips at **95.93%** accuracy
- **abstain below 0.7**: answers 57.9% of clips at **99.03%** accuracy

This converts the most visible failure mode — a fluent-looking wrong answer —
into an honest "sign that again". It is also the one place where the model's
behaviour matches its confidence closely enough to say so out loud.

## Cannot claim — no measurement exists

| Item | Why not |
| --- | --- |
| ~~**On-device latency**~~ | **RETRACTED — it was measured.** This row previously said on-device latency was "much slower than the desktop figure" and quoted CPU-delegate numbers (analyse_p50 84-112 ms, detect_p95 159-187 ms, 8.7-11.6 fps). Those were real but they were the DEFAULT CPU delegate, because `HandTracker.create()` set no delegate at all. Enabling the GPU delegate changed it: **detect p50 34-45 ms, p95 46-54 ms, 19.2-25.0 fps** on the same Vivo V2404. See `results.csv` rows `latency_p95` and `latency_on_device_p50`. Honest reading: p50 passes in every run, p95 passes in 3 of 4 and is 4 ms over in the hottest run, so the scored row is `partial`, not `pass`. |
| ~~**Power ≤ 500 mW**~~ | **RETRACTED — a measurement exists.** `results.csv` row `avg_power` records 2666-2898 mW across three windows on the camera path, marked NOT_APPLICABLE rather than pass because the camera line covers sensor+ISP+encode, not inference alone. The number is measured; the attribution to inference is what is unresolved. |
| ~~**Thermal drift ≤ 20%**~~ | **RETRACTED — a measurement exists.** `results.csv` row `thermal_drift_10min` records 19.2% (SoC 45.8→56.7 °C) over a 615 s run, against a ≤20% target. It passes with 0.8 points of headroom, with the caveat that the trace COOLS from its t=0 peak, so it is an initial transient rather than a runaway. |

An earlier version of this file called these "20% of the rubric and the honest gap
in this submission". That was true when written and is no longer: all three now
have measurements, and the honest statement is the one above — one clean pass,
one partial, one not-attributable.

## Claims made earlier and retracted

Recorded so nobody re-quotes them. All four were inherited figures with no
retained artifact; I checked each on 16 Nov.

| Claim | Was | Actually | Factor |
| --- | --: | --: | --: |
| MediaPipe per frame | 15.7 ms | **p50 15.3 ms (clean runs)** | ~1× — the inherited number was roughly right |
| Per-frame p95 | 32.80 ms (PASS) | **22.95 ms (PASS)** | the retraction of this one was itself wrong |
| Cold start | 8,800 ms | **159,000 ms** | 16× |
| End-to-end p50 | 470 ms | **4,650 ms** | 10× |

The pattern is worth stating plainly: the accuracy numbers were verified
repeatedly and survived, and the latency numbers had never been checked from
source until now.

**A correction to the correction.** On 16 Nov I re-measured per-frame latency at
p50 133 ms / p95 199 ms and retracted the older 15.7 ms / 32.80 ms figures as 8×
optimistic. That retraction was itself wrong. Those measurements were taken while a
`distllm` inference server (a separate project) was saturating this machine's CPU.
Re-run on an unloaded machine, the *same script* reports **p50 14.5–16.4 ms, p95
22.7–25.1 ms** across three consecutive runs — inside the 50 ms budget, and close to
the figure originally inherited. The lesson is the one that keeps recurring here: a
latency number measured under contention describes the contention, not the code.

## One correction that went the other way

Everything accuracy-related was ~9 points **too low** because the landmark cache
was built at the wrong geometry. The shipped 77.02% became 85.45%. That is the
single largest change in this project and it was in our favour — but only
because the earlier number was measured against input the app never produces.

## The one-line summary for a judge

**85.45% top-1 on unseen signers, 16.33 MB total (9 members 8.51 MB + detector
7.82 MB), per-frame detector p50 34-45 ms on the phone (GPU delegate; p95 46-54
ms, passing in 3 of 4 runs), classifier 4 ms — both inside the 50 ms budget —
with pixel robustness passing at +4.48, landmark occlusion failing at +5.76
ungated and passing at +2.39 with the app's acceptance gate, and a 615 s thermal
run at 19.2% drift.**
