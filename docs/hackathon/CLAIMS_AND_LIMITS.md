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
| Model size | **8.51 MB** | nine members, summed from the meta's member list | `ls -l models/*.tflite` |
| Pixel robustness | **+4.48 PASS** | full split, 4 corruptions, all under the 5-pt budget | `scripts/ml/robustness_pixel_parallel.py --cache .scratch/landmarks_native` |
| Classifier latency | **4.2 ms/clip** | nine members, desktop x86 | `scripts/ml/eval_tflite_cache.py` |
| **Per-frame detector latency** | **p50 15.3 ms / p95 23.0 ms** | desktop x86, 160 frames × 3 clean runs, n=192 prior — **inside the 50 ms budget** | `python scripts/ml/latency_per_frame.py` |
| Confidence calibration | **99.7% at >0.8** | 594 of 1,251 clips; 74% of all errors sit below 0.4 | `python scripts/ml/confidence_calibration.py` |
| Abstain-and-reask | **95.93% at 72.6% coverage** | answering only clips above 0.5 confidence | same |

## Can claim, with a caveat that must travel with it

| Claim | Number | The caveat |
| --- | --- | --- |
| Landmark robustness | +5.52 raw / **+1.24 gated PASS** | Unconditional per-clip scoring fails on `drop_hand_25pct` (+5.52); at the app's own margin ≥ 0.15 acceptance gate it is +1.24 PASS (92.13% clean / 90.89% corrupted, 86%/80% coverage). Both rows ship in `results.csv`. A harder-hand-dropout retrain was also tried and lost more clean accuracy than it recovered. See [DROP_HAND_ANALYSIS.md](DROP_HAND_ANALYSIS.md) |
| Cold start | **159 s** | One-time per process, and 135 s of it is MediaPipe building its graph on the first `detect()`. Now paid inside `_init_model()`, so it does not pollute a timed prediction |

### Confidence is the strongest demo claim we have

Re-measured on the native cache after the geometry fix:

| confidence | clips | accuracy | share of all errors |
| --- | --: | --: | --: |
| 0.0–0.2 | 67 | 28.4% | 26.4% |
| 0.2–0.4 | 172 | 59.3% | 38.5% |
| 0.4–0.6 | 189 | 75.1% | 25.8% |
| 0.6–0.8 | 229 | 93.4% | 8.2% |
| **0.8–1.0** | **594** | **99.7%** | **1.1%** |

**74% of all mistakes are below 0.4 confidence.** Above 0.8 confidence the model
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
| **On-device latency** | Never measured. The emulator cannot run MediaPipe (Google's AAR has no x86_64 native library) and the phone disconnected mid-install. The Android path uses `detectForVideo` with tracker reuse at native 640×480, which is a *different* path from the one measured here — it could be faster or slower |
| **Power ≤ 500 mW** | No measurement hardware, no capture |
| **Thermal drift ≤ 20%** | No sustained run performed |

These are **20% of the rubric** and they are the honest gap in this submission.

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

**85.45% top-1 on unseen signers, 8.51 MB, per-frame detector 15 ms and classifier
4 ms — both inside the 50 ms budget — with pixel robustness passing, one landmark
corruption failing, and on-device latency, power and thermal still unmeasured.**
