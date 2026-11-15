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

## Can claim, with a caveat that must travel with it

| Claim | Number | The caveat |
| --- | --- | --- |
| Landmark robustness | +5.76, **FAIL** | Fails on `drop_hand_25pct` alone; other five pass. It is a model limitation, not a measurement artefact — measured three ways, the attention mask is working correctly. See [DROP_HAND_ANALYSIS.md](DROP_HAND_ANALYSIS.md) |
| Per-frame latency | p50 **133 ms** | Desktop x86 Python. The budget is 50 ms and even the floor (~70 ms) exceeds it. Wide run-to-run spread (84–163 ms p50). **Not** an on-device figure |
| Cold start | **159 s** | One-time per process, and 135 s of it is MediaPipe building its graph on the first `detect()`. Now paid inside `_init_model()`, so it does not pollute a timed prediction |

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
| MediaPipe per frame | 15.7 ms | **p50 133 ms** | 8× |
| Per-frame p95 | 32.80 ms (PASS) | **199 ms (FAIL)** | 6× |
| Cold start | 8,800 ms | **159,000 ms** | 16× |
| End-to-end p50 | 470 ms | **4,650 ms** | 10× |

The pattern is worth stating plainly: the accuracy numbers were verified
repeatedly and survived, and the latency numbers had never been checked from
source until now. Every one was optimistic.

## One correction that went the other way

Everything accuracy-related was ~9 points **too low** because the landmark cache
was built at the wrong geometry. The shipped 77.02% became 85.45%. That is the
single largest change in this project and it was in our favour — but only
because the earlier number was measured against input the app never produces.

## The one-line summary for a judge

**85.45% top-1 on unseen signers, 8.51 MB, pixel-robust within budget, with
on-device latency, power and thermal still unmeasured.**
