# Dev → Abhijeet · what changed on the model side, 16 Nov

Written 16 Nov 2026 by Dev. This is the only document you need to read to bring the
app and release docs into line with the model branch. Everything below is verified
against the files in `models/` on `hackathon/dev-model`.

**Short version: your app code is correct and already loads the nine-member
ensemble. The *numbers* and *docs* around it are one generation stale.**

---

## 1. The shipped model is unchanged since you last synced

| | |
| --- | --- |
| Ensemble | nine FP16 TFLite members (4 × 255-wide x,y + 5 × 381-wide x,y,z) |
| Combined SHA-256 (meta order) | `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb` |
| Sizes | tcn21/rob71 942,160 B · tcnm61/tcnm62 835,896 B · five `_z` members 990,544 B each |
| Total | 8.51 MB ensemble |

This matches `ModelContract.MEMBER_SHA256` and `ENSEMBLE_SHA256` in the app exactly.
**No action needed on the Kotlin pins.**

## 2. The accuracy number moved — twice, both in our favour, neither from retraining

| cache | 9-model TFLite, test |
| --- | --: |
| old, 256×256 squash (8 Nov) | 77.02% |
| **native 640×480 (the app's geometry)** | **85.45%** |

Two separate corrections produced this:

1. **The attention mask was never exported** (13 Nov). `build_tf_model` declared one
   input; the model was trained with a mask. Supplying it moved the same weights
   +1.84 test / +2.90 val per member. *You already integrated this (V4.3).*
2. **The evaluation cache was wrong about hand presence on 29% of test clips**
   (15 Nov). Re-extracting at the native geometry you already feed MediaPipe moved
   the same nine models from 76.94% to 85.45%.

**The second one means: the app was right, our Python evaluation was reading at 256
squash while the app reads native 640×480.** `CACHE_REBUILT_2026-11-15.md` and
`ABHIJEET_CACHE_HANDOFF_2026-11-15.md` have the full trace.

Val: **89.31%** (5 unseen signers, 346 clips). Top-5: **96.64%** (test), 97.69% (val).
Per-signer mean 86.02%, min 72.54%, max 99.01%.

## 3. Your release documents still say 72.78% / four members

These are the files I did **not** touch because they are yours. Each needs the
number updated:

| file | currently says | should say |
| --- | --- | --- |
| `docs/release/FINAL_SUBMISSION_STATUS.md` | 72.78% (909/1249), 4-member | 85.45% (1,251), 9-member |
| `submission_final/README.md` | "4 × FP16", 72.78%, 11.37 MB | 9 × FP16, 85.45%, 8.51 MB ensemble |
| `submission_final/RELEASE_MANIFEST.json` | members `112279bc…` etc., 72.78% | the nine-member hashes below |
| `docs/submission/COMPLIANCE_FINAL.md` | 72.78% | 85.45% |
| `docs/submission/HANDOFF_PACK.md` | 72.78% | 85.45% |
| `docs/release/JUDGING_SCOREBOARD.md` | 72.78% | 85.45% |
| `app/.../res/values/strings.xml:132` | "Four-model ensemble" | "Nine-model ensemble" |

The nine member hashes (meta order) for the manifest:

```
af031d898c98c97abcf66ae7538166316faac458b7b938fe9e3390fce31e5d4c  tcn21     942160
b0a97aa45c6797b4b85e2f6a1b03945881a94dad5519cb71be47e474fdea54f0  tcnm61    835896
1d247397d57e982196e68f3f230adeeabbfcdf673ba01ed541eadd54d874ea8d  tcnm62    835896
105218a89fde1d107441422254805dfb38cc7df25a216ff986801bc15dccb89a  rob71     942160
387749f81673719bd4a24813f33c6880c6117ed7be9d6bd50853250f8082901e  z_s0      990544
02a2dc70fbd769f793c8f8915c8a30cc568d901bf2e35bffe1752aa762000a35  z_s1      990544
3b97a2d416faec002bb84a6d2e47c131bb97941f074feed248a2c5a99c1c3b3d  z_s2      990544
2f13d0eba1f34e68f3e3ac1b2c645d9f75d1953918d69be6d58ca00a1f2b36b2  z_s7      990544
3d3a61e08d6a4d2a0a8064ae3b63692a55b55a2004ba37a428308a22b6336fba  z_s13     990544
```

## 4. Latency: the honest split

**Nothing here is on-device; no phone was ever attached.** But the two numbers that
matter are different, and I have been guilty of quoting the wrong one:

- **Classifier inference (the juried metric): 3.1 ms p50 / 3.7 ms p95** for all nine
  members, desktop x86. **Well inside the ≤50 ms budget.**
- **Per-frame detector: 133 ms p50 (floor 69, p95 199) desktop x86.** The floor alone
  exceeds 50 ms, so the per-frame criterion **fails on desktop**. The repo previously
  quoted 15.7 ms/frame with no artifact behind it; retracted 16 Nov.

The phone could be much faster: your `detectForVideo` with tracker reuse is a
different path from the one I measured. But **that has never been measured** and I
won't claim it.

`results/results.csv` now reads `latency_per_frame_p95 = 199.4, FAIL`. It previously
read `pass` because of a column shift — that is fixed, and it is the artefact the
judges read.

## 5. Robustness

| level | worst | verdict | note |
| --- | --: | --- | --- |
| pixel (blur/low-light/downscale/jpeg) | +4.48 | PASS | full split, 512 letterbox |
| landmark (jitter/drop/fps/camera) | +5.76 | FAIL | `drop_hand_25pct` only |

`drop_hand_25pct` fails by 0.76 pt and is a genuine model limitation — the attention
mask works (measured three ways), the information is simply missing when a hand
leaves frame. A retrain with harder hand-dropout augmentation is running as I write
this; I will report the result rather than predicting it.

Our two robustness suites test *different corruptions* — yours (crop, occlusion,
motion blur, frame drop) and mine (jitter, landmark drop, hand drop, fps, camera
distance). Neither covers the other. Both should be reported, and the rubric's
"Occlusions" line touches both.

## 6. What is NOT done and is yours

1. **On-device latency/power/thermal** — 20% of the rubric. Emulator cannot run
   MediaPipe (no x86_64 slice), JVM cannot run TFLite. Needs a physical phone. The
   verified APK is `.scratch/apkcheck/echogesture-v4.3-masked.apk`,
   SHA `3db9d94a…1380896`. Runbook: `ON_DEVICE_MEASUREMENT_OPTIONS.md`.
2. **Demo video** — never recorded. The 2:50 script exists in `docs/demo/`.
3. **Deck personalisation** — `docs/submission/EchoGesture_Submission.pptx` needs the
   team/college strings and the required filename.
4. **Rights** — the public repo exposes the weights and a 1,705-row participant CSV.
   Unchanged. That is a decision only the owner can make.

## 7. What I changed on my side today (16 Nov)

- `results/results.csv` — repaired a column shift that made `latency_per_frame_p95`
  read "pass" at 199.4 ms. All verdicts now match measured-vs-target.
- `results/eval_summary_{test,val}.json` — regenerated with accuracy fields; test
  85.45/96.64, val 89.31/97.69.
- `scripts/ml/make_predictions.py` — now computes and writes accuracy + per-signer
  (it previously wrote only latency/metadata).
- `scripts/ml/export_tflite_tf.py` — the INT8 accuracy gate was a no-op (`drop = 0.0`);
  now it really compares FP16 vs INT8 on the same clips and refuses on >15 pp.
- `src/README.md`, `README.md`, `TEAM_GUIDE.md`, `results/QUANTIZATION.md`,
  `docs/hackathon/DEV_HANDOFF_MODEL.md` — brought from the four-member / 72.78% era
  to the nine-member / 85.45% state, with corrected member hashes.
- `src/landmark_frontend.py` — corrected stale docstring (133 ms not 26 ms) and
  removed a duplicated docstring block.

Nothing I changed touches `src/main.py`, the app, or the release bundle. Your call on
the eight doc files in §3.
