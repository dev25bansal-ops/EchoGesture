# Organizer acceptance matrix — Track 02

Checked against `XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1 (2).pdf`,
pages 9–12 (Track 02) and 24–25 (dates, submission rules). Every row states the
official requirement, the current evidence, and a verdict from
**PASS / FAIL / NOT_TESTED / BLOCKED**.

Verdicts are evidence, not intent. A requirement with no measurement is
`NOT_TESTED`, never `PASS`.

| # | Requirement (source) | Current evidence | Verdict |
| --- | --- | --- | --- |
| **A** | Quantized TFLite/ONNX model < 20 MB (p.9 budget; p.10 "quantized TFLite/ONNX"; p.11 matrix scoring) | 4-member FP16 ensemble 3,544,080 B + FP16 detector 7,819,105 B = **11,363,185 B (10.84 MiB)**, under 20 MB. But the members are **FP16, not INT8**. INT8 candidates exist and are smaller but lose accuracy. | **PARTIAL — size PASS, INT8 format BLOCKED on organizer definition** |
| **B** | Pose-invariant 3D hand features (p.10) | Current features are **2D (x,y) only** — `use_z=false`, 255 dims = pos/vel/acc/geom over 2D landmarks. MediaPipe z is available but deliberately dropped. | **FAIL as literally worded** — the feature is pose-normalised 2D, not 3D. A 3D variant exists only as an unvalidated proposal (`docs/release/DEV_3D_FEATURE_PROPOSAL.md`) |
| **C** | Edge inference < 50 ms (p.9, p.11 target) | Measured `latency_p95 = 275.8 ms` on macOS laptop CPU (AI-Edge-LiteRT 2.3.0, XNNPACK). Ensemble inference itself is 0.41 ms; **hand tracking is ~90% of runtime**. No target-device measurement exists. | **FAIL** on the measured configuration; **NOT_TESTED** on a Galaxy-class phone |
| **D** | Offline, exocentric pipeline (p.9, p.10) | Python `predict()` and the Android app both run fully offline. The app declares **only** `android.permission.CAMERA`, no INTERNET. Third-person camera is the normal use. | **PASS** (offline + exocentric); battery/thermal budget is separately NOT_TESTED |
| **E** | Recognition accuracy on held-out signs (p.11, 35%) | `results.csv`: **top1_accuracy 72.78%** on the ASL Citizen signer-disjoint test split (909/1249 scored, 11 unseen signers). Validation-only re-scoring on this branch: **84.39% top-1 / 93.93% top-5** (n=346). Targets: ≥70% pass. | **PASS on the project split**; **NOT_TESTED on organizer test videos** (none supplied) |
| **F** | Robustness to low light, occlusion, motion, camera change (p.10, 25%) | `results.csv`: **accuracy_drop_robustness = 10.00 pt** (worst condition, signed drop) against a ≤5 pt target → **FAIL**. Per-condition from `submission_final/RELEASE_MANIFEST.json`: crop **10.0**, occlusion **6.67**, low_light −3.33 (improves), motion_blur 0.0, frame_drop 0.0 — 30 clips, 5 signers, seed 42, 210 real predictions. | **FAIL** |
| **G** | Useful HUD and captions (p.10, part of 25%) | Android V2 app: live preview, framing guide, capture countdown, result card with top-5, confidence wording, uncertainty label, local TTS, history, glossary. | **PASS** (built and emulator-verified); **NOT_TESTED** on a phone |
| **H** | Exact `predict(video_path)` interface (p.11) | `src/main.py::predict(video_path: str) -> dict` returns exactly `clip_id, prediction, top5, confidence, latency_ms`. Verified by a real Docker call returning `EAT1`. Extra optional `clip_id` kwarg does not change the one-argument contract. | **PASS** |
| **I** | Official class vocabulary (p.11 "fixed vocabulary list") | **No Track 02 vocabulary was supplied anywhere in the deck.** The project uses its own 100-gloss list (`models/landmark_labels.json`), uppercase and unique. | **BLOCKED** — requires organizer input |
| **J** | Exact evaluation clip IDs (p.11 "clip_id must match the test manifest exactly") | **No Track 02 test manifest was supplied.** `results/predictions.jsonl` uses the project's own ASL Citizen test clip IDs. | **BLOCKED** — requires organizer input |
| **K** | `predictions.jsonl` and `results.csv` (p.11/p.12) | `results/predictions.jsonl` (1249 rows, sha256 `1aacbacb…`) and `results/results.csv` (6 metric rows) both present. `results/predictions.json` (p.12 names `.json`) is **absent**. | **PARTIAL** — `.jsonl` PASS, `.json` missing |
| **L** | `results.csv` with every required metric (p.11) | All six required metrics present: `top1_accuracy, latency_p95, model_size, accuracy_drop_robustness, avg_power, thermal_drift_10min`. Two are `NOT_MEASURED`. | **PARTIAL** — structure PASS, 2 metrics NOT_TESTED |
| **M** | Python 3.11 reproducibility (p.12) | `src/requirements.txt` pins CPython-3.11-compatible versions; Docker image is `python:3.11-slim`; real inference verified in-container. | **PASS** (container); clean-host install NOT_TESTED |
| **N** | GitHub repo, README, `predict()` in `main.py` (p.12) | Public repo exists; `src/main.py` present. README exists but **does not document Docker or the evaluator contract**. | **PARTIAL** |
| **O** | Docker / setup instructions (p.25 "Docker files & other requirements") | `Dockerfile` at repo root builds and runs (verified). Setup instructions **absent from README**. | **PARTIAL** — artifact PASS, documentation FAIL |
| **P** | Final demo video + presentation (p.25) | No video recorded. No presentation produced. Naming convention `CollegeName_TeamName` required for both. | **NOT_TESTED / BLOCKED** on hardware + team metadata |
| **Q** | Legal rights and privacy (p.9 privacy principle) | App is offline with CAMERA-only permission and no cloud path; diagnostics store no images or landmarks. **Model weights are not redistributable** — see `PUBLIC_WEIGHT_EXPOSURE.md`, `ARTIFACT_RIGHTS.md`. Public repo carries source only. | **PASS** on privacy; **BLOCKED** on weight redistribution rights |

## Discrepancies in the official material

These are conflicts inside the supplied deck itself, not project failures.

1. **Model size target is stated twice, differently.** p.9 says "≤ 20 MB
   quantized model (INT8)"; p.11's results-sheet template says `model_size
   <=25 MB`. The project passes both (10.84 MiB) but the *definition* of what
   must be quantized is what matters — see 2.
2. **"INT8" vs "quantized TFLite/ONNX".** p.9 and the deliverable line on p.10
   say INT8; the scoring line on p.11 says "Compilation to TFLite/ONNX <20MB"
   without naming INT8. Under the narrower reading the project fails (FP16);
   under the broader reading it passes.
3. **`predictions.json` vs `predictions.jsonl`.** p.11 requires a single
   `predictions.jsonl`; p.12's bundle diagram shows `results/predictions.json`.
4. **Demo video length.** p.10 says "2–3 min"; p.24 says "5-min demo video";
   p.25 says "max 3 minutes". The safe reading is ≤3 minutes.
5. **Deadline year.** The deck is titled 2026, and p.25 says both "8 Nov 2026"
   and "18th Nov 2026"; p.27 says "5th Nov 2026". If today is 13 Nov 2026, the
   8 Nov registration deadline has already passed and 18 Nov is 5 days away.
6. **Track 02 has no starter kit.** Track 04 (p.21–22) explicitly ships
   `manifest.csv`, `supported_classes.json` and `validate.py`. Track 02's
   pages contain no equivalent, so items I and J have no supplied input.

## Priority

Sorted by disqualification risk, then judging weight, then time to fix.

| Priority | Item | Why | Time |
| --- | --- | --- | --- |
| **P0** | I, J — vocabulary + manifest | p.25: "Teams NOT following the submission guideline would lead to direct disqualification". Without the official list the evaluator contract cannot be satisfied exactly. | Blocked; clarification is minutes, the reply is not |
| **P0** | Registration status | Registration closed 8 Nov 2026. If the team is not registered, the submission link was never shared. | Minutes to check |
| **P0** | K, O — `predictions.json`, README Docker section | Trivial to fix, explicitly required by name. | ~1 h |
| **P0** | A/E — INT8 definition + accuracy claim | 40% of the score is execution/efficiency, 35% accuracy. Cannot be resolved without a clarification reply. | Clarification email |
| **P1** | C, F — latency and robustness FAIL | 20% + 25% of the score. Latency is dominated by hand tracking, not the classifier, so it is addressable by engineering. | Days |
| **P1** | P — video, deck, naming | Hard requirement; needs a phone and team metadata. | Days |
| **P2** | B — 3D features | Literal wording failure. Fixing it properly is a model-contract change and would invalidate the frozen accuracy evidence. | Weeks — out of scope now |
| **P2** | M — clean-host install | Container path already works. | Hours |

## What cannot be claimed

- **Not** "INT8-compliant". The shipped classifier is FP16 and the detector is
  FP16.
- **Not** "<50 ms". Measured 275.8 ms p95 on laptop CPU.
- **Not** "3D pose-invariant features". The features are 2D.
- **Not** "robustness ≤5 pt". Measured 10 pt drop.
- **Not** "phone-verified". No phone session has been run.
- **Not** "official vocabulary / official test IDs". Neither has been supplied.