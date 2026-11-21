# Track 2 compliance matrix — verified against the organizer PDF

Source: `XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf` (28 pages).
Track 2 is pages 9–12; general submission rules are 24–27. Every quote below is
transcribed from that PDF's text layer, not from an earlier repo summary.

**States:** `PASS` · `FAIL` · `PARTIAL` · `BLOCKED_EXTERNALLY` · `NOT_TESTED` ·
`FIXED_AND_VERIFIED`

Compiled 2026-11-12. **Revised 2026-11-20** — see §10. Where the deck
contradicts itself the row says so rather than picking a side.

---

## 1. Scoring rubric (p.11, 100-point jury matrix)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R1 | Model quantized | `Model Quantization (20%): Compilation to TFLite/ONNX <20MB.` | **9 × FP16 TFLite, 8,508,832 B** (8.51 MB); 16,327,937 B (16.33 MB) including the detector | `models/landmark_model_meta.json`, `results/results.csv` rows `model_size` / `model_size_classifiers` | **PASS** (format), see R8 |
| R2 | Edge latency | `Edge Latency (20%): Measured inference time <50ms.` | **on-device, GPU delegate: p50 34–45 ms (passes in all 4 runs), p95 46–54 ms (3 of 4 runs pass)** | `results/results.csv` rows `latency_on_device_p50` / `latency_p95` | **PARTIAL** — p50 passes, p95 does not in every run |
| R3 | Top-1 accuracy | `Top-1 Accuracy (35%): Precision on held-out test videos.` | **1,069 / 1,251 = 85.45%** on 11 unseen signers | `results/results.csv` row `top1_accuracy` | **PASS** (≥70% target) |
| R4 | Robustness & HUD | `Robustness & Delivery (25%): Occlusions & HUD clarity.` | Occlusion is the one weak condition: raw `drop_hand_25pct` **+5.76 pt** (FAIL against ≤5), **+2.39 pt at the shipped acceptance gate** (pass). All other robustness conditions pass. HUD now distinguishes an accepted reading from a declined one. | `results/results.csv` rows `accuracy_drop_robustness` / `_gated`; `docs/ROBUSTNESS_OCCLUSION_2026-11-20.md` | **PARTIAL** — 1 of 5 landmark conditions fails raw; passes gated |
| R4b | Results-sheet template | p.11 table: `metric target measured unit device runtime pass` with rows `top1_accuracy >=70 %`, `latency_p95 <=50 ms`, **`model_size <=25 MB`**, `accuracy_drop_robustness <=5 pt`, `avg_power <=500 mW`, `thermal_drift_10min <=20 %` | `results/results.csv` carries all six metric names in this order, with the p.11 target values, plus supporting rows | `results/results.csv` | **PASS**. Note: the sheet's own target is **25 MB**, not the 20 MB of p.9/p.10; 16.33 MB passes both readings. |

## 2. Hard budgets (p.9)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R5 | Model size | `Budgets: ≤ 20 MB quantized model (INT8)` | **16,327,937 B** total (8.51 MB classifiers + 7.82 MB detector), under both the 20 MB and 25 MB readings | `results/results.csv` row `model_size` | **PASS** |
| R6 | Latency budget | `≤ 50 ms latency` | see R2 — p50 on-device 34–45 ms, p95 46–54 ms | `results/results.csv` | **PARTIAL** (p50 passes, p95 marginal) |
| R7 | Zero cloud | `zero cloud streaming` | APK requests CAMERA only; INTERNET and ACCESS_NETWORK_STATE removed from the merged manifest | `evaluation/device/android_four_model_evidence.md`; `app/android_final/app/src/main/AndroidManifest.xml` | **FIXED_AND_VERIFIED** |
| R8 | INT8 specifically | p.9 `quantized model (INT8)`; p.10 `< 20 MB INT8`; p.11 rubric says only `TFLite/ONNX <20MB` with no INT8 | shipped artifact is **FP16**, 9 members | `models/landmark_model_meta.json` → `"quantization": "float16"` | **BLOCKED_EXTERNALLY** — organizer clarification drafted, **not sent** |

## 3. Architecture constraints (p.10)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R9 | Exocentric third-person | `Exocentric third-person view: moving heads, angled signers, low light` | Android prefers `DEFAULT_BACK_CAMERA` with front fallback, no handedness swap | `MainActivity.kt` | **FIXED_AND_VERIFIED** |
| R10 | **3D hand features** | `Pose-invariant 3D hand features` | **5 of the 9 shipped members consume x,y,z** (381-wide features including the MediaPipe z channel); 4 consume x,y (255-wide). The z-consuming members are the five `mirror_handdrop_tcn_z_s*` seeds. | `models/landmark_model_meta.json` → `members_detail[].use_z` | **PASS** — see §7 |
| R11 | Local processing | `strictly local processing, zero cloud streaming` | same as R7 | — | **PASS** |
| R12 | Captions | `Captions on display or audio output for two-way conversation` | on-screen caption, top-5 alternatives, and on-device TTS | `app/android_final/.../CameraScreen.kt` | **PASS** |

## 4. Required deliverables (p.10, p.11)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R13 | `predict()` | `GitHub repo: predict() video → text, reproducible setup`; `the predict() in main.py` | `predict(video_path, *, clip_id=None, profile=False) -> dict`, thread-safe under a module lock | `src/main.py` | **PASS** |
| R14 | Predictions file | p.10 `predictions.jsonl (one JSON per clip)`; p.11 `Teams submit a single predictions.jsonl file, one JSON object per line, one line per test clip`; but p.11 package bullet says `predictions.json and results.csv in results folder` and p.12 layout says `results/predictions.json` | builder emits **both** `.jsonl` and `.json` with identical JSON-Lines content | `results/predictions.jsonl` | **PASS** (deck contradicts itself; both spellings shipped) |
| R15 | Results sheet | p.11 `A CSV with one row per metric. Each row has a measured value, a target, and a pass/fail column.` plus the template table | 8 p.11 columns; the six template metrics plus supporting rows | `results/results.csv` | **PASS** |
| R16 | Return object | p.11 `{"clip_id","prediction","top5","confidence","latency_ms"}` | exact five keys | `src/main.py` | **PASS** |
| R17 | Manifest IDs | `clip_id must match the test manifest exactly` | no organizer manifest was supplied anywhere in the repo; the builder refuses to fabricate one | `scripts/release/build_bundle.py` | **BLOCKED_EXTERNALLY** |
| R18 | Conversion script | p.10 `Quantized model + conversion script (< 20 MB INT8)`; p.11 `Quantized model file and conversion script` | `scripts/ml/export_tflite_tf.py` is present in the built release ZIP | release bundle listing | **PASS** |
| R19 | Demo video | p.10 `Demo video (2–3 min) or deployed app URL`; p.24 `Prototype code, 5-min demo video and deck due`; p.25 `Demo video, max 3 minutes (YouTube or Drive link)`; p.27 `a 5-minute demo video by 18th November` | **none recorded**; the recording guide targets a 2:50 cut, which satisfies all three readings | `docs/demo/DEMO_RECORDING_GUIDE.md` | **NOT_TESTED** (no video file exists in the repo) |
| R19b | Installable APK | not required by the PDF, but strengthens Delivery | `submission/apk/EchoGesture-2026-11-21.apk`, 82,894,510 B, SHA-256 `96f00458f54de715c7a6f489f9ae92425bc73a089043a91ccbe0ae6d4f060035`, arm64-v8a, minSdk 24, targetSdk 34, built from `29c48b6` | `submission/apk/README.md` | **PASS** |

## 5. Submission rules (p.24–27)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R20 | Deadline | `FINAL SUBMISSION · Last date 18 Nov 2026, 11:59 PM` | **extended to 21 November 2026** | `docs/submission/INT8_CLARIFICATION_2026-11-19.md` records the extension | **PASS** (extension applied; timezone still unstated in 28 pages) |
| R21 | Private repo OK | `Working prototype code — public or shared GitHub repo` | public repo `dev25bansal-ops/EchoGesture`, branch `submission/final` | `git remote -v`, branch listing | **PASS** |
| R22 | Docker | `README with reproducible setup instructions, Docker files & other requirements` | root `Dockerfile`, Python 3.11; `python:3.11-slim` installs the X11/EGL/GLES libs MediaPipe loads lazily | `Dockerfile`, `evaluation/device/DOCKER_ACCEPTANCE.md` | **FIXED_AND_VERIFIED** — fresh build runs `--help` and a real prediction (`WHATFOR1`, 0.8818); see §9.1 |
| R23 | Deck naming | `Presentation file (PPT or PDF) — CollegeName_TeamName` | deck exists; slide 1 placeholders unfilled and filename not yet renamed | `docs/submission/EchoGesture_Submission.pptx` | **BLOCKED_EXTERNALLY** (team/college strings) |
| R24 | ZIP layout | p.12 `team_name.zip` → `src/{requirements.txt,main.py,README.md}` + `results/{predictions.json,results.csv}` | builder produces this and verifies the layout on a clean extraction; adds `models/`, `manifest.json`, `release.json` | `scripts/build_submission_bundle.py` | **PASS** |
| R25 | Python 3.11 | p.12 `Python 3.11 – all scripts are written for Python 3.11` | `FROM python:3.11-slim`; local venv is CPython 3.11.18 | `Dockerfile`, `.venv/bin/python -V` | **PASS** |
| R26 | Disqualification | `Teams NOT following the submission guideline would lead to direct disqualification` | deck naming and team strings still unfilled | — | **BLOCKED_EXTERNALLY** |

---

## 6. The two contradictions in the organizer deck

Recorded rather than resolved, because we cannot resolve them from inside:

1. **INT8.** p.9 and p.10 say `INT8`; the p.11 scored rubric says only
   `Compilation to TFLite/ONNX <20MB`. The 20-point quantization score is
   defined on p.11, which does not require INT8. We ship FP16 and state it
   everywhere. Clarification drafted in
   `docs/submission/INT8_CLARIFICATION_2026-11-19.md` and **not sent**.
2. **Predictions filename.** p.10/p.11 say `predictions.jsonl`; the p.12 bundle
   diagram says `predictions.json`. The builder emits both with identical
   JSON-Lines content so either reading resolves.

## 7. The 3D-feature question (R10) — now satisfied

An earlier revision of this document recorded R10 as a **FAIL on a literal
reading**, on the basis that `src/features.py` has `use_z` defaulting to `False`
and therefore discards the MediaPipe z channel.

That was correct about the default and **wrong about the shipped artefact**. The
default applies only to the four 255-wide members. The five
`landmark_ens_mirror_handdrop_tcn_z_s*` members were trained on and consume
**381-wide x,y,z features**, including the depth channel. They are 5 of the 9
members in the ensemble and they are what the APK loads.

- **Evidence:** `models/landmark_model_meta.json` → `members_detail[].use_z`,
  five entries `true`, four `false`. The meta file's own note says members
  without an entry, or with `use_z` false, take the 255-wide x,y features.
- **Consequence for scoring:** the architecture description
  `Pose-invariant 3D hand features` is met by the shipped system.
- **What is still true:** the four xy members exist, so the ensemble is a mix
  rather than uniformly 3D. If asked, state that plainly rather than claiming
  all nine are 3D.

## 8. What is genuinely not done

| Item | Owner | Why |
| --- | --- | --- |
| Demo video | Abhijeet | not recorded |
| Team/college strings, deck rename | Abhijeet | registration details not in repo |
| INT8 answer | Organizer | clarification unsent |
| Organizer manifest + vocabulary | Organizer | never supplied |
| Weight redistribution | Dev | no documentary permission; marked BLOCKED in `NOTICE` |
| Raw (ungated) occlusion drop ≤5 pt | — | model limitation; a real fix is occlusion augmentation at training time, which is a retrain. Fails raw at +5.76, passes gated at +2.39. |
| p95 latency ≤50 ms in every run | — | 3 of 4 on-device runs pass; the fourth is 4 ms over on a thermally throttled phone |

---

## 9. Corrections made on 2026-11-12 by Agent C (independent PDF re-extraction)

Rows below were **wrong or unverifiable as written** and have been corrected
above. Quotes are re-extracted from the PDF's own text layer with a second,
independent decoder (zlib + content-stream token scan), not copied from
`_slides_text.txt`.

| # | Was | Why it was wrong | Corrected to |
| --- | --- | --- | --- |
| R15 | implied the sheet's target for size is 20 MB | p.11's own table says **`model_size <=25 MB`**; only p.9/p.10 say 20 MB | `results/results.csv` carries the p.11 template values; the note records that 16.33 MB passes both readings |
| R19 | cited only p.10 and p.25 | p.24 says **`Prototype code, 5-min demo video and deck due`** and p.27 says **`a 5-minute demo video by 18th November`**. There are three lengths: 2–3 min (p.10), 5-min (p.24, p.27), max 3 min (p.25) | row now lists all four citations; the 2:50 cut satisfies all three readings |
| R22 | `NOT_TESTED` (no Docker daemon here) | a Docker daemon **is** present. A fresh build revealed a real defect: `--help` worked but `predict()` died on `libGLESv2` because MediaPipe loads GL lazily | **FIXED_AND_VERIFIED** — committed `Dockerfile` installs the libs; re-verified `WHATFOR1` @ 0.8818 in-container, §9.1 |
| R25 | cited `runtime.txt` as evidence | `runtime.txt` **does not exist** in the repo | evidence is `Dockerfile` + `.venv/bin/python -V` (3.11.18) |
| R18 | said "`build_bundle.py` closure" | the claim is verifiable directly | the built ZIP contains `scripts/ml/export_tflite_tf.py` |
| R14 | p.11 attribution was partly wrong | the `predictions.json` spelling is on **p.11 too** (`predictions.json and results.csv in results folder`), not only p.12 | row now quotes both p.11 bullets |

### 9.1 Docker: PASS, after a fix; verified independently

I found this as a **FAIL** on the `Dockerfile` as it stood at `e2b22af`, and
separately from the packaging workstream, which found and fixed the same thing.
Recorded because the failure mode is subtle and worth knowing.

A fresh `docker build` of that `Dockerfile` succeeded, and
`docker run --rm <image> --help` printed the contract. But a **real prediction**
with weights and a clip mounted failed:

```
File "/app/src/landmark_frontend.py", line 74, in get_landmarker
OSError: libGLESv2.so.2: cannot open shared object file: No such file or directory
```

`python:3.11-slim` does not ship the X11/EGL/GLES shared objects. The
`opencv-python-headless` change in `src/requirements.txt` fixed the **cv2**
import, so `--help` worked — but `mediapipe` loads `libEGL`/`libGLESv2`
**lazily**, only when `hand_landmarker` is created, so the defect was invisible
until a real clip was run. That is exactly the class of bug that "the container
starts" testing misses.

Current `Dockerfile` installs `libgl1 libglib2.0-0 libxcb1 libegl1 libgles2`.
Verified by a fresh local build (`eg-verify-d`) at 2026-11-12:

```
$ docker run --rm <image> --help
EchoGesture — on-device isolated sign recognition (100 glosses)
...
$ docker run --rm -v models:/app/models -v clips:/clips:ro <image> "...WHAT FOR.mp4"
  "clip_id": "0283886564670357-WHAT FOR",
  "prediction": "WHATFOR1",
  "confidence": 0.8818,
```

Full evidence, including the missing-weights path and the in-container parallel
tracker, is in `evaluation/device/DOCKER_ACCEPTANCE.md`.

### 9.2 Stale Docker images in the local daemon

`echogesture:rc` and `echogesture:rc2` both fail at `import cv2` with
`libxcb.so.1`. Both predate the headless-OpenCV change. **Do not cite either
image as Docker evidence**; rebuild from the committed `Dockerfile`. The fresh
`eg-verify-c` (pre-fix) and `eg-verify-d` (post-fix) builds above are the
accurate tests.

### 9.3 Claims verified exactly (no change needed)

| Claim | Verification |
| --- | --- |
| 9 × FP16 members sum to 8,508,832 B; detector 7,819,105 B; **total 16,327,937 B** | `stat` on all ten files |
| `"quantization": "float16"`; 5 of 9 members `use_z: true` | `models/landmark_model_meta.json` |
| **1,069 / 1,251 = 85.45%** top-1, 96.64% top-5 | `results/results.csv` rows `top1_accuracy`, `top5_accuracy` |
| `use_z=False` is the default for the four xy members only | `src/features.py`, `models/landmark_model_meta.json` |
| Android merged manifest has CAMERA only | merged manifest; no INTERNET, no ACCESS_NETWORK_STATE |
| `predict()` returns the five p.11 keys | `src/main.py` |
| 100-gloss vocabulary, all uppercase | `models/landmark_labels.json`, 100 keys, all `.isupper()` |
| Python 3.11 | 3.11.18 |
| APK 82,894,510 B, SHA-256 `96f00458…` | `sha256sum -c submission/apk/EchoGesture-2026-11-21.apk.sha256` |
| **139** JVM unit tests pass with 0 failures | `app/android_final/app/src/test`, `./gradlew :app:testDebugUnitTest`, run 2026-11-20 |
| **34** instrumentation test methods across **20** classes; **15 across 8 classes were run green on device** on 2026-11-20 | `app/android_final/app/src/androidTest` (static count). The remaining classes include diagnostic probes that are documented as hanging — MediaPipe cannot recreate a landmarker twice in one process — so they are not evidence of a passing suite. Do not report "all instrumentation tests pass". |

### 9.4 Two denominators, one numerator — stated so the numbers are not read as a contradiction

`results.csv` reports **85.45%** on 1,251 clips. The robustness document
`docs/ROBUSTNESS_OCCLUSION_2026-11-20.md` reports **85.18%** clean on "the full
1,255-clip split". Both are the **same 1,069 correct predictions**. The
difference is the denominator: the robustness harness counts the 4 clips the
pipeline refuses (no hand detected) as wrong, giving 1,255; the scored
accuracy row counts them as not-attempted, giving 1,251. Neither is wrong; do
not quote them interchangeably without saying which denominator is meant.

### 9.5 Resolved since the 2026-11-12 revision

- **APK size disagreement — RESOLVED, TWICE.** The three conflicting values
  (75,072,782 / 75,072,706 B and their hashes) described the four-model APK. A
  second drift then recorded 83,125,166 B for a 83,128,337-byte file, which was
  already stale when written. The artefact is now `EchoGesture-2026-11-21.apk`,
  82,894,510 B, SHA-256
  `96f00458f54de715c7a6f489f9ae92425bc73a089043a91ccbe0ae6d4f060035`, rebuilt
  from `29c48b6` so the binary matches this branch's source, with a `.sha256`
  file beside it.
- **Latency — now measured on device.** The 275.8 / 152.5 / 93.9 ms desktop
  p95 figures are superseded for scoring by the Vivo V2404 on-device numbers
  (`latency_on_device_p50` 34–45 ms, `latency_p95` 46–54 ms). Desktop
  clip-to-JSON is retained only as a non-scored row.
- **Unit-test count — now 139 JVM + 15 instrumentation.** The older "9" and
  "20" figures both predate the current test tree.
- **Deck slide 9 parity figure** — the repo pins `1e-5` (logits), `1e-6`
  (softmax) and `1e-3` (confidence) tolerances; the deck was corrected to
  `<1e-5` / `<1e-6`.

## 10. Revision log

| Date | Change |
| --- | --- |
| 2026-11-12 | Compiled; Agent C re-extracted the PDF and corrected R14, R15, R18, R19, R22, R25 |
| 2026-11-20 | R1/R5 model count and bytes corrected (4 → 9 members, 11.36 → 16.33 MB). R3 accuracy corrected (72.78% → 85.45%). R2/R6 latency moved FAIL → PARTIAL on measured on-device GPU numbers. R4 rewritten around the occlusion diagnosis and the acceptance gate. **R10 changed FAIL → PASS** — 5 of 9 shipped members consume the z channel. R20 deadline updated for the extension. R19b APK row added. §9.4–9.5 added. |