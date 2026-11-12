# EchoGesture — shot-by-shot recording guide (2:50)

Every number below was measured in this repo and is reproducible from the
paths given. Anything not verified is marked **DO NOT SAY**. Nominal cut is
**2:50**, which satisfies every organizer length reading (2–3 min, 5 min,
"max 3 minutes"). Do not pad to 5:00.

Two hard facts that shape every shot:

- `adb` is not installed on this machine and no Android device is attached.
  **There is no device footage and no live-camera footage.** Every shot is
  the offline file path, run on a macOS arm64 laptop.
- Nothing is uploaded. Captures go to a private temp directory, are predicted
  once, and are deleted.

Label the recording honestly, on screen, throughout:
`OFFLINE FILE DEMO · not live camera · not the target hardware`.
The HUD already prints `OFFLINE | integration not release-certified`; keep
it legible and do not crop it out.

---

## What the human must do before recording

| # | Action | Tool | Notes |
| --- | --- | --- | --- |
| 1 | Record the screen at **1920×1080, 30 fps** | QuickTime Player → File → New Screen Recording, or OBS | QuickTime is preinstalled. Show the terminal with a large font (18pt+). |
| 2 | Install the demo dependencies once | `.venv` already exists | Verify: `.venv/bin/python -c "import cv2, ai_edge_litert"` |
| 3 | Rehearse both demo commands (below) | Terminal | Each run takes ~1–2 s to start and ~0.5–0.7 s to predict. |
| 4 | Record narration separately or live | Any microphone | A single continuous take reads better than per-shot audio. |
| 5 | Cut and export H.264 MP4 at 1080p | QuickTime, iMovie, or `ffmpeg` | Target ≤ 3:00 hard, 2:50 nominal. |
| 6 | Watch the export once end-to-end at 1× | — | Confirm no personal paths, no device claims, no 50 ms claim. |

**Do not** record a live camera shot, an Android phone shot, or a fabricated
HUD. If a shot needs a miss on camera, use a clip that genuinely produces
`RETRY`/`ERROR`, or cut the shot.

---

## Pre-flight (run these, do not show them unless the shot calls for it)

```bash
cd /path/to/EchoGesture

# Demo A — the plain offline demo (this is what you record for 0:40–1:10)
.venv/bin/python -I app/offline_demo.py \
  --video "evaluation/device/v5_accuracy/data/val_30/5784034175717614-EASY.mp4" \
  --render-output /tmp/eg_easy_hud.png

# Demo B — the second clip (harder, lower confidence)
.venv/bin/python -I app/offline_demo.py \
  --video "evaluation/device/v5_accuracy/data/val_30/3738312466703404-TEACH.mp4"

# Demo C — the wrapped CLI with an exit code (use if you want a clean
# JSON transcript on screen)
PYTHONPATH=. .venv/bin/python -I app/v7_demo/cli.py \
  --video "evaluation/device/v5_accuracy/data/val_30/5784034175717614-EASY.mp4"
```

Verified outputs from a real run on this machine (record these, not
re-typed numbers):

| Command | Prediction | Confidence | State | Latency |
| --- | --- | --- | --- | --- |
| Demo A (`offline_demo.py`) | `EASY` | **0.9028** | `CAPTION` | 704.3 ms |
| Demo A top-5 | `EASY, DARK1, HOW1, KNIGHT1, CALENDAR1` | | | |
| Demo B (`offline_demo.py`) | `TEACH1` | **0.4715** | `CAPTION` | 495.4 ms |
| Demo B top-5 | `TEACH1, CHEER, HALLOWEEN1, DECIDE1, PARTY1` | | | |
| Demo C (`app/v7_demo/cli.py`) | `EASY` | **0.8976** | `CAPTION` | exit 0 |

Model identity printed by all three:
`landmark:4-model ensemble sha 0e4c392732e7 3544080 B`.

> Demo A and Demo C differ slightly in confidence (0.9028 vs 0.8976) because
> the CLI wrapper re-encodes sampled frames to a temporary clip before
> predicting; the direct path reads the file in place. **The prediction is
> identical.** If you show both, say this; never present the two numbers as
> the same measurement.

---

## 0:00–0:20 — The problem

**Screen:** title card, dark background, one line of text. No app footage.
Optional: a still of hands signing (licensed/own footage only — do not use
any clip from `data/`).

**On-screen text (exact):**
```
100 isolated ASL signs · not continuous translation · not an interpreter replacement
```

**Narration (exact):**
> "Deaf and hard-of-hearing users routinely lose information in one-to-one
> conversation. EchoGesture recognizes a fixed vocabulary of 100 isolated
> ASL signs on-device and returns a single caption. It is not continuous
> translation, and it is not an interpreter replacement."

---

## 0:20–0:40 — The approach

**Screen:** the pipeline diagram, left to right:

```
video clip
  → MediaPipe hand landmarks (21 points × 2 hands, per frame)
  → 24 frames × 255 features
     (position, velocity, acceleration, inter-hand geometry,
      presence masks, canonical [Left, Right] slots)
  → 4-model FP16 TFLite ensemble (mean of logits → one softmax)
  → top-5 gloss → caption
```

**On-screen text (exact):**
```
All local · no network · 4-member FP16 ensemble · 100 classes
```

**Narration (exact):**
> "Frames become hand landmarks, not raw pixels. Landmarks drop signer
> appearance and background, so the classifier can generalize across people.
> The features are 24 frames by 255 values: position, velocity, acceleration,
> inter-hand geometry, explicit presence masks, and canonical left and right
> hand slots. Four independently trained FP16 models vote — their logits are
> averaged and softmaxed once. Everything runs on the device; nothing is sent
> anywhere."

---

## 0:40–1:10 — Live recognition (offline file path)

This is the core shot. It is a **file demo**, not a camera demo.

**Screen:** terminal on the left, the HUD image on the right (open
`/tmp/eg_easy_hud.png` after the run, or use the second monitor). Show the
real command being typed, then the JSON output.

1. Type Demo A and run it. It finishes in about a second.
2. Show the JSON: `"prediction": "EASY"`, `"confidence": 0.9028`,
   `"top5"` list, `"latency_ms": 704.3`.
3. Open the HUD PNG. Point at the caption `EASY`, the `CAPTION` state,
   and the confidence bar.
4. Run Demo B. Show `TEACH1` at 0.4715, also `CAPTION`.

**On-screen text (exact, lower third):**
```
OFFLINE FILE DEMO · not live camera · not the target hardware
```

**Narration (exact):**
> "These are real runs on held-out validation clips, right now, on this
> laptop. The model names the sign, shows its top five, and reports its own
> clip latency. The second clip is harder — same mechanism, lower
> confidence. Confidence is the softmax of the averaged logits. It is
> uncalibrated: it is not a probability that the sign is correct."

---

## 1:10–1:40 — Offline operation

**Screen:** terminal showing the model file listing, then the identity
block from the demo JSON.

```bash
ls -l models/hand_landmarker.task models/landmark_ens_*_fp16.tflite
```

Show these five files and their sizes:

| File | Bytes |
| --- | ---: |
| `hand_landmarker.task` | 7,819,105 |
| `landmark_ens_tcn21_fp16.tflite` | 939,152 |
| `landmark_ens_tcnm61_fp16.tflite` | 832,888 |
| `landmark_ens_tcnm62_fp16.tflite` | 832,888 |
| `landmark_ens_rob71_fp16.tflite` | 939,152 |
| **Total** | **11,363,185 B** (reported as **11.37 MB**) |

**On-screen text (exact):**
```
FP16 (not INT8) · no network · temp captures deleted · 11.37 MB
```

**Narration (exact):**
> "The demo is fully offline. A capture is written to a private temporary
> directory, predicted once, then deleted — nothing retained, nothing
> uploaded. The HUD hashes the files it names, so what you see on screen is
> the model that actually ran. The whole thing is 11.37 megabytes: the four
> ensemble members plus the hand detector."

**Do not show or mention:** a phone, a camera preview, a network icon, or
the word "live".

---

## 1:40–2:05 — Ensemble and architecture detail

**Screen:** a still of the architecture put up in 0:20, plus a small table
of the four members and their SHA prefixes. Do not claim per-member
accuracies; none is recorded in this repo.

```
landmark_ens_tcn21_fp16.tflite    TCN                    112279bc8609
landmark_ens_tcnm61_fp16.tflite   Conv1D, mirror-aug      01a261d8eb6c
landmark_ens_tcnm62_fp16.tflite   Conv1D, mirror-aug      27058a85e032
landmark_ens_rob71_fp16.tflite    TCN, stronger augment   08417e72da59
                                  mean logits → one softmax
```

**On-screen text (exact):**
```
4 × FP16 TFLite · 3,544,080 B · mean of logits → one softmax · uncalibrated
```

**Narration (exact):**
> "Four independently trained classifiers — two temporal convolutional
> networks and two mirror-augmented Conv1D models — each see the identical
> 24-by-255 tensor. Their logits are averaged, and a single softmax is taken
> over the average. That is deliberately simple: the aggregation is four
> multiplies and a softmax, and the Android Kotlin implementation matches the
> Python reference to within about one part in ten million on the same clip."

**Do not say:** "the ensemble adds 3.68 points over its best member" — that
figure is not recorded anywhere in this repo. Do not compare 62.61% (the
previous single model) to 72.78% as a controlled improvement; they are
different models from different runs.

---

## 2:05–2:30 — Accuracy, size, runtime

**Screen:** a results table with N and device on every row. Grey out anything
not measured.

| Metric | Value | N / basis | Status |
| --- | --- | --- | --- |
| Test top-1 | **72.78%** (909/1249) | 1,249 signer-disjoint clips, 11 unseen signers | meets ≥70%; not a pristine blind test |
| Test top-5 | **87.51%** (1,093/1249) | same 1,249 | — |
| Validation top-1 | **75.36%** (260/345) | 345 scored clips | optimistic vs test |
| Model size | **11.37 MB** | 4 members + detector | PASS under 20 MB |
| Full-clip p95, singleton | **274.2 ms** | 12 clips, full `predict()` | **FAIL** vs 50 ms |
| Full-clip p95, parallel-2 | **154.8 ms** | same 12 clips | still over 50 ms |
| Full-clip p95, parallel-4 | **99.8 ms** | same 12 clips, opt-in | still over 50 ms |

Device line, on screen at all times during this shot:
```
macOS arm64 laptop · ai-edge-litert 2.3.0 · XNNPACK CPU · NOT the target hardware
```

**Narration (exact):**
> "On 1,249 signer-disjoint test clips from 11 unseen signers, top-1 is
> 72.78%, top-5 is 87.51%. The honest caveat: we inspected the test split
> during model selection, so this is not a pristine blind number; on the
> strict all-eligible denominator it is 72.43%. The model is 11.37 MB, inside
> the 20 megabyte limit. Runtime is the weak point. Full clip-to-JSON p95 is
> 274 milliseconds on one worker, 155 on two, and 100 on four. All of those
> are on a laptop, not the target hardware, and all of them miss the
> 50 millisecond target."

**Never say:** "meets the 50 ms target", "real-time", "sub-100 ms on the
target device", "Android latency is X". None of those is measured.

---

## 2:30–2:50 — Limitations and close

**Screen:** two columns — "Measured" and "Not measured". Keep it plainly
honest.

```
MEASURED                         NOT MEASURED
top-1 / top-5 (test, val)        on-device camera behaviour
model size 11.37 MB              target-device latency
full-clip latency (laptop)       power / thermal
robustness: low light PASS       live-camera accuracy
 motion blur PASS
 frame drop PASS
 crop -10.00 pp FAIL
 occlusion -6.67 pp FAIL
```

**On-screen text (exact):**
```
100 isolated signs · 4-model FP16 ensemble · 11.37 MB · fully on-device
The 50 ms full-clip target is not met. No physical Android device was available for camera testing.
```

**Narration (exact):**
> "What we have is a working offline recogniser with measured accuracy, size
> and laptop latency, built on landmarks rather than raw pixels. What we do
> not have is a device measurement. No physical Android device was available
> here, so camera behaviour, power and thermal are simply not measured — we
> do not claim them. Crop and occlusion robustness fail our own 5-point
> budget, and we report those failures rather than tuning them away. The next
> step is the same code on the target device, measured there."

---

## What must NOT be said anywhere in the recording

- **No 50 ms compliance claim of any kind.** The 50 ms full-clip target is
  not met on any configuration (best measured: 99.8 ms p95, laptop).
- **No full-sentence or continuous translation claim.** The scope is 100
  isolated signs; the deck and the video both say "not continuous
  translation" and "not an interpreter replacement".
- **No camera or Android-device claim.** No device test exists; `adb` is not
  installed and no device was attached.
- **No INT8 claim.** Shipped weights are FP16. INT8 is not compliance.
- **No "pristine blind test" claim.** The test split was inspected during
  model selection.
- **No per-member accuracy or "3.68-point ensemble gain" claim.** Not
  recorded in this repo.
- **No calibrated-confidence claim.** Confidence is an uncalibrated softmax
  of averaged logits.
- **No fabricated miss.** If you need a `RETRY`/`ERROR` on screen, use a clip
  that genuinely triggers it.