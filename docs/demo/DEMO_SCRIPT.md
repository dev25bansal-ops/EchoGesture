# EchoGesture demo script — 2–3 minute offline clip-path demonstration

**Read this before filming.** The live camera path is **blocked locally**
(macOS denied camera permission) and **Android is not implemented** (no
SDK/device). Record the **offline clip-path** version below. Label it as such
on screen. Never claim a live phone demo, an INT8 model, representative
accuracy, or 50 ms clip p95 — none of those is established.

Only record the actual release candidate (`ECHOGESTURE_MODEL=landmark`,
`models/landmark_classifier_fp16.tflite`). Do not use a scripted caption or
an unlabelled synthetic clip as recognition evidence.

| Time | What happens | What to say |
|---|---|---|
| **0:00–0:25** | External view of a **consenting signer not in training**, signing an isolated word from the **100-sign vocabulary** (`models/landmark_labels.json`). | "EchoGesture recognizes a fixed set of 100 isolated ASL signs locally. It is not continuous translation or an interpreter replacement." |
| **0:25–0:55** | Show the real CLI / manifest run on that clip: visible input clip + the actual JSON output (`PULLCONVINCE`, confidence 0.7507 on the recorded integration run). Name the model file and its FP16 format. | "This runs fully offline. The model is FP16, not INT8 — the INT8 alternative exists but is not selected." |
| **0:55–1:20** | Profile run: show the stage table. | "Tracking dominates: 612 ms of a 642 ms N=1 clip. Features and classifier are under 1 ms combined. One clip is not a p95, and this does not meet a 50 ms clip target." |
| **1:20–1:50** | Architecture overlay: sampled frames → MediaPipe hand landmarks → shared 255-feature transform → TFLite classifier → top-5 JSON. Report shipped bytes: classifier 832,888 + detector 7,819,105 = **8,651,993 bytes**. | "Landmarks discard signer appearance and background. Missing hands are explicit flags, not fake points." |
| **1:50–2:15** | Results slide: the measured / not-measured table from the dashboard (39 tests pass; accuracy, p95, robustness, power, thermal = NOT_MEASURED). State N everywhere. | "No representative accuracy or latency claim. The old pixel baseline is a separate path, not this model." |
| **2:15–2:45** | Limitations: vocabulary is project-defined until the organizer confirms it; camera and Android unverified; INT8, power, and thermal not measured. | "What is missing, and what closes each gap." |
| **2:45–3:00** | Close: repo/commit + dashboard link. No unmeasured claims. | — |

## Fallback if the camera path becomes available later

Swap the clip-path segment for the real `app/offline_demo.py` capture only
after the checklist items pass (permission granted, preview + SPACE capture
visible, a real model caption from a signer held out of training, offline
status shown). Keep the provisional-threshold disclaimer: the 0.25 retry
threshold is **not** validation-calibrated.

## Hard rules

- The HUD/demo code labels itself **"integration not release-certified"** —
  keep that visible.
- Do not edit a correct caption over an unrelated clip.
- Do not describe per-frame timing as clip p95, or capture-to-result as
  either.
- Do not film until the release candidate is frozen and SHA-verified.
