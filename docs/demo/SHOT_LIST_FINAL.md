# Shot list — final demo, 2:50 cut

**Rule for this file: every number below is one this repo measured.** Each
figure carries its source and N. Anything not measured is marked
**ASPIRATIONAL — do not say it as a result**.

Nominal cut is 2:50, inside the strictest stated limit ("max 3 minutes").
The organizer slides disagree on length — 2–3 min, 5 min, and "max 3
minutes" appear in different places. **2:50 satisfies all three; do not pad
to 5:00 and do not rely on the 5-minute reading.**

Camera and Android are blocked here. Every shot is the **offline file
path**. Label the video as the offline clip path; never show or imply a
live camera.

---

## Global on-screen furniture

- Persistent lower-third: `OFFLINE FILE DEMO · not live camera · not the
  target hardware`.
- The demo HUD already prints `OFFLINE | integration not release-certified`.
  Keep it legible; do not crop it out.
- Show the runtime and device on the results slide:
  `ai-edge-litert 2.3.0, XNNPACK CPU — macOS arm64 laptop (not target
  hardware)`, Python 3.11.

---

## 0:00–0:20 — Problem

**Visual:** slow pan or still of a hearing person and a Deaf signer failing
to communicate; a hand signing in frame. No fake app footage.

**Screen text:** `100 fixed isolated ASL signs · not continuous
translation · not an interpreter replacement`.

**Say:** "Deaf and hard-of-hearing users routinely lose information in
one-to-one conversation. EchoGesture recognizes a fixed vocabulary of **100
isolated ASL signs** on-device and returns a caption. It is not continuous
translation and it is not an interpreter replacement."

**Numbers:** 100 signs (`models/landmark_labels.json`, 100 classes).

---

## 0:20–0:40 — Architecture

**Visual:** pipeline diagram, left to right:

`video clip → MediaPipe hand landmarks (21 pts/hand) → shared 255-feature
transform → 4 FP16 TFLite members → mean of logits → one softmax → top-5`

**Screen text:** `All local · no network · 4-member FP16 ensemble`.

**Say:** "Frames become hand landmarks, not raw pixels. Landmarks drop
signer appearance and background, so the classifier generalizes across
people. Four independently trained FP16 models vote; their logits are
averaged and softmaxed once. Everything runs on the device — nothing is
sent anywhere."

**Verified numbers:**
- Ensemble = 4 members, 24 frames, 255 features, 100 classes
  (`models/landmark_model_meta.json`).
- Ensemble members total **3,544,080 B** (measured on disk and shown by the
  HUD); with the MediaPipe hand-landmark detector (**7,819,105 B**) the
  on-device footprint is **11.37 MB** as reported (`results/results.csv`
  `model_size` = 11.37 MB, PASS under the 20 MB limit).
  *Note:* the CSV note text says `3544080 B` for the members; the files on
  disk and the HUD say `3544080 B` — a 1,000 B difference that does not
  change the 11.37 MB figure (11,363,185 B = 11.36 MB decimal).
- Confidence is the softmax of the mean of four members' logits, and is
  **uncalibrated**.

---

## 0:40–1:35 — Real recognition demo (the core shot)

**Visual:** screen recording of an actual CLI run — terminal and HUD
side by side. Run by path (not `-m`):

```bash
.venv/bin/python -I app/v7_demo/cli.py \
  --video "evaluation/device/v5_accuracy/data/val_30/5784034175717614-EASY.mp4" \
  --render-output evaluation/device/demo_final/easy_hud.png
```

**Show, in this order:**

1. The input clip on screen.
2. The HUD appearing: `EASY`, green `CAPTION`, confidence **0.898**,
   top-5 `EASY, DARK1, HOW1, KNIGHT1, CALENDAR1`.
3. The JSON on stdout with `"exit_code": 0`.
4. Then a second clip, `3738312466703404-TEACH.mp4` → **`TEACH1` @
   0.4221**, still `CAPTION` (above the 0.25 threshold).

Both runs are real, exit `0`, and are recorded in
`evaluation/device/demo_final/{easy,teach}_run.json`. HUD PNGs are
`easy_hud.png` and `teach_hud.png`.

**Say:** "These are real runs on held-out validation clips. The model
names the sign, shows its top five, and reports its own clip latency. The
second one is a harder clip — same label, lower confidence."

**Guardrails to say or display:**
- Confidence is **uncalibrated**; it is not "90% sure".
- The demo re-encodes sampled frames to a temp clip, which shifts
  confidence slightly versus calling `predict()` directly (EASY 0.9028 →
  0.8976; TEACH 0.4715 → 0.4221). Predictions are unchanged.
- The 0.25 retry threshold is **provisional**.
- These are two clips, not an accuracy claim. Accuracy comes next.

If a shot needs a miss on camera, do **not** edit a correct caption onto an
unrelated clip. Use a clip that genuinely triggers `RETRY`/`ERROR` or cut
the shot.

---

## 1:35–2:00 — Offline / model explanation

**Visual:** terminal showing no network, temp capture deleted, then the
identity block. Show `models/hand_landmarker.task` + the four member files.

**Say:** "The demo is fully offline. Captures are written to a private
temporary directory, predicted once, and deleted — nothing is retained and
nothing is uploaded. The HUD hashes the actual files it names, so what you
see on screen is the model that ran."

**Verified numbers:**
- Identity shown by the HUD:
  `landmark:4-model ensemble sha 0e4c392732e7 3544080 B`.
- Member SHA-256 prefixes: tcn21 `112279bc8609`, tcnm61 `01a261d8eb6c`,
  tcnm62 `27058a85e032`, rob71 `08417e72da59`.
- Detector SHA-256: `fbc2a300…` (`results/results.csv` `model_size`).
- `live_camera_verified: false` in every report — say "file demo", not
  "camera demo".

**Screen text:** `FP16 (not INT8) · no network · temp captures deleted`.

---

## 2:00–2:25 — Results + limitations

**Visual:** results table with N and device on every row.

| Metric | Value | N / basis | Status |
| --- | --- | --- | --- |
| Test top-1 | **72.78%** (909/1249) | 1,249 signer-disjoint test clips, **11 unseen signers** | meets ≥70%; **not a pristine blind test** (split was seen during model selection) |
| Test top-5 | **87.51%** (1093/1249) | same 1,249 | — |
| Validation top-1 | **75.36%** | 345 scored clips | optimistic vs test |
| Model size | **11.37 MB** | 4 members 3,544,080 B + detector 7,819,105 B | PASS (<20 MB) |
| Full-clip p95, singleton | **274.2 ms** | 12 clips, 1 cold + 12 warm, full `predict()` | **FAIL** vs 50 ms |
| Full-clip p95, parallel-4 | **99.8 ms** | same 12 clips, opt-in tracker | still over 50 ms |
| Robustness, clean | **76.67%** | 30 val_30 clips, 5 signers | baseline |
| Crop severity 0.3 | **−10.0 pp** | same 30 | **FAIL** (limit ≤5 pp) |
| Occlusion severity 0.5 | **−4.25 pp** | same 30 | PASS (limit ≤5 pp) |

**Say:** "On 1,249 signer-disjoint test clips from 11 unseen signers, top-1
is 72.78% and top-5 is 87.51%, above the 70% target. The honest caveat: the
test split was looked at during model selection, so this is not a pristine
blind test.

"Latency and robustness are where we fall short. Full-clip p95 is 274 ms
singleton and 100 ms with the opt-in parallel tracker — both miss the 50 ms
target, and the benchmark is 12 clips on a laptop, not the target device.
Under crop the model loses 10 points, which fails the 5-point robustness
budget.

"Power and thermal are not measured — we have no target device."

**ASPIRATIONAL — do not claim:** meeting 50 ms on target hardware, INT8,
Android, or continuous translation.

---

## 2:25–2:50 — Closing

**Visual:** repo path, commit SHA, and the one-line contract
`predict(video_path) -> dict`. Cut to the HUD for the last beat.

**Screen text:** `Fully on-device · 100 isolated signs · FP16 4-model
ensemble · 11.37 MB`.

**Say:** "EchoGesture runs a 4-model FP16 ensemble fully on-device in 11.37
MB. It meets the top-1 target on held-out signers and misses the latency
and crop-robustness targets. The next step is porting to the target device
and measuring there. Code and evidence are in the repo at this commit."

---

## Shot-status tracking

| Shot | Status |
| --- | --- |
| Problem | to shoot |
| Architecture | to shoot |
| Recognition demo (EASY, TEACH) | **verified runs exist** (`demo_final/*_run.json`, exit 0) |
| Offline / model explanation | to shoot (identity verified) |
| Results + limitations | to shoot (all figures verified) |
| Closing | to shoot |
| Any live-camera shot | **BLOCKED — no camera, TCC denied; do not shoot** |
| Any Android shot | **BLOCKED — not implemented/tested on device; do not shoot** |

---

## Figure provenance (one line each)

- 72.78% / 87.51% / 274.2 ms / 11.37 MB / robustness rows:
  `results/results.csv` (metrics 2–12).
- 99.8 ms parallel-4 p95: `results.csv` `latency_p95_full_clip` note and
  `evaluation/device/performance_final/results_ensemble/benchmark.json`
  (`runs.parallel_4.warm.p95_ms = 99.814`).
- 75.36% / 87.25% validation: `evaluation/device/ensemble_evidence/VAL_RESULT.md`.
- 909/1249, 11 unseen signers: `evaluation/device/ensemble_evidence/RESULT.md`,
  `docs/release/FINAL_EVIDENCE.md`.
- Demo confidences 0.8976 / 0.4221: real runs in
  `evaluation/device/demo_final/{easy,teach}_run.json`.
- Identity SHA/size: HUD identity block in the same reports, matching
  `sha256sum` of `models/`.