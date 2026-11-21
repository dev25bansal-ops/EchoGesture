# HUD evidence, captured on the shipping build

Five screenshots taken on the signed release APK
(`EchoGesture-2026-11-22-release.apk`, `03d51354…ab2f49`) on the physical
Vivo V2404, 2026-11-22, driven over adb. They exist because the
"Occlusions & HUD clarity" criterion is 25% of the quality score and is the
one place a judge can check our claim against a picture rather than a number.

Each shot is unedited — no crop, no annotation. The status bar and the
`9-model ensemble · FP16 TFLite · SHA-256 995cc8d0…` footer are left in
deliberately, so the artifact that produced the screen is identifiable.

| File | What it shows | Claim it supports |
| --- | --- | --- |
| `hud_01_camera_offline.png` | Idle camera screen, green **Offline** badge, footer with the ensemble SHA | The app runs with no network permission; the model identity is on screen |
| `hud_02_glossary.png` | The glossary of the 100 output classes | The vocabulary is exactly what the model emits, inspectable |
| `hud_03_refusal_no_hands.png` | **"No hands in view — Hands were visible for too little of that recording."** | The app declines rather than guesses when no hand is found. This state also produces **no History entry** — a refusal is not recorded as a recognition |
| `hud_04_recording_too_long.png` | **"Perform one sign, then tap Finish"** with a Cancel affordance, mid-recording | Constructive guidance during capture, not an error after the fact |
| `hud_05_result_p95.png` | `MECHANIC1`, **89.3% model score**, alternatives `DEMAND1 · HALLOWEEN1 · FAIL · DECIDE1`, and **"Per-frame p95: detect 59 ms, convert 5 ms — the ≤50 ms target is a p95"** | A real accepted reading, the alternatives are always shown, and the scored latency metric is readable on the artifact that ships |

## What shot 05 also tells us honestly

That capture reported **detect p95 59 ms**, above the 50/51 ms seen earlier on
the same build. Across five file-pick classifications on the release build the
observed detect p95 spans **50–59 ms**. The spread is real and is recorded as a
range in `results/results.csv` (`latency_p95` notes) rather than quoted as the
best value. On this path the target is met at the low end and missed by ~9 ms
at the high end; the live-camera-with-a-hand path remains unmeasured, so the
row stays `FAIL`.

## What these shots do not show

- **No live-camera accuracy.** Every correct reading above is the file-pick
  path. The live camera was tested separately and misreads a still closed fist
  as `RAZOR2`, which is diagnosed in
  `docs/CAMERA_MISREAD_DIAGNOSIS_2026-11-21.md`.
- **No phone accuracy claim.** One device, one signer. A correct clip reading
  is evidence the pipeline works, not a measured accuracy.
- **No "uncertain" label.** None of these captures produced a low-margin
  result, so the uncertain state is not pictured here. It is described in
  `docs/HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md` and was **not** staged.