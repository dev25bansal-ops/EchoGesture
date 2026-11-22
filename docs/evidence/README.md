# HUD evidence, captured on the shipping build

Six screenshots taken on **EchoGesture Auto 6.2**
(`EchoGesture-Auto-6.2.apk`, SHA-256 `b1c0e671…dcf8`) on the physical
Vivo V2404, 2026-11-22, driven over adb. They exist because the
"Occlusions & HUD clarity" criterion is 25% of the quality score and is the
one place a judge can check our claim against a picture rather than a number.

Each shot is unedited — no crop, no annotation. The status bar is left in
deliberately, so the capture time and battery state stay visible.

**These supersede the five `hud_0*.png` shots**, which were taken on
`EchoGesture-2026-11-22-release.apk` (`9930894…4b397`) — the build that
is no longer submitted. The old files are kept in this directory so the
change is auditable, but no claim in the submission may cite them any more.
Two things differ on the shipping build and are visible below: the app is
titled **EchoGesture Auto**, and the result card reports
*Analysis / classifier* instead of a per-frame `Clip detect p95` line.

| File | What it shows | Claim it supports |
| --- | --- | --- |
| `auto62_01_camera_offline.png` | Idle camera screen, green **Offline** badge, **"Searching for hands"** pill, the framing guide, and the "New here?" guidance card | The app runs with no network permission, and the idle state tells the user what it is doing |
| `auto62_02_settings_model_hash.png` | Settings → **MODEL**: "9-model ensemble / logit-averaged, one softmax", "FP16 TFLite", the **ensemble SHA-256 `995cc8d0…b4bb`**, and all nine member filenames. Same screen carries **Export diagnostics…** and the **Capture diagnostics** toggle | The model identity is on screen on the artifact that ships, so a judge can match it to `results.csv` without trusting the README |
| `auto62_03_glossary_100.png` | The glossary, headed **"100 of 100 signs"**, with a search field | The vocabulary is exactly what the model emits, and it is inspectable |
| `auto62_04_refusal_no_hands.png` | **"No hands in view"** in the live pill (amber), and the result panel **"No hands in view — No sign started. Tap Start Recognition and perform one sign."** | The app declines rather than guesses when no hand is found. A refusal produces **no History entry** — a refusal is not a recognition |
| `auto62_05_live_result.png` | `VOICE`, **62.4% model score**, header **"POSSIBLE MATCH"**, alternatives `RABBIT2 · COMB2 · HOSPITAL1 · JEWELRY`, **"Analysis 201 ms · classifier 186 ms"**, **"20 samples over 2217 ms · 9 samples/s"**, and the Copy / Speak / Scan-next / Clear actions | A real reading from the **live camera**, with its alternatives always shown, and the timing the app actually records on the path a user takes |
| `auto62_06_history_saved.png` | History with the toggle on: subtitle **"Saved on this device"**, one entry `VOICE 85% 11:52 AM`, and a **Clear history** affordance | History persistence works, and the user can see and delete what is stored |

## What these shots tell us honestly

**Shot 05 is a low-confidence reading, and it is labelled as one.** The header
reads "POSSIBLE MATCH", not "Recognised sign", and the body says *"Low
confidence. Compare the options below or try the sign on its own."* We are not
showing a cherry-picked 95% card. The 62.4% score and the amber warning
triangle are the gate doing its job.

**The Auto build does not print a per-frame p95 on the result card.** The
release build did, and `hud_05_result_p95.png` is the only screenshot in this
repository that ever showed it. That line does not exist on the shipping
build, so the latency row is evidenced by the app's own diagnostics export
instead: `LATENCY_AUTO62_LIVE_CAMERA_2026-11-22.md` and the raw
`auto6.2_diagnostics_live_camera_2026-11-22.txt` beside it, which carry
`detect_p95` for 19 live-camera sessions. That is a stronger record than a
screenshot, because it is per-session rather than one card.

**Every latency sample on this build is over target.** The 19-session export
gives per-frame detect p95 of 72–100 ms, median 85 ms, against a 50 ms
criterion. The row is `FAIL` and these shots do not pretend otherwise.

## What these shots do not show

- **No accuracy claim.** One device, one signer. A correct reading is evidence
  the pipeline works, not a measured accuracy. The 85.45% figure is a corpus
  number over 1,251 signer-disjoint clips.
- **No power or thermal measurement.** Both need their own run; see
  `docs/POWER_THERMAL_2026-11-19.md` for the figures in `results.csv`.
- **The "recording too long" state is not pictured here.** The Auto build ends
  a capture automatically, so the old `hud_04` state ("Perform one sign, then
  tap Finish") does not exist in the same form on this build.
