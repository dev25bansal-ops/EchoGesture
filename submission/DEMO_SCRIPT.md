# Demo video script — 3 minutes

Recorded with the **shipped nine-model APK** on a physical phone. If no phone is
available, read §7 before recording anything.

The rule this script follows: **every prediction on screen is a live inference
from a real sign.** If an attempt fails, the take is retried honestly or cut.
Nothing is ever overlaid to look like a success.

---

## 0. What this script must not say

Three claims were wrong in earlier drafts. They are corrected here, and the
corrections are not negotiable:

| Wrong | Right |
| --- | --- |
| "Below 0.8 confidence the model is right 99.7%" | "**Above** 0.8 confidence, the model was correct on 99.7% of 594 evaluated clips" — and this is a **laptop/corpus** figure, not phone accuracy |
| "The app asks you to sign again when unsure" | The app **always shows its best guess**. When the margin is below 0.15 the card is headed **"POSSIBLE MATCH"** with *"Low confidence. Compare the options below or try the sign on its own."* It never withholds a prediction. A capture with **no hand** is the different case: it refuses with *"No hands in view — No sign started."* |
| 77.02% top-1 / 92.79% top-5 / 1,249 clips | **85.45%** top-1 / **96.64%** top-5 over **1,251** clips, 11 unseen signers |

**Do not script UI behaviour the app does not implement.** The uncertain state is
a *label on a shown result*, not an abstention.

---

## 1. Pre-flight

| Item | Check |
| --- | --- |
| APK | **`EchoGesture-Auto-6.2.apk`**, SHA-256 `b1c0e671…dcf8` — signed release build, from [release `v2026-11-22-auto6.2`](https://github.com/dev25bansal-ops/EchoGesture/releases/tag/v2026-11-22-auto6.2). This is the shipping app, so film **this** one. Do **not** film `EchoGesture-2026-11-22-release.apk` (superseded) or the older `EchoGesture-2026-11-21.apk` (debuggable). |
| Model identity | Camera footer reads **`9-model ensemble · FP16 TFLite · SHA-256 995cc8d0…`** |
| Network | Airplane mode **ON** for the whole recording |
| Second camera | A separate phone or camera filming the screen *and* the signer |
| Lighting | Even, front-lit; not backlit by a window |
| Background | Plain wall for §4; cluttered shot for the robustness beat |
| Signs | Rehearse 4 glosses from the 100-word vocabulary |

Pick glosses that are visually distinct and that you can sign cleanly. Confirm
each is in `models/landmark_labels.json` before using it.

---

## 2. Storyboard

### 0:00 – 0:15 · Problem

**Shot:** close on the signer's hands, no app yet.
**Overlay:** `EchoGesture — offline recognition of isolated ASL signs`

> "Around 70 million deaf people use sign language as a first language. Most of
> the people standing next to them do not sign. EchoGesture turns a phone camera
> into an offline recogniser — no cloud, no upload, no network."

### 0:15 – 0:35 · The app

**Shot:** over-the-shoulder, hand-held, showing the phone as a user holds it.
Show the camera screen, the framing guide, the **Offline** badge, and let the
footer sit on screen for two seconds.

**Overlay:** `9-member FP16 ensemble · on device`

> "This is the whole product. Camera preview, a framing guide, and one button.
> The badge and the footer are not decoration — the footer is the SHA-256 of the
> exact model files running right now."

### 0:35 – 1:25 · Live recognition — the core

**Shot:** screen recording, plus the wide shot of the signer. Split or
picture-in-picture. **Four signs, unedited, in real time.**

For each sign:
1. Tap **Start sign** — the button changes, the app starts buffering.
2. Sign once, at a natural speed.
3. Tap **Finish sign**.
4. The result card appears — let it sit 3 seconds for the camera.

**Overlay per attempt:** the ground-truth word and the predicted word, both
literal. Do not clean this up in post.

> "Start, sign, finish. It reads the interval I actually recorded — not a single
> photo, not a rolling window over whatever the camera happened to see."

**If a sign is wrong:** keep it in, say "that one missed", and sign it again. An
honest miss with a visible retry is stronger than four suspiciously perfect
takes, and judges notice edited cuts.

### 1:07 – 1:25 · The uncertain label — only if it happens naturally

**Do not stage this.** If a real attempt produces a low-margin result, the card
is headed **"POSSIBLE MATCH"** with *"Low confidence. Compare the options below
or try the sign on its own."* Point at it and say:

> "Notice it still gave me an answer, but it told me not to trust it. That label
> appears when the top two candidates are close together. The app does not hide
> a guess — it flags it."

If no attempt triggers it, **skip this beat entirely.** Do not manufacture it by
signing badly on purpose and calling it a system behaviour.

### 1:25 – 1:50 · How it works

**Shot:** the architecture slide, or a simple animated diagram.

> "The classifier never sees pixels. MediaPipe extracts 21 points per hand, the
> app normalises them into a 24-frame feature sequence, and nine small
> classifiers vote on it. Four take x,y; five take x,y,z. All nine see a
> hand-presence mask so absent hands get zero attention. Their logits are
> averaged, then one softmax."

### 1:50 – 2:15 · Offline and private

**Shot:** pull down the notification shade to show **airplane mode** is on. Then
show Settings → Privacy. Then sign again and get a result **still in airplane
mode**.

**Overlay:** `CAMERA is the only permission`

> "Airplane mode, and it still works. The app declares no internet permission —
> it does not have the ability to upload a frame. Raw camera frames are never
> written to disk."

### 2:15 – 2:40 · Measured results

**Shot:** the evaluation slide.

> "On ASL Citizen, signer-disjoint: 85.45% top-1 over 1,251 test clips from
> eleven unseen signers, 96.64% top-5. Nine members are 8.5 megabytes, well under
> the twenty-megabyte budget. Android and Python agree to one part in a million
> on the same features."

Then the caveat, in the same breath — do not let the number stand alone:

> "That is a desktop measurement on a corpus I have seen during development. It
> is not phone accuracy. No phone accuracy figure exists, and I am not going to
> invent one."

Do **not** say "85% accurate on a phone". See §5.

### 2:40 – 3:00 · Limits and close

**Shot:** signer, direct to camera.

> "What this is not: not continuous translation, not a 2,000-word vocabulary —
> one hundred signs, one at a time. If a hand leaves the frame for a quarter of a
> clip, accuracy drops about 5.8 points. Three of the six scored criteria fail
> and that is one of them.
> On device the detector runs about 34–45 ms median, and 58 to 72 ms at p95
> across four runs on the signed release build — over the 50 ms target on every
> sample. But it runs offline, on a phone, right now."

**End card:** `EchoGesture · Track 02 · VIT_Chennai_Sky_Grazers`

---

## 3. The numbers you may say, and their exact labels

Say the label with the number or do not say the number.

| Number | What it is | What it is NOT |
| --- | --- | --- |
| 85.45% top-1 | nine-member FP16 ensemble, 1,251 clips, 11 unseen signers, native 640×480 landmark cache, **desktop** | not phone accuracy; not the 256-square geometry |
| 96.64% top-5 | same population | — |
| 89.31% val top-1 | separate 5-signer validation split | not the test split |
| 81.53% | best single member | the ensemble beats it by 3.92 points |
| 8.51 MB | nine FP16 members | excludes the 7.82 MB detector |
| ~15.3 ms p50 / 23.0 ms p95 | MediaPipe **detector**, per frame, desktop x86, unloaded machine | not the classifier; not the full pipeline; not Android |
| ~4.2 ms | nine-member classifier, desktop | not capture-to-caption |
| ~4.65 s p50 | full Python `predict()` clip-to-result, **desktop** | not Android |
| +5.76 pp | hand-dropout loss, unconditional, full 1,251-clip split — **FAIL** | do not present the gated figure as the robustness result |
| 2.39 pp | hand-dropout loss **at the app's margin gate, on accepted clips only**, 86% clean / 80% corrupted coverage | NOT proof of unconditional robustness. (An older 1.24 figure used the coords-kept diagnostic and is superseded.) |
| 34–45 ms p50 / 46–54 ms p95 (empty frame) / 68–70 ms p95 (hand in frame) | MediaPipe detector, on-device, GPU delegate, Vivo V2404, **debug build** | not a PASS; not the classifier; the release build now shipping should be faster but was not re-measured |
| 2666–2898 mW | camera hardware floor: batterystats prices "camera on" at a fixed 749.4 mA PowerProfile constant (~2.9 W), on-device — **FAIL** vs 500 mW | not the model's own power — it is the cost of the camera being on at all, ~5.8× the budget, so no camera app can pass; app compute is ~157 mW |
| 19.2% drift / 10 min | SoC max over CPU/GPU/NSP, on-device, debug build — **PASS** vs 20% | only 0.8 points of headroom |
| 99.7% above 0.8 confidence | corpus, 594 of 1,251 clips | not phone accuracy, not a guarantee |

## 4. What must never appear on screen

- A prediction the model did not produce.
- A success caption over a failed attempt.
- "INT8" — the shipped build is FP16, and INT8 compliance is **not** claimed.
- Power or thermal numbers **presented as anything other than measured-FAIL
  (power) and measured-PASS with 0.8 pt headroom (thermal)**. Both were measured
  on 2026-11-19 on the debug build; the old line here said they were unmeasured.
- **Latency stated as a PASS.** On the signed release build the GPU delegate p95
  is **72–100 ms, median 85, across 19 live-camera sessions**, every one over the
  50 ms target — say the live-camera number, because that is the path the video
  is showing.
  p50 (34–45 ms) passes but the criterion is a p95. If a judge asks, that is the
  whole answer; do not round it into "under 50 ms", and do not quote the older
  46–54 / 68–70 DEBUG figures as if they were the shipping configuration.
- "Uncertain — sign again" as a system behaviour. It does not exist.

## 5. The 77% → 85% change, if a judge asks

Say this plainly:

> "My earlier number was 77.02% because the landmark cache I evaluated on had a
> wrong presence mask on 29% of test clips. Re-extracting at the native 640×480
> geometry the app actually uses gives 85.45% on the same models with the same
> weights. The weights never changed — the evaluation was wrong."

Do not present this as a model improvement. It is a measurement correction that
happened to be in our favour.

## 6. Recording with the current APK

The APK to film is the **signed release build**, not tracked in git because 80.8
MB times out the HTTPS push. Install it from the GitHub release
[`v2026-11-22-submission`](https://github.com/dev25bansal-ops/EchoGesture/releases/tag/v2026-11-22-submission)
and check the digest against
`submission/apk/EchoGesture-Auto-6.2.apk.sha256` (`b1c0e671…dcf8`).
Confirm the footer reads `9-model ensemble · FP16 TFLite · SHA-256 995cc8d0…`
before you roll. The older `EchoGesture-2026-11-21.apk` is the **debuggable**
build — it is what every latency, power and thermal figure in `results.csv` was
measured on, so it is the right artifact for a measurement session and the
wrong one to ship or to film.

**Film the clip path, not the live camera.** The same VOICE sign scores 0.969
through the file picker and 0.246 through the camera in the same session; the
camera path is a documented limitation, not a demo. Use DARK or MECHANIC — the
held-out pair — and say so on camera.

If the phone is not available, see §7 — but a phone **was** available on
2026-11-21 and the shipped build was driven on it, so §7 is a fallback, not the
expected case.

## 7. If no phone is available

Do **not** fake a phone shot. Two honest options:

1. **Record the real Android app on an emulator** with a pre-recorded video
   played into its virtual camera, and label the frame
   `EMULATOR — synthetic camera input`. Only do this if the clip actually runs
   through the shipped nine-model path and produces a real prediction. The
   emulator has no hand detector path on x86_64 for MediaPipe in some builds —
   if it cannot produce a prediction, this option is not available.
2. **Record the Python evaluator** producing a real prediction on an authorised
   clip, labelled `DESKTOP EVALUATOR — not the Android app`, and state that the
   phone demonstration is pending.

Say which one you used. Never imply the Android app was demonstrated if it was
not.
