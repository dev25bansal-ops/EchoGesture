# Demo video script — 3 minutes

Recorded with the **shipped nine-model APK** on a physical phone. If a phone is
not available, see §7 before recording anything.

The rule this script follows: **every prediction on screen is a live inference
from a real sign.** If an attempt fails, that take is either retried honestly or
cut. Nothing is ever overlaid to look like a success.

---

## 1. Pre-flight

| Item | Check |
| --- | --- |
| APK | `echogesture-v4.3-masked.apk` (or the P1 build, §6) installed |
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
> the people standing next to them do not sign. EchoGesture turns a phone
> camera into an offline recogniser — no cloud, no upload, no network."

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

**If a sign is wrong:** keep it in, say "that one missed", and sign it again.
An honest miss with a visible retry is stronger than four suspiciously perfect
takes, and judges notice edited cuts.

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

> "On ASL Citizen, signer-disjoint: 77.02% top-1 over 1,249 test clips from
> eleven unseen signers, 92.79% top-5. Nine members are 8.5 megabytes, under
> the twenty-megabyte budget. Android and Python agree to one part in a million
> on the same features."

Do **not** say "77% accurate on a phone". See §5.

### 2:40 – 3:00 · Limits and close

**Shot:** signer, direct to camera.

> "What this is not: not continuous translation, not a 2,000-word vocabulary —
> one hundred signs, one at a time. Robustness to heavy occlusion still fails
> and INT8 is future work. But it runs offline, on a phone, right now."

**End card:** `EchoGesture · Track 02 · <CollegeName_TeamName>`

---

## 3. Technical settings

| | |
| --- | --- |
| Resolution | 1080p, 30 fps minimum |
| Audio | Record clean narration separately; sync in edit |
| Screen capture | `adb shell screenrecord --bit-rate 12000000 /sdcard/demo.mp4` |
| Wide shot | Same frame rate as screen capture, for split-screen |
| Export | H.264 MP4, ≤3:00 hard, target 2:45 |
| Upload | YouTube unlisted or Google Drive link, viewable without sign-in |

```bash
adb shell screenrecord --size 1080x2400 --bit-rate 12000000 /sdcard/demo.mp4
# Ctrl-C on the device, then
adb pull /sdcard/demo.mp4 demo_raw.mp4
```

---

## 4. Sign selection

Use glosses whose predictions can be read at a glance on a phone screen. Before
recording, verify all four are in the shipped vocabulary:

```bash
python3 -c "
import json; v=json.load(open('models/landmark_labels.json'))
print([g for g in v if g in ['HOW','DOG1','ANYONE','SQUEEZE','DOWNSIZE1']])
"
```

Record one plain-background pass and one cluttered-background pass.

---

## 5. Claims you must not make on camera

| Do not say | Because |
| --- | --- |
| "77% accurate on a phone" | 77.02% is a corpus result on a laptop. No phone accuracy measurement exists. |
| "Exactly 184 signs tested" | 184 evaluations on 72 unique clips. |
| "Real-time translation" | It recognises one isolated sign per recording. |
| "Robust to occlusion" | Pixel-level robustness fails at 10.0 pp. |
| "INT8 quantised" | The shipped model is FP16. |
| "Works in any language" | It is ASL, 100 signs, fixed vocabulary. |

Safe: "works offline", "no network permission", "on-device", "seven of nine
members", the corpus numbers with their population named.

---

## 6. Which APK to film

| APK | Footer | Use |
| --- | --- | --- |
| `echogesture-v4.3-masked.apk` | `Four-model ensemble` | **Do not film** — misstates the model |
| P1 rebuild (`150ffeeb…`) | `9-model ensemble` | **Film this one** |

If only the V4.3 APK is available, film it but state the count correctly in
narration; the running model *is* nine members.

---

## 7. If no physical phone is available

**Do not** present emulator footage as phone footage, and do not label the demo
"live on device" while it is an emulator with a scripted input file.

Options, in order of preference:

1. Borrow an Android phone (API 30+, arm64) and record the real thing.
2. Record on an emulator and say so plainly, on screen and in narration:
   *"This is the app running in an Android emulator; no physical-device
   measurement exists yet."* The recognition is genuine inference either way.
3. If neither is possible, the video is **BLOCKED**. Say so in the submission
   notes rather than implying a device test that did not happen.

---

## 8. Final checklist before upload

- [ ] Duration ≤ 3:00
- [ ] Airplane mode visible
- [ ] All displayed predictions are live inferences from real signs
- [ ] No overlaid result that did not come from the app
- [ ] At least one genuine attempt shown unedited end-to-end
- [ ] Any miss kept, or honestly retried on camera
- [ ] 77.02% attributed to the corpus, not the phone
- [ ] Model footer legible in at least one shot
- [ ] `CollegeName_TeamName` on the end card
- [ ] Link viewable without sign-in
- [ ] Link pasted into `submission/RELEASE_MANIFEST.json`