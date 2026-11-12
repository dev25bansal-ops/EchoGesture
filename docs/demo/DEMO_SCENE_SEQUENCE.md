# Demo video — scene sequence, ready to record

Status: **NOT_RECORDED**. A shot list existed before this file; the directive is
explicit that a shot list is not a demo. Below is the recordable sequence with
exact on-screen content and narration, plus the minimum human action.

Target length **2:50**, hard cap **3:00** (deck p.25: `Demo video, max 3 minutes`).

---

## Preconditions — one thing must be true before recording

Two options, in order of preference. Do not mix them.

### Option A — real Android phone (preferred)

Requires a physical device. None is attached to this machine.

1. On the phone: Settings → About phone → tap **Build number** 7× →
   Developer options enabled.
2. Settings → Developer options → enable **USB debugging**.
3. Connect to the Mac with a **data-capable** USB cable (many charge-only cables
   will not enumerate).
4. Accept the phone's **Allow USB debugging** prompt.
5. Verify:

   ```bash
   adb devices -l
   ```

   The phone appears with `device` (not `unauthorized`, not `offline`).

6. Install the APK:

   ```bash
   adb install -r app/android_final/app/build/outputs/apk/debug/app-debug.apk
   ```

   APK under test: `e5ed0a3b…`, 75,072,706 B. Verify with
   `shasum -a 256` before installing; do not install a different build.

7. Record the phone screen with the device's own screen recorder or a second
   camera on a tripod. If using a second camera, film the phone **and** the
   signer in frame — that is the exocentric view deck p.10 asks for.

### Option B — desktop file-based demo (acceptable, must be labelled)

No phone required. Uses the real four-model ensemble on a local clip.

```bash
.venv/bin/python -I app/offline_demo.py \
  --video "evaluation/device/v5_accuracy/data/val_30/5784034175717614-EASY.mp4"
```

Screen-record that window. **Say out loud, on camera, that this is a
file-based demo and not a live camera run.** Never let the edit imply the
prediction came from a live camera.

---

## Scene sequence

### 0:00–0:20 — The problem (20 s)

**On screen:** title card, then a still of two people signing at each other.

**Narration:**
> "Sign language conversation runs faster than most assistive software can keep
> up with. And when a bystander is filmed to make it work, their video leaves
> the device. EchoGesture addresses both: everything runs locally, on the phone."

**Do not:** claim full sentence translation. This model recognises isolated
signs from a 100-gloss vocabulary.

### 0:20–0:40 — What we built (20 s)

**On screen:** the architecture diagram (camera → landmarks → 255 features →
4 models → mean logits → one softmax → caption).

**Narration:**
> "A camera frame goes to MediaPipe for hand landmarks, which become a 24-frame,
> 255-value feature vector — position, motion, and the geometry between the two
> hands. Four independently trained FP16 models each score it. We average their
> raw logits and apply one softmax. Nothing leaves the phone."

### 0:40–1:10 — Real inference (30 s)

**On screen (Option A):** phone, live camera, signer in frame. Tap capture.
Caption appears.

**On screen (Option B):** terminal or HUD running the offline demo.

**Narration:**
> "Here is a capture. The model returns the sign, a confidence, and the next
> four candidates it considered."

**Show:** the gloss, the confidence percentage, and the top-5 list. These are
real values from the run, not mocked.

### 1:10–1:40 — Offline proof (30 s)

**On screen:** enable airplane mode (Option A) or show the process running with
no network available (Option B). Repeat a capture.

**Narration:**
> "This is the same device with the radio off. The app declares only camera
> permission — no internet permission at all, so it cannot open a socket even if
> it wanted to."

### 1:40–2:05 — The ensemble, concretely (25 s)

**On screen:** a simple table of the four members and their sizes.

**Narration:**
> "Four members: two TCN, two Conv1D. 3.5 megabytes of weights, 11.4 with the
> detector. Averaging raw logits before the softmax is what makes them work
> together — averaging probabilities would not give the same answer."

### 2:05–2:30 — Measured results (25 s)

**On screen:** accuracy and size only, as large numbers.

> "72.78 percent top-1 across 1,249 clips from eleven signers the models never
> saw in training. The test split was inspected during development, so this is a
> held-out result, not a blind one."

**Do not** put a latency number on screen without the qualification below.

### 2:30–2:50 — Limitations and close (20 s)

**On screen:** three short lines.

**Narration:**
> "Two honest gaps. Our fastest full pipeline is about 94 milliseconds on a
> laptop — the 50-millisecond target is not met, and most of that time is hand
> tracking, not the model. And heavy occlusion or cropping drops accuracy by six
> to ten points. We are shipping what we measured, not what we hoped for."

**Close:** app name, track, one line on privacy.

---

## Claims that must NOT appear

| Never say | Why |
| --- | --- |
| "under 50 ms" / "meets the latency target" | measured p95 is 93.9 ms at best, on a laptop |
| "real-time continuous translation" | isolated 100-gloss recognition |
| "INT8 quantized" | the shipped artifact is FP16 |
| "3D hand features" | the extractor discards the z channel |
| "works in all lighting" | crop and occlusion both fail the 5-point budget |
| "live camera" over footage that is file-based | must be labelled |
| any accuracy on the organizer's test set | we never received their manifest |

## After recording

1. Keep it **under 3:00** — check the actual file duration, not the script.
2. Upload to YouTube (unlisted is fine) or Drive with link sharing on.
3. Verify the link opens in a private browser window before pasting it into the
   form.
4. Do not publish participant footage from ASL Citizen or WLASL. Use your own
   consenting signer or the file-based demo.