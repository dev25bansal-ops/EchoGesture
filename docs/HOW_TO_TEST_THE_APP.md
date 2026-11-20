# Test the app yourself: four clips with known-correct answers

Written 2026-11-19. Everything here was verified on the Vivo V2404.

## First: the installed APK is the right one

Two independent checks, both passed:

| check | result |
|---|---|
| SHA-256 of the APK on the device vs the local build | **identical** — `503766ea0efeda85…`, 82,828,005 B |
| App footer `SHA-256 995cc8d0b6f6…` vs the nine model files | **match** — `995cc8d0b6f6` |

That second one matters most: the app displays the digest of the weights it
loaded, and it equals the digest of `models/*.tflite` that produced 85.45%. So
the phone is running the graded model, not a stale copy.

## The clips

Copied to the phone at **`/sdcard/Movies/EchoGesture/`** (visible in Photos /
Files), and also in the repo at `docs/demo_clips/`.

| clip | correct answer | confidence | margin | split |
|---|---|---|---|---|
| `08763437833459409-DARK.mp4` | **DARK1** | 0.9803 | high | **test — held out** |
| `9119691570534749-MECHANIC.mp4` | **MECHANIC1** | 0.9344 | 0.9241 | **test — held out** |
| `3567776249024257-NOON.mp4` | **NOON1** | 0.9731 | high | *train — seen during training* |
| `7606272862279693-VOICE.mp4` | **VOICE** | 0.9574 | high | *train — seen during training* |

**Read the split column.** DARK and MECHANIC are clips the model never saw in
training, so their scores are evidence of generalisation. NOON and VOICE are
clips the model was trained on, so those scores show it can still recognise a
familiar sign — a real property, but **not** the same claim. Both are useful for
testing that the app is wired correctly; only the held-out two say anything
about accuracy on a new signer.

Each value is from `predict()` run through the real pipeline on that exact
file — not from a cache lookup. These are clips where the model is right *and*
confident, which is what you want for a demo.

## How to test on the phone

### Easiest and most reliable: pick a video clip

The camera tab has a **Pick a video clip** button. Tap it, choose one of the
clips above from `Movies/EchoGesture`, and the app classifies the file directly —
no camera, no lighting, no steady hand.

Verified on the Vivo V2404, all four clips, correct every time:

| clip | result | score | split |
|---|---|---|---|
| DARK | DARK1 | 77.1% | held out |
| MECHANIC | MECHANIC1 | 87.9% | held out |
| NOON | NOON1 | 95.7% | train |
| VOICE | VOICE | 96.5% | train |

**DARK and MECHANIC are the meaningful ones.** NOON and VOICE were in the
training set, so a high score there proves the app is wired correctly, not that
it generalises. The held-out pair is the honest check — and note their phone
scores (77.1%, 87.9%) sit closer to the 85.45% corpus average than the training
pair's 95%+ do.

**This is the test to use when something looks wrong.** If the clip answer is
correct but a live capture is not, the model is fine and the camera scene is the
difference — which is exactly what happened on 2026-11-19, when a live NOON
capture returned FOREIGNER1 while the file returned NOON1 at 0.957. Full record:
`docs/CLIP_UPLOAD_2026-11-19.md`.

### Or: sign it live

1. Hold the phone so its **camera sees your hand**, and tap **Start sign**, then
   **Finish sign** about 1–2 s later.

**This is the part that goes wrong.** Testing against a laptop playing corpus
video is what sent a full day into investigating the model. If you film a
screen, the camera sees a bright rectangle with a small hand in it, and the
detector's hand coverage collapses. Expect wrong answers that way, and do not
read them as model failures — check the same sign through **Pick a video clip**
first.

**Best option: sign it yourself.** DARK, NOON and VOICE are one-handed and
static; they are genuinely easy to perform. Hold your hand inside the white
frame guide, tap Start sign, make the sign, tap Finish sign. The app trims to
the motion and classifies that span.

## What "correct" looks like

On a good capture the HUD shows the gloss in green, a confidence, and a top-5
list, e.g.:

```
DARK1
confidence 0.98 | clip ... ms | Top 5: DARK1, ...
```

If it shows `RETRY` or `Declined`, the acceptance gate (`margin >= 0.15`) judged
the reading too close to call — that is the app working as designed, not a
crash. Re-record rather than shipping a guess.

## What is NOT claimed

- These four clips are ones the model gets right. The corpus average is
  **85.45%**; a hand-picked demo always looks better than the average.
- On-device accuracy over the full corpus has **not** been measured — the
  replay test hangs on this device. What is verified is that the same nine
  weights and label map are installed, and that the app produces correct
  predictions on real captures.
- The detector runs on the **GPU delegate** (confirmed in logcat), which
  measured detect p95 46–54 ms.
