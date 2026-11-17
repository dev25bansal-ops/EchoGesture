# Recognition pipeline verification — 2026-11-17

**Purpose:** the Android app has never recognised a real hand, because the only
camera available was an emulator's synthetic colour pattern. This records what
was done to verify the *pipeline* independently of the phone camera, and what it
does and does not establish.

---

## What was run

The shipped submission archive (`submission/EchoGesture_Submission.zip`,
SHA-256 `ec7b6f68…61c1`) was extracted to an empty directory and `predict()` was
called on **30 authorised ASL Citizen validation clips** whose ground-truth
labels are in the filenames.

These are the same nine FP16 members, the same meta sidecar, and the same
`src/` code path that the ZIP delivers. The only difference from a phone is that
the frames come from a decoded video file rather than a live CameraX stream.

## Result

**24 / 30 correct = 80.0%**

| Ground truth | Prediction | Confidence | |
| --- | --- | ---: | --- |
| WHATFOR | WHATFOR1 | 0.9749 | hit |
| WHATFOR | WHATFOR1 | 0.2502 | hit |
| NOON | BEE1 | 0.2678 | **miss** |
| SQUEEZE | SQUEEZE | 0.4107 | hit |
| ANYONE | ANYONE | 0.8631 | hit |
| 8HOUR | 8HOUR | 0.2941 | hit |
| TEACH | TEACH1 | 0.7051 | hit |
| CHEW2 | CHEW1 | 0.6429 | hit |
| PONDER | PONDER | 0.7486 | hit |
| SOCCER2 | SOCCER2 | 0.6878 | hit |
| MYSELF | PATIENT2 | 0.4567 | **miss** |
| EASY | EASY | 0.9438 | hit |
| WHATFOR | WHATFOR1 | 0.8900 | hit |
| SHAVE | RAZOR2 | 0.2275 | **miss** |
| BABY2 | BABY2 | 0.9613 | hit |
| UNDERWEAR | UNDERWEAR1 | 0.8533 | hit |
| WHATFOR | WHATFOR1 | 0.9592 | hit |
| SHAVE | SHAVE1 | 0.5686 | hit |
| COMB | COMB2 | 0.9718 | hit |
| BEE | BEE1 | 0.3214 | hit |
| MECHANIC | MECHANIC1 | 0.9523 | hit |
| MAPLE | 8HOUR | 0.0941 | **miss** |
| CEMETERY | CEMETERY | 0.7469 | hit |
| TICKLE | TICKLE | 0.9771 | hit |
| CHEEK | ANYONE | 0.1247 | **miss** |
| PENNY | PENNY | 0.9220 | hit |
| AXE | AXE1 | 0.6327 | hit |
| FLIP | FLIP | 0.2609 | hit |
| BASKETBALL | BASKETBALL1 | 0.8605 | hit |
| BACKPACK | CHEER | 0.5603 | **miss** |

## What this establishes

The end-to-end chain is intact and correctly wired:

video decode → MediaPipe landmarks → 32-frame sample → 24-step resample →
255/381 feature build → presence attention mask → nine TFLite interpreters →
mean-logit aggregation → one softmax → class index → label.

A break in any of those links would not produce 24 correct labels out of 30.

It also corroborates the Android path indirectly: the same clip that the Android
result card renders (`WHATFOR1`, 0.9749, top-5 `WHATFOR1 OFFEND DEAF1 KNIGHT1
PENNY`) produces exactly that from this ZIP. Two independently built code paths
agreeing on the same nine binaries.

## What this does NOT establish

**This is not phone accuracy.** It is a laptop measurement on recorded clips.
Camera optics, exposure, distance, framing, background and compression all
differ from a live phone stream. The number could move either way.

**It is not the 85.45% figure either.** That is 1,251 clips at native geometry
with the ensemble evaluator; this is 30 clips through the ZIP entry point.

**The sample is small.** 30 clips gives a coarse estimate; a 95% interval on
24/30 spans roughly 62–91%.

**The miss distribution matters more than the headline.** Four of the six misses
scored below 0.35, which means the app labels them *"Uncertain — this is the
model's best guess, not a confirmed reading."* The confidence gate is catching
the cases most likely to be wrong. Only two misses were confidently wrong
(`MYSELF→PATIENT2` 0.46, `BACKPACK→CHEER` 0.56).

## Reproduction

```bash
unzip submission/EchoGesture_Submission.zip -d /tmp/zipverify
cd /tmp/zipverify/EchoGesture
python -I -c "
import sys, glob, os; sys.path.insert(0,'src')
import main
for f in sorted(glob.glob('<clips>/*.mp4')):
    out = main.predict(f)
    print(os.path.basename(f), '->', out['prediction'], round(out['confidence'],4))
"
```

The clips are authorised ASL Citizen validation videos held outside Git and are
not redistributed.
