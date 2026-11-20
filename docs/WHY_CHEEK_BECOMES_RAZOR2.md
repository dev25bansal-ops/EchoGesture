# Why CHEEK comes out as RAZOR2

Date: 2026-11-19
Status: **diagnosed, not fixed.** Every number below was produced by a command
recorded in this file. Nothing here is inferred.

## The short answer

Two separate things are true, and they are not the same problem.

1. **CHEEK is genuinely one of the weakest signs in the vocabulary.** Across
   every cached clip at the geometry the phone actually uses, it is 75%
   correct (24/32), and its wrong answers are PATIENT2, APPLE, MYSELF and
   COMB2. It ranks 8th worst of 100 glosses on test.
2. **RAZOR2 is not one of them.** In 32 cached CHEEK clips, and in all 1,251
   test clips, **CHEEK never predicts RAZOR2**. Zero times. So the specific
   error the user saw is not a class-pair confusion the model has.

The third thing follows: the app's acceptance gate is much weaker than its
own documentation claims, so a wrong answer from a weak sign is presented
with a confident-looking percentage instead of being declined.

## What was ruled out, with the measurement that ruled it out

### The Kotlin feature port is not the bug

`FeatureTransform.kt` is a hand-port of `src/features.py`. It was compared
against the Python reference on all 11 test CHEEK clips, both producing the
same 24x255 / 24x381 matrices.

```
clip                            REF        APP        agree   max feature diff
028495375833406555-CHEEK        CHEEK      CHEEK      yes     2.38e-07
1276420379859602-CHEEK          PATIENT2   PATIENT2   yes     2.38e-07
5240585169048262-CHEEK          APPLE      APPLE      yes     1.19e-07
...
CHEEK correct:  REF 7/11   APP 7/11
```

`python .scratch/cheek_diag.py`  (needs torch: `py -3.11`)

Maximum absolute feature difference is `2.4e-07` — float32 rounding, nothing
more. The two implementations are the same function. **Any explanation that
blames the port is wrong.**

### Fixed-window idle padding is not the bug

The corpus clips are trimmed to one sign; the app captures a fixed 3700 ms
window ending at the tap. Padding a CHEEK clip with a held first frame
degrades accuracy but never produces RAZOR2.

```
idle_frac   CHEEK   RAZOR2   other   of 11
       0%       7        0       4
      20%       5        0       6
      50%       3        0       8
      80%       7        0       4

python .scratch/cheek_pad_diag.py
```

RAZOR2 appears zero times at every padding ratio.

### Frame geometry is not the bug

Corpus videos are overwhelmingly 640x480:

```
corpus video geometries (w,h) -> count:
   (640, 480) aspect 1.333  3218
   (960, 540) aspect 1.778   172
```

And the app requests exactly that — `CameraScreen.kt:665` uses
`.setTargetResolution(android.util.Size(640, 480))`. Aspect ratio, and
therefore the x/y anisotropy of MediaPipe's normalised coordinates, matches
training.

### Simple geometric domain shifts are not the bug

Hand scale, position in frame, and motion amplitude are all absorbed by the
root-centring and shared-scale normalisation in `features.py` — by design.

```
shift            CHEEK   RAZOR2   other   most common winner
none                 7        0       4   PATIENT2
scale_up             7        0       4   PATIENT2
scale_down           7        0       4   PATIENT2
shift_left           7        0       4   PATIENT2
shift_up             7        0       4   PATIENT2
static               7        0       4   PATIENT2
mirror_x             0        0      11   TAIL1        <-- destroys it, wrongly
swap_slots           0        0      11   TAIL1        <-- destroys it, wrongly

python .scratch/cheek_shift_diag.py
```

Mirroring and slot-swapping are catastrophic but produce TAIL1, not RAZOR2.
They would also break every other sign, so they are not happening.

### MediaPipe VIDEO mode is not the bug

The corpus was extracted in `RunningMode.IMAGE`; the app runs `VIDEO` with
`detectForVideo`. This does perturb the landmark stream (median per-coordinate
difference 0.069, max 0.556 — enough to move some clips), but it does not
create RAZOR2.

```
32 CHEEK clips    IMAGE correct 24/32    VIDEO correct 24/32    changed 3/32
30 RAZOR2 clips   IMAGE correct 24/30    VIDEO correct 22/30    changed 4/30

CHEEK image-mode winners: CHEEK 24, PATIENT2 4, PENNY 1, APPLE 1, MYSELF 1
CHEEK video-mode winners: CHEEK 24, PATIENT2 5, MYSELF 1, PENNY 1, APPLE 1

python .scratch/imode_vs_vmode.py   (needs torch+cv2+mediapipe: C:/Python314)
```

IMAGE-mode CHEEK never predicts RAZOR2 either. The mode is a real but
secondary difference — worth remembering, not the cause.

## What CHEEK actually collides with

All 32 cached CHEEK clips (train+val+test) at native geometry:

```
top-1 correct    : 24/32 = 75.0%
predicted RAZOR2 :  0/32
gate accepted    : 27/32, correct when accepted 23/27 = 85%
winners          : CHEEK 24, PATIENT2 4, PENNY 1, MYSELF 1, COMB2 1, APPLE 1
```

CHEEK sits in the weakest part of the vocabulary. Test-split recall, n>=9:

```
  #1  CHEESEGRATER   10 clips   40.0%
  #2  SINK           10 clips   50.0%
  #3  SCARF2         12 clips   50.0%
  #4  MYSELF         11 clips   54.5%
  #5  RAZOR2          9 clips   55.6%
  ...
  #8  CHEEK          11 clips   63.6%
  ...
  #22 SHAVE1         15 clips   73.3%
  #53 APPLE          11 clips   90.9%
```

Note that **SINK (50%), SCARF2 (50%) and RAZOR2 (55.6%) are all worse than
CHEEK.** When the user signed SINK and got SCARF2, that is the same weak
region of the model, not a separate fault.

## The real defect: the gate is far weaker than it claims

`ResultPolicy.kt` documents:

> Measured coverage on that population: a 0.15 margin keeps 61% of captures
> at 94% accuracy.

At the native geometry the app actually runs, measured on the whole test
split (`python .scratch/gate_measured.py`, 1,251 clips):

```
--- ALL test clips  (n=1251) ---
 threshold  accepted  coverage  accuracy
      0.10      1124       90%     91.3%
      0.15      1080       86%     92.1%
      0.20      1035       83%     93.4%
      0.50       802       64%     98.4%

ResultPolicy claim:         margin>=0.15 -> 94% accuracy at 61% coverage
measured at native geometry: margin>=0.15 -> 92.1% accuracy at 86% coverage
```

So on the average sign the gate is roughly as advertised — 92% not 94%. The
problem is that it is a **global** number applied to a vocabulary whose
per-sign accuracy ranges from 40% to 100%. On the six-sign face-level cluster
the user is actually testing:

```
--- face-level cluster only  (n=94) ---
 threshold  accepted  coverage  accuracy
      0.15        72       77%     75.0%
      0.30        53       56%     86.8%
      0.50        37       39%     94.6%
```

And per sign, when the gate accepts:

```
CHEEK   n=11  accepted=10  correct when accepted  7/10 = 70%
SINK    n=10  accepted= 6  correct when accepted  4/6  = 67%
APPLE   n=11  accepted= 9  correct when accepted  9/9  = 100%
```

**The gate accepts 10 of 11 CHEEK clips and is only 70% right about them.**
That is exactly the reported experience: a confident-looking answer that is
wrong. And the gate lets through 85 confidently-wrong readings across the
test split, some at extreme margins:

```
truth         predicted      top1   margin
DOWNSIZE1     VLOG          91.3%    0.884
JEWELRY       ZEBRA         83.3%    0.695
CHEESEGRATER  RAZOR2        73.0%    0.605
MYSELF        PATIENT2      78.0%    0.603
```

A 0.15 margin cannot separate right from wrong when the model is wrong with
91% confidence.

## What this means for the RAZOR2 report specifically

The corpus evidence establishes that CHEEK and RAZOR2 are **not** a confused
pair in this model, and that no transform the app performs creates that pair.
So the phone is producing landmarks for CHEEK that sit outside anything in
the 3,388-clip corpus — most plausibly because a live camera, a single
signer with different hand shape, speed and framing, and a background the
model has never seen all push the input off the training manifold.

That last step is **an inference, not a measurement.** Confirming it needs
the phone's own landmarks for the failing sign, which is the one thing not
yet collected. `DiagnosticsLog` records frame counts and presence flags but
**not landmarks**, so it cannot answer this on its own.

## To confirm the last step

Either of these produces the missing evidence:

1. Turn on **Capture diagnostics** in Settings, sign CHEEK, and pull
   `/data/data/org.echogesture.finalapp/files/diagnostics.txt`. This gives
   coverage, frame count and timings — it rules in or out a capture-side
   problem, but not the landmark distribution.
2. Record the failing attempt with the phone's own screen recorder and run
   that video through `src/main.py` with `size="native"`. This is the direct
   test: same pixels, same code path as scoring. If the video also comes out
   RAZOR2, the cause is the input and nothing in the app.

## What would actually improve this

In order of expected effect, all of them measurable before shipping:

- **Per-sign or per-cluster confidence thresholds.** The single global 0.15
  margin is the wrong instrument. A threshold calibrated on the weakest 15
  signs would decline most of the 70%-correct CHEEK readings instead of
  presenting them. This is a `ResultPolicy` change and a calibration table,
  not a model change.
- **Say less, more honestly.** `ResultPolicy` already refuses to claim
  probability-of-correctness. The UI showing "80.7%" invites exactly the
  reading the user made. Showing the top-1 with an explicit "best match, may
  be wrong" and the runner-up is more truthful for a 70%-accurate sign.
- **A "not in my vocabulary" class.** The model has no way to say "none of
  these". Every out-of-distribution input becomes some gloss with a number
  attached. Training a rejection class needs data the project does not have
  today, so this is a larger change.
- **More CHEEK-like training data.** The model is 75% on CHEEK because the
  corpus has 32 clips of it. This is the slow fix and out of scope for the
  deadline.

## Reproduction

All scripts are in `.scratch/`. They need different interpreters because the
environment has torch, cv2 and mediapipe split across them:

| script | interpreter | what it shows |
|---|---|---|
| `cheek_diag.py` | `py -3.11` | Kotlin port == Python reference |
| `cheek_pad_diag.py` | `py -3.11` | idle padding does not create RAZOR2 |
| `cheek_shift_diag.py` | `py -3.11` | geometric shifts do not create RAZOR2 |
| `cluster_diag.py` | `py -3.11` | full-test confusion, RAZOR2 attracts 8 clips |
| `gate_measured.py` | `py -3.11` | the gate's real accuracy and coverage |
| `imode_vs_vmode.py` | `C:/Python314/python.exe` | IMAGE vs VIDEO landmarker mode |

`py -3.11 -I <script>` for the torch-only ones; the 3.14 one must run
without `-I` because its packages are in the user site-packages.
