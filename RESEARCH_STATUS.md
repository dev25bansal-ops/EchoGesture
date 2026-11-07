# Research status — what is done, what is left

Last updated 7 Nov 2026. Six research rounds plus Round 7 are complete.
This file records what is settled, what is genuinely open, and what would
close each gap.

---

## Settled — no further search needed

These were searched to exhaustion and will not change with more rounds.

### 1. No drop-in model exists
Hugging Face, GitHub, Kaggle and PapersWithCode were searched for a
sign-language INT8 TFLite model under 20 MB. The only real INT8 artefact is a
MediaPipe hand-landmark detector (`OpenReasonAI/mediapipe-hands-int8`, 5.9 MB),
which is a front-end, not a classifier. No WLASL-classifier exists.

### 2. The published tiny model does not generalize
S3D-Conv1D's shipped weights: 100% on training signers (17/17 positive
control), **1.47%** on 1,705 signer-disjoint clips vs 1.02% chance.
Root cause is the pixel front-end plus a random 70/20/10 training split.
See [S3D_signer_disjoint_evaluation.md](S3D_signer_disjoint_evaluation.md).

### 3. Landmarks are the answer, not pixels
Google's Isolated Sign Language Recognition competition (1,165 teams) is the
closest analogue: landmarks in, gloss out, TFLite, **held-out signers**.
Winner: 118 landmarks, XY + lag-1 velocity + lag-2 acceleration (708 features),
Conv1D+Transformer, 1.85M params, 11.24 MB TFLite, 0.8929 on a test set of 10
participants disjoint from the 21 training participants. Verified against
primary sources.

### 4. The "50 ms" figure from that competition is wrong
The competition limit was **100 ms**, and the winner's notebook warns it uses
the maximum inference budget. Do not cite 50 ms from GISLR.

### 5. Frame budget: 12–16 frames is NOT safe
Measured directly on 517 WLASL-100 clips with the shipped S3D-Conv1D INT8
pipeline (verify-r7-2):

| Frames | Top-1 | Retention vs all-frames | McNemar p |
|---|---|---|---|
| 32 | 93.81% | 99.4% | — |
| 24 | — | 100.6% | 0.375 |
| 20 | — | 98.1% | 0.012 |
| 16 | 78.72% | 83.9% | 0.006 |
| 12 | 63.83% | 68.0% | 0.0004 |

Uniform-grid phase jitter makes it worse: k=16 drops to 80.8% mean / 57.8% at
k=12. **Use ≥20–24 frames.** If a 16-frame budget is forced, the model must be
*retrained* at 16 frames — the WLASL-300 FP32 run retained 96.6% at both k=16
and k=12, showing the retention is model-dependent, not universal.

### 6. Latency is decode-bound, not model-bound
Measured on 60 real clips: decode 83.4 ms p50, model invoke 2.1 ms p50.
Seeking with `CAP_PROP_POS_FRAMES` is 550 ms — never do it.

### 7. Hackathon logistics
Registration form live, closes 8 Nov 23:59 IST. College dropdown limited to
five institutions; the deck states no such restriction. Deadline 18 Nov.
Rs 1.0 L prize is deck-only — no external corroboration found.

### 8. Samsung has no relevant SDK
`developer.samsung.com` has no Galaxy XR / hand-tracking / on-device-vision
SDK. ARCore for Jetpack XR exposes pinch/stop heuristics only, no classifier.
Do not plan around a Samsung API.

### 9. Accessibility framing
Deaf community critique is well documented (sign-language gloves, hearing-
centric framing, gloss ≠ translation). Deck must state ASL-only scope, position
as complement not interpreter substitute, and avoid "gives Deaf people a voice".

---

## Open — real gaps that research has not closed

### A. Training a landmark model at all (highest value)
**Status: not started.** We have the recipe (118 landmarks, 708 features) but
no trained landmark classifier. This is the single blocker on accuracy.

To close: extract MediaPipe landmarks from WLASL-100/300, apply root-centring +
scale normalisation + velocity/acceleration, train a Conv1D+attention model on
participant-disjoint splits. Expected 70–89% based on competition results at
this vocabulary size. Estimated 2–3 days of work.

### B. Whether MediaPipe extraction fits the latency budget
**Status: partially answered.** MediaPipe is 18–35 ms/frame on CPU, which
makes a 24-frame per-clip budget impossible (432–840 ms). The answer is a
streaming architecture — extract once at 15–20 FPS, hold a ring buffer, run the
sub-millisecond classifier on the buffer. **Not yet prototyped or measured.**

To close: build the streaming loop, measure end-to-end latency from camera
frame to emitted gloss on real hardware.

### C. Robustness harness (scored, 25% of marks)
**Status: not written.** No script applies occlusion / low-light / blur / crop /
frame-drop, and no paired clean-vs-robust comparison exists. The `2.56 pt`
figure in earlier drafts was the paper's SemLex delta, mislabelled.

To close: write the transform script, run it on a held-out set, report the
paired delta honestly. Half a day.

### D. Power and thermal measurement (scored)
**Status: not measured.** No meter, no battery capture, no sustained run.
Requires a physical Android device; cannot be done on this laptop.

To close: run a 10-minute loop on the target device sampling `dumpsys
batterystats` and per-30s p95 latency.

### E. Whether a HUD can be built in the remaining time
**Status: not started.** Design spec exists (gloss, top-5, confidence bar,
latency readout, skeleton toggle, SIGNING/IDLE/DEGRADED states). No code.

To close: OpenCV overlay on webcam, ~1 day.

### F. ISL / non-ASL scope
**Status: decided, not built.** ASL-only for this deadline with a stated
roadmap. ISL datasets exist but are small, lab-collected, and non-transferable
from ASL. No further search needed; the decision is made.

### G. Demo video and deck
**Status: not started.** Script outline exists in the plan. Blocked on (A)
and (E) producing something worth filming.

---

## Searches that returned nothing, and are closed

Recorded so nobody repeats them:

- YouTube transcripts: yt-dlp found 9 relevant videos, zero had captions,
  zero stated accuracy or latency numbers.
- Discord / Reddit / X / Telegram: all blocked or empty. No community signal.
- Devpost / MLH / Unstop: 17 queries, zero Samsung XR Hackathon 2026 listings.
- Patents: Google Patents and Espacenet blocked or returned unrelated results.
  No blocking patent verified — absence of evidence, not evidence of absence.
- CNKI / KCI / J-STAGE: empty via US-based search. Would need native-language
  database access.
- Judging rubric: no published rubric exists. All scoring guidance in this repo
  is inference from the deck, and is labelled as such.

---

## Recommended order of work

1. **A** — train the landmark model. Everything else is decoration without it.
2. **B** — prototype streaming so the latency story is real.
3. **C** — robustness harness, half a day, worth 25% of the marks.
4. **E** — HUD, because robustness and HUD share that 25%.
5. **G** — demo and deck once there is something to show.
6. **D** — power/thermal last; needs hardware and affects fewer marks.
