# Occlusions & HUD clarity — the evidence for the 25-point criterion

Written 2026-11-22. The organizer rubric (deck V3-2, slide 11) awards
**Robustness & Delivery — "Occlusions & HUD clarity" — 25% of the track score**.
[`ROBUSTNESS_OCCLUSION_2026-11-20.md`](ROBUSTNESS_OCCLUSION_2026-11-20.md) covers
the occlusion half. This document covers the half that had no dedicated evidence
file: **what the app shows a user, and what it refuses to show**.

Nothing here is a design intention. Every behaviour below is a string or a code
path in the shipped tree, and the ones marked *observed on device* were exercised
on the physical Vivo V2404 on 2026-11-21/22.

---

## 1. The rule: it declines, and it says why

`SignRecorder` returns `Unusable(reason)` for seven distinct failure modes, each
with its own user-facing message. The app does not fall back to a best guess.

| Code path | User sees |
| --- | --- |
| `Rejection.LOW_COVERAGE` | *"Hands were visible for too little of that recording. Keep both hands in the guide while you sign."* |
| `camera_state_no_hands` | *"No hands in view"* |
| `Rejection.TOO_SHORT` | *"That recording was too short to read. Start, perform the whole sign, then finish."* |
| `Rejection.TOO_FEW_FRAMES` | *"The camera delivered too few frames to read that sign. Hold the phone steady, keep your hand in the guide, and record again."* |
| `Rejection.CAMERA_STALLED` | *"The camera stopped during that sign. Hold still for a moment, then record again."* |
| `Rejection.TOO_LONG` | *"That recording ran past the maximum length and was cut off."* |
| `Rejection.NO_FRAMES` | *"The camera did not send any frames during that recording. Try again."* |

Strings: [`app/android_final/app/src/main/res/values/strings.xml`](../../app/android_final/app/src/main/res/values/strings.xml).
Rejection enum: [`SignRecorder.kt:38-53`](../../app/android_final/app/src/main/java/org/echogesture/finalapp/SignRecorder.kt#L38).

**Observed on device, 2026-11-22.** A 25-second, 102-frame live capture with no
hands in frame produced *"No hands in view — Hands were visible for too little of
that recording. Keep both hands in the guide while you sign."* The app refused
rather than emitting an argmax. The refusal was also **not** written to History —
a declined capture never appears as a recognised sign.

That last point matters more than it looks. The ensemble always returns a
distribution over 100 glosses, so it always has an argmax; on a real phone
recording it returned `FLIP` at 10.6% and `TAIL1` at 24.2% for input that carried
no readable sign. Presenting either as "the recognised sign" would be a false
claim. The app has no notion of "none of these" in the model, so the refusal is
enforced in the delivery layer where it can be seen and tested.

---

## 2. The honest distinction: *declined* vs *uncertain*

These are two different states and the HUD keeps them apart. This is the detail a
judge is most likely to check, because conflating them is the common failure.

| State | When | What the card says |
| --- | --- | --- |
| **Declined** | Too few frames, no hands, low coverage, stall | No reading at all. One of the refusals above. |
| **Uncertain** | A reading exists but `margin < 0.15` | `result_uncertain_label` = *"Uncertain reading"*, plus *"Uncertain — this is the model's best guess, not a confirmed reading"* |
| **Accepted** | `margin ≥ 0.15` and score ≥ 20% | *"Best match — still confirm before relying on it"* |

The gate is one function, [`ResultPolicy.evaluate`](../../app/android_final/app/src/main/java/org/echogesture/finalapp/ResultPolicy.kt#L77), so it can be measured
rather than described. **The app never withholds a prediction on the uncertain
path** — it shows the best guess and labels it. Do not describe the app as
"asking you to sign again"; that behaviour does not exist.

Why margin rather than the raw score: on 82 authorised validation clips the mean
margin was **0.378 when the top-1 was correct and 0.064 when it was wrong**. The
raw score does not separate those cases nearly as well. Recalibrated on 345
validation clips across five unseen signers, the shipped rule accepts **86.1% of
captures at 96.0% accuracy**; the 13.9% it declines are only **35.4% accurate**.
Declining is therefore the correct direction, and the coverage cost is stated
rather than hidden.

---

## 3. What the HUD never claims

The rubric asks for HUD *clarity*, and the clearest thing a HUD can do is not
overclaim. Three specific guards are on screen:

- A model score is captioned: *"A model score ranks the 100 known signs; it is not
  a calibrated probability that the reading is correct. Recognises one isolated
  sign at a time."* (`result_responsibility`)
- The accepted state still says *"still confirm before relying on it"* — not
  "correct".
- The limitations are stated in Settings: *"Recognises one isolated sign at a time
  from a fixed 100-sign vocabulary. It is not a sentence translator and does not
  replace a human interpreter."*

The offline posture is likewise visible rather than asserted: an **Offline** badge
and *"On-device · no network"* on the camera screen, and the app declares no
INTERNET permission, so it cannot upload a frame even if asked to.

---

## 4. The occlusion half, in one table

Full detail and the rejected fix are in
[`ROBUSTNESS_OCCLUSION_2026-11-20.md`](ROBUSTNESS_OCCLUSION_2026-11-20.md).
The numbers a judge should be given, from
[`results/robustness_native.json`](../../results/robustness_native.json) over the
full 1,251-clip signer-disjoint test split:

| condition | top-1 | drop vs clean 85.45% | verdict (≤5 pt) |
| --- | --: | --: | --- |
| clean | 85.45% | — | — |
| jitter σ0.01 | 85.61% | **+0.16** | pass (better than clean) |
| camera further 25% | 85.21% | −0.24 | pass |
| half framerate | 83.93% | −1.52 | pass |
| drop landmarks 15% | 83.29% | −2.16 | pass |
| jitter σ0.03 | 82.25% | −3.20 | pass |
| **`drop_hand_25pct`** | **79.70%** | **+5.76** | **FAIL raw** |
| `drop_hand_25pct`, at the shipped gate | 89.74% | **+2.39** | pass, at 86%/80% coverage |

**How to present this honestly:** one of six conditions fails unconditionally at
+5.76. At the operating point that actually ships it is +2.39 on accepted clips,
but that is a **coverage trade** — the app declines ~20% of occluded captures, and
those declines are correct (of 88 clips accepted clean but declined when occluded,
only 39% would have been right). Quote the coverage whenever the gated number is
quoted. Never present +2.39 as unconditional robustness.

The real fix is training-time occlusion augmentation so the checkpoints learn what
a mid-sign presence drop means. **That was tried and measured, and it lost:** a
harder-hand-dropout retrain (TCN dim 192, 140 epochs, best val 87.3%) scored
**80.26% clean against the shipped 85.45%**, and moved the occlusion drop only
5.76 → 5.12 — it gave up 5.2 points of clean accuracy to buy 0.64 of robustness,
because at 17–24 clips per class harder augmentation trades more than it
recovers. Both checkpoints and the retrain exist locally, so "we could not
retrain" would be false; the accurate statement is "we retrained and it lost."
A presence-hold workaround was also **tested and rejected** because it scores
better the more it hides — it reconstructs the missing input rather than
surviving its absence.

---

## 5. What is *not* claimed

- **No phone accuracy figure exists.** The one phone run confirms the pipeline
  works and that the clip path is correct; it is not a measured accuracy.
- **No HUD usability study with real signers.** The refusal path was observed
  once, by us, on one device.
- **No claim that occlusion is solved.** It is half-passing and disclosed.
