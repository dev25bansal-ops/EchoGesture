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

**Re-observed on the shipping build, 2026-11-22 12:09 IST.** With the one-tap
flow, tapping **Start Recognition** and not signing produced, after the
auto-segmentation window closed, the amber panel **"No hands in view"** over
*"No sign started. Tap Start Recognition and perform one sign."* (`auto_no_sign`).
Screenshot: [`auto62_04_refusal_no_hands.png`](evidence/auto62_04_refusal_no_hands.png).
The wording is the one-tap variant's rather than the record-then-finish variant's,
which is expected — but it means a judge reading an older quote of the
*"Hands were visible for too little"* line should be pointed at this screenshot
for the state they will actually see when they try it.

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
| **Uncertain** | A reading exists but `margin < 0.15` | Header **"POSSIBLE MATCH"** (`result_label_diagnostic`), body *"Low confidence. Compare the options below or try the sign on its own."* (`result_confidence_uncertain`) |
| **Accepted** | `margin ≥ 0.15` and score ≥ 20% | Header **"Recognised sign"** (`result_label`), body *"Best match — still confirm before relying on it"* (`result_confidence_accepted`) |

**These strings are quoted from the shipping build.** Verified 2026-11-22 against
`app/android_final/app/src/main/res/values/strings.xml` on branch
`codex/android-auto-6.2` (commit `33e82b5`), and confirmed on screen in
[`auto62_05_live_result.png`](evidence/auto62_05_live_result.png), where the card
in fact reads "POSSIBLE MATCH" at 62.4%. An earlier version of this section
quoted *"Uncertain — this is the model's best guess, not a confirmed reading"*,
which belongs to the superseded release build and **is not on the shipping app**.

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

### Gap-filling was tested on 2026-11-22 and also does not help

A third intervention after the retrain and the presence-hold: bridge short
interior gaps by linear interpolation between real observations on both sides,
and measure whether the `drop_hand` cost falls. Measured on the shipped
ensemble over the **full 1,251-clip test split**, `.scratch/gapfill_experiment.py`:

| Condition | corrupted top-1 | drop vs clean | clean top-1 | cost to clean |
| --- | --: | --: | --: | --: |
| no fill (shipped) | 79.70% | **+5.76** | 85.45% | — |
| fill ≤ 8 f, mask off | 80.02% | +5.44 | — | — |
| fill ≤ 8 f, mask on | 80.42% | **+5.04** | 84.33% | −1.12 |
| fill 3–8 f only, mask on | 77.38% | +8.07 | 84.49% | −0.96 |
| fill 5–12 f only, mask on | 76.26% | +9.19 | 82.41% | −3.04 |

The harness reproduces the shipped numbers **exactly** before any intervention
(85.45% clean, +5.76), which is the gate that makes the deltas trustworthy — it
is the same experiment, not a neighbouring one.

The best variant reaches **+5.04, still 0.04 over the 5-point line**, and costs
1.12 points of clean accuracy to get there. Narrowing the window to
occlusion-length gaps only (3–8 frames) makes it **worse** (+8.07), which says
the small benefit came from smoothing detector flicker, not from anything
specific to occlusion. There is no setting that passes, so this does not ship.

The reason is the finding worth keeping: **34 of 59 uncorrupted clips contain
real interior MediaPipe gaps** (145 frames in that sample). The model has
learned that an absent hand is meaningful — it is the difference between a
one-handed and a two-handed moment — so interpolating across it fabricates
motion the signer did not make.

Absence is signal here, not missing data to be tidied up. That is also why both
the retrain and the presence-hold lost: each trades clean accuracy for a
corruption score, because the corruption asks the model to unlearn something
true. This is a property of the task, not a bug left in it.

The remaining honest lever is aggregation, not the model: `drop_hand` costs
**5.76 points unconditional** and **2.39 at the shipped acceptance gate at
86%/80% coverage**. Quoting the second number alone would overstate the result,
and quoting the first alone understates the artifact's real behaviour. Both
ship, labelled.

### Why a matched retrain would not have helped either — the decisive control

The retrain above changed *how much* dropout was applied. A separate question is
whether the shipped model loses accuracy because of a **train/eval convention
mismatch** rather than because the information is gone. It is worth ruling out
explicitly, because it is the one hypothesis under which a cheap fix would exist.

The training augmentation (`hand_dropout`, `scripts/ml/search.py:64`) zeroes
**only the presence flag** and leaves the landmark coordinates intact. The
evaluation corruption (`c_drop_hand`, `scripts/ml/robustness.py:88`) zeroes
**both**. So training simulated "flag says gone, numbers still meaningful" while
scoring asks about "flag says gone, numbers are zeros". If the model were
sensitive to that difference, the fix would be a one-line change to the
augmentation.

It is not sensitive. Measured on the shipped 9-member ensemble over the full
1,251-clip split, same corruption seeds, `.scratch/test_flag_only_drop.py`:

| corruption | top-1 | delta vs clean 85.18% |
| --- | --: | --: |
| coordinates **and** flag zeroed (what the harness scores) | 79.92% | **−5.26** |
| **flag only** (what training actually simulated) | 79.92% | **−5.26** |
| coordinates only (no mask headroom) | 78.49% | −6.69 |

Flag-only and coords+flag are **identical to a tenth of a point**. The model
already discards an absent hand's coordinates entirely, purely on the flag — the
mask does exactly the job it was designed for. Zeroing the coordinates as well
therefore adds no new information and costs nothing. The mismatch is real in the
code but inert in the model, so aligning the two conventions would change
nothing.

This is the mechanism behind "absence is signal, not missing data." The model
does not see zeros and get confused; it sees the flag go to zero and stops
reading the hand. What remains is a genuine deficit of evidence: a hand absent
for a quarter of the clip cannot be recovered from the frames in which it is
present. Together with the retrain (5.76 → 5.12 at a 5.2-point clean cost) and
gap-filling (+5.04 at a 1.12-point clean cost, still failing), that is three
independent repairs tested and rejected, and a fourth — the convention fix —
shown to be a no-op before it was attempted. The row stays `FAIL`, and it is
known to be information-limited rather than unattempted.

---

## 5. What is *not* claimed

- **No phone accuracy figure exists.** The one phone run confirms the pipeline
  works and that the clip path is correct; it is not a measured accuracy.
- **No HUD usability study with real signers.** The refusal path was observed
  once, by us, on one device.
- **No claim that occlusion is solved.** It is half-passing and disclosed.
