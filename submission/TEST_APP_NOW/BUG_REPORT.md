# Bug ledger

All findings from real device interaction on `eg34` (Android 14, ARM64,
API 34). Findings B-01…B-06 were raised against the baseline APK
`150ffeeb…cb7a`; the rotation case was then retested on a clean cold-booted
instance against the rebuilt APK `6653160e…d37c`. Nothing here is inferred from
reading code alone unless it says so.

Statuses: **OPEN** · **FIXED** · **VERIFIED** · **BLOCKED** · **NOT REPRODUCED**

---

## Summary

| ID | Severity | Area | Status |
| --- | --- | --- | --- |
| B-01 | P2 (test integrity) | Instrumentation | **FIXED** |
| B-02 | P2 (test coverage) | JVM unit tests | **FIXED** |
| B-03 | P3 (documentation) | Shipped source | **FIXED** |
| B-04 | — | Camera gating | **NOT REPRODUCED** |
| B-05 | — | Emulator stability | **NOT AN APP DEFECT** |
| B-06 | P2 (UX) | Camera scroll | **VERIFIED OK** |
| B-07 | **P1 (claims)** | Demo script | **FIXED** — reversed statistic |
| B-08 | **P1 (claims)** | Demo script | **FIXED** — scripted behaviour the app lacks |
| B-09 | P2 (claims) | Dev's latency artifacts | **DOCUMENTED, escalated to Dev** |
| B-10 | P2 (claims) | Dev's robustness artifact | **DOCUMENTED, escalated to Dev** |
| B-11 | P2 (test coverage) | Instrumentation | **FIXED** — uncertain branch untested |

**Eleven findings. Zero app crashes. Zero broken buttons. No P0 defects.**

B-07 and B-08 are graded P1 even though they are documentation, because they
would have been **spoken to judges** and one of them would have been contradicted
live on screen. A wrong number in a shipped narration is not a cosmetic issue.

That is a real result, not a clean bill of health by omission — §"What I tried to
break" below lists what was exercised to reach it.

---

## B-01 — Dead test asserted a four-member contract against a nine-member fixture

| | |
| --- | --- |
| **Severity** | P2 — no user impact; a protection that could never fire |
| **Module** | `app/src/androidTest/.../EnsembleInferenceTest.kt` |
| **Status** | **FIXED** |

**Reproduction (code inspection, then confirmed by reading the test report):**
`loadFixture()` contained `assertEquals("fixture must describe four members", 4, memberCount)`.
The fixture on disk is `9 100`.

**Expected:** the assertion runs, so the suite fails loudly if the fixture and
the app disagree on member count.

**Actual:** `loadFixture()` and its `Fixture` class were **never called** by any
`@Test`. The live test iterates `ModelContract.MEMBER_SHA256`, so the stale
assertion was unreachable — it could not fail, and therefore protected nothing.
It also carried a comment claiming the fixture "predates the masked nine-member
bundle", which was true when written and false now.

**Root cause:** the fixture was upgraded to nine members and the ensemble test
was rewritten to read from `ModelContract`, but the old loader was left behind
rather than deleted.

**Fix:** removed `loadFixture()` and the `Fixture` class (49 lines of dead code).
No assertion strengthened or weakened — the dead ones are gone, the live ones are
untouched.

**Evidence:** `./gradlew connectedDebugAndroidTest` → 17 tests, 0 failed,
1 skipped. `allMembersAllocateWithTheMaskedTensorContract` still passes and still
iterates all nine.

---

## B-02 — Parity test named "every member" only checked four of nine

| | |
| --- | --- |
| **Severity** | P2 — a real coverage hole, not a runtime defect |
| **Module** | `app/src/test/.../EnsembleRuntimeParityTest.kt` |
| **Status** | **FIXED** |

**Reproduction:** `fun everyMemberAloneWouldPredictTheEnsembleTop1OnThisClip()`
contained `val memberLogits = Array(4) { floats(l[2 + it], classes) }`.

**Expected:** a test claiming "every member" covers all nine shipped members.

**Actual:** members 5–9 were never read. Had one of the five z-feature members
been broken or mis-ordered, this test would still have passed.

**Root cause:** hardcoded `4` from the pre-ensemble generation, in a test that was
otherwise correctly parameterised (it already read `members` from the fixture
header a few lines earlier for a different assertion).

**Fix:** reads the member count from the fixture, asserts it equals 9, and builds
`Array(members)`. Verified against the real fixture before changing anything —
all nine members independently argmax to index 8, so the assertion is still true,
just no longer partial. Corrected three comments that said "four models".

**Evidence:** JVM suite **110 run, 0 failed, 0 errors, 0 skipped**, including
this test now covering nine members.

---

## B-03 — Shipped class docstring said four members

| | |
| --- | --- |
| **Severity** | P3 — documentation only |
| **Module** | `app/src/main/.../MainActivity.kt` |
| **Status** | **FIXED** |

**Actual:** the `MainActivity` KDoc read *"[LandmarkClassifier] runs four members
through their own interpreters"*. It runs nine.

**Fix:** corrected to nine. No behaviour change.

**Note:** the *user-visible* footer was already correct — it reads
"9-model ensemble" because it is derived from the loaded member count, not from a
constant. This was a comment that had drifted, which matters because it is the
first thing a reader trusts.

---

## B-04 — "Start sign" appeared unresponsive — NOT REPRODUCED

| | |
| --- | --- |
| **Severity** | — (test error, not a defect) |
| **Screen** | Camera |
| **Status** | **NOT REPRODUCED** |

**What I observed:** tapped Start sign at `(540, 1397)`; nothing happened; the
status pill stayed on "Searching for hands".

**Investigation:** dumped the view hierarchy. `captureButton` was at
`[42,1617][1038,1764]` — 220 px lower than where I tapped. Earlier in the session
I had toggled the framing guide **off**, which removed the guidance overlay and
shifted the whole column upward, so my recorded coordinate was stale.

**Conclusion:** my tap missed the button. The button was `enabled=true`,
`clickable=true`. Re-tapping the live bounds started recording immediately.
**No app defect.** Recorded because a false bug report is also a cost.

---

## B-05 — Emulator system services crashed during rotation — NOT AN APP DEFECT

| | |
| --- | --- |
| **Severity** | — |
| **Status** | **NOT AN APP DEFECT** |

**What I observed:** after `settings put system user_rotation 1`, adb calls began
hanging and `logcat -b crash` accumulated 12 FATAL exceptions.

**Investigation:** every FATAL belonged to `com.google.android.gms`,
`com.google.android.gms.persistent`, `com.google.android.googlequicksearchbox`
or `com.google.android.apps.messaging`. Grepping the crash buffer for
`echogesture` returned **0**. System services then degraded to the point that
`cmd: Can't find service: activity` — the emulator, not the app, was broken.

**Conclusion:** this is a Play-Services-on-emulator instability triggered by
rotation on an image with `-gpu swiftshader_indirect`, not an EchoGesture crash.
The app process reported no fault at any point.

**Retest — rotation now verified.** The first instance's system services had died,
so I cold-booted a clean `eg34` instance and repeated the test against the
rebuilt APK `6653160e…d37c`:

| Step | Observation |
| --- | --- |
| Rotate to landscape | `topResumedActivity=…org.echogesture.finalapp/.MainActivity`, **0 echogesture crashes** |
| Rotate back to portrait | App still resumed, **0 crashes** |
| Camera after rotation | Preview live, framing guide correct |

The landscape screen renders correctly and is captured as
`screenshots/18_landscape.png`. **Rotation is a pass**, on the rebuilt build.
The emulator instability is recorded above because it explains the earlier
inconclusive run, not because it is still outstanding.

---

## B-06 — "New here?" card appeared clipped by the nav bar — VERIFIED OK

| | |
| --- | --- |
| **Screen** | Camera |
| **Status** | **VERIFIED OK (not a defect)** |

**What I observed:** on a 2400 px-tall screen the "New here?" card and the model
footer were cut off at the bottom edge.

**Investigation:** `screen_camera.xml` is wrapped in a `<ScrollView>` whose header
comment says *"Everything here is inside a ScrollView so a large-font …"*. Swiping
up revealed the full card, the footer, and the help button intact.

**Conclusion:** the content is taller than the viewport by design and scrolls.
Not clipping. Verified by actually scrolling and screenshotting the result
(`screenshots/12_scrolled.png`).

**Minor note for later:** when a result card is present, the page is ~1.5 screens
tall, so the framing guidance is not visible while a result is on screen. That is
a layout preference, not a bug, and I did not change it.

---

## What I tried to break (and what happened)

Checked as a red-team pass, not as a user walkthrough:

| Attempt | Result |
| --- | --- |
| Deny camera permission | Clear "Camera permission needed" card with an explanation and a retry button. No crash. |
| Re-request after denial | Permission dialog reappears; granting resumes the preview. |
| Background the app with the camera live | Preview survives; camera session resets and recovers. No crash. |
| Tap Start sign with no hands in frame | Recording runs, then rejects with "No hands in view" and a specific instruction. It does **not** guess. |
| Tap Start sign repeatedly | Second tap while PROCESSING is a no-op (`captureButton.isEnabled=false` in PROCESSING). |
| Rotate the device | **PASS** on a clean cold-booted instance: stays resumed, 0 app crashes, camera recovers (see B-05) |
| Search the glossary for a non-existent term | List empties correctly; pill reflects the count. |
| Open a glossary entry from a filtered list | Sheet opens with the correct word. |
| Scroll the camera page | Content scrolls as designed. |
| Read the footer model identity | `9-model ensemble · FP16 TFLite · SHA-256 995cc8d0b6f6…` — matches the pinned hash prefix. |

---

## What could not be tested, and why

| Item | Reason |
| --- | --- |
| Live-camera sign recognition with a real hand | No physical device attached, and the emulator's camera is a synthetic test pattern with no hand in it. The MediaPipe detector correctly reports "No hands in view" against that pattern — which is itself evidence the detector is running, but it is **not** a recognition test. |
| Recognition accuracy on a phone | Same. **No phone accuracy number exists for this project.** |
| The 18-clip `DirectClipParityTest` | Needs an authorised ASL Citizen clip on the device; none present in this environment. It **skips** by design rather than failing. |

---

# V4.7 findings — claims and coverage

These were found auditing Dev's branch `hackathon/dev-model` @ `465166f` and our
own submission docs. None is an app defect; all would have reached a judge.

## B-07 — Demo script inverted the confidence statistic

| | |
| --- | --- |
| **Severity** | **P1** — would be spoken to judges, and is the reverse of the truth |
| **Module** | `docs/hackathon/DEMO_SCRIPT_2026-11-16.md` (Dev), and our own script |
| **Status** | **FIXED in ours** |

**What it said:** *"Below zero point eight confidence the model is right
ninety-nine point seven percent of the time — over 594 clips."*

**What is true** (`CLAIMS_AND_LIMITS.md`, `confidence_calibration.py`): **ABOVE
0.8** confidence, the model is correct on **99.7% of 594** clips. Below 0.8 it is
correct on roughly **72.6%**.

The sentence markets *low* confidence as reliable — the exact inverse of the
finding. Any judge with the calibration table would catch it.

**Fix:** rewritten in `submission/DEMO_SCRIPT.md` with the correct direction, the
correct 594-clip denominator, and an explicit statement that it is a corpus
figure, not phone accuracy.

## B-08 — Demo script narrated behaviour the app does not implement

| | |
| --- | --- |
| **Severity** | **P1** — the presenter would say it while the screen contradicted them |
| **Module** | `submission/DEMO_SCRIPT.md`, `docs/hackathon/DEMO_SCRIPT_2026-11-16.md` |
| **Status** | **FIXED in ours** |

**What it scripted:**

> *(Sign badly → the app shows 'Uncertain — sign again'.)*
> "Rather than guess, it asks me to repeat."

**What the app actually does** (`CameraScreen.kt:958`):

```kotlin
val certain = result.margin >= 0.15f && percent >= 20f
```

When `certain` is false the card **still renders the prediction** and labels it:

> *"Uncertain — this is the model's best guess, not a confirmed reading"*

There is no abstention path. `UiState` has no "ask again" state; the only states
that withhold a result are `NO_HANDS`, `LOW_FRAMES` and `ERROR`, which are input
failures, not low confidence.

**Why this matters:** the scripted beat tells the presenter to claim a refusal
while the audience sees a guess on screen. That is worse than saying nothing.

**Fix:** the beat now says the app shows its best guess and flags it, and it may
only be used if a real attempt happens to trigger the label. It must not be
staged. The 0.5 abstention figure is described as an *offline analysis*, not as
live behaviour.

## B-09 — Dev's branch reports two incompatible per-frame latencies

| | |
| --- | --- |
| **Severity** | P2 — a judge who reads two files will find it |
| **Module** | `results.csv`, `latency_per_frame_distribution.json`, `LATENCY_DECOMPOSITION.md`, `src/landmark_frontend.py` |
| **Status** | **DOCUMENTED** — escalated to Dev, not silently picked |

`results.csv` and the distribution JSON report **15.34 ms/frame PASS**.
`LATENCY_DECOMPOSITION.md` reports **135 ms/frame** and explicitly pre-empts the
CPU-contention explanation. `src/landmark_frontend.py` still says **"FAILS"**.

They cannot all hold. `detect_hands()` loops one frame at a time, so 32 frames at
15.34 ms is ~0.5 s — but the measured end-to-end `predict()` is **4,650 ms**,
which only ~135 ms/frame explains.

**Resolution taken in our docs:** quote the classifier (~4.2 ms) and the
end-to-end (4,650 ms); claim **no** per-frame PASS. See
`docs/release/LATENCY_RECONCILIATION.md`.

## B-10 — Dev's robustness artifact mislabels a diagnostic as the raw result

| | |
| --- | --- |
| **Severity** | P2 |
| **Module** | `results/robustness_gated.json` |
| **Status** | **DOCUMENTED** — escalated to Dev |

`robustness_gated.json` labels `drop_pct: 79.94` as `"raw (every clip)"`, but
`DROP_HAND_ANALYSIS.md` states 79.94 is the **coords-kept diagnostic**; the
harness corruption (both zeroed) gives **79.70**, delta **5.76**.

The file also reports delta 5.52 for 85.45 − 79.94, which is 5.51.

**Resolution:** our docs quote **5.76 FAIL** as the unconditional result, and
always state the 86%/80% coverage alongside the gated 1.24. See
`docs/release/ROBUSTNESS_RECONCILIATION.md`.

## B-11 — The uncertain branch of the result card had no test

| | |
| --- | --- |
| **Severity** | P2 — coverage hole |
| **Module** | `androidTest/.../ResultCardRenderTest.kt` |
| **Status** | **FIXED** |

`resultCardShowsTheRealEnsembleOutput` drives a real clip whose margin is far
above the acceptance cut, so it **only ever exercised the accepted branch**.

**Fix:** added `lowMarginResultIsShownButLabelledUncertain`, which renders
`margin = 0.02` and asserts the prediction is still shown, is labelled uncertain,
and that the label does not read as a request to re-sign.

**Verified it can fail:** the test goes red when the margin is changed to `0.90`,
so it is a genuine assertion. Instrumentation moved 17 → 18 run, 0 failed.
