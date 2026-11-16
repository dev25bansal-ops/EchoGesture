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

**Six findings. Zero app crashes. Zero broken buttons. No P0/P1 defects found.**

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
