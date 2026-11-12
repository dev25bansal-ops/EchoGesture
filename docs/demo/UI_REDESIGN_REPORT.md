# EchoGesture V2 — Android UI/UX redesign

Commit range: `c7368d5` (redesign) through `5060ead`, on
`hackathon/abhijeet-integration`. APK built from `5d90063` + `5060ead`
(evidence only); the APK binary is unchanged by the last commit.

## 1. Before → after

Before, the app was a single black screen with a preview and a button. There
was no navigation, no glossary, no history, no settings, no result card, no
permission state, no error state and no help. A user could capture and see a
label; everything else was absent.

After, it is a four-destination product: Camera (capture, six live states,
result card), Glossary (all 100 real output classes, searchable, with detail
sheet and Speak/Copy actions), History (with copy and clear), and Settings
(camera choice, guidance, haptics, caption size, speech, history persistence,
model provenance, privacy, limitations). The recognition core is untouched.

## 2. Screen architecture

Single-activity, four-destination shell in [MainActivity.kt](app/android_final/app/src/main/java/org/echogesture/finalapp/MainActivity.kt).
Views are swapped, not destroyed, so the four TFLite interpreters and the
MediaPipe HandLandmarker are constructed once and survive tab switching. This is
deliberate: Compose re-composition would not have made the app look better and
would have risked reloading the 3.6 MB of weights on every tab change.

| Area | Class | Layout |
| --- | --- | --- |
| Shell | `MainActivity` | `activity_main.xml`, `menu/bottom_nav.xml` |
| Camera | `CameraScreen` | `screen_camera.xml`, `sheet_help.xml` |
| Glossary | `GlossaryScreen`, `GlossaryAdapter` | `screen_glossary.xml`, `row_gloss.xml`, `sheet_gloss.xml` |
| History | `HistoryScreen`, `HistoryAdapter` | `screen_history.xml`, `row_history.xml` |
| Settings | `SettingsScreen` | `screen_settings.xml` |

## 3. Working functions

Every control below performs a real operation. There are no dummy clicks and no
fake toggles.

| Control | What it actually does |
| --- | --- |
| **Recognise sign** | Clears the frame ring, collects a fresh 32-frame window (32→24 resample preserved), runs the four-model ensemble, renders the result |
| **Switch camera** | Rebinds CameraX to the other lens; hidden when only one exists |
| **Torch** | Reads `TorchState` and toggles the flash unit; hidden when `hasFlashUnit()` is false |
| **Copy** (result, history row, gloss sheet) | Writes the label to `ClipboardManager`, toasts "Copied" |
| **Speak** (result, gloss sheet) | Android `TextToSpeech`, device voice packs only; reports "No offline voice available" if absent |
| **Recognise again** | Runs a fresh capture, not a replay |
| **Clear** | Dismisses the result card and returns to WAITING |
| **Retry** (error card) | Restarts the camera after a tracker/inference failure |
| **Allow camera access** | Launches the system runtime permission request |
| **How to use EchoGesture** | Opens the five-step help sheet |
| **Glossary row** | Opens the detail sheet: label, "One of the 100 output classes", Speak, Copy |
| **Glossary search** | Filters all 100 labels live; shows a no-match state; clear button appears only with text |
| **History row copy** | Same clipboard path as the result card |
| **Clear history** | Confirmation dialog with the actual count, then clears persistence |
| **Settings: next launch camera** | Persisted in `SharedPreferences`, read by the camera on start |
| **Settings: framing guide** | Toggles the guide overlay and guidance pill |
| **Settings: haptics** | Gates every `performHapticFeedback` call |
| **Settings: result text size** | Small/Medium/Large → 28/36/46sp on the recognised label |
| **Settings: speak results** | When on, speaks the label automatically after inference |
| **Settings: keep history** | On = `SharedPreferences`; off = memory only, cleared on close |
| **Settings: clear saved history** | Deletes the pref-backed entries only, not session memory |

### Not implemented, and not faked

- **Gallery import** — no picker exists. The app stores no images, so a gallery
  path would need a second decode-and-track pipeline; it was not attempted.
- **Favourites** — no star control exists on any screen. The `ic_star` drawable
  is present but unreferenced by any layout.
- **Continuous translation** — out of scope by design; the app recognises one
  isolated sign at a time and says so in Settings and on the result card.

## 4. Design system

Semantic tokens in [colors.xml](app/android_final/app/src/main/res/values/colors.xml),
spacing and radii in [dimens.xml](app/android_final/app/src/main/res/values/dimens.xml),
type roles in [types.xml](app/android_final/app/src/main/res/values/types.xml).
No layout contains a raw hex literal.

| Token | Value | Role |
| --- | --- | --- |
| `primary` | `#2563EB` | the one action that advances the task |
| `primary_dark` | `#1D4ED8` | pressed/emphasis |
| `success` / `success_soft` | `#16A34A` / `#DCFCE7` | a real positive result |
| `warning` / `warning_soft` | `#F59E0B` / `#FEF3C7` | actionable guidance |
| `danger` / `danger_soft` | `#DC2626` / `#FEE2E2` | an actual failure |
| `background` / `surface` | `#F8FAFC` / `#F1F5F9` | page / raised surfaces |
| `text_primary` / `text_secondary` | `#0F172A` / `#64748B` | copy hierarchy |
| `border` | `#E2E8F0` | card outlines |

Spacing follows an 8dp rhythm (`space_1`…`space_8` = 4…32dp); card radii are
12/16/20dp; the capture button is 56dp and every other target is at least 48dp.
Type roles are 12 / 14 / 16 / 20 / 24 / 32sp with a 14sp letter-spaced section
label. State is never carried by colour alone — each state also changes text and
icon.

## 5. Screenshots

In [evaluation/device/ui_redesign_evidence/](../device/ui_redesign_evidence/),
with a README recording exactly how each was reached. They were captured on the
build at `974f610`; the two commits since change no rendered pixel.

`01_camera`, `02_permission`, `03_capturing`, `04_processing`, `05_result`,
`06_no_hands`, `07_glossary`, `08_history`, `09_settings`, `10_help`,
`11_gloss_detail`.

## 6. Visual defects found by inspecting the captures, and fixed

1. **Unchecked switch thumb rendered near-black.** Material3 defaults
   `SwitchMaterial`'s unchecked thumb to `colorOnSurface`, which on this light
   theme resolves to `text_primary` (#0F172A) — the off-state looked like a black
   blob. Fixed with explicit `app:thumbTint`/`app:trackTint` (`974f610`).
2. **Gloss sheet said "One of the 100 output classes" hard-coded** and its
   Speak/Copy buttons had no icons. Now derived from the real class count, with
   `app:icon` drawables (`02fe0a0`).
3. **Gloss and help sheet action buttons had invisible icons** — `MaterialButton`
   ignores `android:drawableStart`; the icons only appeared after switching to
   `app:icon` + `app:iconGravity` + `app:iconTint`.
4. **Dead strings implying nonexistent controls**, most notably `action_save`
   ("Saved to history") with no Save button. Removed in `5d90063`.

## 7. Build result

```
BUILD SUCCESSFUL
36 actionable tasks
```

Kotlin compiles with one pre-existing deprecation warning
(`HAPTIC_FEEDBACK_ENABLED`, `CameraScreen.kt:659`).

## 8. Test results

| Suite | Result |
| --- | --- |
| JVM unit tests | **20 passed, 0 failed** — `EnsembleParityTest` 4, `EnsembleRuntimeParityTest` 3, `FeatureTransformTest` 5, `ModelContractTest` 8 |
| Instrumentation, `eg34` (Android 14) | **8 passed, 0 failed, 0 skipped** |
| Python suite in CI | **363 passed, 0 failed, 119 skipped, RESULT: PASS** |

The skipped Python tests are the weight-evidence suites that require the
restricted weights, which are not in the repository by design.

Instrumentation tests, all on-device:

- `TfliteEnsembleInferenceTest.fourRealInterpretersMatchThePythonReference`
- `TfliteEnsembleInferenceTest.attentionToTensorIndexResolutionPerMember`
- `EnsembleInferenceTest.productionClassifierReproducesPythonOnFrozenFeatures`
- `EnsembleInferenceTest.allFourMembersAllocateWithTheFrozenTensorContract`
- `CameraFeaturePathTest.cameraReportsValidRotationAndTrackerAcceptsFrames`
- `CameraFeaturePathTest.missingHandFramesKeepTheirPosition`
- `ResultCardRenderTest.resultCardShowsTheRealEnsembleOutput`
- `ProcessingStateRenderTest.processingStateRendersItsOwnAffordances`

## 9. Parity

Four FP16 interpreters are real and separate. Aggregation is mean of raw logits
then one softmax — no per-model softmax. The 24×255 feature tensor, the
missing-hand zeroing (frames keep their position rather than being dropped) and
the 32→24 resample are unchanged. Parity is asserted against frozen vectors
produced by the Python side, not against values transcribed by hand.

## 10. APK

| | |
| --- | --- |
| Path | `app/android_final/app/build/outputs/apk/debug/app-debug.apk` |
| Size | 77,837,927 bytes |
| SHA-256 | `43f175dcc11701d7be8bd167688143187a4b60506cc74f685da707ba563d411b` |
| Signature | debug cert `2e7e40e24ccff12d99457a4267a377cb5b9ae14c00750ba4b406e8df6be37e2c` |
| Permissions | `CAMERA` + the framework's `DYNAMIC_RECEIVER_NOT_EXPORTED` only. No `INTERNET`. |
| Assets | `hand_landmarker.task`, and the four `landmark_ens_*_fp16.tflite` members |
| Target | minSdk 24, targetSdk 34, versionName 1.0 |

**This debug APK contains the restricted model weights and must not be
published.**

## 11. GitHub / CI

- Branch `hackathon/abhijeet-integration`, pushed.
- Run `43212312518` on `5060ead`: **success**.
- Dev's branch `hackathon/dev-model` untouched. No force-push, no merge.
- Only source, tests, strings, layouts and screenshots were committed.

## 12. Device install status

Installed and launched on emulator `eg34`, camera permission granted, all
eleven screenshots taken from this install. Not yet tested on a physical device.

## 13. Remaining work

- **Physical device run.** Everything here is emulator evidence; the synthetic
  camera cannot exercise real hand tracking.
- **Demo video** of a live recognition.
- **a real end-to-end capture with a hand in frame** — the one claim this build
  cannot evidence on an emulator.
- Dev sign-off on the merged branch; organizer submission.

## 14. What is deliberately not claimed

- No INT8 anywhere; the weights are FP16.
- No continuous or sentence-level translation; one isolated sign at a time.
- No replacement for a human interpreter.
- `05_result.png` is a real ensemble inference over a pinned parity vector, not
  a live-camera recognition.

## 15. Option 3 (Mac camera on the emulator)

Blocked. The emulator can present the Mac camera as a virtual device, but macOS
must first grant camera access to the emulator process, which only the user can
do in System Settings → Privacy & Security → Camera. Until that grant exists,
the emulator continues to synthesise its own scene.