# Instrumentation tests on real ARM hardware — finally run

Date: 2026-11-19
Device: Vivo V2404, Android 16, arm64-v8a, connected over WiFi (mDNS)
Command:

```bash
JAVA_HOME=/d/jdk17/jdk-17.0.21+9 ANDROID_HOME=/d/android-sdk \
ANDROID_SERIAL=adb-10BF3T19GJ0069L-kupA38._adb-tls-connect._tcp \
./gradlew :app:connectedDebugAndroidTest --offline
```

Every prior attempt failed because the x86_64 emulator cannot load the
MediaPipe AAR (it ships `arm64-v8a`, `armeabi-v7a` and `x86` JNI libraries and
**no `x86_64` build** — verified by unzipping `tasks-vision-0.10.14.aar`). This
was the reason "25 tests need running on real ARM hardware" stayed open.

## First full run: 25 tests, 0 failed, 5 skipped

```
Starting 25 tests on V2404 - 16
V2404 - 16 Tests 26/25 completed. (5 skipped) (0 failed)
Finished 30 tests on V2404 - 16
BUILD SUCCESSFUL in 12m 3s
```

The 5 skips were all `ClipReplayAccuracyTest` / `DirectClipParityTest`, and
they skip for a specific, honest reason: the corpus is not bundled, so
`clipDir()` finds no `.mp4` and `assumeTrue` marks the run INCOMPLETE rather
than passing it.

## The replay tests, with a corpus present

Pushing 40 corpus clips to the app's external files dir un-skipped them. They
then **failed**, and the failure was worth having:

```
ClipReplayAccuracyTest > syntheticTimestampClockLosesHandsComparedToRealTimestamps
ClipReplayAccuracyTest > videoModeAgreesWithImageModeOnTheSameClips
  com.google.mediapipe.framework.MediaPipeException: failed precondition:
  The received packets having a smaller timestamp than the processed timestamp.
```

**This was a defect in the test, not the app.** Both cases passed raw container
PTS to `detectForVideo`. `sampleFrames()` keeps the frame nearest each target
time, so two targets can resolve to the same frame and yield an equal PTS, and
VIDEO mode rejects any timestamp that fails to strictly advance. One of the two
carried the comment "monotonically increasing, exactly as the analyser supplies
them" — which is simply false. The app never hits this because its analyser
clock is `fi * 33`.

Fixed in `ClipReplayAccuracyTest.kt` by clamping each timestamp to at least
`previous + 1`, which preserves real spacing wherever it is valid and nudges
only the degenerate ties. Commit `c86965e` in the app worktree.

`nativeFramesDetectAtLeastAsManyHandsAsASquareResample` **passed** on hardware,
which is a real result: native frames detect at least as many hands as a
256-square resample on the actual device, matching the Python finding that the
square resample costs ~8.5 points.

### The second defect: a use-after-free that took a native crash

With the PTS clamp in place the timestamp exception disappeared and the test
then crashed the process instead:

```
F libc: Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x18
       in tid 14505 (pool-7-thread-1), pid 6751 (esture.finalapp)
```

`runTwoClocks` walks the same frame list **twice** — once with the synthetic
33 ms clock, once with real PTS — and every `Decoded` shares the same source
bitmaps. The cleanup under the inner loop recycles `df.bitmap`, i.e. the
**source**, so the second pass called `df.bitmap.copy()` on freed pixels and the
native bitmap code segfaulted. Fixed by not recycling the intermediate copy at
all (one ARGB_8888 frame at a time, reclaimed by GC). Commit `a7a18d0`.

The other recycle sites were audited: the conditional one in the decoder is
correct (it keeps the scaled bitmap and frees only the distinct intermediate),
and the single-pass tests recycle sources safely because they walk the list once.

**Both defects had survived because these tests had never executed.** A skip
looks like a pass in a summary line, and neither bug was reachable until a real
device and a real corpus were both present.

### Trap: reinstalling the app wipes the pushed corpus

Android clears the app's external files dir on reinstall. Each
`connectedDebugAndroidTest` run reinstalls, so **clips pushed before a gradle run
are gone by the time the tests execute** — the run then silently skips
everything and reports success. This bit twice during this session. The working
order is:

```bash
./gradlew :app:installDebug :app:installDebugAndroidTest   # install FIRST
adb push ... /sdcard/Android/data/org.echogesture.finalapp/files/   # THEN clips
adb shell am instrument -w -e class ...ClipReplayAccuracyTest \
  org.echogesture.finalapp.test/androidx.test.runner.AndroidJUnitRunner
```

Running via `am instrument` rather than gradle also avoids the reinstall
entirely, which is the more reliable form.

## What is still not established

**`androidPipelineAccuratelyClassifiesLabelledClips` has NOT completed**, after
three distinct attempts and three distinct failure modes — which is itself the
useful result:

| attempt | behaviour |
|---|---|
| unbounded, `-e class` via gradle | process crash, no output |
| unbounded, single method via `am instrument` | process crash, no output |
| bounded to 12 clips | **hung — killed at the 30-minute timeout, having printed zero `CLIP_ACC` lines** |

The progression matters. Bounding the clip count removed the memory crash but
did not produce a result: the test now simply never finishes and never prints.
Zero output in 30 minutes means it hangs inside the **first** clip, before the
first `println`, not that it is slowly working through many clips.

What is known, and what is not:

- The class **is** in the installed test APK (dex scan), so `-e class` is not
  the problem. An earlier `unzip -l | grep` check suggesting otherwise was
  wrong — test classes live inside `.dex`, not as file entries.
- The first SIGSEGV was a bitmap use-after-free and **is fixed**; this is not
  that.
- Reinstalling wipes the pushed corpus (three occurrences), so every run needs
  clips pushed *after* install — now done deliberately.
- The remaining suspect is `sampleFrames()` / `MediaExtractor` on these
  specific clips under instrumentation, since the *unrelated* tests that also
  decode video (`CameraFeaturePathTest`) pass fine.

**Consequence, stated plainly: there is still no on-device accuracy figure under
the GPU delegate.** The 85.45% is a Python / desktop-CPU-MediaPipe measurement.
GPU landmark parity is **unmeasured, not disproven**.

This does not block submission. It is worth knowing that the claim "the app
recognises signs" rests on (a) the same nine TFLite members and the same
label map verified present in the installed APK, (b) a documented
feature-parity test between the Kotlin and Python feature transforms, and
(c) the app visibly producing correct predictions on device — the diagnostics
file recorded `pred=SHAVE1 score=0.6758` from a real capture. What is missing
is a *rate over labelled clips*, not evidence that it works at all.

## Reproduce

```bash
D=adb-10BF3T19GJ0069L-kupA38._adb-tls-connect._tcp
# install FIRST, then push - reinstalling clears the corpus
./gradlew :app:installDebug :app:installDebugAndroidTest
adb -s $D shell mkdir -p /sdcard/Android/data/org.echogesture.finalapp/files
mapfile -t F < .scratch/replay_clips_lf.txt      # 40 clips, LF endings
adb -s $D push "${F[@]}" /sdcard/Android/data/org.echogesture.finalapp/files/
# run without gradle so nothing reinstalls
adb -s $D shell am instrument -w \
  -e class org.echogesture.finalapp.ClipReplayAccuracyTest#androidPipelineAccuratelyClassifiesLabelledClips \
  org.echogesture.finalapp.test/androidx.test.runner.AndroidJUnitRunner
```
