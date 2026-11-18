# Android instrumentation tests — first real run

**Date** 18 Nov 2026. Replaces the "25 instrumentation tests, not runnable" line with an
actual measurement.

## What was run

```
export JAVA_HOME=/d/jdk17/jdk-17.0.21+9 ANDROID_HOME=/d/android-sdk
./gradlew connectedDebugAndroidTest
```

Emulator: AVD `echogesture34`, Android 14 (API 34), **x86_64**, booted with
`-no-snapshot-load`. The AVD already existed; nothing was configured.

## Result

```
tests=25  passed=10  failed=10  skipped=5  errors=0
```

## Why the ten fail, established not guessed

Every one fails with the same cause:

```
java.lang.UnsatisfiedLinkError: dlopen failed:
  library "libmediapipe_tasks_vision_jni.so" not found
at com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.<clinit>
```

Checked the APK rather than assuming packaging was at fault:

```
$ python -c "<list lib/<abi>/ entries in app-debug.apk>"
arm64-v8a    3 libs | mediapipe: True
armeabi-v7a  3 libs | mediapipe: True
x86          3 libs | mediapipe: True
x86_64       2 libs | mediapipe: False     <-- no libmediapipe_tasks_vision_jni.so
```

Then checked the dependency itself:

```
$ python -c "<list jni/ entries in tasks-vision-0.10.14.aar>"
AAR ABIs: {'arm64-v8a': 1, 'armeabi-v7a': 1, 'x86': 1}
```

**The AAR ships no x86_64 build.** So this is an upstream limitation of
`com.google.mediapipe:tasks-vision:0.10.14`, not a defect in the Gradle config, not a
packaging mistake, and not something this repository can fix.

The emulator is x86_64, so the native library cannot load and every test that touches
`HandLandmarker` dies in static initialisation. The five skips and the ten passes are the
tests that never reach it.

## What this does and does not mean

**Does not mean the app is broken.** The shipping target is `arm64-v8a`, which has the
library. These ten failures say nothing about ARM behaviour.

**Does mean the tests cannot be validated on an x86_64 emulator.** Anyone trying to
reproduce the Android test suite on a standard x86_64 AVD will see the same ten failures
and no explanation, which is worse than a documented skip. An arm64 system image, or
`abiFilters` restricted to `arm64-v8a` plus an arm64 AVD, would exercise them.

**Status change:** from "25 instrumentation tests, not runnable" to **"25 run, 10 pass,
10 fail on x86_64 because the MediaPipe AAR has no x86_64 binary; untested on ARM."**
That is strictly more honest than the previous line and strictly more useful.