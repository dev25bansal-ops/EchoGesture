# FINAL_BLOCKERS

Listed most severe first. Nothing here is hidden behind a softer word.

## 1. No physical phone was ever attached — BLOCKING

```
adb devices -l  ->  emulator-5554 only
system_profiler ->  no USB Android device
```

Consequences, stated plainly:

- **Live sign recognition is unverified.** Not "partially verified". Unverified.
- The auto-completion thresholds have never met a real signer's motion.
- Gate E of the brief is **BLOCKED**, not passed.
- Every accuracy number in this package is clip replay on an emulator.

## 2. The user's reported failure may still be present

The user reports "most signs are wrong" on their phone. Nothing in this session
observed that on a phone, so nothing in this session can claim it is fixed.

What *is* established: on 30 labelled clips the pipeline is 86.7% correct and
95.7% precise on what it presents, and Android/Python tensors are bit-identical.
Those are consistent with each other and do not predict "most signs wrong".

So the cause of the phone failure remains **NOT ESTABLISHED**.

## 3. The result-UI fix changes what the app claims, not what it knows

An uncertain reading now shows "Sign not recognised reliably" instead of the
guess. That is a correctness fix for honesty.

**It does not improve recognition accuracy, and no claim here says it does.**
If the phone is producing poor features, this fix makes the app quieter, not
better.

## 4. Model error rate on hard instances

6 of 43 errors on the 345-clip set are high-confidence wrong answers, e.g.
`8HOUR -> MAPLE` at 0.863 margin 0.791. Both platforms make the identical
mistake on the identical clip. This is a model limitation that Android code
cannot fix.

## 5. The release branch cannot be rebuilt from a fresh clone

Model weights, `hand_landmarker.task`, `landmark_labels.json` and
`landmark_model_meta.json` are all gitignored. The branch reproduces **source**
but not the **artifact**, because the repository is public and redistribution
rights are unresolved. This is the owner's decision, not one to make silently.

## 6. Auto-stop is proven on synthetic frames only

15 JVM and 9 instrumentation tests drive the real `HandTracker`, but with
synthetic landmark sequences. Whether a real person's motion crosses the onset
and end thresholds in the intended way is untested.

## What would resolve most of this in ~20 minutes with a phone

1. Install the APK.
2. Settings → enable **Capture diagnostics**.
3. Tap Start Recognition, perform five known signs, three times each.
4. `adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt`

The `hand_frames` and `limit=` fields separate detection failure from throughput
failure from model failure in a single pass.
