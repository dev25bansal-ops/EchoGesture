# 02 — Regressions I introduced and removed

Recorded because one of them shipped in a build before it was caught.

## Regression A — clearing `sAppInstance` by reflection crashed the app

**What I did.** The camera could not recover from a failed start, because
`ProcessCameraProvider` caches its initialisation Future. I cleared the static
field `sAppInstance` by reflection:

```kotlin
// WRONG — removed
ProcessCameraProvider::class.java
    .getDeclaredField("sAppInstance")
    .apply { isAccessible = true }
    .set(null, null)
```

**What the bytecode actually does.** Verified with `javap -p -c` on
`camera-lifecycle-1.3.4.aar`:

```
getInstance(Context)
  5: getstatic     sAppInstance       <- reads the field
  9: invokespecial getOrCreateCameraXInstance   <- calls it as an INSTANCE method
```

`sAppInstance` is created by `<clinit>` and is **never null**. Setting it to
null does not reset the cache; it makes the very next `getInstance()` throw.

**Reproduction.** Grant the permission externally, then tap "Allow camera
access". Every time:

```
java.lang.NullPointerException: Attempt to invoke direct method
'ListenableFuture ProcessCameraProvider.getOrCreateCameraXInstance(Context)'
on a null object reference
    at ProcessCameraProvider.getInstance(ProcessCameraProvider.java:182)
    at CameraScreen.startCamera(CameraScreen.kt:726)
    at CameraScreen.permissionLauncher$lambda$1(CameraScreen.kt:213)
```

**The correct reset.** The cache is the instance field `mCameraXInitializeFuture`.
The public `shutdown()` nulls exactly the right fields:

```
shutdown() -> mCameraXConfigProvider   = null
              mCameraXInitializeFuture = null
              mCameraXShutdownFuture   = <future>
              mCameraX                 = null
              mContext                 = null
```

`shutdown()` cannot be reached the normal way after a failed init, because
`getInstance()` returns a *transformed* Future whose `.get()` rethrows the same
failure — there is no provider in hand to call it on. The class exposes no other
route to the instance, so the static field is **read** (never written) to obtain
it, and `shutdown()` is called on the result.

**Verification after the fix.** Three paths, all on the emulator, no crash:

| Path | Result |
| --- | --- |
| Grant at the prompt | camera starts |
| Deny at the prompt | card shown, correct |
| Grant externally, tap the card's button | recovers to "Searching for hands" |

## Regression B — my accuracy test under-reported the model

**What I did.** Scored predictions against the filename gloss as plain strings:

```kotlin
if (o.pred.equals(o.truth, ignoreCase = true)) correct++   // WRONG
```

Corpus filenames carry the organiser gloss (`TEACH`, `CHEW 2`, `BABY 2`); the
vocabulary keys carry the dataset's disambiguation suffix (`TEACH1`, `CHEW1`,
`BABY2`). Correct predictions were scored WRONG.

**Effect on the reported figure**, same APK and same clips:

| Scoring | native |
| --- | --- |
| Plain string compare | 25/30 = 0.833 |
| Vocabulary-key resolution | **26/30 = 0.867** |

**Fix.** `vocabKeyFor()` resolves a gloss exactly first, then by unique stem,
and returns `null` when two keys share a stem so an ambiguous gloss fails
loudly. All 30 glosses resolve, each to exactly one key; the shipped vocabulary
has 100 distinct stems and zero collisions.

## Regression C — a measurement bug that reported 0% accuracy

**What I did.** In a throwaway evaluation script I read the prediction as
`out['top5'][0][0]`. `predict()` returns `top5` as a **list of label strings**,
not as pairs, so `[0][0]` yielded the first *character* of the top label.

**Symptom.** The first 60-clip run reported `correct=0 wrong=60`, i.e. 0.0%
accuracy, with predictions like `B`, `C`, `D` against truths `BEE1`, `CALENDAR1`,
`DEAF1`. That pattern — one capital letter per prediction — is the signature of
this bug, not of a broken model.

**Fix.** Read `out['prediction']`. The same clips then scored 54/60.

**Why this is recorded.** A 0% result looked like a catastrophic model failure
and would have justified retraining. It was a bug in my own harness. Any future
0% or near-0% figure must be checked for this before it is believed.