# RECOGNITION_ROOT_CAUSE

**Verdict: NOT ESTABLISHED for the phone. The Android path is exonerated by
measurement; one real defect was found and fixed, but it is not the phone cause.**

## What was measured this session

| Path | Input | n | Correct | Accuracy |
| --- | --- | --- | --- | --- |
| A — Python | labelled clips | 30 | 24 | **0.800** |
| B — Android device | same clips | 30 | 26 | **0.867** |
| C — screen recapture | — | 0 | — | **NOT RUN** (no phone) |
| D — real person on phone | — | 0 | — | **NOT RUN** (no phone) |

## The decisive measurement: A vs B agreement

Same 30 clips through two independent implementations:

```
PYTHON vs ANDROID agreement: 26/30
```

The four disagreements:

| Clip | Python | Android | Python conf | Android score / margin | Presented? |
| --- | --- | --- | --- | --- | --- |
| NOON | BEE1 | NOON1 | 0.268 | 0.438 / 0.143 | REJECTED |
| SHAVE | RAZOR2 | PARTY1 | 0.228 | 0.062 / 0.002 | REJECTED |
| MAPLE | 8HOUR | MAPLE | 0.094 | 0.118 / 0.024 | REJECTED |
| CHEEK | ANYONE | COMB2 | 0.125 | — | REJECTED |

**All four divergences are low-confidence and all four are declined by the
acceptance policy.** Not one reached the user as a recognised sign.

That is the discriminating result the brief asked for:

> If both agree but predict the wrong sign, investigate model generalization and
> incorrect or unsupported input signs.

Android and Python agree on 26 of 30, and where they disagree both are unsure.
**There is no Android preprocessing defect to fix.**

## The defect that WAS found: unverified vocabulary substitution

`src/main.py` carried `PINNED_LABELS_SHA256 = ""` — an empty pin whose comment
named a `_load_pinned_labels` function that does not exist. Model weights were
hash-verified; the vocabulary was not.

Reproduction: copy `models/label100.json` over `models/landmark_labels.json` and
load:

```
RESULT: LOADED WITHOUT COMPLAINT
  0 -> BOOK
  1 -> DRINK
  2 -> COMPUTER
```

Every prediction silently renamed, every shape check passing, because both files
are 100 entries with contiguous indices.

**This is a plausible mechanism for "plausible-looking wrong words"** — but I
have **no evidence it occurred on the phone**. The APK's asset is hash-pinned in
`build.gradle.kts`, so a shipped APK cannot carry the wrong list. Treat this as
a hazard closed, not as the cause of the user's report.

After the fix the same reproduction is refused:

```
RESULT: REFUSED -> ValueError: Vocabulary 'landmark_labels.json' hash mismatch:
741dd6a632c4… != e78f5028e97e…. The class ORDER defines what every model index
means; a different file of the same length silently renames every prediction.
```

## What remains unexplained

The user reports that most real-phone attempts are wrong. Against that:

- the ensemble is 86.7% on labelled clips through the Android path,
- Python agrees with Android on 26/30,
- accepted readings are 95.7% precise.

None of those predict "most signs wrong". **The failure therefore lives in the
one place no measurement here reaches: a person signing into a phone camera.**

Candidate explanations that remain open, and what would settle each:

| Candidate | Settled by |
| --- | --- |
| Detector finds too few/one hand on a real signer | `hand_frames` in phone diagnostics |
| Analyser too slow, frames dropped | `fps=` / `dropped=` in phone diagnostics |
| Auto-segmenter picks the wrong interval | `selected_duration_ms` vs actual sign length |
| Real-signer domain shift | Path D vs Path B on the same gloss |

**No code change was made for any of these**, because none is demonstrated. The
brief forbids exactly that.

## A measurement error I made and corrected

My first PATH A run reported 11/30 = 36.7%. That was **my scoring bug**, not the
model: it compared `WHATFOR1` against the filename gloss `WHAT FOR` as strings.
Re-scored with vocabulary-key resolution: 24/30 = 80.0%.

I had already fixed this bug once in the Android harness and reintroduced it in a
fresh Python script. Both are now correct.
