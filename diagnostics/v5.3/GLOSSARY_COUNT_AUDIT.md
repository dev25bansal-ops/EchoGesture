# GLOSSARY_COUNT_AUDIT

**Verdict: the model recognises exactly 100 classes. There is no 112-class model.**

## Evidence

### The nine TFLite members output 100 logits each

Read directly from every shipped member with an interpreter:

```
landmark_ens_mirror_handdrop_tcn_z_s0_fp16.tflite   out=[1, 100] OK
landmark_ens_mirror_handdrop_tcn_z_s13_fp16.tflite  out=[1, 100] OK
landmark_ens_mirror_handdrop_tcn_z_s1_fp16.tflite   out=[1, 100] OK
landmark_ens_mirror_handdrop_tcn_z_s2_fp16.tflite   out=[1, 100] OK
landmark_ens_mirror_handdrop_tcn_z_s7_fp16.tflite   out=[1, 100] OK
landmark_ens_rob71_fp16.tflite                      out=[1, 100] OK
landmark_ens_tcn21_fp16.tflite                      out=[1, 100] OK
landmark_ens_tcnm61_fp16.tflite                     out=[1, 100] OK
landmark_ens_tcnm62_fp16.tflite                     out=[1, 100] OK
members=9  all_100=True
```

The output tensor is `[1, 100]`. A 112-class model would emit 112 logits.
**The architecture cannot answer with 112 classes.**

### Both branches carry the same 100-entry vocabulary

| Source | Path | Entries | SHA-256 |
| --- | --- | --- | --- |
| Release branch | `models/landmark_labels.json` | 100 | `e78f5028…` |
| Dev branch | same path, `origin/hackathon/dev-model` | 100 | `e78f5028…` |

The two files are **byte-identical**. Dev is not shipping a different
vocabulary.

### The APK's own asset is 100 entries

`app/android_final/app/src/main/assets/landmark_labels.json` → 100.
`ModelContract.EXPECTED_CLASSES = 100`. The glossary renders the list it is
handed from the verified model, so it shows 100.

### No 112 anywhere in either branch

Searched all `.json`, `.md`, `.csv`, `.kt`, `.py` on both branches. Every "112"
match is a hash fragment, a millisecond figure, a frame count, or a byte length —
none is a class count. Dev's documentation contains no such claim.

## What "112" most likely is

**Cannot be confirmed from the repository.** Nothing in either branch supports it.
The nearest candidates, none of which is a 112-class model:

| Candidate | Reality |
| --- | --- |
| 112 unique source-dataset glosses | Not present in any manifest here |
| 100 classes + 12 aliases | **No alias file exists**; no code reads one |
| 112 glossary display entries | Glossary reads the 100 model labels directly |
| 112 candidates, 100 trained | Possible historically, unverifiable here |

**Honest conclusion: the 112 figure has no supporting artifact in this
repository.** If Dev is stating it verbally or in a document outside the repo,
that document is the place to correct it. I will not invent an explanation.

## Consequence for the app

Class count stays **100**. No vocabulary entries were added, no aliases were
invented, and no model was changed.

## The real vocabulary hazard that WAS found

Two different 100-entry vocabularies exist:

| File | Content | Example index 0 |
| --- | --- | --- |
| `models/landmark_labels.json` | ASL Citizen glosses | `DOG1` |
| `models/label100.json` | Legacy generic words | `book` |

Both have 100 entries and contiguous indices. They share exactly **one** word
(`APPLE`); 99 of 100 differ. Substituting one for the other renames every
prediction while leaving every shape check passing.

**The Android build already pins the vocabulary hash** (`build.gradle.kts`,
`textAssets`). **The Python evaluator did not** — `PINNED_LABELS_SHA256` was an
empty string whose comment referred to a `_load_pinned_labels` function that does
not exist. Reproduced: copying the legacy file over the canonical one loaded
without complaint and made index 0 read `BOOK`.

Fixed, and covered by
`evaluation/device/v5_accuracy/tests/test_vocabulary_identity.py`.
