submission_final — final package contents and provenance

`README.md` in this directory is the entry point for a reviewer, and
`FINAL_CHECKLIST.md` is the authoritative state table. This file records where
each artifact came from so nothing here is a claim without a source.

## Verified against the working tree, 2026-11-11

| Artifact | Path | Verified how |
| --- | --- | --- |
| Four ensemble members | `models/landmark_ens_*.tflite` | `shasum -a 256`, compared to `models/landmark_model_meta.json` and `evaluation/device/ensemble_evidence/benchmark_ensemble_evidence.json` |
| Detector | `models/hand_landmarker.task` | `shasum -a 256`, 7,819,105 B |
| Release ZIP | `evaluation/device/release_final2/echogesture_final2_ensemble.zip` | SHA-256 `48a062f2…`, 17 entries, extracts and predicts `WHATFOR1` |
| APK | `app/android_final/app/build/outputs/apk/debug/app-debug.apk` | 75,072,782 B, signed, 4 members embedded |
| Results | `results/results.csv` | 13 rows, each bound to model/features/labels SHAs in its notes column |
| Deck | `docs/submission/EchoGesture_Submission.pptx` | 11 slides |

## Size arithmetic, done explicitly

The one number that must not be wrong is the model budget.

```
ens_tcn21   939,152 B
ens_tcnm61  832,888 B
ens_tcnm62  832,888 B
ens_rob71   939,152 B
            -----------
ensemble  3,544,080 B   = 3.54 MB

detector  7,819,105 B   = 7.82 MB

total    11,363,185 B   = 11.36 MB decimal
```

11.37 MB, inclusive of the detector. Under a 20 MB limit whether or not the
detector is counted. This is stated as bytes, not rounded, because a previous
draft of these documents published 3,543,080 B — off by 1,000 — and that kind of
error is fatal in a size-gated submission.

## What is deliberately absent

| Absent | Why |
| --- | --- |
| `*.tflite`, `*.task` weights | no confirmed redistribution permission; gitignored |
| APK binary in the public branch | 75 MB; distributed locally, not in Git |
| Release ZIP in the public branch | contains weights |
| Raw clips and feature caches | ASL Citizen / WLASL derived; never uploaded |
| Organizer manifest predictions | no organizer manifest was supplied; inventing clip IDs would be fabrication |

## Reproducing any number here

```bash
# model identities
shasum -a 256 models/landmark_ens_*_fp16.tflite models/hand_landmarker.task

# full test suite (each file in its own process)
python3 -I scripts/run_all_tests.py --quiet

# recompute accuracy from frozen predictions, both denominators
python3 -I evaluation/device/ensemble_evidence/recompute.py

# release bundle integrity
shasum -a 256 evaluation/device/release_final2/echogesture_final2_ensemble.zip
```

Accuracy and latency were measured on a macOS arm64 laptop with
`ai-edge-litert 2.3.0` on XNNPACK CPU. Neither is target-hardware measurement,
and the submission says so.