# Team workflow — two-person sprint

EchoGesture is built by two people on one repository. This file is the
authoritative summary; the per-owner detail lives in `docs/hackathon/`.

## Ownership

| Area | Owner | Files |
| --- | --- | --- |
| Dataset, splits, features, training, export, quantization | **Dev** | `src/features.py`, `src/landmark_frontend.py`, `scripts/ml/`, `models/*.tflite`, `models/landmark_model_meta.json` |
| Evaluator API, runtime, benchmarks, Android, demo, release | **Abhijeet** | `src/main.py`, `app/`, `evaluation/device/`, `scripts/release/`, `results/`, `docs/release/` |

Shared file to coordinate before editing: `src/requirements.txt`, `README.md`.

## Branches

| Branch | Purpose |
| --- | --- |
| `main` | Stable, reviewed, integrated. No direct pushes during active collaboration. |
| `hackathon/abhijeet-integration` | Runtime, evaluator, Android, demo, release, evidence |
| `hackathon/dev-model` | Data, features, training, model export, quantization |
| `hackathon/two-person-sprint` | Shared integration target for PRs |

Each developer pushes their own branch and opens a PR against
`hackathon/two-person-sprint`. Merge only after review.

## Stable interfaces (do not change unilaterally)

```text
Model input:   [1, 24, 255] float32
Model output:  [1, 100] float32 logits (softmax in src/main.py)
Vocabulary:    100 ordered glosses in models/landmark_labels.json
Feature semantics: Dev-owned
predict(video_path) -> dict:       Abhijeet-owned
```

A change to label order or feature layout requires a new `landmark_model_meta.json`
version and a re-run of the validation evidence. Abhijeet re-verifies parity
independently after every model handoff.

## Model handoff checklist (Dev → Abhijeet)

- Commit SHA and artifact SHA-256
- Ordered label map and feature-config version
- Input/output contract and quantization type
- Training provenance and validation split
- Validation top-1 (and held-out test top-1 if available)
- Conversion/export commands

## Runtime handoff checklist (Abhijeet → Dev)

- Runtime commit SHA and the model SHA it was verified against
- Any change to `predict()` behavior
- Benchmark configuration and measured numbers
- Known limitations

## Licensing

Model weights are **not redistributable** (ASL Citizen / WLASL terms). See
[`NOTICE`](../NOTICE) and [`models/README.md`](../models/README.md). Never
`git add` a `.tflite`/`.task` file; `.gitignore` blocks them by default.