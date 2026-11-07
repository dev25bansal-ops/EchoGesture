# Dev and Abhijeet handoff

## Branches and reviews

Abhijeet owns integration into `hackathon/two-person-sprint`. Work and push
from your own branch, then request the other teammate's review. See the
[GitHub workflow](GITHUB.md) for publishing the plan, branch commands, PR
targets and recovery steps.

| Files or proposed area | Primary editor | Boundary |
| --- | --- | --- |
| `src/features.py`, `models/`, `scripts/ml/` | Dev | Model interface, features, training and export |
| `src/main.py`, `app/`, `scripts/release/` | Abhijeet | Evaluator API, phone app and packaging |
| `evaluation/model/` | Dev | Data checks, unseen-signer model comparison and reproducible corruption fixtures |
| `evaluation/device/`, `results/` | Abhijeet | Full-path timings, end-to-end robustness and final submission outputs |
| `src/requirements.txt`, `src/README.md`, sprint docs | Abhijeet coordinates | Dev hands off dependency, model and reproduction changes for the same PR or a follow-up |

The proposed paths do not exist until the owner adds them. Agree on any
change that crosses these boundaries before editing. For interface changes,
merge Dev's model-side PR first, then have Abhijeet sync and integrate against
the exact artifact SHA. Abhijeet regenerates final results after the model
is pinned.

## Abhijeet → Dev · Evaluator inputs

Provide the starter-kit version, vocabulary and exact label spellings, allowed
sources, manifest shape, and unresolved organizer questions. For benchmarks,
include the model SHA, runtime and delegate, device, clip set and sample count,
end-to-end p95 and stage breakdown. Send any parity failures with a reproducible
fixture or a local clip reference both teammates can access.

## Dev → Abhijeet · Model bundle

| Field | Include |
| --- | --- |
| Candidate | Branch, commit SHA, artifact SHA-256 and bytes |
| Artifacts | TFLite file, exact ordered label map and versioned feature config |
| Input contract | Shape, dtype and scale/zero point, number of frames, sampling, coordinate convention, masks and normalization |
| Output contract | Class axis and whether values are logits or probabilities, plus dequantization parameters where applicable |
| Reproduction | Commands for feature extraction, training and conversion, plus one permitted smoke fixture and expected feature or inference shape |
| Evidence | Data version, split, clip and signer counts, float and INT8 validation top-1, known failure cases |
| Data scope | Sources used and what can be shared in the release |

A handoff is ready when Abhijeet can run the candidate from the documented
artifacts and verify feature and label parity. A change to label order or
features requires a new config version and a new benchmark of the actual
candidate.

## Daily update template

```text
Date (IST):
Owner and task ID:
Branch and commit SHA:
What changed:
Command(s) and result:
Dataset/split or device/runtime:
Artifact or config SHA:
Blocker or decision:
Next action and owner:
```

## Final release sign-off

- Dev verifies model bytes and format, vocabulary and split integrity, and
  reproducibility from the release bundle.
- Abhijeet verifies the actual evaluator API, manifest coverage, device and
  robustness logs, demo and deck, and final form requirements.
- Both agree which commit and artifact SHAs the submitted numbers describe.
  Abhijeet submits the package and keeps the receipt.
