# Artifact rights

Every artifact the submission touches, with its provenance, the evidence that
exists for its licence, and whether we may redistribute it.

**Rule applied:** an artifact appearing in this repository or on GitHub is not a
licence grant. A right is only "cleared" where documentary evidence exists in
this repository. Where it does not, the row says so.

Updated 2026-11-11. Supersedes the artifact table in `NOTICE` §2, which listed
only the single-classifier model and predates the four-model ensemble.

---

## 1. Code we wrote

| Artifact | Origin | Licence evidence | Internal use | Redistribution | Status |
| --- | --- | --- | --- | --- | --- |
| `src/`, `app/`, `scripts/`, `evaluation/device/` | EchoGesture team | `LICENSE` (MIT) in repo | yes | yes | **CLEARED** |

---

## 2. Shipped model artifacts

These are the four members the runtime actually loads. They are **not** the
single `landmark_classifier_fp16.tflite` the older NOTICE documented.

| Artifact | Bytes | Origin | Licence evidence | Internal use | Redistribution | Status |
| --- | --: | --- | --- | --- | --- | --- |
| `models/landmark_ens_tcn21_fp16.tflite` | 939,152 | Trained by Dev on ASL Citizen / WLASL-derived features | None in repo | yes | **no** | **NOT CLEARED** |
| `models/landmark_ens_tcnm61_fp16.tflite` | 832,888 | same | None in repo | yes | **no** | **NOT CLEARED** |
| `models/landmark_ens_tcnm62_fp16.tflite` | 832,888 | same | None in repo | yes | **no** | **NOT CLEARED** |
| `models/landmark_ens_rob71_fp16.tflite` | 939,152 | same | None in repo | yes | **no** | **NOT CLEARED** |
| `models/landmark_labels.json` | 1,492 | 100-gloss vocabulary we assembled | None in repo | yes | **no** | Follows the weights |
| `models/hand_landmarker.task` | 7,819,105 | Google MediaPipe | Apache-2.0 per MediaPipe terms; upstream URL recorded in `NOTICE` | yes | attribution required | **CLEARED WITH ATTRIBUTION** |

The four ensemble members derive from ASL Citizen and WLASL, both of which carry
academic / non-commercial restrictions. No written permission covers
redistribution. They are therefore gitignored and excluded from every published
commit and from the public branch.

**Consequence, stated plainly:** the public repository is source-only. It is not
a runnable package without the weights, and the submission must say so.

---

## 3. Datasets

| Artifact | Origin | Licence | Internal use | Redistribution | Status |
| --- | --- | --- | --- | --- | --- |
| ASL Citizen clips | ASL Citizen (2023) | academic, signer-consented for research | yes | **no** | **NOT CLEARED** |
| WLASL clips | WLASL | C-UDA academic / non-commercial | yes | **no** | **NOT CLEARED** |
| Rendered HUD frames (`evaluation/device/demo_final/*.png`) | Our renderer over the above clips | inherits the dataset terms | yes | **no** | **EXCLUDED** — gitignored |

The HUD frames are pictures of real participants. They are ignored by
`.gitignore` and were never committed.

---

## 4. Built artifacts

| Artifact | Contains | Can we distribute it? | Status |
| --- | --- | --- | --- |
| `evaluation/device/release_final2/echogesture_final2_ensemble.zip` (9,007,144 B, SHA-256 `34a0e667…`) | the four members + detector | **no** — bundles weights | private / local |
| `app/android_final/…/app-debug.apk` (75,072,782 B) | same weights as assets | **no** | private / local |

Both are held locally. Neither is on the public branch. `results/results.csv`,
`results/predictions.jsonl` and all markdown are publishable — they contain
numbers, not weights.

---

## 5. What the submission says about this

The honest position, already in the deck and the README:

- The repository link is offered as **source code for review**, not as a
  complete runnable artifact.
- Weights are supplied separately if the organizer requires a runnable package,
  subject to whatever permission Dev can obtain from the data rights holders.
- No licence is claimed that we do not hold.

## 6. What would change a status

Only written permission from the ASL Citizen and WLASL rights holders, or
explicit organizer guidance that the competition's own terms cover the derived
weights. Neither exists in this repository as of 2026-11-11.

This is `WAITING_FOR_PERMISSION`, owned by Dev, and it does not block
submission — it blocks only publishing the weights.