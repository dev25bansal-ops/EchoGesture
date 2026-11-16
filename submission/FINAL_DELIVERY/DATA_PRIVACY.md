# Data and privacy posture

Date 2026-11-16.

This is the organizer-facing statement. The full internal audit, including the
inventory of pre-existing exposures, is deliberately **not** in this bundle.

---

## What is in this submission

| | |
| --- | --- |
| Model weights | Nine FP16 TFLite members plus a hand detector |
| Source | Python evaluator, Kotlin Android app, evaluation harnesses |
| Results | Aggregate accuracy figures, per-class and per-signer summaries |
| Predictions | Per-clip predictions, **without any participant identifier** |

## What is not, and has never been

- No video or images from the ASL Citizen corpus
- No participant names, contact details, or demographic records
- No participant-to-pseudonym mapping table
- No raw preprocessed feature caches for any corpus clip
- No network access at inference; `INTERNET` is absent from the Android manifest,
  and `CAMERA` is the only permission requested

## Predictions file

`evaluation/asl_citizen_predictions.csv` carries 1,705 rows of
`clip_id, true_label, predicted_label, confidence` style fields. The participant
identifier column has been **removed**, not pseudonymised — the rows cannot be
traced to a signer from this repository alone.

## Per-signer tables in the results documents

Where a result is broken down per signer, signers appear as short `S…` labels
derived by a one-way function with a fixed project salt. The same salt produces
the same label every time, which keeps documents comparable with each other.
The mapping back to a participant is not recorded in this repository.

## Known limitation you should be aware of

The corpus corpus-accuracy figures come from the ASL Citizen research dataset and
are used under its research licence. The model weights were trained on that data.
**Redistribution permission for the trained weights has not been confirmed** —
see the checklist. If the organizers require that confirmation, it is outstanding
and it is a human action, not a technical one.

## Scope of the accuracy claims

Every accuracy figure in this package is:

- measured on a laptop CPU, not a phone
- measured on a corpus test split, not on live signing
- reported with its population and clip count attached

No accuracy number here is a phone measurement. There is no phone measurement.