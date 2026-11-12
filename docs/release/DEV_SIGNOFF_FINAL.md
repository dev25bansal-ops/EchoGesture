# Dev sign-off request — final

**To:** Dev (`dev25bansal-ops`)
**From:** Abhijeet
**PR:** https://github.com/dev25bansal-ops/EchoGesture/pull/1
**Published head:** `bd852764631265b0e754c997e19e4babae4eb298`
**Deadline:** 18 Nov 2026, 23:59 (timezone unconfirmed)
**Updated:** 2026-11-12

Nothing below asks for retraining. This is freeze-and-confirm.

**Item 5 is the urgent one.** It is a live rights problem, not a formality.

---

## 1. Confirm the four member hashes

| Member | SHA-256 |
| --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc86090470b82b20b510746a946220322c25ec84cb3706a538c73dc22a` |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8eb6c2c4d8b501f0ce1a36db338365d17fddabc63528a7941b1390f63` |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85e032bee3af444fb1538ff0bf32fd38fa2b01ce01a2f25c3a428d20b3` |
| `landmark_ens_rob71_fp16.tflite` | `08417e72da59150ac9adf749e6324c30fb001d4a401394e32f125d3276d1ab21` |

3,544,080 B total; 11,363,185 B with the detector.

## 2. Confirm the feature contract

`[1, 24, 255]` float32, `STRICT_RESAMPLE=False`. Please confirm this matches
training — if training used `STRICT_RESAMPLE=True`, every published metric is
invalid.

## 3. Confirm the evaluation denominator

909/1249 scored = 72.78% top-1; six further eligible clips produced no hand and
are excluded from the scored denominator but disclosed. Confirm 1,249 is the
intended scored count and not a filter artifact.

## 4. Confirm runtime/packaging approval

The release bundle now ships `src/parallel_tracker.py` (your tracker, moved into
`src/` so `ECHOGESTURE_TRACKER=parallel` works after extraction). Singleton and
parallel paths produce identical top-1 and top-5 on 12/12 clips. No model byte
changed. Confirm you are content with that move.

## 5. Redistribution permission — the urgent item

The repository is **public**, and `hackathon/dev-model` exposes:

- all four ensemble members, byte-identical to the shipped artifacts
- `hand_landmarker.task` (Apache-2.0 upstream, fine to redistribute with attribution)
- `landmark_classifier_int8.tflite`
- `evaluation/asl_citizen_predictions.csv` — 1,705 rows with **11 raw participant
  IDs** (`P15`, `P17`, `P18`, `P22`, `P35`, `P42`, `P47`, `P48`, `P49`, `P6`, `P9`)

I earlier reported that no ensemble weights were published. That was wrong — I
checked one branch instead of the repository.

Full inventory: `docs/release/PUBLIC_WEIGHT_EXPOSURE.md`.

**What I need from you, as repository owner:**

1. Do you hold written permission to redistribute these weights? If yes, please
   supply it. If no, say so plainly so the submission states it.
2. Do you agree to making the repository **private** and publishing a
   source-only public repository instead? This is the only option that actually
   removes public access to already-committed blobs.
3. If you prefer to leave it public, confirm that explicitly, and I will record
   the exposure as disclosed rather than fixed.

I have not changed visibility, deleted a branch, or rewritten history. All three
need your decision.

## 6. INT8 fallback feasibility

If the organizers confirm INT8 is mandatory, is your INT8 candidate usable, what
is its accuracy, and would you approve shipping it? I will not substitute a
lower-accuracy model without your sign-off.

## 7. Final review

A review on PR #1 — even "LGTM with the licensing caveat" — is what unblocks the
merge gate.

---

## What I need back, minimum

1. **Item 5** — the rights answer (either direction is fine; silence is not).
2. **Item 2** — `STRICT_RESAMPLE` confirmation.
3. **Item 1** — hashes confirmed against what you intended to ship.
4. **Item 7** — a PR review.

Items 3, 4 and 6 are confirmations. If any is wrong, say so now rather than on
the deadline.