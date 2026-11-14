# V4.4 — Python evaluator fix

**Result: `predict(video_path)` works on the nine-member masked ensemble.**

This was the blocking defect. The Android app ran nine members while the
competition evaluator could not load them.

## What was broken

`src/main.py` still ran the V4.2 single-input path:

| Defect | Line-level symptom |
| --- | --- |
| Positional input | `interp.get_input_details()[0]` returned the **mask**, not features |
| One shared width | `expected_shape = [1, frames, feature_dim]` for all members |
| No mask fed | `_run_ensemble` called `set_tensor(in_idx, ...)` only |
| Mask discarded | `tensor, _ = build_features(...)` threw away the returned mask |
| 381 members unusable | the five z members failed the shape check |
| Not Dev's reference | diverged from `hackathon/dev-model`'s runtime path |

`src/features.py` had a second, quieter defect: `normalise_hands` allocated
`np.zeros((HANDS, NUM_LANDMARKS, 2))` — **hardcoded to 2 coordinates** — so
`use_z=True` silently produced 255-wide features, never 381. The flag existed
and had never run. `feature_dim(use_z)` did not exist at all.

## What was changed

### `src/features.py`

Replaced with Dev's version at `11bbddf`. Verified it is a strict superset of
the production copy — the only differences are the z handling, `feature_dim()`,
and the `STRICT_RESAMPLE` flag (default `False`, which is the behaviour the
shipped checkpoints were trained under; changing it drops the ensemble, per
Dev's own measurement, so it is left at `False`).

Nothing production-specific was lost: `features.py` carried no instrumentation.

### `src/main.py` — surgical, not replaced

Dev's `main.py` is 663 diff lines shorter and lacks the production
instrumentation, so it was **not** copied over. The following were ported into
the production file:

1. **`_members_detail()`** — parses per-member `use_z` from the sidecar. A
   member with no entry is a hard error; width is never inferred from a
   filename.
2. **Name-resolved inputs** — each member's `features` and `mask` tensors are
   found by name and asserted to be different tensors.
3. **Per-member width** — `expected_shape = [1, frames, feature_dim(use_z)]`.
4. **`_MEMBER_USE_Z`** — tracked alongside the existing member list.
5. **`_run_ensemble(tensor, mask)`** — accepts a `{use_z: matrix}` dict so each
   member gets its own width, and feeds the same mask to every member. A member
   that requires a mask and receives none raises rather than running unmasked.
6. **`predict()`** — builds both widths **and** the mask from one landmark
   extraction, after temporal resampling.

Preserved untouched: `_record_stage` timing, `_tracker_config` /
`_get_tracker` parallel tracking, `_init_model` commit-only-after-full-validation
safeguard, `clip_id`, error handling, Python 3.11, CLI, `predict()` signature.

### Hash pinning — added

The evaluator previously pinned **nothing**. A corrupt or substituted binary
loaded silently and produced a plausible wrong label. Added:

- `PINNED_MEMBERS` — all nine per-member SHA-256.
- `PINNED_ENSEMBLE_SHA256` — `995cc8d0…`, in meta order.
- `PINNED_DETECTOR_SHA256`.
- Class-count check against the vocabulary on the landmark path (previously
  only the `s3d` path checked classes).

`ECHOGESTURE_ALLOW_UNPINNED_MODELS=1` opts out for deliberately evaluating a
candidate. Off by default: an unpinned binary is an unverified binary.

## Verification

### Real MP4 through production `predict()`

| Clip | Gloss | Prediction | Confidence | Latency |
| --- | --- | --- | ---: | ---: |
| DOWNSIZE1 | DOWNSIZE1 | **DOWNSIZE1** | 0.915 | 982 ms |
| 8HOUR | 8HOUR | MAPLE | 0.863 | 304 ms |
| SHAVE1 | SHAVE1 | RAZOR2 | 0.228 | 322 ms |
| HOW1 | HOW1 | **HOW1** | 0.972 | 279 ms |
| DOG1 | DOG1 | **DOG1** | 0.592 | 245 ms |

Latency is first-call warm-up plus per-clip; it is a **laptop** measurement.

### Contract validation — all PASS

Exactly five keys; `clip_id` str; `prediction` str; `top5` a list of five;
`top5[0] == prediction`; all labels non-empty and in vocabulary; no duplicates;
confidence in [0,1]; `latency_ms` a positive float.

### Deliberate failure cases — all now fail loudly

| Case | Result |
| --- | --- |
| Missing model member | ValueError: ordered ensemble hash `d7118a22…` != pinned |
| Member without `members_detail` | ValueError: no members_detail entry for `[...]` |
| Corrupt member bytes | ValueError: hash mismatch `c62bf352…` != `af031d89…` |
| Wrong class count | ValueError: declares 50 classes, vocabulary has 100 |
| Reversed member order | ValueError: ordered ensemble hash `60d6c6e9…` != pinned |
| Empty `members_detail` | ValueError: no members_detail entry for all nine |

Before pinning, **four of these six loaded silently** — including corrupt
bytes and a reversed member order.

### Test suites

- **Python: 312 passed, 0 failed**, 29 skipped.
- The release-bundle test fixture was pinned to four members and now stages all
  nine from the sidecar. Before that change, 13 bundle tests failed because the
  fixture could not represent the release model — a fixture pinning four would
  have let a nine-member bundle regress to four and still pass.

## Files changed

| File | Change |
| --- | --- |
| `src/features.py` | Dev's version: `use_z`, `feature_dim()`, resample doc |
| `src/main.py` | Two-input name resolution, per-member width, mask, hash pins, class check |
| `evaluation/device/release_official/test_official_bundle.py` | Fixture stages nine members |

Commit `6ccf26d`.