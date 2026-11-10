# V7 tracking evaluation (A03)

Date: 2026-11-09  
Owner: A03, isolated `/tmp/echogesture-v7-agents/A03/evaluation/device/v7_tracking/` only.

## Verdict

No safe production-default improvement was found. The candidate repair is
available as an opt-in experiment, but it changes predictions and regresses
clean INT8 and crop FP16. Do not integrate it silently. The original Dev
frontend, feature builder, model weights, and `src/main.py` were not edited.

The corrected slot repair was harmful on the full clean fixture:

- IMAGE baseline: FP16 272/345, INT8 254/345.
- Slot repair: FP16 252/345, INT8 232/345.
- One-frame internal gap interpolation: FP16 273/345, INT8 246/345.
- Combined slot + gap: FP16 256/345, INT8 229/345.

The 346-clip clean run has one failure: `5989539775823847-INTUITIVE`, no hand
detected. Counts below retain failures in the requested run and show the
successful denominator.

## What was tested

All candidates use the same IMAGE-mode detector output and unchanged Dev
`build_features(expected_frames=24)`. `slot` only repairs adjacent one-hand
slot changes when the crossed pose distance is materially smaller. `gap`
linearly interpolates only internal one-frame missing slots; prefixes,
suffixes, and longer gaps remain absent. `slot_gap` combines both. The
baseline is an exact detached copy.

This deliberately avoids repeating the prior majority canonicalization and
VIDEO-mode changes. Raw clean 30-clip data contained 26 one-hand left/right
transitions across 13 clips and 34 short internal missing slot frames across
17 clips; no strict two-hand swap passed the prior geometric guard.

The V6 `single_slot_continuity` implementation was checked but not blindly
reused; the corrected V7 test and implementation are in `candidates.py` and
`test_candidates.py`.

## Clean results

The following are top-1 counts on successful clips. FP16 and INT8 receive
identical candidate feature tensors.

| Fixture | Candidate | FP16 | INT8 |
|---|---|---:|---:|
| 30 clean | baseline | 22/30 | 21/30 |
| 30 clean | slot | 20/30 | 18/30 |
| 30 clean | gap | 21/30 | 20/30 |
| 30 clean | slot + gap | 20/30 | 16/30 |
| 346 clean | baseline | 272/345 | 254/345 |
| 346 clean | slot | 252/345 | 232/345 |
| 346 clean | gap | 273/345 | 246/345 |
| 346 clean | slot + gap | 256/345 | 229/345 |

Clean 346 FP16/INT8 top-1 parity was 81.45% (64/345 disagreements) for
baseline and 83.19% (58/345) for `gap`. Higher parity is not evidence of
higher accuracy; `gap` loses eight correct clean INT8 predictions.

A separate `max_gap=2` clean check did not improve the outcome: gap FP16
272/345 and INT8 248/345. Longer gaps were not promoted.

## Corrupted versus clean, 346-clip fixture

Seed 42, severity 0.5, deterministic stock corruptions and re-encoding.
`dFP`/`dI8` are candidate minus baseline corrupted correct counts. Failures
remain in the run denominator; successful counts are shown.

| Corruption | Baseline FP16 | Baseline INT8 | Gap FP16 | Gap INT8 | dFP | dI8 | Slot FP16 | Slot INT8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crop | 246/345 | 220/345 | 242/345 | 222/345 | -4 | +2 | 223/345 | 196/345 |
| frame drop | 267/345 | 245/345 | 269/345 | 253/345 | +2 | +8 | 246/345 | 229/345 |
| low light | 270/346 | 239/346 | 271/346 | 242/346 | +1 | +3 | 243/346 | 211/346 |
| motion blur | 248/346 | 228/346 | 252/346 | 230/346 | +4 | +2 | 227/346 | 209/346 |
| occlusion | 215/343 | 201/343 | 218/343 | 201/343 | +3 | 0 | 199/343 | 181/343 |

Corrupted failure counts: crop 1 (`9465582990654366-BELT`), frame drop 1
(`5989539775823847-INTUITIVE`), low light 0, motion blur 0, occlusion 3
(`647818976594698-BABY 2`, `7803719476282813-BABY 2`, `09934450273931295-CHEEK`).
All failures were `ValueError: No hand detected in any frame` at the detector
stage, not hidden model errors.

Paired clean/corrupt agreement and correctness flips:

| Corruption | Baseline FP16 C→W/W→C | Gap FP16 C→W/W→C | Baseline INT8 C→W/W→C | Gap INT8 C→W/W→C |
|---|---:|---:|---:|---:|
| crop | 41/16 | 45/20 | 55/22 | 49/25 |
| frame drop | 19/14 | 21/17 | 28/19 | 19/26 |
| low light | 24/22 | 24/23 | 35/20 | 31/22 |
| motion blur | 37/13 | 39/18 | 41/15 | 41/25 |
| occlusion | 63/9 | 63/11 | 66/15 | 65/22 |

C→W means clean correct, corrupted wrong; W→C is the reverse. Pair counts
exclude clips that failed either arm. The gap candidate does not universally
reduce flips.

30-clip frame-drop control: baseline FP16 18/30, gap FP16 18/30; baseline
INT8 17/30, gap INT8 17/30. This small control is not an accuracy claim.

## Parity observations

For the 346-clip clean set, `gap` increases FP16/INT8 agreement from 81.45%
to 83.19% and reduces disagreements from 64 to 58. On corrupted sets:

| Corruption | Baseline FP16/INT8 agreement | Gap FP16/INT8 agreement |
|---|---:|---:|
| crop | 77.68% (77 disagreements) | 79.13% (72) |
| frame drop | 81.74% (63) | 85.80% (49) |
| low light | 79.48% (71) | 80.92% (66) |
| motion blur | 77.17% (79) | 78.61% (74) |
| occlusion | 69.68% (104) | 72.30% (95) |

Again, parity does not imply correctness. Clean INT8 regression is a blocker
for a default change, and the crop FP16 regression is another blocker.

## Reliability measurements

On the 30-clip clean raw detector output, 32 frames/clip: 42.71% of frames
had any hand, 10.42% both, 32.29% exactly one, and 57.29% none. There were
22 imputed slot frames for one-frame internal gaps. On 346 clean successful
clips (11,040 frames): 42.77% any-hand, 12.69% both, 30.08% one, 57.23%
none; 325 short internal slot frames were imputed. The gap candidate marks
all imputed frames explicitly in `candidate_contract.imputed_slot_frames`.

## Reproduction

Run from the isolated A03 checkout. The existing Python 3.11.18 environment
contains NumPy 2.4.6, OpenCV 5.0.0, MediaPipe 1.2.2, and ai-edge-litert.

```bash
cd /tmp/echogesture-v7-agents/A03
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I -B \
  evaluation/device/v7_tracking/test_candidates.py

# Clean 30 (results/clean_30_rerun)
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I -B \
  evaluation/device/v7_tracking/run_tracking.py \
  /tmp/echogesture-validation-30/manifest.csv --arm clean \
  --output-dir evaluation/device/v7_tracking/results/clean_30_rerun

# Clean 346 (final verification)
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I -B \
  evaluation/device/v7_tracking/run_tracking.py \
  /tmp/echogesture-v5-agents/B/evaluation/device/v5_accuracy/data/val_346/clips/fetched_manifest.csv \
  --arm clean --output-dir evaluation/device/v7_tracking/results/clean_346_verified

# One corruption per process; replace kind as needed
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I -B \
  evaluation/device/v7_tracking/run_tracking.py \
  /tmp/echogesture-v5-agents/B/evaluation/device/v5_accuracy/data/val_346/clips/fetched_manifest.csv \
  --arm corrupted --corruption occlusion --severity 0.5 --seed 42 \
  --output-dir evaluation/device/v7_tracking/results/corrupt_occlusion_346

# Pair clean/corrupt and inspect FP16/INT8 parity
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I -B \
  evaluation/device/v7_tracking/compare_runs.py \
  --clean evaluation/device/v7_tracking/results/clean_346_verified/run.json \
  --corrupted evaluation/device/v7_tracking/results/corrupt_occlusion_346/run.json \
  --output evaluation/device/v7_tracking/results/verified_compare_occlusion_346.json
```

All source/model hashes are recorded in the run JSON. Required hashes were
verified: FP16 `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`,
INT8 `9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b`.
Hand task: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
Dev frontend: `c73ef64b72f21a86638c4a126ad642b64c4ee29851cd2f8fd8ea4ed516fa97b0`.
Dev features: `baf28d05bbd36e2783ea7ef6f78e793b89ed832d0eaad5dffed23d35c775aa84`.
Val-346 manifest: `3f06798af00e69314da87414bf0c39bb2ec49e578f560c111754fa00966f6c82`.
30-clip manifest: `004e5effb7e6c19b515fdd00b048459fff236d8b35c6b7f7b49e91a1efa5d618`.

## Artifacts

- `candidates.py`: opt-in baseline/slot/gap/slot_gap transforms.
- `test_candidates.py`: deterministic synthetic contract tests, passed.
- `run_tracking.py`: one IMAGE detection per clip, all candidates, raw tracks,
  both pinned models, failure preservation, variable-length track handling.
- `compare_runs.py`: clean/corrupt pair and FP16/INT8 parity aggregation.
- `check_gap_masks.py`: narrower masked-gap diagnostic.
- `results/*/run.json`, `records.jsonl`, raw track arrays/JSON: full evidence.
- `results/summary.json`, `verdict_data.json`: compact machine-readable results.

## Final blocker / handoff

The requested focused reliable improvement is not established. `gap` is a
promising targeted experiment for frame drop and some corrupted conditions,
not a production default. Slot repair is rejected. Keep IMAGE mode and Dev's
canonical semantics until Dev approves a separately validated tracking change.
No target-phone measurement was performed; these are desktop Python 3.11.18
results only.
