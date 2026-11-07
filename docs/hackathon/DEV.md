# Dev's workflow — data, features and INT8 model

**Own** data preparation, shared preprocessing, training and export.

**Hand off to Abhijeet** a model bundle with reproducible accuracy and explicit
input and output semantics.

**Start here** [Sprint schedule](README.md) · [Handoff contract](HANDOFF.md)

## DEV-01 · Freeze the data rules by 8 November

1. Check with Abhijeet for the organizer vocabulary and starter kit. Build a
   manifest with `clip_id`, `path`, `gloss`, `signer_id`, `split` and
   `source`. Keep restricted source paths in your local copy.
2. Write a deterministic gloss-to-index map. Report per-class clip counts,
   distinct signers, label conflicts and unreadable clips. Check that no signer
   or duplicate clip crosses splits. Keep the official held-out set locked.
3. While waiting for the official list, use a small provisional subset with
   several one- and two-handed signs and multiple signers. Mark those metrics
   provisional. Replace the vocabulary from config when the official list arrives.
4. If the starter data needs supplementation, inspect overlap and usage terms
   before adding a source. Keep videos and landmark caches local.
   [ASL Citizen splits and dataset](https://www.microsoft.com/en-us/research/project/asl-citizen/) ·
   [Dataset license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/)

**Done when** Abhijeet has the manifest schema, vocabulary status, split audit
and a list of any classes that lack usable examples.

## DEV-02 · Prove extraction and training by 10 November

1. Start with MediaPipe tracking for two hands. Preserve handedness, per-hand
   presence masks, frame order and timestamps. Record reacquisition or missing
   landmarks and inspect several overlaid clips. Check mirroring against the
   canonical left and right slots.
2. Keep hand shape, relative wrist positions and temporal motion. Preserve the
   relationship between hands and body where the input provides that signal.
   Normalize consistently. Handle missing hands explicitly so valid zeros and
   missing observations are distinguishable.
3. Cache raw landmarks locally with an extractor version and sample settings.
   Put the feature transform used by training and Python inference in
   `src/features.py` or another agreed module under `src/`. Share a fixture and
   expected output so Abhijeet can verify feature parity on Android.
4. Train a small temporal CNN. Check that it can overfit a tiny labelled sample
   before the full run. Compare 16 and 24 frames sampled across the clip. Add
   seeded time jitter, hand masking and geometric variation after establishing a
   clean baseline. Keep the evaluation split and conditions identical across
   candidates.
5. Record top-1, sample count, per-class confusion, signer-level results and
   landmark failure rate. If the classifier is weak, inspect coverage and
   confusions before adding complexity.

**Done when** a candidate and its float validation result are reproducible
from documented commands. Keep the current S3D baseline available for comparison.

## DEV-03 · Export and hand off by 12 November

1. Implement a conversion command using representative training inputs that
   include normal, missing-hand and varied-length cases after preprocessing.
   Try integer post-training quantization first. Check exported input and output
   dtypes, supported ops, file size and output order. Consider
   quantization-aware training if the measured validation drop warrants it.
   [LiteRT integer export](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant)
2. Ship the INT8 file, label map, feature config and exact reproduction commands
   in one handoff. Run the same validation on float and INT8 and give Abhijeet
   the corresponding model and config SHAs.
3. If Abhijeet's profile shows landmark or decode cost dominates, test a smaller
   frame budget with matching training settings. Compare accuracy on the same
   validation set and keep the strongest usable tradeoff.

**Done when** Abhijeet can load and run the candidate from the handoff without
asking for a private file or a hard-coded path.

## DEV-04 · Freeze and review by 17 November

1. Pick the release candidate jointly. Freeze the vocabulary, runtime feature
   config, conversion command and model SHA. Evaluate the frozen candidate on
   the locked split and record what was measured.
2. Review the bundle from a clean directory on Python 3.11. Run a sample clip
   and verify the prediction fields, model size, claimed metrics and demo scope.

## Prompt for Dev's agent

> I am Dev on EchoGesture. Read AGENTS.md and the Dev and handoff workflows.
> Start with the first unfinished DEV task. I own data, shared features,
> training and INT8 export. Coordinate interface changes with Abhijeet before
> changing the evaluator contract. Run the smallest useful verification and
> return the branch and SHA, commands, measured result and next handoff.
