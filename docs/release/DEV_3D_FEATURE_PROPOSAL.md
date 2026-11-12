# Dev proposal: an xy+z feature variant (3D hand features)

**Status:** proposal only. Nothing in the shipped ensemble changes from this
document. Dev owns the decision.

## Why this exists

Organizer deck p.10 lists `Pose-invariant 3D hand features` as an architecture
constraint. The shipped extractor at `src/features.py:151` does:

```python
xy = landmarks[..., : 3 if use_z else 2]
```

with `use_z=False`, so the MediaPipe `z` channel is discarded. The 255-value
vector is x/y position, velocity, acceleration, inter-hand geometry, presence
masks and canonical slots.

This is a scoring risk on the architecture description, not a disqualification
rule — the rubric that carries points (p.11) does not mention 3D. But a judge
reading p.10 and then the code will notice.

## Why we are not changing it now

1. `FEATURE_DIM` is 255 and is asserted in the metadata, the Kotlin contract, the
   Android tests and the release builder. Enabling `use_z` changes it.
2. Every published number — 72.78% top-1, the robustness drops, the latency
   figures — was measured on xy features. Changing the input invalidates all of
   them at once, days before submission.
3. The four FP16 members were trained on 255. A 255→new-dimension input would
   need retraining, which is Dev's call.

## What a proper evaluation would look like

Only worth doing if there is time after the submission deadline, or as a
post-hackathon improvement:

1. Build the xy+z variant: `use_z=True`, `FEATURE_DIM` recomputed, model input
   tensor regenerated. This is a **new candidate**, not an edit to the shipped
   one.
2. Train on the same signer-disjoint split with the same seed and schedule.
3. Report top-1 on the same 1,249 scored clips, plus the same 5 robustness
   conditions, so the comparison is like-for-like.
4. Only ship it if it is better on the scored split **and** does not regress the
   robustness conditions — and only with Dev's sign-off.

## Honest caveat

Monocular `z` from MediaPipe is an estimate, not a measurement. It may help or
may add noise. The current xy-only result is measured; a 3D result would need to
earn its place.

## Recommendation

Ship the measured xy model. Describe the features accurately in the deck as
"2D landmark trajectories with temporal and inter-hand geometry" rather than
claiming 3D, and note the xy+z ablation exists behind `use_z` as future work.
Understating a feature is survivable; overstating one is what a judge punishes.