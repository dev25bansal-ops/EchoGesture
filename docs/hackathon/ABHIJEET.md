# Abhijeet's workflow — integration, device, demo and release

**Own** organizer contract, evaluator API, benchmarks, live app and release.

**Hand off to Dev** the official vocabulary and data constraints, full-path
profiles and feedback tied to a model SHA.

**Start here** [Sprint schedule](README.md) · [Handoff contract](HANDOFF.md)

## ABH-01 · Lock the evaluator rules by 8 November

1. Confirm the team has registered. Obtain the starter kit and record the
   vocabulary, manifest fields, reference clips and allowed training sources.
   Send the stable contract to Dev before locking the training manifest.
2. Confirm the required predictions filename, clip ID mapping, evaluation
   device and runtime, what the 50 ms includes, and how detector weights count
   toward the model budget. Mark each unresolved item and its working assumption.
3. Reserve an Android phone. On Python 3.11, run the current `src/main.py`
   against one readable clip. Record the repo SHA and expected JSON keys.
   Set up the phone build and run a camera preview while Dev trains.
   [MediaPipe Android camera sample](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/hand_landmarker/android)

**Done when** Dev has the best available organizer contract and you have a
repeatable baseline and a physical device for the demo.

## ABH-02 · Build the evaluator and benchmark by 12 November

1. Own `src/main.py` and preserve `predict(video_path) -> dict`. Integrate Dev's
   extractor, feature transform and model bundle. A supplied clip must work
   without network access or a precomputed feature cache. Keep artifact paths
   relative to the submitted code.
2. Add a manifest runner that invokes the actual API. Verify exact IDs, one
   row per clip, uppercase labels from the fixed vocabulary, finite confidence
   in [0, 1] and finite nonnegative latency. A failed clip or missing ID should
   fail the bundle check. Preserve the API's label output even when the demo
   would ask the signer to retry.
3. Instrument decode, tracking, feature preparation and inference. Measure
   full clip-to-JSON p50 and p95 directly on a representative batch. Record
   cold start and warm timing separately and log the device, resolution, frame
   count, runtime, delegate and model SHA. Send the stage breakdown to Dev.
4. Run a Python 3.11 end-to-end smoke check and take a first phone profile.
   Label any dev-laptop timing with its device. Measure live app latency
   separately on the phone and state which device ran each path.

**Done when** Dev can reproduce your profile and you can select a candidate
together using its real accuracy and timing.

## ABH-03 · Build the live demo and robustness run by 14 November

1. Use a local camera pipeline and a bounded buffer of recent features. Show
   a large caption, a waiting state and a retry state when evidence is weak.
   Use a confidence or stability threshold measured on validation clips.
   Replace any mock caption path before filming.
2. Run the demo in airplane mode on the physical phone with a consenting
   signer held out of training. Rehearse both one- and two-handed signs. Use a
   clearly labelled tap-to-capture flow if automatic sign boundaries are
   unreliable. Measure sign-end-to-caption delay for the live flow.
3. Run seeded low-light and occlusion conditions on matched clean clips.
   Report clean accuracy, degraded accuracy, percentage-point drop and clip
   counts for each condition. Run a repeatable ten-minute device workload and
   compare early and late p95. Report power with a documented measurement
   method, or mark it unmeasured.

**Done when** captions work offline, failure states are visible and results
are tied to the exact model in the demo.

## ABH-04 · Package and submit by 17 November

1. Generate predictions from the organizer manifest if supplied. Keep
   existing research outputs separate. Produce the official `results.csv`
   with target, measured value, unit, device, runtime and pass status.
2. Package the Python 3.11 entry point, pinned dependencies, README, Docker
   setup, model, vocabulary, conversion script and requested results layout.
   Verify the actual final form for the prediction extension and ZIP structure.
3. Record a 2–3 minute demo and prepare the named PPT or PDF. Start with real
   offline captions. Include architecture, a compact metrics table and the
   supported scope. Use isolated ASL signs as the product claim.
4. Give the extracted ZIP to Dev for a clean-room run. Check the final
   manifest count, names, permissions and links. Submit the reviewed package
   and save the receipt and release SHA. The hard deadline is 18 November,
   11:59 PM IST.

## Prompt for Abhijeet's agent

> I am Abhijeet on EchoGesture. Read AGENTS.md and the Abhijeet and handoff
> workflows. Start with the first unfinished ABH task. I own the evaluator,
> device benchmarks, demo and release. Use Dev's model handoff and preserve
> its label order and preprocessing. Return the branch and SHA, commands,
> measured result and the next dependency.
