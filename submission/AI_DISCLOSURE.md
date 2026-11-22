# AI disclosure

Team **VIT_Chennai_Sky_Grazers** — Samsung PRISM XR Hackathon 2026, Track 02 (EchoGesture).
Written 2026-11-22.

Two different things in this submission are "AI", and they need separate statements:

- **AI used to build the submission** — coding assistants, language models.
- **AI inside the submitted product** — the models that do the recognising.

The first is disclosed because it materially shaped the code and the prose. The
second is disclosed because the project *is* a machine-learning system and the
judges are entitled to know exactly what it is made of.

Neither the organizer's 27-slide brief nor the Phase-1 registration form carries
an AI-disclosure field or a disclosure requirement. This document exists because
the honest thing is to state it anyway, not because a rule demanded it.

---

## 1. AI tools used to build this project

| Tool | What it did |
|---|---|
| **Claude Code** (Anthropic; models in the Claude 5 family, principally Sonnet 5 and Fable 5) | Agentic coding assistant used throughout the build. It wrote and edited source, ran the test suite and the on-device toolchain, authored documentation, performed the audits recorded in `docs/`, and produced review passes over the deck and the results. |
| **Codex** (OpenAI) | Used to build the Android **Auto 6.2** branch, which is the app that ships as `EchoGesture-Auto-6.2.apk`. Four commits in this repository are authored by `Codex <codex@users.noreply.github.com>`. |
| **Claude Sonnet 5** | Recorded in commit trailers on **165 of 418 commits** in the repository's history. 157 of those were co-authored under Abhijeet Miskin's commits, 8 under Dev Bansal's. GitHub Copilot is not present in the history. |

Agentic assistance here was not autocomplete. Claude Code ran the test suites,
drove the phone over ADB to capture the on-device screenshots in
`docs/evidence/`, built and measured the Docker image, and wrote the audit
documents. A reader who assumes every line was hand-typed should know otherwise.

### What the human team did, and what it means for the numbers

Every **measurement** in `results/results.csv` was produced by running the code
on hardware and reading the output. The accuracy figure is reproducible from
`results/predictions.jsonl` against the published labels; the latency, power and
thermal figures came from a physical Vivo V2404. An AI assistant wrote the
scripts that took those measurements, but it did not invent their values.

Every **claim** in the deck, the README and this document was checked against an
artifact on disk or a recorded device run before it was committed. Where a claim
could not be substantiated it was withdrawn rather than softened — the withdrawn
15.7 ms latency figure and the withdrawn 4.40 pp robustness figure are both still
recorded in `submission/FINAL_DELIVERY/CHECKLIST.md` with the reason they were
pulled. The deck's slide 8 ("What this is not") and `docs/HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md`
state the project's limits in its own failures, not only its successes.

What the team is accountable for: **the two members named on slide 1 reviewed
and approved this submission as their own work.** The delegation above shifts
how the code was produced; it does not shift who is answerable for its accuracy
or for the honesty of its claims.

---

## 2. AI inside the submitted product

This is not an AI-assisted project that happens to run on a phone. The shipped
artifact is an inference system, and everything below runs **entirely on the
device**.

| Component | What it is | Licence / origin |
|---|---|---|
| `models/hand_landmarker.task` (7.82 MB) | Google **MediaPipe** hand-landmark detector. 2 hands, 21 landmarks each, `RunningMode.VIDEO`. Runs under the GPU delegate with a CPU fallback. | Apache-2.0, published by MediaPipe |
| Nine `landmark_ens_*_fp16.tflite` members (8.51 MB total) | A **nine-member ensemble of temporal convolutional classifiers**, each trained by this team on hand-landmark sequences. FP16 weights, mean-logit aggregation. These produce the actual gloss prediction. | Trained by this team on ASL Citizen-derived features |
| `models/landmark_classifier_int8.tflite` | An INT8 conversion kept in the tree for the export pipeline. **This is not what ships** — the shipped app loads the nine FP16 members. See the INT8 note below. | Trained by this team |

The pipeline is: camera frame → MediaPipe landmarks → a recorded gesture interval
→ 24 frames of 255- or 381-wide features → nine classifiers → aggregated logits →
one of 100 ASL glosses. There is no cloud component at any point.

### On-device claim, verified

The shipping app's `AndroidManifest.xml` explicitly removes the `INTERNET` and
`ACCESS_NETWORK_STATE` permissions that the manifest merger would otherwise
inject. The app cannot make a network request. Zero raw video leaves the device,
which is the privacy principle Track 02 sets out. Speech output uses the
device's own installed voice pack, which is local.

### Training data and its restrictions

> **Correction, 2026-11-22.** `docs/release/PUBLIC_WEIGHT_EXPOSURE.md` described
> four of the ensemble members as trained on "ASL Citizen / WLASL-derived data".
> Four is wrong (there are nine) and so is the WLASL part. Both are corrected in
> that file on this date. `ARCHITECTURE_R8.md` had *proposed* a WLASL-100
> auxiliary set "if time permits"; the proposal was never carried out.

The nine classifiers were trained on features derived from **ASL Citizen**
alone — `scripts/ml/extract_landmarks.py` reads the ASL Citizen `splits/*.csv`
and nothing else, and `scripts/ml/train.py` trains on the landmark caches it
produces. **WLASL was not used for training**; it appears only as a 17-clip
in-distribution positive control in the evaluation. Both datasets carry
redistribution restrictions that this submission does **not** resolve:

- ASL Citizen video is under the Microsoft Research **non-commercial** licence.
- WLASL is under **C-UDA** (academic / non-commercial).

**The model weights are redistributed in the public GitHub repository and inside
the submitted APK with no confirmed permission covering that redistribution.**
This is the single largest legal exposure in the package and it is stated here
rather than left for someone to discover. `NOTICE` and
`docs/release/ARTIFACT_RIGHTS.md` mark the weight files "BLOCKED for
publication"; they are nevertheless present.

### The demo video

The submitted demo (`EchoGesture_demo_2026-11-22.mp4`, 1:20) is a **real
person on camera** with their own recorded narration. It is not synthetic voice
and not an AI-generated presenter.

For completeness: an earlier, **superseded** 2:55 walkthrough
(`docs/submission/`, not the submitted video) *was* machine-generated — its
narration came from Windows SAPI's "Zira" text-to-speech voice via
`scripts/media/build_demo_video.py`. It is not the demo anyone will watch, and no
current claim cites it.

---

## 3. Things this disclosure wants to prevent being said

- **Do not call the quantisation INT8.** The brief asks for INT8; the ensemble
  that ships is **FP16**. The INT8 file in `models/` is not the shipped path.
  Claiming INT8 compliance would be false.
- **Do not call this a translator.** It recognises isolated signs from a fixed
  100-word vocabulary. Continuous signing is not solved here and the deck says so.
- **Do not quote a phone accuracy.** Every accuracy number is a laptop, on a
  corpus, in a signer-disjoint test split. One phone run confirms the pipeline
  works; it is not a measured phone accuracy.
- **Do not describe the assistant's output as independently verified.** It was
  checked against artifacts by the same process that produced it, then reviewed
  by the team. That is human accountability, not third-party audit.

---

## 4. Where the rest of the evidence lives

| Claim | Artifact |
|---|---|
| Six scored metrics and their verdicts | `results/results.csv` |
| Accuracy reproduces from stored predictions | `results/predictions.jsonl`, 1,251 records → 85.45% |
| On-device behaviour, refusal path, HUD | `docs/HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md`, `docs/evidence/` |
| Live-camera latency, measured | `docs/evidence/LATENCY_AUTO62_LIVE_CAMERA_2026-11-22.md` |
| What was withdrawn and why | `submission/FINAL_DELIVERY/CHECKLIST.md` |
| Weight redistribution status | `NOTICE`, `docs/release/ARTIFACT_RIGHTS.md` |
| Artifact hashes | `submission/RELEASE_MANIFEST.json` (13 entries, all verified) |
