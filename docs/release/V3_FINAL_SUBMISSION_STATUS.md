# V3 final submission status

**Verdict: SUBMISSION_BLOCKED.**

Not because the app is broken — the capture path is now correct and tested —
but because the Samsung submission cannot be marked ready while the phone
evidence, the official vocabulary/contract and the delivery artifacts are all
still outstanding. Each blocker is named below with what would clear it.

Date of this status: 2026-11-14.

## 1. The final decision

Per the V3 directive the decision must be one of the seven named values. The
evidence supports:

| Candidate | Verdict | Why |
| --- | --- | --- |
| LIVE_PHONE_VERIFIED | **No** | No physical device was ever attached |
| PIPELINE_FIX_STILL_REQUIRED | **No** | Both reported capture defects are fixed and covered by tests |
| MODEL_FINE_TUNING_REQUIRED | **No** | No valid phone data shows repeated cross-signer errors — see `V3_MODEL_DECISION.md` |
| MODEL_CANDIDATE_VALIDATED | **Yes, for the model** | Frozen FP16 ensemble unchanged, hashes verified, 97/100 genuine training clips correct off-phone |
| PHONE_EVIDENCE_BLOCKED | **Yes, for phone accuracy** | `adb devices -l` → only `emulator-5554` |
| SUBMISSION_READY | **No** | Section 3 |
| SUBMISSION_BLOCKED | **Yes** | Section 3 |

The two non-exclusive verdicts that apply are **MODEL_CANDIDATE_VALIDATED** and
**PHONE_EVIDENCE_BLOCKED**, giving **SUBMISSION_BLOCKED** overall.

## 2. What changed in this branch

Branch `diag/live-phone-recognition-v3`, based on `12feeef`. Files touched:

| File | Change |
| --- | --- |
| `HandTracker.kt` | Strict synchronous `beginCapture()`; pure `selectCapture()`; removed the waiting path, the stale fallback and all `capturing`/`captureCallback` state |
| `CameraScreen.kt` | Consumes `CaptureRequest`; distinct messages per rejection reason; `UiState.CAPTURING` removed |
| `strings.xml` | Three rejection messages added; three now-unused strings removed |
| `PreTapCaptureContractTest.kt` | New — 22 behavioural tests over the real selector |
| `PreTapOnDeviceTest.kt` | New — 2 instrumentation tests |

The model, vocabulary, features and ensemble ordering are untouched.

### Candidate APK

`dist/echogesture-v3-12feeef-pretap.apk`
SHA-256 `ce83cf4c0e080b210f3b3cb160d650177cf759f29e190b8951730dcbbdda8aa4`,
77,716,903 bytes. Full manifest in `docs/release/V3_FINAL_APK_MANIFEST.json`.

**This is built from an uncommitted working tree.** Commit it before
distributing, so the APK corresponds to a SHA anyone can check out.

## 3. Remaining blockers

### Blocker 1 — no physical phone evidence (hard)

`adb devices -l` has returned only `emulator-5554` throughout. Every claim
about real-phone recognition is therefore unverified. The exact sequence that
closes it is in `docs/release/PHONE_GATE_INSTRUCTIONS.md`; the APK to install
is the candidate above, not the older `echogesture-rc-a2c0a6c.apk`.

**Clears when:** `diagnostics.txt` from a real device is analysed by
`accuracy_report.py` and the coverage + accuracy numbers are recorded.

### Blocker 2 — official vocabulary and manifest not supplied

The app ships 100 labels from `models/landmark_labels.json`. The organizer's
final evaluation vocabulary and clip manifest have not been provided. If the
official set differs, the evaluator contract changes.

**Clears when:** the organizer publishes the vocabulary and clip IDs.
Do not invent either.

### Blocker 3 — INT8 not shipped

The submission criteria ask for INT8. The app ships FP16 because the measured
cost of INT8 was 5.2 pt. That is a deliberate, recorded trade-off, but it is
still a deviation from the stated requirement.

**Clears when:** either an INT8 conversion lands within an acceptable accuracy
delta, or the deviation is accepted explicitly in the submission text.

### Blocker 4 — latency target not met on the stated measure

Stated target: ≤ 50 ms. Measured 275.8 ms p95 on warm desktop evaluation
(`LATENCY_ANALYSIS.md`) — not a phone. The new capture path removes a possible
up-to-3.7 s wait, which helps end-to-end tap-to-result, but inference itself is
unchanged.

**Clears when:** either latency is optimised below the target, or the delta is
documented as a known limitation.

### Blocker 5 — robustness gap

Robustness under the organizer's perturbation suite shows a 10 pt drop under
crop (frame-drop is 0.0). Recorded in the release manifest.

**Clears when:** either the augmentation gap is closed, or it is reported as a
measured limitation.

### Blocker 6 — delivery artifacts not produced

- Submitted video: **not created.** Must be ≤ 3 minutes per the official final
  submission instructions.
- Video filename: must follow `CollegeName_TeamName`.
- Google Form: **not submitted** — requires explicit approval.

**Clears when:** the video exists, is correctly named, and the form is submitted
with the team's approval.

## 4. What is safe to state in the submission today

These are backed by verified evidence:

- Four-member FP16 TFLite ensemble, mean raw logits → one softmax → top-5,
  combined SHA-256 `0e4c392732e7da7e77632620dc72e4afce8b1a5aa00234dbf7a13a1049a7ad93`.
- 24 frames × 255 features, layout `pos[84] | vel[84] | acc[84] | geom[3]`.
- The capture window is the 3.7 s **ending** at the tap, resampled to 32 real
  frames by timestamp — matching the training sample step.
- Off-phone: 97/100 genuine training clips correct across previously-failing
  classes; no class fails on every clip.
- Android and Python agree on the identical file.
- Rejection gating: 73.6% micro top-1 → 94.1% among the 64.2% accepted.
- 64 JVM tests and 5 instrumentation tests pass; lint reports no new findings.

## 5. What must NOT be stated

- Any phone accuracy number. There is none.
- Any claim the app is "verified working" on a real device.
- Any claim of 100% recognition.
- That INT8 is shipped.
- That the ≤ 50 ms latency target is met.
- Any organizer instruction, vocabulary or clip ID not actually supplied.

## 6. Next highest-value action

One action dominates the rest: **run the phone gate on a real device.** It is
the only thing that can move PHONE_EVIDENCE_BLOCKED, and it is the input every
remaining model decision depends on. The APK and the exact command sequence
already exist.