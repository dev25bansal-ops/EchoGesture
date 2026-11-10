# Demo verification checklist

**Live-camera items are blocked locally.** macOS denied camera permission
during the engineering check, so camera-path shots cannot be verified yet.
Use the offline clip-path recording (see `DEMO_SCRIPT.md`) and keep the
unchecked boxes unchecked — do not tick them for clips that were never shot.

## Verified (integration evidence)

- [x] Release candidate SHA matches evidence: classifier
  `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`,
  labels `e78f5028…`, features `baf28d05…`, detector `fbc2a300…`.
- [x] Real model produces the displayed caption: `PULLCONVINCE` at confidence
  0.7507, executed from a verified extracted bundle. No mock result path.
- [x] Offline mode; no cloud inference. Temporary captures are deleted.
- [x] Timing shown is capture-to-result / clip latency, never described as
  clip p95.
- [x] Provisional 0.25 retry threshold is not described as calibrated.
- [x] Supported vocabulary stated: 100 isolated signs
  (`models/landmark_labels.json`).
- [x] FP16/INT8 status described accurately: FP16 classifier, not INT8;
  INT8 alternative unselected.

## Blocked — complete before any live-camera recording

- [ ] Consenting signer was not part of training.
- [ ] Camera permission works in the actual launch environment.
- [ ] Real camera preview and tap/SPACE capture are visible.
- [ ] Retry and camera/model error states are exercised on camera.
- [ ] No raw footage is retained unintentionally.
- [ ] Final 2–3 minute video is reviewed before submission.
- [ ] Android claims are made only after testing on a physical Android device.
