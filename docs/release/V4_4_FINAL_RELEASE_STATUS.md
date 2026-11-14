# V4.4 — final release status

Branch `fix/v4-4-final-release`. Date 2026-11-14. Deadline 2026-11-18.

## Decision summary

| Question | Answer |
| --- | --- |
| Python evaluator working | **YES** — `predict(video_path)` on real MP4s, 9 members |
| Nine-model numerical parity | **PASS** — features 0.000000, masks exact, logits 5e-6, top-1 9/9 |
| Android tests | **PASS** — 110 JVM, 11 instrumentation (unchanged from V4.3) |
| Python tests | **PASS** — 312 passed, 0 failed, 29 skipped |
| Physical phone | **NOT TESTED** — no device attached |
| Rights | **BLOCKED** — see §8 |
| INT8 compliance | **FAIL** — FP16 ensemble |
| Submission | **BLOCKED** — five named items |
| Recommended APK | **V4.3**, `3db9d94a…` — unchanged, see §5 |

## 1. What was actually fixed

Four defects, all real:

1. **Production `predict()` could not load the nine-member ensemble.** It took
   `get_input_details()[0]` positionally (which is the mask), assumed one shared
   feature width, never fed the mask, and discarded the mask `build_features()`
   returned. The Android app worked; the competition evaluator did not.
2. **`features.py` hardcoded 2 coordinates.** `use_z=True` silently produced
   255-wide features, never 381. The flag existed and had never run.
3. **The Python evaluator pinned nothing.** Corrupt bytes, a reversed member
   order, a wrong class count and a missing member all loaded silently and
   produced plausible wrong labels. Four of six deliberate-corruption cases
   passed as if healthy. Now all six fail loudly.
4. **The parity fixture generator was machine-specific.** It hardcoded
   `/tmp/eg-rc/.pynref` and a fixed output path. Now portable, with model-hash
   verification before it writes anything and a provenance sidecar.

## 2. Files changed

| File | Change |
| --- | --- |
| `src/features.py` | Dev's version: `use_z`, `feature_dim()`, n_coords from input |
| `src/main.py` | Name-resolved two-input contract, per-member width, mask plumbing, SHA-256 pinning, class-count check |
| `evaluation/device/release_official/test_official_bundle.py` | Fixture stages nine members, not four |
| `evaluation/device/v4_3_parity/make_parity_fixture.py` | Rewritten portable, with verification |
| `results/{v4_2_baseline,v4_3_masked,v4_4_release}/results.json` | New, versioned, self-describing |

Commits: `6ccf26d` (evaluator), `5857801` (parity + provenance), plus this report.

## 3. Verification performed

### Primary acceptance test — fresh extracted package

The bundle was built, extracted to an empty directory, and run **from the
extracted copy** with no imports from the working tree:

```
cd /tmp/fresh_v44
python -c "sys.path.insert(0,'/tmp/fresh_v44/src'); import main; main.predict(clip)"
→ module file: /tmp/fresh_v44/src/main.py
→ {"clip_id": "5803218002565746-DOWNSIZE", "prediction": "DOWNSIZE1",
   "confidence": 0.915, "latency_ms": 848.97}
```

Identical to the reference. **The extracted package produces the correct model
result on a real MP4.**

### Docker

Image built from the extracted bundle; weights mounted at runtime (never baked
in, by design). The container's evaluator entrypoint returns `DOWNSIZE1` for the
same clip — same answer as the host.

### Parity

Nine clips, five signers, 46–181 frames, confidence 0.21–0.97. Feature diff
0.000000, mask mismatches 0, logit diff 5e-6, top-1 agreement 9/9. One clip is
misclassified identically by both platforms, which is stronger evidence than a
correct agreement.

### Test suites

| Suite | Result |
| --- | --- |
| Python `evaluation/device` | **312 passed, 0 failed**, 29 skipped |
| Kotlin JVM | **110 passed, 0 failed** |
| Android instrumentation | **11 passed, 0 failed** |

Skipped tests are 29 optional ones requiring assets or harness state not present;
they are **not** counted as passes.

## 4. What is NOT verified

- **Physical phone.** Only `emulator-5554` was attached. No phone accuracy, no
  phone latency, no power or thermal measurement exists.
- **Pixel-level robustness.** Not measured in this cycle.
- **Real-camera robustness.** Only synthetic landmark corruption exists, and a
  synthetic "camera further" coordinate scaling is not a camera test.
- **The organizer's ≤50 ms criterion.** Inference is ~276 ms-class on a laptop
  and capture interaction is seconds by design. **FAIL**, not unmeasured.
- **INT8.** FP16 ships; no new INT8 candidate was attempted.
- **A clip with a hand absent for the whole recording.** The all-zero mask is
  covered by unit tests, not by a real clip.

## 5. APK

**No production Android change was made in V4.4.** The V4.4 work is Python,
parity tooling, provenance and release process. `git diff 5251bf8..HEAD --
app/android_final/app/src/main` is empty.

Per the brief's instruction not to rebuild an identical APK unnecessarily, the
validated artifact remains:

| | |
| --- | --- |
| Path | `dist/echogesture-v4.3-masked.apk` |
| Full SHA-256 | `3db9d94ad761a76bdcf5dd9b3eaeff7e91dd14a51eaee2eca44acb98f1380896` |
| Commit | `5251bf8` |
| Members | 9 (4 × 255-wide, 5 × 381-wide) |
| Ensemble SHA-256 | `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb` |
| Permissions | CAMERA only |

Rollback preserved: `dist/echogesture-v4.2.apk`, `5490e029…`.

## 6. Provisional submission bundle

Built and verified at commit time. Contains all nine members, the four source
files, requirements, README, Dockerfile and the export script. Status stamp is
`structure_verified`, **not** official-complete — the organizer manifest, exact
clip IDs and rights are unavailable, so an official build would be fabricated.

## 7. Results provenance

Three versioned result files now exist, each carrying model version, commit,
combined hash, feature-width contract, mask status, split, unique clip count,
unique signer count, method and metric definitions.

**Corrected statement about the 184-evaluation figure:** it is **72 unique
clips** from **37 signers**, repeated under synthetic idle. It is not 184
independent signs. 79.3% must be quoted as "on 72 authorised clips across 184
evaluations including repeated idle conditions", never as independent accuracy
on 184 videos.

Dev's stale artifacts are documented but **not rewritten**: `eval_summary_test.json`
and `robustness.json` are the old four-member model, `results.csv` is the new
one, and they disagree about the same robustness claim. None records a model
hash, which is why a stale artifact reads as authoritative.

## 8. Rights — BLOCKED

Restricted-derived weights may already exist in public Git history on
`origin/main` and other branches, predating this work. **Deleting them in a new
commit does not remove them from history.** Remediation needs a history rewrite,
which is the owner's decision; it was not performed.

No new restricted material was added to Git in this branch (`.tflite` is
gitignored and none were committed). Full path-by-path assessment is in
`V4_4_FINAL_COMPLIANCE.md` §Rights.

**Rights are not claimed as cleared merely because the repository is public.**

## 9. Submission status

**BLOCKED.** Five items, all requiring a human:

1. **Physical phone test** — attach a device and run
   `docs/release/PHONE_GATE_INSTRUCTIONS.md`.
2. **Rights decision** — §8.
3. **Demo video** ≤3 minutes, and the presentation that depends on it.
4. **Exact `CollegeName_TeamName`** string.
5. **Google Form approval.**

## 10. Exact human-only actions

```bash
adb devices -l                    # must show a real phone, not emulator-5554
adb install -r dist/echogesture-v4.3-masked.apk
```

Then: Settings → Capture diagnostics → on; 10 supported signs × 5 attempts
across lighting/distance conditions; record every rejection and wrong answer.

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt > diagnostics.txt
python evaluation/device/accuracy_report.py diagnostics.txt annotations.txt
```

## 11. Proven / hypothesis / untested

**Proven**
- The extracted package runs `predict()` on a real MP4 and produces the correct
  result for the shipped model.
- Android and Python agree numerically to 1e-6 on features and masks across
  nine clips, both feature widths, five signers.
- The evaluator refuses corrupt, substituted, reordered and mislabelled model
  sets rather than producing a plausible wrong label.
- The nine-member ensemble beats the old four-member one by 15.7 points
  top-1 on 72 clips, and the gain is concentrated where idle frames exist.

**Hypothesis**
- That this improves recognition on a real phone. Supported by every off-phone
  measurement; none of it was taken on a phone.
- That the confidence gate transfers. Re-measured on validation it is the best
  of the options tested, but n=40 with 10 errors is thin.

**Untested**
- Any physical phone, any live signer, any camera condition.
- Pixel-level and real-camera robustness.
- Power and thermal behaviour.