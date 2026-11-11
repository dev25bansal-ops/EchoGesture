# Final submission checklist

Maintained as a real state table. No row is "pending" without an owner and a
next action.

## Status vocabulary

`DONE_VERIFIED` · `DONE_PUBLISHED` · `READY_FOR_USER` · `WAITING_FOR_DEV` ·
`WAITING_FOR_ORGANIZER` · `WAITING_FOR_HARDWARE` · `WAITING_FOR_PERMISSION` ·
`FAILED_WITH_CAUSE`

---

## Core gates

| Gate | State | Evidence |
| --- | --- | --- |
| G1 Ensemble integration | `DONE_VERIFIED` | 4 members load, per-member tensor indices, mean logits; `WHATFOR1` @ 0.8821 |
| G2 All tests discovered | `DONE_VERIFIED` | 572 local / 500 published, 0 failures |
| G3 Source-only CI | `DONE_PUBLISHED` | GitHub Actions **green** on `79278a7` and later |
| G4 Android APK | `DONE_VERIFIED` | 75,072,782 B, signed, 4 members embedded, 9/9 unit tests |
| G5 Performance / robustness frozen | `DONE_VERIFIED` | p95 274.2 / 154.8 / 99.8 ms; robustness 3 PASS / 2 FAIL |
| G6 Public-safe code committed | `DONE_PUBLISHED` | 7 commits on `hackathon/abhijeet-public` |
| G7 Branch pushed, remote matches | `DONE_PUBLISHED` | remote `26b2990` == local |
| G8 PR #1 updated | `DONE_PUBLISHED` | [PR #1](https://github.com/dev25bansal-ops/EchoGesture/pull/1), draft, Dev requested |
| G9 Release package | `DONE_VERIFIED` | `echogesture_final2_ensemble.zip`, 9,007,144 B, SHA `34a0e667…`, extracts → `WHATFOR1` @ 0.8821 |
| G10 Public source-only distribution | `DONE_PUBLISHED` | weights, APK, ZIPs, signer screenshots all gitignored |
| G11 Presentation | `DONE_VERIFIED` | `docs/submission/EchoGesture_Submission.pptx`, 11 slides |
| G12 Demo materials | `READY_FOR_USER` | runbook + 2:50 shot list; **no video recorded** |
| G13 Dev review request | `WAITING_FOR_DEV` | PR review requested; 9 specific questions in the PR body |
| G14 Organizer clarification | `READY_FOR_USER` | draft below; requires your send |
| G15 Registration | `REGISTRATION_USER_CONFIRMED` | confirmed by the team owner; the ID and exact college/team strings are held privately and are needed for the form and filenames |
| G16 Model licensing | `WAITING_FOR_PERMISSION` | weights excluded; NOTICE records status |
| G17 Device tooling | `WAITING_FOR_HARDWARE` | commands ready; no device attached |
| G18 Submission performed | `WAITING_FOR_USER` | needs the recorded college/team strings, the demo video and your explicit approval |

---

## The three things only a human can do

### 1. Registration — confirmed by the team owner

The team owner has confirmed the team completed registration. This is recorded
as `REGISTRATION_USER_CONFIRMED` and is not an open question.

What is still not in the repository: the confirmation ID and the exact
registered college and team strings. Those drive form fields, the deck filename
and the package filename (`CollegeName_TeamName`, per deck p.25), so they are
needed at submission time even though registration itself is settled.

**Action:** supply the exact registered college string, team string and team
member details when filling the form. Do not submit a duplicate registration.

### 2. Send the INT8 clarification

The official material contradicts itself and the scored rubric is ambiguous:

> p.9  "Budgets: ≤ 20 MB quantized model **(INT8)**, ≤ 50 ms latency"
> p.10 "Quantized model + conversion script (< 20 MB **INT8**)"
> p.11 "Model Quantization (20%): Compilation to **TFLite/ONNX** <20MB."

Our shipped model is **FP16** TFLite at 11.37 MB.

**Draft to send:**

> Subject: Track 2 — clarification on the model quantization requirement
>
> Hello, we are a Track 2 team. The consolidated slides state the model budget
> as "≤ 20 MB quantized model (INT8)" (p.9) and list "Quantized model +
> conversion script (< 20 MB INT8)" under deliverables (p.10), while the
> 100-point jury matrix on p.11 scores "Compilation to TFLite/ONNX <20MB"
> without naming INT8. Could you confirm whether an FP16 TFLite model under
> 20 MB satisfies the quantization criterion, or whether INT8 weights are
> mandatory? Our current submission is FP16 and we want to be accurate.

### 3. Supply team and college names

The deck uses explicit placeholders `<TEAM NAME>` / `<COLLEGE NAME>`. The
organizer requires the deck to be named by college and team. Fill these in
before submitting — I did not invent them.

---

## What still needs a device

No Android device is attached, so these are genuinely unmeasured:

- on-device install and live-camera inference
- capture-to-caption latency on real hardware
- sustained 10-minute thermal drift
- power draw

Commands and a measurement procedure exist in
[`docs/demo/DEMO_RUNBOOK.md`](../demo/DEMO_RUNBOOK.md); the tooling is ready,
only the hardware is missing.

---

## Known limitations to state, not hide

- 50 ms full-clip p95 is **not met** (best 99.8 ms on a laptop).
- Crop (−10.00 pp) and occlusion (−6.67 pp) exceed the 5 pp robustness budget.
- The 1,249-clip test split was inspected during model selection; it is not a
  pristine blind evaluation.
- No physical device: on-device latency, power and thermal are unmeasured.
- No redistribution permission confirmed for the weights.