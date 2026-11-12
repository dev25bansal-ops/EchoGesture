# Public model-weight exposure — inventory and remediation

**Status: UNSATISFIED RELEASE GATE.** This supersedes any earlier claim in this
repository that no restricted weights are publicly available. That claim was
false at repository level, and it was false because it was checked against a
single branch.

Compiled 2026-11-12. Every fact below was read from the remote, not from local
state.

---

## 1. Repository visibility

```
gh repo view dev25bansal-ops/EchoGesture --json visibility
  -> visibility=PUBLIC, private=false, owner=dev25bansal-ops
```

The repository is public, and the owner account is `dev25bansal-ops` (Dev).
Branch refs:

```
refs/heads/main
refs/heads/hackathon/two-person-sprint
refs/heads/hackathon/abhijeet-public
refs/heads/hackathon/dev-model
```

Anyone can `git clone` and read any of these. Git history is also public: once
a blob is committed, deleting it in a later commit does not remove it from
history, and it remains reachable by SHA from the remote.

---

## 2. Publicly accessible model artifacts

### 2a. `hackathon/dev-model` — the serious exposure

| File | Bytes | SHA-256 (first 16) | Identical to what we ship? |
| --- | --: | --- | --- |
| `models/landmark_ens_tcn21_fp16.tflite` | 939,152 | `112279bc86090470` | **yes, byte-identical** |
| `models/landmark_ens_tcnm61_fp16.tflite` | 832,888 | `01a261d8eb6c2c4d` | **yes, byte-identical** |
| `models/landmark_ens_tcnm62_fp16.tflite` | 832,888 | `27058a85e032bee3` | **yes, byte-identical** |
| `models/landmark_ens_rob71_fp16.tflite` | 939,152 | `08417e72da59150a` | **yes, byte-identical** |
| `models/hand_landmarker.task` | 7,819,105 | `fbc2a30080c3c557` | yes — unmodified upstream |
| `models/landmark_classifier_int8.tflite` | 509,720 | `9d24640a23244a3c` | legacy classifier |

All four final ensemble members are public, and they match the shipped artifacts
**byte for byte**. First introduced in `4594bfa` (and a 3-member precursor in
`cdafe12`), so they are in reachable history regardless of working-tree state.

### 2b. `main`, `hackathon/two-person-sprint`, `hackathon/abhijeet-public` — the older exposure

| File | Bytes | SHA-256 (first 16) | Note |
| --- | --: | --- | --- |
| `models/ptq_100_int8.tflite` | 904,256 | `067c743c81c51e4a` | third-party research weights |
| `models/s3d_conv1d_qat_int8.tflite` | 901,200 | `107756f3a5ec118d` | Kimpinde & Olukanmi, *Algorithms* 2026; WLASL-derived |

These are **not** the four members we ship, but they are still model weights with
unresolved redistribution terms, and they are on every branch including `main`.

---

## 3. Participant-identifiable data, also public

`evaluation/asl_citizen_predictions.csv` on `hackathon/dev-model`:

- 1,705 prediction rows
- **11 distinct raw participant identifiers** (`P15`, `P17`, `P18`, `P22`, `P35`,
  `P42`, `P47`, `P48`, `P49`, `P6`, `P9`)
- each row ties a dataset clip filename to a participant ID, a true gloss and a
  prediction

This is a privacy exposure independent of the weights, and it is arguably the
more serious one: ASL Citizen consent covers research use of the video, not
publication of a per-participant evaluation table.

---

## 4. Proven grants vs unresolved permissions

| Artifact | Claim | Evidence in repo | Verdict |
| --- | --- | --- | --- |
| `hand_landmarker.task` | Apache-2.0, Google MediaPipe | Upstream URL recorded in `NOTICE`; file is an unmodified download of the published MediaPipe asset | **Grant exists upstream.** Attribution must ship. The Apache-2.0 text is **not** vendored in the tree — add it. |
| Four `landmark_ens_*_fp16.tflite` | Trained by Dev on ASL Citizen / WLASL-derived data | **None.** No written permission anywhere in the repo. | **UNRESOLVED** |
| `landmark_classifier_int8.tflite` | Same training lineage | None | **UNRESOLVED** |
| `ptoq_100_int8` / `s3d_conv1d_qat_int8.tflite` | Third-party, WLASL-derived | `README.md` previously said "MIT-licensed"; `NOTICE` says no permission. The upstream *code* may be MIT; the *weights* inherit WLASL's academic/non-commercial terms. | **UNRESOLVED** |
| ASL Citizen participant table | — | None | **UNRESOLVED** |

Summary: exactly one artifact class has a grant, and it is the one we did not
train. Everything we derived is unresolved.

---

## 5. Responsibilities

| Who | Owns | Concrete ask |
| --- | --- | --- |
| **Dev** (`dev25bansal-ops`) | Repository owner, model trainer, holder of the dev-model branch | Supply redistribution evidence or confirm there is none; decide on the branch and history |
| **Abhijeet** (integration/submission) | What we publish, what we submit, how evaluators obtain artifacts | Choose the visibility/takedown route and the delivery method |
| **Organizers** | Whether evaluation delivery of weights is covered by competition terms | Ask, in the existing clarification email |

Only the repository owner can change visibility or delete branches. I have not
done either.

---

## 6. Remediation options, ranked

### Option A — Make the repository private, then re-expose code only *(recommended)*

1. Owner sets visibility to **private** (`Settings → General → Danger Zone`).
   This is reversible.
2. Create a **new public repository containing source only** — no `models/`
   binary, no prediction CSVs with participant IDs. Push from a fresh history if
   necessary, so no blob is reachable.
3. Give organizers access to whichever repository the submission requires. A
   private repo can still be supplied: add the evaluator as a collaborator, or
   deliver a read-only archive through the official submission channel.

**Trade-off:** the submission stays open to the "public or shared GitHub repo"
wording on p.25, which explicitly permits a *shared* repository. This is the
only option that actually removes public access to already-committed blobs.

### Option B — Leave the repository public, disclose the exposure

Publish a `RIGHTS.md` stating plainly which artifacts are exposed, that
redistribution permission is unresolved, and that they are present for research
provenance. This is honest but does not fix the problem, and it does not undo
the participant table.

### Option C — History rewrite to purge the blobs

Would remove the weights from history for anyone who clones afterwards. It also
rewrites every SHA, invalidates the commit references in this submission pack,
breaks open PRs, and **does not** recall the objects from anyone who already
cloned. Highest risk, and it requires Dev's explicit agreement as owner.

**Recommendation: A.** It is the only option that removes public access rather
than explaining it.

## 7. What I did not do

- Did not change visibility.
- Did not delete or rewrite any branch.
- Did not force-push.
- Did not publish anything new.

All of these need your decision, and the branch/visibility changes need Dev's
cooperation as repository owner.

---

## 8. Consequence for the submission

Until this is resolved, the correct classification for "model artifacts may be
distributed to evaluators" is **WAITING_FOR_PERMISSION**. Sending the weights to
the organizers is not automatically authorized by the fact that the competition
asked for a working model — check this in the same email.