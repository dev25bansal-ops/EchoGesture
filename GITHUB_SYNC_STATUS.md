# GitHub sync status

Date **2026-11-14**. Deadline 2026-11-18.

Every claim below was read back **from the remote ref**, not from the local
worktree. `git ls-remote` and `git cat-file -e origin/<branch>:<path>` are the
only sources treated as evidence that GitHub has something.

---

## 1. Headline

| | |
| --- | --- |
| Repositories with unpushed unique commits | **none** |
| Commits reachable only from this machine | **0** |
| Files staged because they were unreviewed | 52 |
| Restricted binaries newly pushed | **0** |
| Push status for this session's work | **SUCCESS — remote HEAD verified** |

---

## 2. What was pushed this session

Branch **`hackathon/abhijeet-integration`**, commit
`1be2cab9d04435e29f1a4aba99723994a706db85`.

```
d67d468..1be2cab  hackathon/abhijeet-integration -> hackathon/abhijeet-integration
```

**Remote verification**

| Check | Result |
| --- | --- |
| `git ls-remote origin refs/heads/hackathon/abhijeet-integration` | `1be2cab9d04435e29f1a4aba99723994a706db85` |
| Local `HEAD` | `1be2cab9d04435e29f1a4aba99723994a706db85` |
| **Match** | **YES** |
| Files in the commit present in the remote tree | **52 / 52**, 0 missing |
| `git diff 1be2cab origin/…` | empty — trees identical |

Included: ensemble and robustness evidence, staged performance profiling, the
final QA audit, unified release-candidate captures, demo runbook and shot list,
the PR #2 pre-merge gate, the published-handoff record, the organizer
correspondence README, and an APK-evidence record.

Also included: `.gitignore` now blocks `/dist/`, `*.apk` and `*.aab`.

### Deliberately excluded

`test-report.json` — a machine-generated artifact whose inputs are not
reproducible from this tree; it references suite directories by name only. Left
untracked, not committed and not deleted.

---

## 3. Privacy work done before committing

The repo's own sanitizer (`scripts/release/sanitize_export.py`) defines a bare
`P<n>` as a **raw participant-ID token** to strip before publication, and the
QA audit in this same tree had raised that exposure as its item #1, severity
**High**.

Three of the files pending commit carried raw ASL Citizen participant
identifiers in prose. They were rewritten before their **first** commit, so the
raw IDs were never published:

| File | Change |
| --- | --- |
| `evaluation/device/ensemble_evidence/RESULT.md` | Per-signer table relabelled S1–S11; a note states the raw labels are held locally |
| `evaluation/device/robustness_ensemble/REPORT.md` | Signer list replaced with "5" |
| `evaluation/device/qa_final3/QA_REPORT.md` | §6.2 and open item #1 updated to record the resolution |

Accuracy figures were preserved exactly — only the labels changed.

The seven `robustness_ensemble/results/*.json` files already used
`signer_<hash>` pseudonyms, matching the convention in tracked code
(`v7_int8/test_v7_int8.py`: *"Signer keys are pseudonymized (signer_&lt;hash&gt;) in
published evidence"*), so they were committed unchanged.

A scan of staged content from the index returned **no** raw participant IDs and
**no** secrets. The single secret-pattern hit is the QA report's own sentence
stating that no secrets exist.

---

## 4. Pre-commit gate

| Check | Result |
| --- | --- |
| `git diff --cached --check` | PASS |
| Restricted binaries staged (`.tflite`, `.task`, `.apk`, `.aab`, video, `.npy`) | **none** |
| Raw participant IDs in staged content | **none** |
| Secrets in staged content | **none** |
| Largest staged file | 892 KB (emulator screenshot) |
| Total staged | 1.91 MB |

Staging was explicit, file by file and directory by directory. **`git add .` was
not used.**

---

## 5. Restricted assets — not newly pushed

`origin/hackathon/abhijeet-integration` contains exactly two `.tflite` files,
`models/ptq_100_int8.tflite` and `models/s3d_conv1d_qat_int8.tflite`. Both
predate this work (present on `origin/main` since the initial commit) and
**neither was added by this commit**:

```
git diff-tree --no-commit-id --name-only -r 1be2cab | grep -Ei '\.(tflite|task|apk|aab)$'
→ (empty)
```

`.gitignore` blocks `*.tflite`, `*.apk`, `*.aab` and `/dist/`, so a future build
cannot be committed by accident.

**These two pre-existing files remain an unresolved rights question.** They are
in public history already; deleting them in a new commit would not remove them
from that history. Remediation needs a history rewrite, which is the repository
owner's decision and was not performed. See §8.

---

## 6. Branch inventory — nothing stranded

Three local branches reported "unpushed" commits against a naive tracking count.
That count was misleading — those branches simply have no upstream configured.
Measured properly:

```
git rev-list <branch> --not --remotes
```

| Branch | Commits unreachable from **any** remote ref |
| --- | --- |
| `diag/live-phone-recognition-v3` | **0** |
| `diag/training-clip-repro` | **0** |
| `hackathon/solo-final-submission` | **0** |
| `fix/true-video-recognition-v4` | 0 (in sync) |
| `fix/v4-4-final-release` | 0 (in sync) |
| `hackathon/abhijeet-public`, `hackathon/echogesture-v2-rc`, `main` | 0 (in sync) |

**Every local commit is reachable from at least one remote branch.** Nothing is
stranded on this machine.

---

## 7. Where each piece of work actually lives

This is the finding that matters most, and it is **not** what a single-branch
check would suggest.

| Work | Branch | On remote |
| --- | --- | --- |
| **Nine-member masked evaluator** (`src/main.py` name-resolved two-input, nine pinned hashes, mask plumbing; `src/features.py` 381-wide) | **`fix/v4-4-final-release`** | **YES**, verified |
| V4.4 release reports (evaluator fix, parity, provenance, compliance, status) | **`fix/v4-4-final-release`** | **YES**, all five present |
| Nine-member release-bundle test fixture | **`fix/v4-4-final-release`** | YES |
| V4.2 segmentation work | `fix/true-video-recognition-v4` | YES |
| V4.4 evidence, demo, QA records | `hackathon/abhijeet-integration` | YES (this session) |

### `hackathon/abhijeet-integration` still carries the **old** evaluator

```
git show origin/hackathon/abhijeet-integration:src/main.py | grep 'get_input_details()\[0\]'
→ 315:  in_details = interp.get_input_details()[0]
→ 357:  new_input_details = new_single.get_input_details()[0]
```

And it has **0** matches for the nine pinned member hashes. It is a genuine
ancestor of the release line — 51 commits behind it — with 3 commits of its own
(the two pre-existing `docs(private)` records plus this session's evidence
commit).

**This is deliberate, not a sync failure.** The brief for this phase forbids
merging into `main` or Dev's branch automatically. `fix/v4-4-final-release` is
where the working evaluator was built and verified; the integration branch is
where its evidence is published. Reconciling them is a review decision for the
branch owners.

---

## 8. Reproducibility from the remote

| Requirement | Status |
| --- | --- |
| Authorised asset provisioning documented | **YES** — `models/README.md` and `NOTICE` present on the remote and state that weights are supplied separately, with the SHA-256 to verify |
| Bundle builder present | **YES** — `scripts/release/build_bundle.py` |
| Python tests present | 46 test files |
| Kotlin tests present | 10 test files (on the integration branch) |
| Weights in Git | **NO, by design** — `*.tflite` gitignored |

A clean clone reproduces the code and the harness. It does **not** reproduce
inference, because the weights are intentionally not in Git. That is the
documented, authorised provisioning model, not a missing step.

**Caveat:** on `hackathon/abhijeet-integration` the release-bundle test fixture
is the older four-member form. The nine-member fixture is on
`fix/v4-4-final-release`. Anyone running the bundle suite against the integration
branch should expect it to exercise four members.

---

## 9. What is still blocked — all human

| # | Item | Owner action |
| --- | --- | --- |
| 1 | **Physical phone test** | Attach a device; run `docs/release/PHONE_GATE_INSTRUCTIONS.md` |
| 2 | **Rights decision** on the two pre-existing `.tflite` files in public history | Repository owner |
| 3 | **Demo video** ≤3 min and the presentation | Human |
| 4 | **Exact `CollegeName_TeamName`** | Human |
| 5 | **Google Form approval** | Explicit sign-off required |

None of these was worked around, faked, or marked complete.

---

## 10. Honest summary

**What is verified now:** every local commit is reachable from GitHub; this
session's 52 files are on `origin/hackathon/abhijeet-integration` and its HEAD
matches local exactly; no restricted binaries were added; the privacy item the
tree's own QA audit flagged as High was closed **before** first publication.

**What is not:** no physical phone has run this. INT8 and the ≤50 ms criterion
remain **FAIL**, not unmeasured. The nine-model evaluator is verified on
`fix/v4-4-final-release`, not on the integration branch, and the two were left
unmerged on purpose.