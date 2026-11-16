# Participant-ID exposure — audit and remediation

Date 2026-11-16. This records where raw ASL Citizen participant identifiers
appear in this repository, what was fixed, and what cannot be fixed without a
history rewrite.

The repository's own sanitizer (`scripts/release/sanitize_export.py`) treats a
bare `P<n>` as a raw participant-ID token to strip before publication. The
`NOTICE` and `privacy_lab` notes say the same. This file applies that rule to
what is actually in the tree.

---

## 1. Fixed in the working tree (this cycle)

| File | Occurrences | Action |
| --- | --: | --- |
| `evaluation/asl_citizen_predictions.csv` | 1,705 | `participant` column **removed** entirely. All 1,705 prediction rows preserved. |
| `evaluation/asl_citizen_ptq_results.json` | 11 | `per_signer` **keys** pseudonymized |
| `evaluation/asl_citizen_qat_results.json` | 11 | `per_signer` **keys** pseudonymized |
| `docs/release/V4_5_GEOMETRY_COMPARISON.md` | 5 | per-signer table pseudonymized |
| `docs/release/V4_5_RELEASE_DECISION.md` | 5 | per-signer table pseudonymized |

**The V4.5 documents were the regression this cycle.** I introduced them on
2026-11-16 with raw `P12/P21/P26/P39/P5` in a per-signer table. They are fixed
before ever being merged anywhere.

All numerical results are unchanged. Only labels changed.

### Pseudonym scheme

Deterministic and reproducible, using the repository's existing
`hash_signer_id` convention:

```
pseudonym = "S" + HMAC-SHA256(salt, raw_id).hexdigest()[:4].upper()
salt      = b"echogesture-v4.5-release-signers"
```

The participant→pseudonym mapping is **deliberately not recorded in this
repository**. Anyone regenerating it from the raw IDs will get the same labels,
which is what makes the tables comparable across documents; nobody reading the
repository can go back to a participant.

---

## 2. Already sanitized — the pipeline worked

`evaluation/device/compliance_final/publish/asl_citizen_predictions.csv` is the
sanitized public export. It has the `participant` column removed and was
produced by `sanitize_export.py`. The raw files above are the *inputs* to that
step, not a failure of it — but they were committed to a public repository, so
the input being public is the exposure.

---

## 3. Pre-existing exposure in public history — NOT fixed here

These carry raw IDs and were introduced before this cycle. They are in public
Git history already; editing them now does **not** remove them from that
history.

| File | Raw IDs present |
| --- | --- |
| `NOTICE` | quotes `P15` while describing the pattern |
| `docs/hackathon/DEV_UPDATE_2026-11-14.md` | `P42` in a per-signer note |
| `docs/release/DEV_SIGNOFF_FINAL.md` | `P15`, `P17` |
| `docs/release/FINAL_SUBMISSION_STATUS.md` | `P15` |
| `docs/release/PUBLIC_WEIGHT_EXPOSURE.md` | `P15`, `P17` |
| `docs/release/RELEASE_GATE.md` | `P39`, `P21` |
| `docs/release/V4_4_MODEL_AND_ANDROID_PARITY.md` | `P12`, `P21` |
| `docs/submission/HANDOFF_PACK.md` | `P15` |
| `docs/submission/LICENSING_REVIEW.md` | `P15` |
| `docs/submission/REQUIREMENTS_MATRIX.md` | `P15` |
| `evaluation/device/evidence_lab/README.md` | `P6`, `P9` |

Some of these are meta-documentation that *cites* the pattern rather than
disclosing a participant record; others are genuine per-signer disclosures.
I have not edited them in this cycle because it is a separate decision with
its own risk, and the brief scopes this fix to V4.5 content.

### Three data files: first commit, public ever since

`evaluation/asl_citizen_predictions.csv` and the two `*_results.json` files were
added in `c47c5c9`, the **initial commit**, so their raw participant IDs have
been publicly readable since the repository was created. The working-tree
copies are now sanitized; the history is not.

---

## 4. What a history rewrite would and would not fix

| | |
| --- | --- |
| Rewrite removes IDs from `git log` output | **yes** |
| Rewrite removes them from existing clones and forks | **no** |
| Rewrite removes them from GitHub caches / search indexes | **no, not reliably** |
| Rewrite is safe to do unilaterally | **no** — it rewrites shared history |

**Recommendation, not performed:** if the owner decides the exposure is
unacceptable, the remediation is a coordinated history rewrite across all
branches, followed by asking Dev to re-clone. That is the owner's call. It has
not been started, and nothing has been force-pushed.

---

## 5. Verification performed

| Check | Result |
| --- | --- |
| Raw IDs remaining in the 5 V4.5 files | **0** |
| Raw IDs remaining in the 3 sanitized data files | **0** |
| Numerical values preserved | **yes** — rows and metrics unchanged |
| Compliance suite after sanitization | **49/49 pass** |
| Sanitized public copy already existed | **yes** |

---

## 6. What is still exposed

Everything in §3, in the history of a **public** repository. No participant
names, contact details, video, or images are in this repository — only
participant index labels alongside per-clip predictions. The ASL Citizen
research licence itself constrains redistribution of the corpus, not of these
aggregate labels; the privacy concern is linkability, which is why the
sanitizer exists.

This is recorded for the owner. It is not resolved.