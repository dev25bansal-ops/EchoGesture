# Licensing review — final check before submission

Compiled 2026-11-12 by Agent C. Project-internal compliance record, not legal
advice. This is a **review**, not a rewrite: nothing here changes history,
licence text, or artifact visibility.

Documents read: `LICENSE`, `NOTICE`, `models/README.md`,
`docs/release/ARTIFACT_RIGHTS.md`, `README.md`, `submission_final/RELEASE_MANIFEST.json`,
plus the actual git trees of every branch on `origin`.

---

## 1. Verdict

| Question | Answer |
| --- | --- |
| Does any file claim a right we do not hold? | **No.** `LICENSE` is deliberately narrow, `NOTICE` §5 lists what is *not* claimed, and every weight row is marked `LOCAL/PRIVATE` / `BLOCKED`. The prose is honest. |
| Is the repo's description of itself consistent with the tree? | **No — one material inconsistency.** `RELEASE_MANIFEST.json` says `"model_weights_published": false` and disclosures say weights "are excluded from the public repository", but **two weight files (1,705,456 B) are tracked on every branch, including `origin/main` and `origin/hackathon/abhijeet-public`.** See §2. |
| Is anything restricted attached to a public surface? | **Yes** — and it is already on GitHub. See §2. |
| Can we ship the submission as planned? | **Yes, with the controls in `HANDOFF_PACK.md` §6.** The restriction binds public redistribution, not delivery to the evaluators. |

---

## 2. FINDING — restricted weights are committed to every branch

This is the one thing that contradicts our own documentation and needs an
owner decision. It is flagged, not fixed: rewriting history is explicitly out
of scope and is a destructive operation.

### 2a. A second, independent contradiction: `README.md` asserts MIT on the S3D weights

`README.md` makes two statements that `NOTICE` §2 contradicts:

| File:line | Wording | `NOTICE` §2 says |
| --- | --- | --- |
| `README.md:106` | `We downloaded the published S3D-Conv1D INT8 weights (**MIT-licensed**, shipped by Kimpinde & Olukanmi, Algorithms 2026)` | `**NO confirmed redistribution permission for these specific weight files. LOCAL/PRIVATE only pending written authorization.** WLASL C-UDA (academic / non-commercial) restriction reattaches.` |
| `README.md:226` | `S3D-Conv1D architecture and weights: MIT (Kimpinde & Olukanmi)` | same |

There is **no licence file or grant in the repo** supporting the README's MIT
claim for those weights — `NOTICE` records the opposite, and the paper's
licence is not vendored. `NOTICE` is the audited document; `README.md` is the
stale one.

This is the closest thing in the repo to "claiming a right we do not hold", and
it is the kind of sentence a reviewer or rights holder notices immediately.
`README.md` is not owned by this workstream, so it is **flagged, not edited**.
**Recommended fix (owner):** change `README.md:106` and `:226` to say the
weights are used under the paper's terms and are **not redistributed**, matching
`NOTICE` §2. This is a two-line edit with no effect on any measurement.

### 2b. Restricted weight files committed to every branch

This is the one thing that contradicts our own documentation and needs an
owner decision. It is flagged, not fixed: rewriting history is explicitly out
of scope and is a destructive operation.

### What is committed

| Path | Bytes | Blob | `NOTICE` §2 status |
| --- | --: | --- | --- |
| `models/ptq_100_int8.tflite` | 904,256 | `371380d9…` | **NO confirmed redistribution permission; local/private only** |
| `models/s3d_conv1d_qat_int8.tflite` | 901,200 | `04453dc8…` | **NO confirmed redistribution permission… LOCAL/PRIVATE only**; WLASL C-UDA non-commercial |
| **Total** | **1,705,456** | | |

### Where they are

```
origin/main                                   models/ptq_100_int8.tflite
                                              models/s3d_conv1d_qat_int8.tflite
origin/hackathon/two-person-sprint            same two files
origin/hackathon/abhijeet-public              same two files   <-- the branch we link as the submission
origin/hackathon/dev-model                    models/hand_landmarker.task
                                              models/landmark_classifier_int8.tflite
                                              models/landmark_ens_{rob71,tcn21,tcnm61,tcnm62}_fp16.tflite
hackathon/abhijeet-integration                same two files
```

These are **real blobs**, not pointers (`git cat-file -s` returns 904,256 and
901,200). `.gitignore` lines 67–68 list `*.tflite` and `*.task`, but ignore
rules do not affect files already tracked, which is how these survived.

### Why it matters

- `NOTICE` §2 says these files have **no** redistribution permission.
- `docs/release/ARTIFACT_RIGHTS.md` marks them **BLOCKED for publication**.
- `RELEASE_MANIFEST.json` asserts `"model_weights_published": false`.
- The repository is public (`https://github.com/dev25bansal-ops/EchoGesture`)
  and `hackathon/abhijeet-public` is the branch the submission pack links.

All four statements cannot be true at once. The tracked files are the fact;
the documentation is wrong about them.

Note the distinction: the **four FP16 ensemble members** — the ones that
actually produce the reportable numbers — are correctly absent from every
public branch. The leaked files are the older **research lineage**
(`ptq_100_int8`, `s3d_conv1d_qat_int8`) that the shipped runtime does not use.

### Owner decision required

This is not mine to make and not mine to execute. There are three honest
options, in ascending order of effort:

1. **Document it.** If the owner judges that the two research files are
   acceptable to leave public, amend `NOTICE` §2, `ARTIFACT_RIGHTS.md`, and
   `RELEASE_MANIFEST.json` to say what is in fact published. Cheapest, and
   keeps the docs truthful.
2. **Untrack going forward.** `git rm --cached` the two files on
   `hackathon/abhijeet-public` and add an explicit ignore rule. Stops new
   clones receiving them; **does not remove them from existing history**, and
   GitHub may still serve blobs by SHA to anyone who has them.
3. **Purge history.** Requires a rewrite (e.g. `git filter-repo`) and a
   force-push to a public branch. Destructive, coordination-heavy, and can
   break the PR. Only worth it if option 1 is judged unacceptable.

`docs/release/ARTIFACT_RIGHTS.md` "Open actions" already lists the repo
visibility question; this review sharpens it from "a public link exposes the
unresolved weights" (it named only the ensemble) to the specific two files.

### Recommendation

Do **option 1 now** so the docs stop asserting something false, and decide
between 2 and 3 after submission. Neither affects the submission itself: the
organizers are being handed the ZIP under the controls in
`HANDOFF_PACK.md` §6 either way.

---

## 3. What the licence files claim, checked against the artifact

| Artifact | Claimed | Supported by evidence in repo? | Verdict |
| --- | --- | --- | --- |
| `src/`, `app/`, `scripts/`, `evaluation/device/` harnesses | MIT (`LICENSE`) | `LICENSE` is an MIT grant explicitly scoped to repo-authored code | **Consistent.** The grant is correctly narrowed and does not purport to cover weights or datasets. |
| `LICENSE` as a whole repo grant | explicitly **not** claimed | `LICENSE` closing paragraph: *"Do not describe a release that contains them as 'MIT licensed' as a whole."* | **Consistent.** No over-claim. |
| 4 × `landmark_ens_*_fp16.tflite` | no redistribution permission | `NOTICE` §2, `models/README.md` `**no** (restricted)` | **Consistent**, and correctly absent from `origin/main` and `hackathon/abhijeet-public`. |
| `models/landmark_labels.json`, `label100.json` | follows the weights | `NOTICE` §2 | **Consistent.** |
| `hand_landmarker.task` | Apache-2.0 claimed | `NOTICE` §2 itself says *"Re-verify current MediaPipe terms"*; only the `mediapipe` **wheel's** LICENSE is in the environment, not the binary's | **Claimed, evidence indirect.** `ARTIFACT_RIGHTS.md` already flags that no Apache-2.0 text or MediaPipe attribution ships inside the ZIP or APK. That is a **fixable omission**, not an unresolved right. |
| ASL Citizen / WLASL datasets | non-redistributable | `NOTICE` §3; upstream licence text not vendored | **Consistent.** Clips and caches are correctly outside Git and outside the ZIP. |
| APK, release ZIP | no public redistribution | both embed restricted weights | **Consistent**, provided `HANDOFF_PACK.md` §6 is followed. |
| `s3d_conv1d_qat_int8.tflite`, `ptq_100_int8.tflite` family | `README.md:106`/`:226` say **MIT-licensed** | no licence text or grant in repo; `NOTICE` §2 records the opposite | **INCONSISTENT** — §2a above. |
| `ptq_100_int8.tflite`, `s3d_conv1d_qat_int8.tflite` | "excluded from the public repository" | **contradicted by the tree** | **INCONSISTENT** — §2b above. |

### No artifact claims a right we do not hold

Every licence statement in `LICENSE`, `NOTICE`, `models/README.md` and
`ARTIFACT_RIGHTS.md` is either a correctly scoped MIT grant, a correctly
scoped third-party attribution, or an explicit statement that a right is
**not** held. The documents err toward disclosing restriction, not toward
over-claiming. The one failure is a **factual** statement about what is
published (§2), not a licence overreach.

---

## 4. Separate pre-existing research artifacts — flagged, not fixed

These are in the public repo, came from the original research phase, and are
**not** needed for the submission. Each carries a redistribution or
attribution question that is a separate owner decision. None changes our
evidence; all are noise on a link an evaluator will click.

| Artifact | Concern | Owner decision |
| --- | --- | --- |
| `clemchris/asl-sign-detection-pytorch-lightning.ipynb` | Kaggle notebook by another author, including its own text: *"Leave an upvote if you find it useful or fork it"*. Kaggle notebook content is under Kaggle's terms, not MIT. | Confirm the notebook may be redistributed, or remove it. |
| `hengck/lb-0-67-one-pytorch-transformer-solution.ipynb` | Same — another Kaggle author's notebook, which itself credits `clemchris`. | Same. |
| `hoyso_infer/1st-place-solution-inference.ipynb`, `hoyso_train/1st-place-solution-training.ipynb` | Third-party competition solution notebooks. | Same. |
| `islr_1st.ipynb` (515 KB, repo root) | A Kaggle notebook (starts `cd ./drive/MyDrive`, installs `git+https://github.com/hoyso48/tf-utils`) by another author, sitting at the repo root as if first-party. | Confirm provenance and licence, or remove. |
| `2nd_place_discussion.md`, `3rd_place.md`, `4th_place.md`, `5th_place.md`, `disc_406684.md`, `leaderboard.md`, `hoyso48_inference.md` | Copies of other competitors' discussion posts and leaderboard text. Under Kaggle terms, not ours. | Confirm or remove. |
| `wlasl100_asllex_vocab_justification.csv` | Derived from WLASL, which is C-UDA non-commercial. `NOTICE` §3 says WLASL-derived data stays outside Git. | **Directly conflicts with `NOTICE` §3.** Move out of Git or amend NOTICE. |
| `evaluation/asl_citizen_predictions.csv` and `evaluation/asl_citizen_{ptq,qat}_results.json` | Carry **raw `P15`-style participant IDs** — ASL Citizen per-participant data under a research licence. `NOTICE` §3 already calls these "LOCAL RESEARCH INPUTS … a separate owner decision". | Publish-safe copies exist at `evaluation/device/compliance_final/publish/`. Decide whether to untrack the raw ones. |

`NOTICE` labels the first six as nothing at all — it never mentions the vendored
notebooks. `ARTIFACT_RIGHTS.md` also does not cover them. If a reviewer reads
the repo, these are the files that look least like ours. **Recommendation for
the owner:** for the submission window, either remove them from the public
branch or add one paragraph to `NOTICE` §3 recording their provenance. Do not
rewrite history over this before 18 November.

---

## 5. What is correctly excluded — verified

| Excluded | How verified |
| --- | --- |
| Four FP16 ensemble members | absent from `origin/main`, `origin/hackathon/abhijeet-public`, `origin/hackathon/two-person-sprint` |
| `hand_landmarker.task` | absent from every public branch (present only on `dev-model`) |
| `landmark_classifier_*` | absent from public branches |
| Raw clips and landmark caches | never tracked; `build_bundle.py` stages an explicit allowlist |
| Demo screenshots / signer images | not tracked |
| Local paths, tokens, keys | `build_bundle.py` scans for absolute-path leaks in `results.csv`, `release.json` and `predictions.jsonl` |

## 6. Submission-window recommendation

1. Follow `HANDOFF_PACK.md` §6: **link** the public repo, **deliver the ZIP
   under access control to the organizers only**, never attach weights to a
   public field.
2. Do not describe any release as "MIT licensed" as a whole — `LICENSE`
   already forbids this and the pack does not do it.
3. Either vendor the Apache-2.0 text and MediaPipe attribution into the ZIP, or
   state in the submission note that the detector is Apache-2.0 from MediaPipe
   and link the upstream URL. `models/README.md` has the URL.
4. Make the two decisions in §2 (document vs untrack) and §4 (notebooks,
   WLASL-derived CSV, participant CSV) after 18 November, not before. Neither
   blocks submission; both should be recorded so they are not lost.