# Hugging Face dataset mirror — what is in it and how to use it

**Repo:** https://huggingface.co/datasets/Dev2506/echogesture-full
**Visibility: PRIVATE.** See [Rights](#rights--do-not-publish) before sharing it.

This is a complete mirror of the EchoGesture working tree, uploaded so the
development environment can be rebuilt byte for byte without a network install.
It is **not** a model release and **not** a redistributable dataset.

Written 12 Nov 2026, corrected 13 Nov 2026.

> **The full-tree upload did not finish.** It stopped after 2.73 GB / 5,751
> files, and never reached `.scratch/landmarks/`. What the repo actually holds
> is recorded under [Actual contents](#actual-contents) below. The 16.9 GB
> figures further down are the *local* measurement of the tree, not what is
> remote; they are kept only to show what an upload would need to carry.

---

## Actual contents (verified 13 Nov 2026)

| | |
| --- | --- |
| Files on the remote | **9,194** |
| Bytes on the remote | **~2.77 GB** |

| path | remote | what it is |
| --- | --- | --- |
| `.scratch/landmarks/{train,val,test}/` | **3,390 npz, complete** | The training input. 1,789 train / 346 val / 1,255 test — exact against the split CSVs |
| `.scratch/aslcit_vocab_videos/` | 3,390 clips | ~1,400 pulled before the download was stopped |
| `.scratch/eval_research/aslcit/` | splits, records | Split CSVs and the participant records |
| `.scratch/claim2_probe/`, `claim3_probe/` | partial | verification probes |
| `models/` | 8 files, complete | The shipped four-member FP16 ensemble + labels + meta + detector |
| `src/`, `scripts/`, `results/`, `docs/` | complete | Code, measurements, handoff notes |
| `.scratch/vocab_best100.json` | complete | The 100-class vocabulary |
| `.scratch/s3d_repro/`, most of `eval_research/` | **absent** | 13 GB of venvs and research sweeps — reinstallable, not needed to train |

`scripts/release/hf_push_for_pod.py` is what completed the landmark and code
push. It is resumable per batch, so rerunning it only sends what is missing.

### Verified end to end

A fresh PAI-DSW A10 instance pulled the repo with

```python
snapshot_download("Dev2506/echogesture-full", repo_type="dataset",
    allow_patterns=[".scratch/landmarks/*/*.npz", ".scratch/vocab_best100.json",
                    ".scratch/eval_research/aslcit/splits/*.csv",
                    "models/*", "src/*", "scripts/*", "results/*", "docs/*"],
    local_dir="/mnt/workspace/eg")
```

and reproduced the shipped numbers exactly from that download alone:

```
val  289/345  = 83.77%
test 909/1249 = 72.78%
```

See [POD_SETUP.md](POD_SETUP.md) for the full recipe.

---

## Local tree size, measured

| | |
| --- | --- |
| Local tree, everything | **17,466,790,331 bytes** (16.27 GiB) across 348,004 files |
| Uploaded scope | **16,905,750,619 bytes** (16.91 GB) across 181,083 files |
| Excluded | `__pycache__`, `node_modules`, and the Hub client's own staging directory |

The difference between the two figures is bytecode and caches — nothing that
cannot be regenerated.

## What is inside

| path | size | files | what it is |
| --- | --: | --: | --- |
| `.scratch/eval_research/` | 7.24 GB | 19,168 | Research sweeps: ASL Citizen runs, paper artifacts, similarity/handbook lookups, a 2.3 GB `semlex/test_full.tar.gz` |
| `.scratch/s3d_repro/` | 5.50 GB | 60,563 | The S3D-Conv1D reproduction environment: two Python venvs (TF + PyTorch), notebooks, INT8 experiments, licence text |
| `.scratch/aslcit_vocab_videos/` | 2.05 GB | 3,390 | **The ASL Citizen clips** for our 100 glosses. Restricted — see Rights |
| `_altbench/pinto_hand/` | 0.95 GB | 2 | Third-party hand-detection model archives, for the benchmark comparison |
| `.scratch/venv311_check/` | 0.38 GB | 4,331 | Clean Python 3.11 runtime used to verify the shipped bundle |
| `.scratch/claim2_probe/`, `claim3_probe/` | 0.33 GB | 1,894 | Verification probes for paper claims |
| `_altbench/` (other) | 0.15 GB | — | RTMPose, RTMDet, TinyCNN comparison models |
| `.scratch/landmarks/` | 0.02 GB | 3,390 | **Extracted landmark caches** — one `.npz` per clip, the actual training input |
| `.scratch/runs/` | 0.04 GB | 22 | PyTorch training checkpoints, the four shipped members included |
| `models/` | 0.01 GB | 8 | The shipped four-member FP16 TFLite ensemble + labels + meta + detector |
| `.scratch/apk_out/` | 0.08 GB | 1 | The built `EchoGesture-debug.apk` |
| everything else | <0.1 GB | — | `scripts/`, `src/`, `results/`, `docs/`, `evaluation/`, `.git/` |

## What Abhijeet actually needs

Most of the 16.9 GB is development history. For the submission, **four things
matter** and everything else is optional:

1. **`models/`** — the four FP16 members, `landmark_labels.json`,
   `landmark_model_meta.json`, `hand_landmarker.task`. 11.4 MB total.
2. **`src/`** — `main.py`, `features.py`, `landmark_frontend.py`, and the
   `predict(video_path) -> dict` contract.
3. **`.scratch/landmarks/`** — the extracted features, if he wants to
   re-evaluate accuracy without running MediaPipe over 3,390 clips (35 min).
4. **`.scratch/apk_out/EchoGesture-debug.apk`** — the built Android app,
   75 MB, with all four members embedded.

If bandwidth or quota is tight, those four are about **2.1 GB**, and the rest
can be skipped with `--exclude`.

## How to download

Everything:

```bash
hf download Dev2506/echogesture-full --repo-type dataset --local-dir ./echogesture
```

Only the shipping subset (recommended for Abhijeet):

```bash
hf download Dev2506/echogesture-full --repo-type dataset \
  --include "models/*" "src/*" "results/*" \
  --local-dir ./echogesture
```

The video clips specifically:

```bash
hf download Dev2506/echogesture-full --repo-type dataset \
  --include ".scratch/aslcit_vocab_videos/*" --local-dir ./echogesture
```

You need to be authenticated and be a collaborator on the repo:

```bash
hf auth login
```

## Verification

The upload is committed in batches, so the remote file count climbs over time
while the total bytes stay correct only once it finishes. To check:

```python
from huggingface_hub import HfApi
fs = HfApi().dataset_info("Dev2506/echogesture-full").siblings
print(len(fs), sum(f.size or 0 for f in fs))
```

Compare against the [actual contents](#actual-contents-verified-13-nov-2026)
above — **9,194 files**. The 181,083-file figure this section used to name was
the intended scope of the full-tree upload, not what landed.

Spot-check the models against the handoff:

```bash
sha256sum echogesture/models/landmark_ens_*.tflite
```

They must match `docs/hackathon/DEV_HANDOFF_MODEL.md` — `112279bc…`, `01a261d8…`,
`27058a85…`, `08417e72…`.

## Rights — do not publish

**The repository is private for a reason.** It contains:

- **3,390 ASL Citizen video clips** (2.05 GB). ASL Citizen is under a
  **Microsoft Research non-commercial** licence, signer-consented for research.
- **Landmark caches derived from those clips.** They are derived works and
  inherit the same restriction.
- **Third-party model archives** under `_altbench/`, with their own terms.

`NOTICE` and `docs/release/ARTIFACT_RIGHTS.md` record this. There is **no
written permission covering redistribution** of any of it.

Private storage is not publication. **Making this repository public would be
redistribution of licensed research data.** Do not do it without written
permission from the rights holders — not to demo, not to share with a judge.

The four shipped TFLite members derive from the same data and carry the same
restriction. That is why `hackathon/abhijeet-public` ships them separately and
does not commit them.

## What this is not

- Not a model card. The shipped model is documented in `RESULTS.md` and
  `docs/hackathon/DEV_HANDOFF_MODEL.md`.
- Not a clean dataset release. It is a working-tree snapshot with venvs, caches
  and intermediate artifacts included, because reproducing the environment
  exactly was the point.
- Not reproducible across machines for the venvs — they are Windows binaries.
  Reinstall from `src/requirements.txt` elsewhere; the pin they encode is in
  that file.

## Regenerating the upload

`scripts/release/hf_upload_all.py` does this. It is resumable: each batch is one
commit and `.scratch/hf_upload_state.json` records what landed, so a dropped
connection resumes instead of restarting 16.9 GB.

```bash
python scripts/release/hf_upload_all.py --dry-run   # plan only
python scripts/release/hf_upload_all.py             # upload
```

Measured throughput on this machine: **4.3 MB/s** on a single file, **0.6–1.3
MB/s** averaged across small-file batches. Expect a few hours for a full run.