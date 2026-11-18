"""Push the parts of the project a training machine actually needs.

The previous full-tree upload stopped after 2.73 GB of ~16 GB and never reached
`.scratch/landmarks/`, which is the only expensive-to-reproduce artifact here:
3,390 npz files (19 MB) produced by ~1 hour of MediaPipe over the clip corpus,
and byte-exact with respect to the split CSVs. Re-extracting them on the pod is
possible but pointless when the cache already exists.

Also pushes the code, models and results the pod needs to train and evaluate,
so the pod can work from the repo alone.

Every batch is one commit and recorded in a state file, so a failure resumes
instead of re-uploading. Verified after each batch by re-listing the repo.

Usage:
    python scripts/release/hf_push_for_pod.py --dry-run
    python scripts/release/hf_push_for_pod.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ID = "Dev2506/echogesture-full"
REPO_TYPE = "dataset"
STATE = Path(".scratch/hf_pod_push_state.json")

# Ordered by cost of not having them. Landmarks first: everything downstream
# (training, evaluation, the ensemble comparison) needs them and nothing else
# can substitute.
PLAN = [
    (".scratch/landmarks", 120, "hand-landmark cache, 3390 clips"),
    ("models", 40, "shipped TFLite ensemble + labels/meta"),
    ("src", 20, "features.py contract, predict() entry point"),
    ("scripts", 20, "training, extraction, evaluation, release"),
    ("results", 20, "measured metrics"),
    ("docs", 20, "handoff notes"),
    (".scratch/vocab_best100.json", 1, "100-class vocabulary"),
    (".scratch/eval_research/aslcit/splits", 20, "train/val/test split CSVs"),
]


def collect(paths):
    """Expand the plan into (path, size) pairs that exist locally."""
    out = []
    for rel, _, _ in paths:
        p = Path(rel)
        if p.is_file():
            out.append((rel, p.stat().st_size))
        elif p.is_dir():
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = [d for d in dirnames
                               if d not in {"__pycache__", "node_modules"}]
                for f in filenames:
                    fp = Path(dirpath) / f
                    try:
                        if fp.stat().st_size:
                            out.append((fp.as_posix(), fp.stat().st_size))
                    except OSError:
                        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-files", type=int, default=400)
    ap.add_argument("--batch-bytes", type=int, default=300_000_000)
    ap.add_argument("--force", action="store_true",
                    help="re-upload batches recorded as done")
    args = ap.parse_args()

    entries = collect(PLAN)
    total = sum(s for _, s in entries)
    print(f"{len(entries):,} files, {total/1e6:.1f} MB to push\n")
    for rel, limit, why in PLAN:
        n = sum(1 for p, _ in entries if p == rel or p.startswith(rel + "/"))
        b = sum(s for p, s in entries if p == rel or p.startswith(rel + "/"))
        print(f"  {rel:46s} {n:6d} files {b/1e6:8.1f} MB   {why}")

    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    done = set()
    if STATE.exists() and not args.force:
        done = set(json.loads(STATE.read_text()))

    t0 = time.time()
    pushed = 0
    for gi, (rel, _, _) in enumerate(PLAN, 1):
        group = [(p, s) for p, s in entries if p == rel or p.startswith(rel + "/")]
        if not group:
            continue

        # Chunk so one commit stays small enough to retry cheaply.
        batches, batch, bbytes = [], [], 0
        for p, s in group:
            if batch and (len(batch) >= args.batch_files
                          or bbytes + s > args.batch_bytes):
                batches.append(batch)
                batch, bbytes = [], 0
            batch.append(p)
            bbytes += s
        if batch:
            batches.append(batch)

        for bi, batch in enumerate(batches, 1):
            key = f"{rel}#{bi}"
            if key in done:
                print(f"[{gi}/{len(PLAN)}] {rel} {bi}/{len(batches)} - done")
                continue
            size = sum(s for p, s in group if p in set(batch))
            print(f"[{gi}/{len(PLAN)}] {rel} {bi}/{len(batches)}  "
                  f"{len(batch):,} files {size/1e6:.1f} MB", flush=True)
            api.upload_folder(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                folder_path=".",
                path_in_repo="",
                allow_patterns=batch,
                commit_message=f"add {rel} ({bi}/{len(batches)})",
            )
            done.add(key)
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps(sorted(done), indent=2))
            pushed += size
            el = time.time() - t0
            print(f"    {pushed/1e6:.1f} MB pushed in {el/60:.1f} min", flush=True)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    print(f"repo: https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
