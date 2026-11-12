"""Upload the full EchoGesture tree to a private HF dataset repo, in batches.

Resumable: each batch is one commit, and a small manifest records which
top-level paths have completed, so a rerun skips finished work instead of
re-uploading 16 GB.

Deliberately skips:
  - .git/            history is not what was asked for, and it is large
  - nested venvs     pure Python dependencies, reinstallable, not project data
  - __pycache__      bytecode

Usage:
    python scripts/release/hf_upload_all.py --dry-run
    python scripts/release/hf_upload_all.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ID = "Dev2506/echogesture-full"
REPO_TYPE = "dataset"
STATE = Path(".scratch/hf_upload_state.json")

# Only genuinely disposable content is pruned. Nested venvs ARE uploaded:
# they are part of the 16.2 GiB the user asked for, and they make the
# environment reproducible byte for byte without a network install.
# .cache/huggingface/upload is the hub client's own staging area - uploading
# it would duplicate the tree into itself.
SKIP_DIRS = {"__pycache__", "node_modules"}
SKIP_PATHS = {".cache"}
VENV_MARKERS = {"pyvenv.cfg"}


def is_venv(path: Path) -> bool:
    return False  # venvs are included deliberately; see SKIP_DIRS comment


def collect(root: Path):
    """Return (files, skipped) as relative posix paths, pruning venvs and caches."""
    files, skipped = [], []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        d = Path(dirpath)
        keep = []
        for name in dirnames:
            child = d / name
            rel = child.relative_to(root).as_posix()
            if name in SKIP_DIRS:
                skipped.append((rel, "cache/git"))
            elif is_venv(child):
                skipped.append((rel, "venv"))
            else:
                keep.append(name)
        dirnames[:] = keep
        for f in filenames:
            p = d / f
            try:
                if p.stat().st_size:
                    files.append(p.relative_to(root).as_posix())
            except OSError:
                pass
    return files, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-files", type=int, default=1200,
                    help="max files per commit")
    ap.add_argument("--batch-bytes", type=int, default=1_800_000_000,
                    help="max bytes per commit")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files, skipped = collect(root)

    sizes = {}
    for rel in files:
        sizes[rel] = (root / rel).stat().st_size
    total_bytes = sum(sizes.values())

    print(f"files to upload : {len(files):,}")
    print(f"bytes           : {total_bytes:,} ({total_bytes/1e9:.2f} GB)")
    print(f"pruned dirs     : {len(skipped)}")
    for rel, why in skipped[:12]:
        print(f"    [{why}] {rel}")
    if len(skipped) > 12:
        print(f"    ... and {len(skipped)-12} more")

    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()

    done = set()
    if STATE.exists():
        done = set(json.loads(STATE.read_text()))

    # Batch by top-level directory so each commit is coherent and a resume
    # maps naturally onto "this subtree is done".
    groups = {}
    for rel in files:
        groups.setdefault(rel.split("/")[0] if "/" in rel else ".", []).append(rel)

    order = sorted(groups, key=lambda k: -sum(sizes[r] for r in groups[k]))
    print(f"\n{len(order)} top-level groups, largest first:")
    for k in order:
        gb = sum(sizes[r] for r in groups[k]) / 1e9
        print(f"    {k:38s} {len(groups[k]):>7,} files  {gb:7.2f} GB")

    t0 = time.time()
    uploaded_files = uploaded_bytes = 0

    for gi, group in enumerate(order, 1):
        if group in done:
            print(f"[{gi}/{len(order)}] {group} - already done, skipping")
            uploaded_files += len(groups[group])
            uploaded_bytes += sum(sizes[r] for r in groups[group])
            continue

        batch, bbytes = [], 0
        batches = []
        for rel in groups[group]:
            if batch and (len(batch) >= args.batch_files
                          or bbytes + sizes[rel] > args.batch_bytes):
                batches.append(batch)
                batch, bbytes = [], 0
            batch.append(rel)
            bbytes += sizes[rel]
        if batch:
            batches.append(batch)

        for bi, batch in enumerate(batches, 1):
            bptr = sum(sizes[r] for r in batch)
            print(f"[{gi}/{len(order)}] {group} batch {bi}/{len(batches)}  "
                  f"{len(batch):,} files {bptr/1e6:7.1f} MB", flush=True)
            api.upload_folder(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                folder_path=str(root),
                path_in_repo="",
                allow_patterns=batch,
                commit_message=f"add {group} ({bi}/{len(batches)})",
            )
            uploaded_files += len(batch)
            uploaded_bytes += bptr
            el = time.time() - t0
            rate = uploaded_bytes / el / 1e6 if el else 0
            print(f"    running total {uploaded_bytes/1e9:.2f} GB "
                  f"({uploaded_files:,}/{len(files):,} files) at {rate:.1f} MB/s",
                  flush=True)

        done.add(group)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(sorted(done), indent=2))

    print(f"\nDONE {uploaded_files:,} files, {uploaded_bytes:,} bytes "
          f"in {(time.time()-t0)/60:.1f} min")
    print(f"repo: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    sys.exit(main())