"""Pixel robustness, parallelised by clip.

`robustness.py --level pixel` is serial: 1,249 clips x 5 corruptions x 32 frames
at 512px projects to roughly 9-10 hours on one core, because every corruption
re-runs MediaPipe over every frame. MediaPipe releases the GIL, so this splits
the clip list across processes and merges the flags.

Corruptions, the model set and the aggregator are all imported from
`robustness.py`, so this file cannot drift from the serial harness. It is a
scheduling change, not a second implementation.

Usage:
    python scripts/ml/robustness_pixel_parallel.py --workers 6 --size 512 --letterbox
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

CKPTS = [
    ".scratch/runs/tcn21.pt", ".scratch/runs/tcnm61.pt",
    ".scratch/runs/tcnm62.pt", ".scratch/runs/rob71.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s0.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s1.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s2.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s7.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s13.pt",
]


def _build_models(ckpts):
    import torch
    from train import TemporalModel

    models, frames = [], None
    for c in ckpts:
        ck = torch.load(ROOT / c, map_location="cpu", weights_only=True)
        frames = int(ck["frames"])
        m = TemporalModel(len(ck["vocab"]), dim=int(ck["dim"]), frames=frames,
                          backbone=str(ck["backbone"]),
                          in_dim=int(ck["model"]["stem.weight"].shape[1]))
        m.load_state_dict(ck["model"])
        m.eval()
        models.append((m, str(ck["backbone"])))
    return models, frames


def score_shard(job):
    """Score every corruption for the clips this worker owns.

    Returns {corruption: [0/1, ...]} in the same clip order the shard received.
    """
    import numpy as np
    import robustness as R
    from landmark_frontend import read_frames, detect_hands

    items, ckpts, size, letterbox, seed = job
    models, frames = _build_models(ckpts)
    videos = ROOT / ".scratch" / "aslcit_vocab_videos"

    out = {}
    for name, fn in R.PIXEL_CORRUPTIONS.items():
        rng = np.random.default_rng(seed)
        flags = []
        for p, _lm, _pres, label, _signer in items:
            src = videos / (p.stem + ".mp4")
            if not src.exists():
                continue
            # size=None here means "use read_frames' default" (256 squash), NOT
            # the native path - the native path is requested explicitly by the
            # caller. So only pass size when one was given.
            kw = {"size": size, "letterbox": letterbox} if size is not None else {}
            fr = read_frames(str(src), max_frames=32, **kw)
            if not fr:
                continue
            if fn is not None:
                fr = fn(fr, rng)
            lm, pres = detect_hands(fr, mode="image")
            if lm.shape[0] == 0 or float(pres.max()) < 0.5:
                flags.append(0)          # no hand found under this corruption
                continue
            pr = R.ensemble_probs(models, lm, pres, frames)
            flags.append(int(pr.argmax() == label))
        out[name] = flags
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--split", default="test")
    # None means "same as the serial harness's default" (FRAME_SIZE = 256,
    # squash). It must NOT default to 512: that silently scores a different
    # geometry than `robustness.py --level pixel` would.
    ap.add_argument("--size", type=int, default=None,
                    help="detector input size (default: FRAME_SIZE = 256, squash)")
    ap.add_argument("--letterbox", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", default=None,
                    help="landmark cache root (default .scratch/landmarks)")
    ap.add_argument("--out", default=None,
                    help="output path (default: results/robustness_pixel_full.json; "
                         "a --limit run defaults to results/robustness_pixel_limit<N>.json)")
    args = ap.parse_args()

    import multiprocessing as mp
    import torch
    import robustness as R

    if args.cache:
        R.CACHE_ROOT = ROOT / args.cache
    ck = torch.load(ROOT / CKPTS[0], map_location="cpu", weights_only=True)
    items = R.load_split(args.split, {g: i for i, g in enumerate(ck["vocab"])})
    if args.limit:
        items = items[: args.limit]

    n = len(items)
    w = max(1, args.workers)
    print(f"{n} clips, {w} workers, size {args.size}, letterbox={args.letterbox}")

    # Strided shards, so each worker sees a representative slice of the split
    # rather than one contiguous block (the split is per-participant ordered).
    jobs = [(items[i::w], CKPTS, args.size, args.letterbox, args.seed)
            for i in range(w)]

    with mp.Pool(w) as pool:
        parts = pool.map(score_shard, jobs)

    merged = {}
    for part in parts:
        for k, v in part.items():
            merged.setdefault(k, []).extend(v)

    clean = merged["clean"]
    # A run that scored nothing is an ERROR, not a 0.00%/PASS measurement. The
    # old `100*sum(v)/max(1,len(v))` turned an empty sample into 0.00% and every
    # row "passed", and the default --out then overwrote the tracked
    # results/robustness_pixel_full.json (the jury-criterion evidence) with an
    # all-zero PASS block, exit code 0. Same defect class as BUG_HUNT H10,
    # which prescribed exactly this guard.
    if not clean:
        sys.exit(f"REFUSING: scored 0 clips (n=0 clean) - check --cache "
                 f"({args.cache!r}) and --split {args.split!r}; no accuracy "
                 f"exists for this input and no artifact was written")
    base = 100 * sum(clean) / len(clean)
    results = {}
    print()
    for k, v in merged.items():
        acc = 100 * sum(v) / max(1, len(v))
        results[k] = {"top1": round(acc, 4), "n": len(v)}
        print(f"  {k:24s} top1 {acc:6.2f}%   n={len(v)}")

    print("\n  drop vs clean (target <=5.00 pts):")
    worst = 0.0
    for k, v in results.items():
        if k == "clean":
            continue
        drop = base - v["top1"]
        worst = max(worst, drop)
        print(f"    {k:24s} {drop:+6.2f} pts  {'PASS' if drop <= 5.0 else 'FAIL'}")
    print(f"\n  worst-case drop: {worst:+.2f} pts  "
          f"({'PASS' if worst <= 5.0 else 'FAIL'})")

    results["_meta"] = {
        "split": args.split, "size": args.size,
        "letterbox": args.letterbox, "workers": w,
        "limit": args.limit,
        "clean_top1": round(base, 4),
        "worst_drop": round(worst, 4),
        "aggregator": "mean-logit (shipped)",
    }
    # A --limit smoke run must not overwrite the tracked full-split artifact
    # unless --out names it explicitly.
    if args.out is None:
        out_rel = (f"results/robustness_pixel_limit{args.limit}.json"
                   if args.limit else "results/robustness_pixel_full.json")
    else:
        out_rel = args.out
    out_path = Path(out_rel)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    if args.limit and out_path.resolve() == (ROOT / "results/robustness_pixel_full.json").resolve():
        print(f"\nWARNING: this is a {args.limit}-clip LIMIT run writing over the "
              f"tracked full-split artifact {out_rel}.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, allow_nan=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()