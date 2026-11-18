"""Measure the full predict() path: cold start, warm per-clip, and stage split.

`results.csv` carries latency_end_to_end_p50 = 4,650 ms and latency_cold_start =
159,000 ms, both measured on 16 Nov while a `distllm` inference server from a
different project was saturating the CPU. Those figures are almost certainly
contaminated the same way the per-frame number was (133 ms -> 15 ms on re-measure).

This re-measures both, and additionally reports whether the machine was under
load, so a reader can tell a clean number from a contended one.

What it times, in the order the evaluator sees them:
  cold   — a fresh process: import + _init_model() + first predict()
  warm   — subsequent predict() calls, the timer inside predict() is what is scored
  stages — decode / detector / features / classifier, so a failure localises

Usage:
    python scripts/ml/latency_end_to_end.py --clips 6
    python scripts/ml/latency_end_to_end.py --clips 6 --affinity 20-27   # pin away
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
VIDEOS = ROOT / ".scratch" / "aslcit_vocab_videos"
OUT = ROOT / "results" / "latency_end_to_end.json"


def machine_load():
    """Relative CPU busy-ness, so a contended measurement is labelled as such."""
    try:
        import psutil
        return {"cpu_percent": psutil.cpu_percent(interval=0.5),
                "cpu_count": psutil.cpu_count()}
    except Exception:
        return {"cpu_percent": None, "cpu_count": os.cpu_count()}


def competing_processes():
    """Name-count of other heavy python processes, for the record."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object {$_.CommandLine -like '*distllm*'} | Measure-Object | "
             "Select-Object -ExpandProperty Count"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out or 0)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--affinity", default=None,
                    help="CPU range to pin to, e.g. 20-27; leaves other cores to the culprit")
    ap.add_argument("--cold-timeout", type=float, default=400.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    pinned = False
    if args.affinity:
        try:
            import psutil
            lo, hi = args.affinity.split("-")
            cores = list(range(int(lo), int(hi) + 1))
            psutil.Process().cpu_affinity(cores)
            pinned = True
            print(f"pinned to cores {cores}")
        except Exception as e:
            print(f"affinity not applied: {e}")

    clips = sorted(glob.glob(str(VIDEOS / "*.mp4")))[: args.clips]
    if not clips:
        sys.exit(f"no clips under {VIDEOS}")

    load_before = machine_load()
    competing = competing_processes()

    t_import = time.perf_counter()
    import main as M
    import_time_ms = (time.perf_counter() - t_import) * 1000.0

    # ---- cold start: includes the MediaPipe graph build, which now happens
    # inside _init_model() rather than in the first detect().
    t_init = time.perf_counter()
    M._init_model()
    init_ms = (time.perf_counter() - t_init) * 1000.0

    t_cold0 = time.perf_counter()
    first = M.predict(clips[0])
    cold_ms = (time.perf_counter() - t_cold0) * 1000.0

    # ---- warm: the number predict() itself reports
    warm = []
    for c in clips:
        r = M.predict(c)
        warm.append(r["latency_ms"])
    warm.sort()
    n = len(warm)
    p50 = statistics.median(warm)
    p95 = warm[min(n - 1, int(round(0.95 * (n - 1))))]

    # ---- stage split on the last clip. Decode and detection are timed
    # SEPARATELY: clip_to_landmarks = read_frames + detect_hands, and the
    # committed artifact's hand-assembled `stage_split_ms.detect` (1,136 ms)
    # actually included decode while the csv note paired it with the isolated
    # detector-only 20.3 ms/frame - two different quantities under one name.
    lm = None
    from landmark_frontend import read_frames, detect_hands
    from features import build_features
    n_frames = max(32, M._FRAMES)
    t0 = time.perf_counter()
    frames = read_frames(clips[-1], max_frames=n_frames)
    t_decode = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    lm, pres = detect_hands(frames, mode="image")
    t_detect = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    built = {}
    for uz in set(M._MEMBER_USE_Z):
        built[uz] = build_features(lm, pres, expected_frames=M._FRAMES, use_z=uz)
    t_feat = (time.perf_counter() - t0) * 1000.0

    # The nine-member classifier alone, on the features just built - the
    # quantity results.csv's latency_model_only row refers to.
    import numpy as _np
    t0 = time.perf_counter()
    for interp, (in_idx, out_idx, mask_idx), uz in zip(
            M._INTERPRETERS, M._MEMBER_IO, M._MEMBER_USE_Z):
        f, mk = built[uz]
        interp.set_tensor(in_idx, f[None, ...].astype(_np.float32))
        if mask_idx >= 0:
            interp.set_tensor(mask_idx, mk[None, ...].astype(_np.float32))
        interp.invoke()
    t_cls = (time.perf_counter() - t0) * 1000.0

    load_after = machine_load()

    rec = {
        "measured": time.strftime("%Y-%m-%d"),
        "clips": len(clips),
        "pinned_cores": args.affinity,
        "competing_distllm_processes": competing,
        "load_before": load_before, "load_after": load_after,
        "import_ms": round(import_time_ms, 1),
        "init_ms_incl_mediapipe_graph": round(init_ms, 1),
        "cold_first_predict_ms": round(cold_ms, 1),
        "cold_start_total_ms": round(import_time_ms + init_ms + cold_ms, 1),
        "warm_p50_ms": round(p50, 2),
        "warm_p95_ms": round(p95, 2),
        "warm_min_ms": round(warm[0], 2),
        "warm_max_ms": round(warm[-1], 2),
        "warm_all_ms": [round(x, 2) for x in warm],
        # One clip, same process, timed in sequence. decode + detect + features
        # + classifier is the decomposition of a single predict() pass - unlike
        # the artifact this replaces, whose stage keys did not exist in any
        # committed version of this script.
        "stage_split_ms": {"decode": round(t_decode, 1),
                           "detect": round(t_detect, 1),
                           "features": round(t_feat, 2),
                           "classifier_nine_members": round(t_cls, 2)},
        "stage_split_note": ("single last clip, timed in sequence after the warm "
                             "loop; decode and detect are separate stages"),
        "stage_detect_ms": round(t_detect, 1),
        "stage_features_ms": round(t_feat, 2),
        "stage_classifier_ms": round(t_cls, 2),
        "first_prediction": first.get("prediction"),
        "note": ("Warm latency_ms is the value predict() reports and the one the "
                 "evaluator scores. Cold start is one-time per process. If "
                 "competing_distllm_processes > 0 the CPU was shared; re-run with "
                 "--affinity to confirm."),
    }
    Path(args.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(f"\ncompeting distllm procs: {competing}   cpu load: "
          f"{load_before.get('cpu_percent')}% -> {load_after.get('cpu_percent')}%")
    print(f"  import                 {import_time_ms:8.0f} ms")
    print(f"  init (incl MP graph)   {init_ms:8.0f} ms")
    print(f"  first predict (cold)   {cold_ms:8.0f} ms  -> {first.get('prediction')}")
    print(f"  cold start TOTAL       {import_time_ms+init_ms+cold_ms:8.0f} ms (one time)")
    print(f"  warm p50 / p95         {p50:8.1f} / {p95:.1f} ms   (scored)")
    print(f"    range                {warm[0]:.1f} .. {warm[-1]:.1f} ms")
    print(f"  stage (one clip): decode {t_decode:.0f} ms | detect {t_detect:.0f} ms "
          f"| features {t_feat:.1f} ms | classifier {t_cls:.1f} ms")
    print(f"\nwrote {args.out}")
    if pinned:
        print("(pinned run — compare against an unpinned one)")


if __name__ == "__main__":
    main()
