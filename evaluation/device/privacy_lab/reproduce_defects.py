"""A15 privacy lab — reproductions for every FAIL in COMPLIANCE_AUDIT.md.

Each function returns (name, reproduced, evidence). No network use except
an explicit, opt-in --allow-network flag that only demonstrates the
fetchers are research-only (never run it in CI). All clip fixtures are
synthetic (solid-colour OpenCV writes) - no real signing footage.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evaluation/device"))
sys.path.insert(0, str(ROOT / "scripts/release"))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))


def repro_robustness_leak() -> dict:
    """D1: robustness workdir survives a decode failure."""
    import robustness
    from corruptions import CorruptionSpec
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bad = root / "bad.mp4"
        bad.write_bytes(b"not a video")
        try:
            robustness.paired_evaluate(
                [("c1", bad, "X", "")],
                lambda p, **k: {"clip_id": "c1", "prediction": "X",
                                "confidence": 1.0, "latency_ms": 1.0},
                {"X"}, CorruptionSpec("low_light"), root / "out")
        except Exception as exc:
            raised = f"{type(exc).__name__}: {str(exc)[:50]}"
        else:
            raised = "no-raise"
        leaked = (root / "out" / "work").exists()
        return {"id": "D1-robustness-workdir-leak",
                "reproduced": bool(leaked),
                "evidence": f"raised={raised!r}; workdir survived={leaked}"}


def repro_manifest_traversal() -> dict:
    """D2: did '../' escape get accepted? False now means containment works."""
    import run_manifest
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sub = root / "sub"
        sub.mkdir()
        outside = root / "outside.mp4"
        outside.write_bytes(b"x")
        manifest = sub / "m.csv"
        manifest.write_text("clip_id,path\nc1,../outside.mp4\n", encoding="utf-8")
        try:
            items = run_manifest.read_manifest(manifest)
        except Exception as exc:  # containment check fired: the defect is closed
            return {"id": "D2-manifest-path-traversal",
                    "reproduced": False,
                    "evidence": f"{type(exc).__name__}: {exc}"}
        return {"id": "D2-manifest-path-traversal",
                "reproduced": bool(items and items[0][1].is_file()),
                "evidence": f"resolved={items[0][1]}"}


def repro_bundle_colon() -> dict:
    """D3: safe_member accepts colon-bearing names (Windows drive-rel)."""
    from build_bundle import safe_member
    verdicts = {n: safe_member(n) for n in ("data:file", "a:b", "ok/file", "../x", "C:/evil")}
    return {"id": "D3-bundle-safe-member-colon",
            "reproduced": bool(verdicts["data:file"]),
            "evidence": json.dumps(verdicts)}


def repro_temp_dir_is_shared() -> dict:
    """D4: offline demo temp files land in the shared system temp dir."""
    import tempfile
    import numpy as np
    from offline_demo import PredictionWorker
    seen: list[Path] = []

    def predict(path, *, clip_id=None):
        seen.append(Path(path))
        return {"clip_id": clip_id, "prediction": "X", "confidence": 0.9, "latency_ms": 1.0}

    worker = PredictionWorker(predict)
    try:
        worker.submit([np.zeros((8, 8, 3), np.uint8)], 12, "c1")
        worker.results.get(timeout=10)
    finally:
        worker.close()
    return {"id": "D4-temp-dir-shared",
            "reproduced": bool(seen) and str(seen[0]).startswith(tempfile.gettempdir()),
            "evidence": f"temp={seen[0] if seen else 'none'}; shared={tempfile.gettempdir()}"}


def repro_temp_deleted_on_exception() -> dict:
    """D5 (control): temp clip IS deleted even when predict raises."""
    import numpy as np
    from offline_demo import PredictionWorker
    seen: list[Path] = []

    def boom(path, *, clip_id=None):
        seen.append(Path(path))
        raise RuntimeError("boom")

    worker = PredictionWorker(boom)
    try:
        worker.submit([np.zeros((8, 8, 3), np.uint8)], 12, "c2")
        kind, _ = worker.results.get(timeout=10)
    finally:
        worker.close()
    import time
    time.sleep(0.2)
    return {"id": "D5-temp-deleted-on-exception (control)",
            "reproduced": kind == "error" and bool(seen) and not seen[0].exists(),
            "evidence": f"kind={kind}; exists_after={seen[0].exists() if seen else 'n/a'}"}


def repro_hostname_in_benchmark() -> dict:
    """D6 (code evidence): benchmark.json embeds platform.node()."""
    src = (ROOT / "evaluation/device/run_manifest.py").read_text(errors="ignore")
    return {"id": "D6-hostname-in-benchmark",
            "reproduced": '"device_model": platform.node()' in src,
            "evidence": "run_manifest.py summary writes platform.node() as device_model"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-network", action="store_true",
                    help="Opt-in: only prints the research fetcher URLs; never downloads.")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    results = [repro_robustness_leak(), repro_manifest_traversal(),
               repro_bundle_colon(), repro_temp_dir_is_shared(),
               repro_temp_deleted_on_exception(), repro_hostname_in_benchmark()]
    if args.allow_network:
        results.append({"id": "NET-research-fetchers",
                        "reproduced": True,
                        "evidence": "fetch_smoke_clip.py + eval_asl_citizen.py "
                                    "use urllib.request against download.microsoft.com; "
                                    "informational only, no request made"})
    out = {"root": str(ROOT), "reproductions": results}
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
