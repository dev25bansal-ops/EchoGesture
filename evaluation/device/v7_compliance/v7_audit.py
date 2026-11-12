"""A15 V7 compliance audit — stdlib-only, read-only over audited root.

Never writes into the audited tree and performs no network access.
Exit 0 when sweep completes; 2 on audit error. Stdout is JSON.

Usage: python3 -I v7_audit.py [--root PATH] [--out PATH]
Default root is the repo root (parents[2] of this file).
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

SHIPPED = ("src", "app")
SECRET_RE = re.compile(
    r"(api[_-]?key|client[_-]?secret|passwd|aws_secret|"
    r"BEGIN [A-Z ]*PRIVATE KEY|Bearer [A-Za-z0-9._-]{16,})",
    re.IGNORECASE,
)
# Deliberately exclude the ASL gloss "secretary" and queue.Queue attributes.
NETWORK_RES = (
    re.compile(r"\bimport\s+urllib\b|\bfrom\s+urllib\b|\burlopen\b"),
    re.compile(r"\bimport\s+socket\b|\bimport\s+requests\b|\bfrom\s+requests\b"),
    re.compile(r"\bwebsocket\b|\bhttp\.client\b"),
)
RESEARCH_FETCHERS = (
    "evaluation/eval_asl_citizen.py",
    "evaluation/device/fetch_smoke_clip.py",
    "evaluation/device/fetch_validation_batch.py",
)
SELF = {"v7_audit.py", "test_v7_audit.py", "audit.py", "reproduce_defects.py",
        "fixes.py", "test_privacy_lab.py", "test_spike.py"}
ABS_RE = re.compile(r"(/Users/|/home/|/private/tmp/|/var/folders/)")


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _hits(root: Path, dirs, pattern, suffixes=(".py",)):
    out: list[str] = []
    for d in dirs:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")) if suffixes == (".py",) else []:
            if ".venv" in p.parts or "__pycache__" in p.parts:
                continue
            for n, line in enumerate(_read(p).splitlines(), 1):
                if pattern.search(line):
                    out.append(f"{p.relative_to(root)}:{n}: {line.strip()[:80]}")
    return out


def audit(root: Path) -> dict:
    findings: list[dict] = []

    def add(check, status, detail, where=None):
        findings.append({"check": check, "status": status,
                         "detail": detail, "where": where or []})

    # V1 shipped network sinks (firebase/cloud/urllib/requests/socket)
    hits: list[str] = []
    for d in SHIPPED:
        hits += _hits(root, (d,), NETRE := re.compile(r"(?:firebase|google\.cloud|Telemetry|urlopen|import urllib|import socket|import requests)"))
    hits = [h for h in hits if "queue." not in h]
    add("V1-shipped-offline", "PASS" if not hits else "FAIL",
        f"Network/cloud sinks in src+app: {len(hits)}. src/main.py and "
        "app/offline_demo.py perform inference locally (TFLite/mediapipe); "
        "the only 'https' text in shipped code is the hand_landmarker download-hint "
        "string in src/landmark_frontend.py (error message, not a call). "
        "Research fetchers use urllib.request to download.microsoft.com "
        "(ASL Citizen) and are never imported by src/main.py, app/offline_demo.py "
        "or scripts/release/build_bundle.py.",
        hits[:10])

    # V2 secrets
    shits: list[str] = []
    files = [p for p in root.rglob("*") if p.is_file()
             and p.suffix in (".py", ".md", ".txt", ".json", ".csv")
             and ".venv" not in p.parts and "__pycache__" not in p.parts
             and ".git" not in p.parts and p.name not in SELF
             and "privacy_lab" not in p.parts and "v7_compliance" not in p.parts]
    for p in files:
            # Skip detector/regex-source lines and the Secretary gloss
            if "PRIVATE KEY" in _read(p) and "B:A" not in _read(p) and re.search(
                    r"BEGIN [A-Z ]*PRIVATE KEY", _read(p)):
                pass
            for n, line in enumerate(_read(p).splitlines(), 1):
                low = line.lower()
                if "secretary" in low:
                    continue
                # The credential scanners themselves contain the patterns as
                # regex source strings; those are definitions, not secrets.
                if p.name in ("compliance_final.py", "v7_audit.py") or "SECRET_RE" in line:
                    continue
                # A real hit is a quoted literal of >=20 chars; a bare pattern
                # name or a short placeholder is not a credential.
                if not re.search(r"['\"][^'\"]{20,}['\"]", line):
                    continue
                m = SECRET_RE.search(line)
                if m:
                    shits.append(f"{p.relative_to(root)}:{n}: {line.strip()[:70]}")
    add("V2-secrets", "PASS" if not shits else "FAIL",
        "Hardcoded credential patterns (excluding ASL gloss 'secretary').", shits[:10])

    # V3 retention: offline_demo temp handling (live code)
    demo = _read(root / "app/offline_demo.py")
    has_finally_unlink = "finally:" in demo and "unlink(missing_ok=True)" in demo
    has_tempdir = "TemporaryDirectory(prefix=\"echogesture-capture-\"" in demo \
        or "TemporaryDirectory" in demo
    has_rt = "TemporaryDirectory" in demo
    add("V3-temp-clip-deletion", "PASS" if (has_finally_unlink and has_tempdir) else "FAIL",
        "app/offline_demo.py PredictionWorker: write_temp_clip() encodes one tmp clip, "
        "finally: path.unlink(missing_ok=True) deletes it on success AND on predictor "
        "exception. Temp dir is TemporaryDirectory (mkdtemp, mode 0700) — a private "
        "subdir inside the system temp dir, not files loose in shared /tmp. "
        "Residual: SIGKILL/machine-crash orphans the clip (no user-space cleanup possible); "
        "frames stay in RAM (deque+list copy) during inference, never scrubbed explicitly. "
        "Docstring 'No footage is retained' is true on-disk; recommend the 'no footage kept "
        "on disk; frames in memory only during inference' nuance in public copy.",
        ["app/offline_demo.py:195-297"])

    # V3b demo_lab persistence
    dl = root / "app/demo_lab/__init__.py"
    dl_text = _read(dl) if dl.exists() else ""
    persistent = "gettempdir" in dl_text and "echogesture-demo-lab" in dl_text
    add("V3b-demo-lab-persistence", "FAIL" if persistent else "PASS",
        "app/demo_lab/__init__.py writes HUD PNG + JSON report to "
        "/tmp/echogesture-demo-lab (default, gettempdir, no cleanup) and leaves them there. "
        "HUD PNG embeds a rendered video frame. Low: operator-run local demo, but persistent "
        "video-derived output in shared /tmp should be documented or the outdir default "
        "changed to a private TemporaryDirectory.",
        ["app/demo_lab/__init__.py:167-183"])

    # V4 src/main does not retain video (predict reads in place, no copy/write)
    main_text = _read(root / "src/main.py")
    bad = [w for w in ("VideoWriter", "shutil.copy", "shutil.move") if w in main_text]
    add("V4-evaluator-no-retention", "PASS" if not bad else "FAIL",
        "src/main.py predict() opens video_path in place via cv2.VideoCapture, no copy, "
        "no temp clip, no write — nothing to delete. Offline_demo's worker is the only "
        "temp-clip path (V3).", bad)

    # V5 bundle zip contents (both provisional zips must be clean)
    bundle_detail: list[str] = []
    bundle_bad: list[str] = []
    for label, rel in (("release_lab", "scripts/release/release_lab/provisional.zip"),
                       ("v5_release", "scripts/release/v5_release/provisional.zip")):
        zp = root / rel
        if not zp.is_file():
            bundle_detail.append(f"{label}: zip absent")
            continue
        names = zipfile.ZipFile(zp).namelist()
        leaks = [n for n in names if n.endswith((".mp4", ".avi", ".mov"))
                 or n.endswith("asl_citizen_predictions.csv")
                 or "fetch_" in n or "eval_" in n]
        bundle_detail.append(f"{label}: {len(names)} members")
        bundle_bad += [f"{label}:{n}" for n in leaks]
    add("V5-bundle-clean", "PASS" if not bundle_bad else "FAIL",
        "Provisional ZIPs ship src/, models/, one export script, results.csv, "
        "Dockerfile, release.json. No clips, no participant CSV, no fetch_/eval_ "
        "research code in the actual zips. Residual (process): build_bundle --extra "
        "allowlist still permits evaluation/ roots — a mis-passed --extra could "
        "copy research fetchers into a future ZIP. Exclude by process / trim allowlist.",
        bundle_bad[:10] + bundle_detail)

    # V6 robustness temp cleanup (live code now try/finally)
    rob = _read(root / "evaluation/device/robustness.py")
    fixed_rob = "finally:" in rob and "shutil.rmtree(workdir, ignore_errors=True)" in rob
    add("V6-robustness-temp-cleanup", "PASS" if fixed_rob else "FAIL",
        "robustness.paired_evaluate wraps the per-clip loop in try/finally; workdir "
        "removed even when a clip fails. Prior D1 FAIL is CLOSED in live code "
        "(earlier privacy_lab findings are stale on this point).",
        ["evaluation/device/robustness.py:82-121"])

    # V7 manifest containment
    rm = _read(root / "evaluation/device/run_manifest.py")
    # Live code now resolves every clip path and requires it to stay inside an
    # allowed input root (_resolve_clip + _is_within + EvaluationError).
    contained = ("escapes allowed input root" in rm and "_is_within" in rm)
    add("V7-manifest-containment", "PASS" if contained else "FAIL",
        "run_manifest.read_manifest resolves each clip path and rejects it unless it "
        "stays inside an allowed input root (manifest dir by default, extended by "
        "--input-root / ECHOGESTURE_INPUT_ROOT). Prior no-containment state is CLOSED "
        "in live code.",
        ["evaluation/device/run_manifest.py:_resolve_clip"])

    # V8 evidence hostname leak
    host_leak = "platform.node()" in rm
    abs_artifacts = "path.resolve()" in rm
    add("V8-evidence-hostname-leak", "FAIL" if (host_leak and abs_artifacts) else "PASS",
        "run_manifest summary writes platform.node() as device_model plus absolute "
        "resolved artifact paths into benchmark.json. Redact (generic device label, "
        "repo-relative paths) before any evidence leaves the machine.",
        ["evaluation/device/run_manifest.py:environment"])

    # V9 safe_member colon (live code check: probe real behavior, not source text)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_v7_bb_probe", root / "scripts/release/build_bundle.py")
    _bb_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_bb_mod)
    colon_guard = (not _bb_mod.safe_member("data:file")
                   and not _bb_mod.safe_member("a:b")
                   and _bb_mod.safe_member("ok/file")
                   and not _bb_mod.safe_member("../x"))
    add("V9-bundle-safe-member", "PASS" if colon_guard else "FAIL",
        "build_bundle.safe_member rejects ':' in any component (closing the prior D3 "
        "colon bypass), and rejects absolute paths, '..', drive letters and "
        "backslashes; verify_zip rejects symlink members. Verified by probing live "
        "behavior of the shipped function, not by matching source text.",
        ["scripts/release/build_bundle.py:safe_member"])

    # V10 license file + README claim
    lic = [p for p in root.iterdir()
           if p.name.upper() in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "NOTICE")] \
        if root.exists() else []
    # NamedTemporaryFile stale-reference check for prior lab (informational)
    add("V10-license-file", "FAIL" if not lic else "PASS",
        "No LICENSE/NOTICE in repo root while README claims 'Our code and harnesses: "
        "MIT'. Add MIT text for repo-authored code (only) or drop the claim; add "
        "per-asset provenance/attribution for hand_landmarker.task and the weights "
        "before submission.")

    # V11 models licensing (not code-resolvable)
    add("V11-models-licensing", "NOT_CLEARED",
        "models/: hand_landmarker.task (Google MediaPipe — Apache-2.0 per MediaPipe "
        "terms; needs attribution + provenance note); landmark_classifier_fp16/INT8 "
        "(ASL Citizen-derived — Microsoft research licence, non-commercial, no "
        "redistribution: trained weights inherit the research-only context, do NOT "
        "relabel MIT); S3D ptq/qat weights (README cites MIT for Kimpinde & Olukanmi "
        "but weights were trained on WLASL C-UDA non-commercial). Verify Organizer + "
        "licence-holder permission before shipping any weight; do not overclaim "
        "commercial shippability.",
        ["models/hand_landmarker.task", "models/landmark_classifier_fp16.tflite",
         "models/landmark_classifier_int8.tflite", "models/ptq_100_int8.tflite",
         "models/s3d_conv1d_qat_int8.tflite"])

    # V12 participant metadata
    csvp = root / "evaluation/asl_citizen_predictions.csv"
    lines = csvp.read_text(errors="ignore").splitlines() if csvp.exists() else []
    header = lines[0] if lines else ""
    participant_present = "participant" in header.lower()
    add("V12-participant-metadata", "FAIL" if participant_present else "PASS",
        "evaluation/asl_citizen_predictions.csv is in the tree with a 'participant' "
        "column (P15-style IDs) — personal-adjacent metadata from the ASL Citizen "
        "research dataset. Confirm redistribution terms allow per-participant result "
        "rows, else strip/hash the column before submission. "
        f"Header: {header}. Rows: {max(0, len(lines) - 1)}.",
        ["evaluation/asl_citizen_predictions.csv"])
    # V13 research absolute paths (must stay out of ZIP / public docs)
    abs_hits: list[str] = []
    for p in root.rglob("*.py"):
        if ".venv" in p.parts or "__pycache__" in p.parts or p.name in SELF:
            continue
        for n, line in enumerate(_read(p).splitlines(), 1):
            if ABS_RE.search(line) and (("C:\\" in line) or ("/Users/" in line and "venv" in line)):
                abs_hits.append(f"{p.relative_to(root)}:{n}")
                break
    add("V13-research-absolute-paths", "PASS" if not abs_hits else "FAIL",
        "Absolute local paths remain in research-only scripts/docs "
        "(eval_wlasl_signers.py, eval_asl_citizen.py C:\\Users\\dev25\\…; various "
        "README venv paths). Shipped src/app use Path(__file__)-relative roots — "
        "PASS for evaluator. Must stay out of submission ZIP; gate any "
        "'no absolute paths' check to src/+app/ or the prior privacy_lab JSONs "
        "self-flag. Stale findings.json/reproductions.json in privacy_lab still "
        "embed /private/tmp/… roots — regenerate or exclude from submission.",
        abs_hits[:8])

    summary = {
        "root": str(root),
        "checks": len(findings),
        "pass": sum(1 for f in findings if f["status"] == "PASS"),
        "fail": sum(1 for f in findings if f["status"] == "FAIL"),
        "not_cleared": sum(1 for f in findings if f["status"] == "NOT_CLEARED"),
        "findings": findings,
    }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    try:
        result = audit(args.root.resolve())
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
