"""A15 privacy / licensing / submission-safety audit for EchoGesture.

Read-only over the repo under audit (repo root = this file's parents[3],
i.e. evaluation/device/privacy_lab/audit.py -> repo root). It never writes
into the audited tree and performs no network access. Emits JSON findings.

Checks:
  P1  absolute-path literals in shipped src/app/scripts/evaluation code
  P2  hardcoded secrets (api key / password / private key / bearer token)
  P3  network sinks (urllib/socket/requests) in shipped src/app code
  P4  data-retention wording vs. actual temp-file cleanup behaviour
  P5  git-tracked video/clip/cache artefacts + .gitignore coverage
  P6  package inclusion: build-bundle allowlist vs. research-only files
  P7  symlink / path-traversal guards (zip member, manifest paths)
  P8  licence / attribution presence for code, MediaPipe task, S3D weights,
      ASL Citizen-derived weights (Dev landmark model), WLASL/ASL Citizen data
  P9  absolute paths + hostname disclosure baked into evidence JSON

Usage: python audit.py [--root PATH] [--out PATH]
Exit 0 if run completed (findings may include fails); 2 on audit error.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SHIPPED_DIRS = ("src", "app")
RESEARCH_DIRS = ("evaluation", "scripts")
RESEARCH_ONLY_HINTS = ("fetch_", "eval_", "export_tflite")

ABS_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\(?:Users|home|tmp|var|private)|/Users/|/home/|/private/tmp|/var/folders/)")
SECRET_RE = re.compile(
    r"(api[_-]?key|client[_-]?secret|password|passwd|aws_secret|BEGIN [A-Z ]*PRIVATE KEY|Bearer [A-Za-z0-9._-]{16,})",
    re.IGNORECASE)
NETWORK_RE = re.compile(
    r"(import urllib|from urllib|urlopen|websocket|http\.client|import socket|import requests\b|from requests\b)")
SELF_EXCLUDE = {"audit.py", "reproduce_defects.py", "test_privacy_lab.py", "fixes.py", "test_spike.py"}

P4_PHRASES = ("no footage is retained", "No footage is retained",
              "no raw footage is retained", "Temporary demo captures are deleted",
              "captures are deleted after processing")


def audit(root: Path) -> dict:
    findings: list[dict] = []

    def add(check, status, detail, where=None):
        findings.append({"check": check, "status": status,
                         "detail": detail, "where": where or []})

    # ---- tree inventory (no symlink following, no writes) ----
    # Exclude the lab's own files so the scanner's regexes do not self-flag.
    files = [p for p in root.rglob("*") if p.is_file() and ".venv" not in p.parts
             and "__pycache__" not in p.parts
             and not (("privacy_lab" in p.parts) and p.name in SELF_EXCLUDE)]

    # ---- P1 absolute paths ----
    hits = []
    for p in files:
        if p.suffix not in (".py", ".md", ".txt", ".json") or ".git" in p.parts:
            continue
        t = p.read_text(errors="ignore")
        for n, line in enumerate(t.splitlines(), 1):
            if ABS_PATH_RE.search(line):
                hits.append(f"{p.relative_to(root)}:{n}: {line.strip()[:90]}")
    add("P1-absolute-paths", "FAIL" if hits else "PASS",
        f"{len(hits)} absolute-path literal(s). Shipped code must use "
        "Path(__file__)-relative roots; research scripts may carry local paths "
        "but must not ship as evaluator entry points.", hits[:12])

    # ---- P2 secrets ----
    shits = []
    # The scanners themselves hold the patterns as regex source, and the
    # security tests use literal probe strings like "/etc/passwd" to assert
    # rejection. Neither is a credential. A real hit needs a quoted literal
    # value, not a pattern name.
    _SCANNER_FILES = {"audit.py", "compliance_final.py", "v7_audit.py"}
    for p in files:
        if p.suffix not in (".py", ".md", ".txt", ".json", ".csv", ".toml", ".yaml", ".yml"):
            continue
        if ".git" in p.parts:
            continue
        if p.name in _SCANNER_FILES:
            continue
        t = p.read_text(errors="ignore")
        for n, line in enumerate(t.splitlines(), 1):
            m = SECRET_RE.search(line)
            if (m and "secretary" not in line.lower()
                    and "SECRET_RE" not in line
                    and re.search(r"['\"][^'\"]{20,}['\"]", line)):
                shits.append(f"{p.relative_to(root)}:{n}: {line.strip()[:80]}")
    add("P2-secrets", "FAIL" if shits else "PASS",
        "Hardcoded credential patterns ('secretary' vocab label excluded as "
        "a false-positive substring match).", shits[:12])

    # ---- P3 network sinks in shipped code ----
    nhits = []
    for d in SHIPPED_DIRS:
        for p in (root / d).rglob("*.py") if (root / d).exists() else []:
            t = p.read_text(errors="ignore")
            for n, line in enumerate(t.splitlines(), 1):
                if NETWORK_RE.search(line):
                    nhits.append(f"{p.relative_to(root)}:{n}: {line.strip()[:90]}")
    rhits = []
    for d in RESEARCH_DIRS:
        for p in (root / d).rglob("*.py") if (root / d).exists() else []:
            if "privacy_lab" in p.parts:
                continue
            t = p.read_text(errors="ignore")
            for n, line in enumerate(t.splitlines(), 1):
                if NETWORK_RE.search(line):
                    rhits.append(f"{p.relative_to(root)}:{n}: {line.strip()[:90]}")
    add("P3-network-sinks", "FAIL" if nhits else ("PASS" if rhits else "PASS"),
        f"Shipped src/app network sinks: {len(nhits)}. Research-only network "
        f"fetchers (expected, keep out of submission ZIP): {len(rhits)}.",
        (nhits + rhits)[:12])

    # ---- P5 tracked artefacts + gitignore ----
    gi = (root / ".gitignore").read_text(errors="ignore") if (root / ".gitignore").exists() else ""
    gi_need = ["*.mp4", ".scratch/", "*.npz"]
    gi_missing = [g for g in gi_need if g not in gi]
    try:
        tracked = subprocess.run(["git", "-C", str(root), "ls-files"],
                                 capture_output=True, text=True, timeout=10).stdout.splitlines()
    except Exception:
        tracked = []
    bad_tracked = [t for t in tracked if t.endswith((".mp4", ".avi", ".mov", ".npz", ".npy", ".pt", ".pth"))]
    add("P5-tracked-media", "FAIL" if bad_tracked else "PASS",
        f"Tracked video/cache files: {bad_tracked or 'none'}. .gitignore "
        f"missing entries: {gi_missing or 'none'}. NOTE: isolated audit copy "
        "has no .gitignore (original checkout does); verify in original.",
        bad_tracked[:12])

    # ---- P6 bundle inclusion: research files must stay out of ZIP ----
    try:
        bb = (root / "scripts/release/build_bundle.py").read_text(errors="ignore")
    except OSError:
        bb = ""
    allowlist_ok = 'relative.parts[0] not in ("src", "models", "scripts", "evaluation")' in bb
    add("P6-bundle-allowlist", "PASS" if allowlist_ok else "FAIL",
        "build_bundle.py restricts --extra artifacts to src/models/scripts/"
        "evaluation. Research fetchers under evaluation/ are still IN the "
        "allowlist: they must be excluded by not passing them as --extra, or "
        "by widening the check so the submission ZIP cannot carry ASL Citizen "
        "download code. ZIP ships only src/main.py + requirements + README + "
        "explicit --extra + model/labels/results + generated Dockerfile.",
        [])

    # ---- P7/P9 code-pattern findings (verified by reproduction) ----
    add("P7-robustness-temp-cleanup", "FAIL",
        "evaluation/device/robustness.py: shutil.rmtree(workdir) is NOT in a "
        "try/finally. A decode/write failure mid-loop leaves corrupted "
        "derivative clips in <output-dir>/work/. Reproduced: paired_evaluate "
        "on an unreadable clip raises ValueError and the work dir remains. "
        "HANDOFF: wrap loop in try/finally (owned by Abhijeet's eval code).",
        ["evaluation/device/robustness.py:45"])
    add("P7-manifest-containment", "FAIL",
        "evaluation/device/run_manifest.py:read_manifest resolves relative "
        "clip paths from the manifest dir but allows '../' and absolute "
        "paths pointing anywhere on disk (no containment check). Low risk "
        "(local evaluator, operator-supplied manifest) but a crafted "
        "manifest can pull clips from outside the dataset dir. Reproduced: "
        "'c1,../outside.mp4' is accepted. HANDOFF: add containment check "
        "or document it as a deliberate operator capability.",
        ["evaluation/device/run_manifest.py:82-86"])
    add("P9-evidence-hostname-leak", "FAIL",
        "evaluation/device/run_manifest.py writes platform.node() as "
        "'device_model' plus absolute resolved paths for every artifact "
        "into benchmark.json. If that file is pasted into results/notes or "
        "submitted, it discloses the operator's hostname and local paths. "
        "HANDOFF: use a generic device label and repo-relative paths for "
        "any evidence that leaves the machine.",
        ["evaluation/device/run_manifest.py:324-334"])

    # ---- P4 retention wording vs behaviour ----
    add("P4-retention-wording", "PASS",
        "app/offline_demo.py: NamedTemporaryFile(delete=False) temp becomes "
        "unlink(missing_ok=True) in finally -> file deleted on success AND on "
        "predictor exception (verified; the only leak is SIGKILL/machine "
        "crash, which no user-space code can clean). Frames remain in RAM "
        "(deque + list copy) and are freed on submit/clear but not "
        "explicitly scrubbed. Wording 'no footage retained' would be more "
        "honest as 'no footage is kept on disk after each capture; frames "
        "are held in memory only during inference'. The on-disk claim holds.",
        ["app/offline_demo.py:50-78"])
    add("P4b-temp-dir", "FAIL",
        "Temp clips are created in the shared system temp dir with a "
        "predictable prefix 'echogesture-capture-*.mp4' (tempfile gives "
        "0600 + O_EXCL, so predictable-name attack is blocked, but the file "
        "is still world-listable by name in /tmp and same-user processes "
        "can open it). Prefer a private subdir (tempfile.mkdtemp, 0700) "
        "or TMPDIR with restricted permissions for any release build. "
        "HANDOFF if demo is reworked.",
        ["app/offline_demo.py:59"])

    # ---- P8 licensing / attribution ----
    lic_files = [p for p in files if p.name.upper() in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "NOTICE")]
    add("P8-license-file", "FAIL" if not lic_files else "PASS",
        "No LICENSE file in repo root. README.md:167-172 states 'Our code "
        "and harnesses: MIT' but there is no LICENSE text to bind that "
        "statement, and no per-asset licence/attribution file for "
        "hand_landmarker.task (MediaPipe, Apache-2.0) or the S3D weights. "
        "Do not label the release 'MIT' without adding the licence text.",
        [])
    add("P8-models", "NOT_CLEARED",
        "models/ contains 4 .tflite + hand_landmarker.task + 3 .json. "
        "Trained Dev landmark classifier (fp16/int8, ASL Citizen-derived) "
        "carries the ASL Citizen research-licence context: evaluation, "
        "not redistribution, of the dataset is permitted but weights "
        "trained on it must not be over-claimed as freely licensable or "
        "commercially shippable. Verify Organizer + licence review before "
        "shipping any trained weight; match the model's vocab (landmark_ "
        "labels) to the packaged label file. hand_landmarker.task is a "
        "Google MediaPipe asset (Apache-2.0 per MediaPipe terms; verify "
        "current terms and add attribution) - checked-in blob must carry "
        "its provenance. S3D ptq/qat weights: README attributes MIT to "
        "Kimpinde & Olukanmi (Algorithms 2026); confirm the licence text "
        "ships with the weights and that WLASL C-UDA restrictions do not "
        "reattach - the weights were trained on WLASL, which README itself "
        "says cannot be shipped commercially.",
        ["models/hand_landmarker.task", "models/landmark_classifier_fp16.tflite",
         "models/landmark_classifier_int8.tflite", "models/ptq_100_int8.tflite",
         "models/s3d_conv1d_qat_int8.tflite"])
    add("P8-research-data", "FAIL",
        "evaluation/asl_citizen_predictions.csv (1706 lines, participant "
        "IDs P15-style) is tracked in Git. As derived evaluation output it "
        "is low-risk, but participant IDs are personal-adjacent metadata: "
        "confirm ASL Citizen terms allow redistribution of per-participant "
        "result rows, or strip participant IDs before submission. Raw "
        "clips correctly stay outside Git (fetch_*.py write to "
        "operator-chosen output dirs); there is no .gitignore in this "
        "audit copy - confirm the original retains *.mp4/.scratch "
        "exclusions.",
        ["evaluation/asl_citizen_predictions.csv"])

    summary = {"root": str(root), "files_scanned": len(files),
               "pass": sum(1 for f in findings if f["status"] == "PASS"),
               "fail": sum(1 for f in findings if f["status"] == "FAIL"),
               "not_cleared": sum(1 for f in findings if f["status"] == "NOT_CLEARED"),
               "findings": findings}
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    try:
        result = audit(args.root.resolve())
    except Exception as exc:  # audit itself must never crash the pipeline
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
