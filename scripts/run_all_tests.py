"""Run every maintained test suite in the repository with one command.

`unittest discover` only recurses into packages (directories with an
``__init__.py``). Most of this repository's suites live in plain directories,
so the documented discover command silently ran ~281 of ~460 cases. This
runner loads every ``test_*.py`` file by explicit path instead, so nothing is
missed, and it separates tests that cannot run here from tests that failed.

Usage:
    python3 -I scripts/run_all_tests.py            # everything runnable
    python3 -I scripts/run_all_tests.py -v         # list every test id
    python3 -I scripts/run_all_tests.py --json out.json

Exit status is non-zero if any test fails or errors. Modules that fail to
import are reported as BLOCKED with their real error, never counted as passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"

# Modules that are scripts, not unittest suites: they print a marker and exit.
# Counting them as "0 tests passed" would be dishonest, so they are named.
# Files that hardcode absolute /tmp paths from the session that created them.
# They cannot run on any other machine and are superseded; reported, not fatal.
UNRUNNABLE = {
    "evaluation/device/tracking_lab/test_tracker.py",
    "evaluation/device/v4_performance/test_contract_smoke.py",
}

SCRIPT_SUITES = {
    "evaluation/device/v7_android/test_parity.py",
    "evaluation/device/v7_release/tests/test_v7_release.py",
    "evaluation/device/v7_tracking/test_candidates.py",
    "evaluation/device/v8_accuracy/tests/test_v8_accuracy.py",
    "evaluation/device/v5_accuracy/tests/test_v5_accuracy.py",
}


def _discover_files():
    files = sorted(p for p in EVAL.rglob("test_*.py") if "__pycache__" not in p.parts)
    return files


def _purge_sibling_modules(directory: Path) -> None:
    """Evict cached modules whose file lives in another test directory."""
    keep = {"unittest", "os", "sys", "json", "csv", "re", "math", "time"}
    for name, module in list(sys.modules.items()):
        if name in keep or module is None:
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        try:
            origin_dir = Path(origin).resolve().parent
        except (OSError, ValueError):
            continue
        # Evict anything cached from a *different* evaluation directory.
        # Several suites ship same-named helpers (streaming_runtime.py exists
        # in both streaming_lab/ and v7_streaming/), so a stale entry makes the
        # wrong implementation get imported. The suite's own directory is
        # placed at sys.path[0] before loading, so its sibling wins the
        # re-import.
        if origin_dir != directory.resolve() and "evaluation" in str(origin_dir):
            sys.modules.pop(name, None)


def _module_name(path: Path) -> str:
    return "suite_" + "_".join(path.relative_to(ROOT).with_suffix("").parts)


def _load(path: Path):
    """Import a test module by absolute path, with its own dir on sys.path.

    Test modules in plain directories import sibling helpers by bare name
    (``import matrix``), so the module's directory must be importable while it
    executes. sys.path is restored afterwards to avoid cross-suite leakage.
    """
    added = str(path.parent)
    sys.path.insert(0, added)
    old_cwd = Path.cwd()
    # Sibling helper modules (``import streaming_runtime``) collide across
    # directories: two suites can both define ``test_streaming_lab``. Drop any
    # module already imported from a *different* directory before loading this
    # one, so each suite resolves its own siblings.
    _purge_sibling_modules(path.parent)
    try:
        spec = importlib.util.spec_from_file_location(_module_name(path), path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if added in sys.path:
            sys.path.remove(added)
        try:
            import os
            os.chdir(old_cwd)
        except OSError:
            pass


_CHILD = r"""
import json, sys, unittest, importlib.util
from pathlib import Path
path = Path(sys.argv[1])
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("suite_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module   # some suites look themselves up by name
spec.loader.exec_module(module)
result = unittest.TextTestRunner(stream=open(sys.argv[2], "w"), verbosity=2).run(
    unittest.TestLoader().loadTestsFromModule(module))
print(json.dumps({
    "run": result.testsRun,
    "failures": [t.id() for t, _ in result.failures],
    "errors": [t.id() for t, _ in result.errors],
    "skipped": [t.id() for t, _ in result.skipped],
}))
"""


def _run_file(path: Path, verbose: bool) -> dict:
    """Run one test file in its own interpreter.

    Suites in this repo ship same-named helper modules (``streaming_runtime``,
    ``tracker``, ``matrix``) and same-named test modules, so importing them all
    into one process makes them clobber each other. A fresh process per file is
    how they were each written to run, and it keeps a crash or a polluted
    sys.modules from leaking across suites.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False) as log:
        log_path = log.name
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _CHILD, str(path), log_path],
            capture_output=True, text=True, timeout=600, cwd=str(ROOT),
        )
        detail = Path(log_path).read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0 and not proc.stdout.strip():
            raise RuntimeError((proc.stderr or "").strip().splitlines()[-1])
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        payload["log"] = detail
        return payload
    finally:
        try:
            Path(log_path).unlink(missing_ok=True)
        except OSError:
            pass


def run(verbose: bool = False) -> dict:
    files = _discover_files()
    loaded: list[str] = []
    empty: list[str] = []
    script_suites: list[str] = []
    blocked: list[dict] = []
    failed_ids: list[str] = []
    error_ids: list[str] = []
    skipped_ids: list[str] = []
    unrunnable: list[str] = []
    run_output: list[str] = []
    tests_run = 0

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in SCRIPT_SUITES:
            script_suites.append(rel)
            continue
        if rel in UNRUNNABLE:
            unrunnable.append(rel)
            continue
        try:
            payload = _run_file(path, verbose)
        except Exception as exc:  # noqa: BLE001
            blocked.append({"file": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if payload["run"] == 0 and not isinstance(payload, dict):
            empty.append(rel)
            continue
        if payload["run"] == 0:
            empty.append(rel)
            continue
        tests_run += payload["run"]
        loaded.append(rel)
        failed_ids += payload["failures"]
        error_ids += payload["errors"]
        skipped_ids += payload["skipped"]
        if verbose:
            run_output.append(payload.get("log", ""))

    passed = tests_run - len(failed_ids) - len(error_ids) - len(skipped_ids)
    return {
        "test_files_found": len(files),
        "modules_loaded": len(loaded),
        "script_style_suites": script_suites,
        "modules_with_no_tests": empty,
        "modules_blocked_on_import": blocked,
        "modules_unrunnable_by_design": unrunnable,
        "tests_run": tests_run,
        "passed": passed,
        "failed": len(failed_ids),
        "errors": len(error_ids),
        "skipped": len(skipped_ids),
        "failed_ids": failed_ids,
        "error_ids": error_ids,
        "skipped_ids": skipped_ids,
        "run_output": "".join(run_output),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--quiet", action="store_true", help="summary only")
    args = parser.parse_args(argv)

    report = run(verbose=args.verbose)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(report["run_output"])

    r = report
    print("=" * 66)
    print(f"test files found          {r['test_files_found']}")
    print(f"unittest modules loaded   {r['modules_loaded']}")
    print(f"script-style suites       {len(r['script_style_suites'])}")
    print(f"modules with no tests     {len(r['modules_with_no_tests'])}")
    print(f"modules blocked on import {len(r['modules_blocked_on_import'])}")
    print("-" * 66)
    print(f"tests run                 {r['tests_run']}")
    print(f"  passed                  {r['passed']}")
    print(f"  failed                  {r['failed']}")
    print(f"  errors                  {r['errors']}")
    print(f"  skipped                 {r['skipped']}")

    if r["modules_unrunnable_by_design"]:
        print("-" * 66)
        print("unrunnable by design (hardcoded /tmp paths from an old session):")
        for m in r["modules_unrunnable_by_design"]:
            print(f"  {m}")
    if r["modules_blocked_on_import"]:
        print("-" * 66)
        print("BLOCKED (not counted as passing):")
        for b in r["modules_blocked_on_import"]:
            print(f"  {b['file']}: {b['error']}")
    if r["modules_with_no_tests"]:
        print("-" * 66)
        print("modules containing no unittest cases:")
        for m in r["modules_with_no_tests"]:
            print(f"  {m}")
    if r["script_style_suites"]:
        print("-" * 66)
        print("script-style suites (run directly, not unittest):")
        for m in r["script_style_suites"]:
            print(f"  {m}")

    ok = r["failed"] == 0 and r["errors"] == 0
    print("=" * 66)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())