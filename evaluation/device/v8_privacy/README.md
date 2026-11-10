# EchoGesture — v8_privacy release-safety patches

Hardened replacements for the A15 privacy-lab defects (D2–D7), delivered
as a self-contained tree in `evaluation/device/v8_privacy/`. Nothing in
the original checkout or Dev-owned code has been edited here; every fix
is a new module plus a unified diff in `patches/` that the owning agent
can `git apply` after review, plus regression tests that import the
parent modules read-only.

## Layout

| File | What it fixes |
|---|---|
| `secure_manifest.py` | D2 — manifest traversal, with configurable secure input root |
| `sanitize_evidence.py` | D6 — hostname / private-path redaction; D7 — participant-ID exclusion |
| `secure_demo.py` | D4 + retention wording — private 0700 temp dir, 0600 clips, race-free close |
| `secure_bundle.py` | D3 residual — colon in *any* bundle-member component rejected |
| `test_secure_manifest.py`, `test_sanitize_evidence.py`, `test_secure_demo.py`, `test_secure_bundle.py` | Targeted regression tests (attacks rejected; legitimate paths still work) |
| `patches/*.patch` | Unified diffs against the original files (owner applies) |
| `LICENSE`, `NOTICE` | Accurate licence + attribution; no availability-expanded redistribution claims |

## Import contract

Tests and patched modules import parent code read-only:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # evaluation/device/
from v8_privacy.secure_manifest import read_manifest_secure
# (run_manifest is the parent module already on that path)
import run_manifest  # original, untouched
```

`run_manifest` needs `evaluation/device/` on the path;
`app.offline_demo` needs the repo root on the path (it already does
`sys.path.insert(0, str(ROOT))` internally). All v8_privacy modules are
Python 3.11 compatible and stdlib-only (cv2/numpy only where the
original demo already needs them).

## Manifest traversal: behavior and configuration

`read_manifest_secure(path, allowed_roots=None, *, env=None)`:

* **Default:** only the manifest's own directory (resolved) is allowed.
  Relative paths resolve from it; absolute paths must fall inside it.
  This preserves the `fetch_validation_batch.py` layout (manifest.csv
  beside the clips) and every existing test fixture that keeps clips in
  the manifest tree.
* **Judge / organizer layout:** pass `allowed_roots=[Path("/organizer/clips")]`
  or set `ECHOGESTURE_INPUT_ROOT` (`os.pathsep`-separated). Entries are
  additive containment roots; a relative `../clips/x.mp4` that lands
  inside a configured root is then legitimate, while `../outside.mp4`
  landing outside *every* root is still rejected.
* Containment is checked on **resolved** paths (symlinks followed), so
  symlink escapes are rejected. `~` is expanded before resolution.
* Every rejection is `SecureManifestError` (subclass of `ValueError`)
  with `Manifest line N:` context. Missing-but-inside files keep the
  `clip not found` message so `assertRaisesRegex(..., "clip not found")`
  suites keep passing.

## Evidence sanitisation

```python
from v8_privacy.sanitize_evidence import (
    sanitize_benchmark_dict, sanitize_benchmark_file,
    sanitize_predictions_csv, sanitize_robustness_summary,
)
clean = sanitize_benchmark_dict(bench, repo_root=REPO_ROOT)
sanitize_benchmark_file(Path("benchmark.json"))
sanitize_predictions_csv(Path("asl_citizen_predictions.csv"), Path("public.csv"))
```

* `device_model` → `"generic-device"` (never a hostname).
* Artifact/manifest `path` → repo-relative POSIX path when under
  `repo_root`, otherwise basename only. Embedded private paths in any
  other string → `[redacted-path]`.
* SHAs, bytes, counts, timings, code commit, dataset/split, timestamps
  pass through verbatim — `generate_results` stale-evidence binding
  (`evidence_lab.provenance.require_bound_to`) is preserved.
* `assert_sanitized(value)` lists remaining leaks (empty when clean) —
  wire it as a CI gate mirroring `build_bundle._scan_no_absolute_leak`.
* `sanitize_predictions_csv` drops `participant` / `signer_id` /
  `signer` / `Participant ID` columns; with a private `salt` it adds a
  `signer_pseudonym` HMAC column instead of raw IDs. The salt file
  must never ship.

## Demo media retention

`SecurePredictionWorker` is a drop-in for `PredictionWorker` (same
`submit` / `results` / `close` / `tmpdir_path` surface):

* Clips live in a worker-owned private `0700` directory (a caller
  `tmpdir=` becomes the *parent*, never the landing dir).
* Files are `mkstemp` + `O_EXCL`, `0600`, with per-file chmod (the
  `VideoWriter` umask alone yields `0644`).
* Deleted on success **and** on predictor exception (verified D5
  control still holds).
* `close()` drains with a bounded join, purges leftovers, closes the
  `TemporaryDirectory`, then sweeps again — so a close racing a slow
  predictor cannot silently orphan clips.
* Frames are cleared/zeroed after encode. Use
  `RETENTION_WORDING` verbatim in place of every
  "no footage is retained" string:

  > No footage is kept on disk after each capture; frames are held in
  > memory only during inference.

## Bundle member check

`safes` the residual D3 bypass: `safe_member("foo/bar:baz")` was `True`;
the hardened check rejects a colon in *any* component (Windows NTFS
`file:stream` form), plus the original drive/absolute/`..`/backslash
guards. Legitimate names (`models/hand_landmarker.task`, …) still pass.

## Verification

```bash
V=.venv/bin/python  # 3.11 venv (cv2/numpy); run from the repo root
$V -m unittest -v \
  v8_privacy.test_secure_manifest \
  v8_privacy.test_sanitize_evidence \
  v8_privacy.test_secure_bundle
$V -m unittest -v v8_privacy.test_secure_demo  # needs cv2/numpy (venv)
$V -I -m unittest ...  # also run any test touching downloaded/untrusted data with -I
```

Expected: all attacks rejected; sibling/absolute-inside/judge-root
paths, digit-only ZIP-safe member names, verbatim SHAs, and
delete-on-success/error all green. See `patches/` for the exact
original-file hunks these modules encode.
