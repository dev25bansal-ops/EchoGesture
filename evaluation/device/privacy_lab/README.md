# Privacy lab (A15)

Offline, read-only audit lab for EchoGesture submission safety. It never
writes into the audited tree, never contacts the network, and uses no real
signing footage (all fixtures are synthetic).

## Files

| File | What it does |
|---|---|
| `audit.py` | Read-only sweep: absolute paths, secrets, network sinks, tracked media, bundle inclusion, symlink/traversal code patterns, licence/attribution, evidence hostnames. Emits JSON findings. |
| `reproduce_defects.py` | One reproduction per FAIL (D1–D6). Synthetic fixtures only. `--allow-network` prints research-fetcher URLs, never downloads. |
| `fixes.py` | Proposed diffs for Abhijeet-owned files (reviewed, not applied). |
| `test_privacy_lab.py` | Unit tests pinning the reproduced defects so a later fix can be verified by test. |
| `test_spike.py` | Spike: asserts the demo temp-clip cleanup actually deletes on success and on predictor exception (temporary experiment; delete once temp handling is reworked). |

## Run

```bash
V=.venv/bin/python  # Python 3.11 venv with cv2/numpy
$V evaluation/device/privacy_lab/audit.py --root . \
  --out evaluation/device/privacy_lab/findings.json
$V evaluation/device/privacy_lab/reproduce_defects.py \
  --out evaluation/device/privacy_lab/reproductions.json
$V -m unittest evaluation.device.privacy_lab.test_privacy_lab -v
```

## Exit codes

`audit.py` exits 0 when the sweep completes (findings may include FAILs)
and 2 only if the audit itself errors. `reproduce_defects.py` always exits
0; check `reproduced: true` per defect.

## Ownership

This lab is owned by A15 (Privacy, Licensing and Security). It may edit
only `evaluation/device/privacy_lab/` and `docs/release/COMPLIANCE_AUDIT.md`.
Defect fixes in `evaluation/device/*.py`, `app/`, `src/` or
`scripts/release/` are handed off with exact repro commands (see
`fixes.py`); they are never applied silently.
