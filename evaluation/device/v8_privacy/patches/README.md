Patch integration notes (v8_privacy → original checkout)

These diffs encode exactly what the v8_privacy modules implement.
They are NOT applied here — the owning agent reviews and applies them.
Line numbers are against the audited original
(``hackathon/abhijeet-integration`` as staged in this copy). If a hunk
fails to apply, re-derive it from the v8_privacy module (single source
of truth) rather than editing by hand.

Files:
- run_manifest.patch      → evaluation/device/run_manifest.py (D2 + D6)
- build_bundle.patch      → scripts/release/build_bundle.py (D3 residual)
- offline_demo.patch      → app/offline_demo.py (D4 + retention wording)
- robustness.patch        → evaluation/device/robustness.py (D1 verification stub)
- generate_results.patch  → scripts/release/generate_results.py (D6/D7 wiring + note)

How to apply (after review):
    git apply evaluation/device/v8_privacy/patches/<name>.patch
    git diff --stat
    .venv/bin/python -m unittest -v <relevant suite>

v8_privacy tests are the contract: after applying, the patches' in-tree
behaviour must match ``test_secure_manifest`` / ``test_sanitize_evidence`` /
``test_secure_demo`` / ``test_secure_bundle`` expectations.

--- run_manifest.patch -------------------------------------------------
D2: import secure reader as the single containment gate;
    read_manifest gains (allowed_roots=..., env=...) while keeping the
    (path) positional signature for existing callers.
D6: artifacts[]/manifest[] paths → _sanitized_relative(path, root);
    environment.device_model → "generic-device" (no platform.node()).
    Adds --input-root (repeatable) + ECHOGESTURE_INPUT_ROOT passthrough.
    Post-write, main() re-sanitises the on-disk benchmark.json via
    sanitize_evidence.assert_sanitized and fails loudly on any leak.

--- build_bundle.patch -------------------------------------------------
D3: safe_member() colon check widened from parts[0] to any(component).
    Everything else byte-identical.

--- offline_demo.patch -------------------------------------------------
D4: write_temp_clip → mkstemp 0600 via _unique_clip_path; PredictionWorker
    always creates a private 0700 child TemporaryDirectory;
    close() → bounded drain + _purge_dir + cleanup + second sweep;
    frames zeroed/cleared after encode. Retention docstring/header/HUD
    strings replaced with secure_demo.RETENTION_WORDING. Public names,
    submit/results/close semantics unchanged so test_offline_demo passes.

--- robustness.patch --------------------------------------------------
D1 is already fixed in the audited original (try/finally around the loop;
    shutil.rmtree(workdir) in finally). The patch is a no-op anchor that
    pins the finally with an explicit comment so future edits cannot
    silently drop it; it also wires the sanitised summary path
    (signer_ids → count only) for public exports. Behaviour unchanged.

--- generate_results.patch --------------------------------------------
D6/D7: notes strings are built from the sanitised provenance trace
    (sanitize_evidence.sanitize_benchmark_dict before extract_provenance);
    results CSV writer asserts assert_sanitized(rows) == [] and, for
    per-signer rows, hashes keys only when a private salt env
    (ECHOGESTURE_SIGNER_SALT) is set, otherwise emits aggregate-only
    rows. No schema change (same 8 columns).

Licence files:
- LICENSE + NOTICE in v8_privacy/ are the exact files to promote to the
  repo root (LICENSE covers repo code only; NOTICE is the per-asset
  attribution). Do not promote README's blanket "MIT" sentence to cover
  models/ or datasets.
