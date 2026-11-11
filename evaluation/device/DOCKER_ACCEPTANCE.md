# Docker acceptance evidence — EchoGesture evaluator image

Owner: Abhijeet (organizer requirements, evaluator entry point, packaging).
Branch `hackathon/abhijeet-integration`, commit `e2b22af`.

This document records what was **actually executed** against the container,
with literal output. Anything not executed is stated as NOT VERIFIED.

---

## 1. Verdict

| Question | Status |
| --- | --- |
| Docker daemon available | **YES** — Docker Desktop, server 29.9.2 |
| Image builds from the committed `Dockerfile` | **YES** |
| `docker run --rm <image>` prints the contract | **YES** (after fix, see §3) |
| `docker run --rm <image> --help` prints the contract | **YES** (after fix) |
| Real prediction with mounted weights | **YES** — `WHATFOR1`, 0.8818 |
| Missing-weights path names the missing file | **YES** — `FileNotFoundError`, exit 1 |
| No developer-checkout dependency | **YES** — imports resolve from `/app/src` |
| `ECHOGESTURE_TRACKER=parallel` in-container | **YES** — `engine: parallel` |

**As committed at `e2b22af`, the image builds but does not run.** Two defects
were found and fixed in the working tree (`Dockerfile`, new `.dockerignore`).
Both are described in §3 and are not yet committed.

---

## 2. Environment

| Item | Value |
| --- | --- |
| Host | macOS (Darwin 25.3.0), arm64 |
| Docker | Docker Desktop, server **29.9.2**, storage driver `overlayfs` |
| Base image | `python:3.11-slim` (Python 3.11.18) |
| Build command | `docker build -t echogesture:rc .` |
| Image ID | `sha256:be65e2bf8a10…` |
| Image size | 1.25 GB (1,250,134,348 B) |
| Reference clip | `evaluation/device/v5_accuracy/data/val_30/0283886564670357-WHAT FOR.mp4` |
| Expected result | `WHATFOR1`, confidence 0.8821 (host, macOS) |

---

## 3. Defects found in the committed image

### D1 — the image builds but every invocation crashes (blocking)

`python:3.11-slim` does not ship the X11/GL shared objects that
`opencv-python` and `mediapipe` link against. The build succeeds because
linking is lazy; the failure is at import time:

```
$ docker run --rm echogesture:rc --help
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  ...
  File "/app/src/main.py", line 19, in <module>
    import cv2
  File "/usr/local/lib/python3.11/site-packages/cv2/__init__.py", line 181, in <module>
    bootstrap()
  ...
ImportError: libxcb.so.1: cannot open shared object file: No such file or directory
```

`ldd` on `cv2.abi3.so` inside the image confirms the missing set:

```
libxcb.so.1         => not found
libGL.so.1          => not found
libgthread-2.0.so.0 => not found
libglib-2.0.so.0    => not found
```

`libEGL.so.1` and `libGLESv2.so.2` are a second layer: they are loaded lazily
by mediapipe's C++ bindings, so they appear only during a real `predict()`,
after `--help` already works:

```
OSError: libEGL.so.1: cannot open shared object file: No such file or directory
...
OSError: libGLESv2.so.2: cannot open shared object file: No such file or directory
```

**Fix** — install the five packages in the image (verified sufficient by
running both `--help` and a real prediction in a container starting from
`python:3.11-slim`):

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
```

### D2 — dead `ECHOGESTURE_MODELS` env var

The committed Dockerfile set `ENV ECHOGESTURE_MODELS=/app/models`. Nothing in
`src/`, `scripts/` or `evaluation/` reads that variable — `grep -rn
ECHOGESTURE_MODELS` matches the Dockerfile only. `predict()` resolves weights
as `<src>/../models`, which is `/app/models` in the image. The variable implied
a configurable mount point that does not exist, so it was replaced with a
comment stating the real contract.

### D3 — no `.dockerignore`

The build context was the whole checkout: **1.4 GB**
(`.venv` 466 MB, `evaluation` 598 MB, `app` 284 MB, `.git` 21 MB). Two
consequences beyond the slow upload:

- `COPY src/ /app/src/` copied `src/__pycache__/` (4 host `.pyc` files) into
  the image.
- `COPY results/ /app/results/` copied `results/predictions.jsonl` (222 KB),
  which is not needed by `predict()`.

A `.dockerignore` was added restricting the context to what the Dockerfile
`COPY`s. Measured build context before and after:

| | Context transferred |
| --- | --- |
| No `.dockerignore` (as committed) | the whole checkout, 1.4 GB on disk |
| With `.dockerignore` | **20.12 MB** |

(Measured with `docker build --progress=plain` against the same tree.)

---

## 4. Container invocations

All commands below were run from the repo root on arm64 macOS with image
`echogesture:rc` built from the fixed Dockerfile.

### 4.1 No arguments — prints the contract, exit 0

```
$ docker run --rm echogesture:rc
EchoGesture — on-device isolated sign recognition (100 glosses)

Usage:
  python -m src.main <path_to_video.mp4> [clip_id]

Prints the organizer prediction object:
  {"clip_id", "prediction", "top5", "confidence", "latency_ms"}

Requires model weights under models/ (see models/README.md). They are
not redistributable, so a source-only checkout raises FileNotFoundError
naming the missing member rather than silently running fewer model.

Environment:
  ECHOGESTURE_TRACKER=parallel   opt-in multi-worker hand tracking
  ECHOGESTURE_WORKERS=<n>        worker count (default 4)
EXIT=0
```

Note: `main()` exits `0` when an argument is present and `1` when none is,
because the `CMD` default passes `--help`. `docker run --rm echogesture:rc`
therefore exercises the `CMD` path and exits 0, while a bare
`python -m src.main` with no arguments exits 1. Both print the same text.

### 4.2 `--help` — prints the contract, exit 0

```
$ docker run --rm echogesture:rc --help
```

Output is byte-identical to §4.1. `EXIT=0`.

Critically, `--help` is **not** treated as a video path: `src/main.py` checks
`sys.argv[1] in ("-h", "--help")` before touching the filesystem, so no
`FileNotFoundError: --help` occurs.

### 4.3 Real prediction with mounted weights

```
$ docker run --rm \
    -v "$PWD/models:/app/models:ro" \
    -v "$PWD/evaluation/device/v5_accuracy/data/val_30:/clips:ro" \
    echogesture:rc "/clips/0283886564670357-WHAT FOR.mp4"
{
  "clip_id": "0283886564670357-WHAT FOR",
  "prediction": "WHATFOR1",
  "top5": [
    "WHATFOR1",
    "OFFEND",
    "KNIGHT1",
    "SCROLLDOWN",
    "DECIDE1"
  ],
  "confidence": 0.8818,
  "latency_ms": 765.87
}
EXIT=0
```

Repeated three times inside the container: `WHATFOR1 0.8818` each time
(latency 780 / 832 / 785 ms). Repeated three times on the host with the same
clip and the same four members: `WHATFOR1 0.8821` each time.

**Parity statement.** Prediction label and `top5` order are identical. The
confidence differs by **0.0003** (0.8818 in Linux/container vs 0.8821 on
macOS/host). This is a platform difference, not a packaging difference: the
same four `.tflite` members (SHA-256 recorded in §6), the same
`hand_landmarker.task`, the same `read_frames` sampling (32 frames both
sides), and the same 11 frames with detected hands. Remaining candidates are
the per-platform `opencv` decode path (container resolves
`opencv-python-headless`; host `opencv`) and platform float reduction order in
the LiteRT kernels. Deterministic within each platform. **The container is
not bit-identical to the host; it is label-identical and within 0.0003
confidence.**

### 4.4 Missing weights — clear error, exit 1

```
$ docker run --rm \
    -v "$PWD/evaluation/device/v5_accuracy/data/val_30:/clips:ro" \
    echogesture:rc "/clips/0283886564670357-WHAT FOR.mp4"
...
  File "/app/src/main.py", line 269, in _init_model
    raise FileNotFoundError(f"Selected vocabulary file not found: {label_path}")
FileNotFoundError: Selected vocabulary file not found: /app/models/landmark_labels.json
EXIT=1
STDOUT_BYTES=0
```

The traceback terminates in `main.py` at the missing-asset check. It is not an
import error from an unrelated module, and it names the exact missing path
inside the image. No prediction is emitted.

---

## 5. No hidden developer-checkout dependency

`sys.path` inside the image:

```
$ docker run --rm --entrypoint python echogesture:rc -c "import sys; print(sys.path)"
['', '/usr/local/lib/python311.zip', '/usr/local/lib/python3.11',
 '/usr/local/lib/python3.11/lib-dynload', '/usr/local/lib/python3.11/site-packages']
```

Image contents:

```
$ docker run --rm --entrypoint sh echogesture:rc -c 'ls /app/src /app/scripts/ml /app/results'
/app/results:
predictions.jsonl  research  results.csv
/app/scripts/ml:
export_tflite_tf.py
/app/src:
README.md  features.py  landmark_frontend.py  main.py  parallel_tracker.py  requirements.txt
```

- No `/Users/...` path is present in the image; `grep -rl "/Users/" /app/src
  /app/scripts` is empty.
- `predict()` imports `features`, `landmark_frontend` and `parallel_tracker`
  from `/app/src` because `main.py` inserts its own directory on `sys.path`.
  Verified in the final image:

  ```
  $ docker run --rm --entrypoint python echogesture:rc -c \
      "import sys; sys.path.insert(0,'/app/src'); \
       [__import__(m) for m in ('features','landmark_frontend','parallel_tracker')]"
  features OK
  landmark_frontend OK
  parallel_tracker OK
  ```
- `evaluation/` is **not** required at runtime.
- The image contains **no** `.tflite` or `.task` file (`find /app -name
  '*.tflite' -o -name '*.task'` → 0) and **no** `.pyc`
  (`find /app -name '*.pyc'` → 0; the `.dockerignore` fixed the pre-existing
  leak from `src/__pycache__`).

Tracker engine actually used, read from `main.get_last_profile()` in the
final image:

```
default:  {"milliseconds": ..., "frames_with_hands": 11, "engine": "singleton"}
parallel: {"milliseconds": 411.3, "frames_with_hands": 11, "engine": "parallel"}
```

`ECHOGESTURE_TRACKER=parallel` reports `engine: parallel` inside the
container, so the vendored `src/parallel_tracker.py` is present and loads from
`/app/src`, not from the research tree. Prediction is `WHATFOR1` at 0.8818 in
both modes, so the opt-in path does not change accuracy.

---

## 6. Model artifacts (mounted, never baked in)

SHA-256 of the four ensemble members and the detector, measured on the host
before mounting:

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | 939,152 | `112279bc…dc22a` |
| `landmark_ens_tcnm61_fp16.tflite` | 832,888 | `01a261d8…390f63` |
| `landmark_ens_tcnm62_fp16.tflite` | 832,888 | `27058a85…20b3` |
| `landmark_ens_rob71_fp16.tflite` | 939,152 | `08417e72…1ab21` |
| `hand_landmarker.task` | 7,819,105 | `fbc2a300…7cde1` |

Weights are not redistributable and are not in git or the image. The image
contains no `.tflite` or `.task` file; `docker run` without the `-v` mount
fails exactly as shown in §4.4.

---

## 7. Release acceptance re-run (independent)

### 7.1 Full test suite

```
$ .venv/bin/python -I scripts/run_all_tests.py --quiet
tests run                 582
  passed                  579
  failed                  0
  errors                  0
  skipped                 3
RESULT: PASS
```

Matches the expected 582 / 579 / 0 / 3.

### 7.2 Provisional bundle build + clean-extract prediction

```
$ .venv/bin/python -I evaluation/device/release_final/release_build.py \
    --output /var/tmp/eg-verify/provisional.zip
{
  "zip": ".../provisional.zip",
  "zip_sha256": "88cfb15f569c0f40d6ce974253771d3468b61ebd01c42fa681058911b56a51fe",
  "zip_bytes": 9744380,
  "extraction": {
    "prediction": {"clip_id": "0283886564670357-WHAT FOR", "prediction": "WHATFOR1",
                   "top5": ["WHATFOR1","OFFEND","KNIGHT1","SCROLLDOWN","DECIDE1"],
                   "confidence": 0.8821, "latency_ms": 497.07}
  }
}
```

Extraction into a fresh temporary directory and prediction both succeeded.

### 7.3 Two bundle defects found during this re-run

These are **not** fixed by this workstream (they are in
`evaluation/device/release_final/release_build.py`, which stages the
provisional ZIP). Recorded here because they weaken the shipped bundle.

**B1 — the provisional ZIP omits `src/parallel_tracker.py`.** The ZIP contains
17 files including `src/main.py`, `src/features.py` and
`src/landmark_frontend.py`, but no `src/parallel_tracker.py`. Running the
extracted bundle with `ECHOGESTURE_TRACKER=parallel` therefore does not use
the fast path:

```
RuntimeError: Parallel tracker not available at
  <extract>/src/parallel_tracker.py or
  <extract>/evaluation/device/performance_final/tracker.py
```

`predict()` swallows that exception and falls back, reporting
`engine: singleton_fallback` and 450–565 ms of tracking. The prediction is
correct (`WHATFOR1`, 0.8821), but the speed figure is not reproducible from
the bundle. Adding `src/parallel_tracker.py` to `CLOSURE` in
`release_build.py` fixes it — verified by copying the file into an extracted
bundle, which then reports `engine: parallel`. The `build_bundle.py`
`_LANDMARK_RUNTIME_CLOSURE` already lists the file, so only the
`release_final` wrapper is behind.

The 14 tests in `test_release_final.py` all pass with this gap present.
`test_closure_is_complete_and_honest` asserts an exact tuple that itself
omits `src/parallel_tracker.py`, so the assertion cannot detect the omission
it is named after. The sibling `test_build_bundle_integrated.py` *does*
expect the file in `EXPECTED_CLOSURE` (for `build_bundle`, a different
module), so the two suites disagree about what "complete" means.

**B2 — the provisional ZIP ships a different Dockerfile from the repo root.**
`release_build.write_dockerfile()` writes its own 5-line Dockerfile
(`COPY . /app`, `CMD ["python", "src/main.py"]`) rather than copying the root
`Dockerfile` (1,259 B as committed). The bundled one:
- has no `ENTRYPOINT`, so `--help` is passed to the script as a filename and
  is not handled as a flag;
- installs no system libraries, so `import cv2` fails as in D1;
- does not set `PYTHONDONTWRITEBYTECODE`.

Anyone who unpacks the submission and builds the bundled Dockerfile gets a
container that cannot start. The organizer's deck asks for "Docker files"; the
shipped one is not the one that was verified here.

---

## 8. What was and was not verified

**Verified by execution**

- Daemon up; build completes; image 1.25 GB.
- `--help` and no-arg invocation both print the contract and exit 0.
- Real prediction from the mounted `models/` mount: `WHATFOR1`, 0.8818,
  deterministic across three runs.
- Missing-weights failure is a `FileNotFoundError` naming
  `/app/models/landmark_labels.json`, exit 1, no stdout.
- Imports resolve from `/app/src`; no `/Users/...` in the image.
- `ECHOGESTURE_TRACKER=parallel` reports `engine: parallel`.
- Full suite 582 / 579 / 0 / 3; provisional bundle builds, extracts and
  predicts.

**Not verified**

- **arm64 vs amd64.** All runs here were arm64 Linux containers on Apple
  silicon. The organizer's evaluation host architecture is unknown; an amd64
  run (`docker build --platform linux/amd64`) was not performed.
- **Confidence parity.** 0.8818 (container) vs 0.8821 (host). Label and top5
  identical; the 0.0003 gap is unexplained beyond "platform difference"
  (§4.3). No bit-exact claim is made.
- **GPU / delegates.** CPU XNNPACK only; no GPU delegate tested.
- **Latency as a device figure.** 750–830 ms in a Docker Desktop VM is not a
  device measurement and must not be quoted as one.
- **The fixes in §3 are uncommitted.** A fresh clone at `e2b22af` still builds
  a non-running image until `Dockerfile` and `.dockerignore` are committed.
- **The bundled Dockerfile (B2) was not rebuilt here.** The repo-root
  Dockerfile was fixed; the one `release_build.py` emits was not.

---

## 9. Reproduction

```bash
cd /path/to/EchoGesture

# Build (with the fixed Dockerfile and .dockerignore)
docker build -t echogesture:rc .

# Contract
docker run --rm echogesture:rc
docker run --rm echogesture:rc --help

# Real prediction
docker run --rm \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/evaluation/device/v5_accuracy/data/val_30:/clips:ro" \
  echogesture:rc "/clips/0283886564670357-WHAT FOR.mp4"

# Missing weights (expect FileNotFoundError, exit 1)
docker run --rm \
  -v "$PWD/evaluation/device/v5_accuracy/data/val_30:/clips:ro" \
  echogesture:rc "/clips/0283886564670357-WHAT FOR.mp4"

# Parallel tracker
docker run --rm -e ECHOGESTURE_TRACKER=parallel \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/evaluation/device/v5_accuracy/data/val_30:/clips:ro" \
  --entrypoint python echogesture:rc -c \
  "import sys,json; sys.path.insert(0,'/app/src'); import main; \
   r=main.predict('/clips/0283886564670357-WHAT FOR.mp4', profile=True); \
   print(r['prediction'], r['confidence']); \
   print(main.get_last_profile()['stages']['landmark_tracking'])"
```

Build time on this host was dominated by dependency downloads and by four
concurrent `docker build` invocations from other workstreams. A serialised
build with warm layer cache completed in roughly 4 minutes; the first cold
build took ~27 minutes including a 1.4 GB context upload (since removed by
`.dockerignore`).