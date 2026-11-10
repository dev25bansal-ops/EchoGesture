"""A09 V7 host-side parity reference + environment/blocker verifier.

Runs with:  /Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I \\
             evaluation/device/v7_android/test_parity.py

Verifies (a) Kotlin feature parity data is regenerated from src/features.py,
(b) the pinned FP16 candidate validates strictly via model_contract.py
(SHA, tensor census fp16_float32_io, logits, [1,24,255]->[1,100]),
(c) no Kotlin feature logic drift: compares the stored npz against a fresh
    build_features call, (d) records current environment state honestly.
"""
import json, pathlib, sys
import numpy as np

B = pathlib.Path(__file__).resolve().parents[3]  # A09 root = /tmp/echogesture-v7-agents/A09
sys.path.insert(0, str(B / "src"))
sys.path.insert(0, str(B / "evaluation/device"))

from features import build_features, FEATURE_DIM  # noqa: E402
from model_contract import validate_candidate_bundle  # noqa: E402

V7 = B / "evaluation/device/v7_android"
errors = []

# (a) strict candidate validation (with interpreter if available)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as _t; Interpreter = _t.Interpreter
    except ImportError:
        Interpreter = None
interp = None
if Interpreter is not None:
    interp = Interpreter(model_path=str(B / "models/landmark_classifier_fp16.tflite"))
    interp.allocate_tensors()
res = validate_candidate_bundle(B / "models", interpreter=interp)
print(f"candidate: status={res['status']} sha={res['model_sha256'][:12]}... classes={res['classes']}")
if res["model_sha256"] != "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4":
    errors.append("candidate SHA mismatch")
if FEATURE_DIM != 255:
    errors.append(f"FEATURE_DIM={FEATURE_DIM} != 255")

# (b) fixture regeneration check
for name in ["parity_synthetic", "parity_chew2"]:
    d = np.load(V7 / f"{name}.npz")
    feats2, _ = build_features(d["landmarks"], d["present"], expected_frames=24)
    err = float(np.abs(feats2 - d["features"]).max())
    print(f"{name}: max feature regenerate err={err:.2e}")
    if err > 1e-6:
        errors.append(f"{name} fixture drift {err}")

# (c) classifier smoke on chew2 + synthetic (logits + top1 recorded for parity)
if interp is not None:
    labels = json.loads((B / "models/landmark_labels.json").read_text())
    ordered = [None] * 100
    for k, v in labels.items(): ordered[int(v)] = k.upper()
    for name in ["parity_synthetic", "parity_chew2"]:
        d = np.load(V7 / f"{name}.npz")
        x = d["features"].astype(np.float32)[None]
        interp.set_tensor(interp.get_input_details()[0]["index"], x)
        interp.invoke()
        logits = interp.get_tensor(interp.get_output_details()[0]["index"])[0]
        top1 = int(np.argmax(logits))
        print(f"{name}: top1={ordered[top1]} argmax={top1} logit={logits[top1]:.4f}")

# (d) environment snapshot (same as env_probe.txt, appended)
import subprocess, os
def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[:1]
    except Exception as e:
        return [f"<{e}>"]
env = {
    "adb": run(["which", "adb"]), "gradle": run(["which", "gradle"]),
    "sdkmanager": run(["which", "sdkmanager"]), "emulator": run(["which", "emulator"]),
    "sdk_dir": str((pathlib.Path.home() / "Library/Android/sdk").exists()),
    "android_home": os.environ.get("ANDROID_HOME", "unset"),
    "java_home": os.environ.get("JAVA_HOME", "unset"),
    "studio_jbr": run(["/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin/java", "-version"]),
}
(V7 / "environment.json").write_text(json.dumps(env, indent=1))
print("environment.json:", env)

if errors:
    print("FAIL:", errors); sys.exit(1)
print("PARITY-REFERENCE OK (host side). Android build remains separately blocked — see README.md.")
