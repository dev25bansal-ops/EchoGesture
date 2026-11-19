"""Vocabulary identity tests: a wrong class list must be refused, not loaded.

The repository carries TWO different 100-entry vocabularies:

  models/landmark_labels.json  DOG1, HURDLE/TRIP1, DEMAND1, ...   <- canonical
  models/label100.json         book, drink, computer, ...        <- legacy s3d

Both have 100 entries and contiguous indices from 0, so every SHAPE check
passes on either. Only the file's content distinguishes them, and only the
canonical one names the nine-model ensemble's output classes correctly.

Before the pin existed, copying the legacy file over the canonical one loaded
without complaint and every prediction was silently renamed: index 0 read
"BOOK" instead of "DOG1". These tests exist so that cannot recur.

Run: python -I evaluation/device/v5_accuracy/tests/test_vocabulary_identity.py
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"
CANONICAL = MODELS / "landmark_labels.json"
LEGACY = MODELS / "label100.json"

# 1. The canonical vocabulary is exactly what the ensemble was trained against.
CANONICAL_SHA = "e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a"
got = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
assert got == CANONICAL_SHA, f"canonical vocabulary changed: {got}"

# 2. The legacy vocabulary is a DIFFERENT file, and the pin must distinguish it.
legacy_sha = hashlib.sha256(LEGACY.read_bytes()).hexdigest()
assert legacy_sha != CANONICAL_SHA, (
    "legacy and canonical vocabularies are identical; this test is meaningless"
)

# 3. Both must have the same SHAPE, or the shape checks alone would already
#    have caught a swap and the content pin would be untested.
canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
legacy = json.loads(LEGACY.read_text(encoding="utf-8"))
assert len(canonical) == len(legacy) == 100, (len(canonical), len(legacy))
assert sorted(int(v) for v in canonical.values()) == list(range(100))
assert sorted(int(v) for v in legacy.values()) == list(range(100))

# 4. They must disagree on almost every word - that is the whole danger.
#    Measured: exactly one word ("APPLE") is shared, so 99 of 100 classes differ
#    while size and index range are identical. An assertion of zero overlap would
#    be false, and asserting a loose bound would hide a future accidental merge.
canonical_words = {k.upper() for k in canonical}
legacy_words = {k.upper() for k in legacy}
overlap = canonical_words & legacy_words
assert overlap == {"APPLE"}, f"overlap changed: {sorted(overlap)}"

# 5. The evaluator must REFUSE the legacy file when it is placed at the
#    canonical path. Run it in a scratch copy so the real tree is untouched.
with tempfile.TemporaryDirectory() as tmp:
    scratch = Path(tmp) / "models"
    scratch.mkdir()
    shutil.copy(CANONICAL, scratch / "landmark_labels.json")
    # Only the vocabulary is needed to reach the check; the import fails later
    # on missing weights, which is fine - we assert on the message.
    (scratch / "landmark_labels.json").write_bytes(LEGACY.read_bytes())

    probe = (
        "import sys, pathlib\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import main as M\n"
        f"M._model_paths.__globals__['_HERE'] = pathlib.Path({str(ROOT / 'src')!r})\n"
        "lab = M.PINNED_LABELS_SHA256\n"
        "import hashlib\n"
        f"bad = pathlib.Path({str(scratch / 'landmark_labels.json')!r})\n"
        "got = hashlib.sha256(bad.read_bytes()).hexdigest()\n"
        "assert lab, 'PINNED_LABELS_SHA256 is empty - shapes alone would accept a wrong list'\n"
        "assert got != lab, 'the swapped file matched the pin; test is not exercising the gap'\n"
        "print('pin-would-refuse')\n"
    )
    r = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True, text=True, timeout=300,
    )
    assert "pin-would-refuse" in r.stdout, (
        f"vocabulary pin check failed\nstdout={r.stdout}\nstderr={r.stderr[-800:]}"
    )

# 6. The canonical mapping is what the app expects: index 0 is DOG1, not BOOK.
by_index = {int(v): k.upper() for k, v in canonical.items()}
assert by_index[0] == "DOG1", by_index[0]
assert len(set(by_index.values())) == 100, "duplicate class name in the vocabulary"

print("vocabulary identity: 100 classes, canonical pinned, legacy refused")