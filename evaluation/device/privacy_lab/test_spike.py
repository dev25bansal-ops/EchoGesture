"""SPIKE - temporary experiment, delete after temp handling is reworked."""
import tempfile, unittest
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parents[3] / "app"))
from offline_demo import PredictionWorker  # noqa: E402

class SpikeTest(unittest.TestCase):
    def test_temp_clip_always_deleted(self):
        seen = []
        def predict(path, *, clip_id=None):
            seen.append(Path(path)); return {"clip_id": clip_id, "prediction": "X", "confidence": 1.0, "latency_ms": 1.0}
        w = PredictionWorker(predict)
        w.submit([np.zeros((8, 8, 3), np.uint8)], 12, "s")
        w.results.get(timeout=10); w.close()
        self.assertFalse(seen[0].exists())

if __name__ == "__main__":
    unittest.main()
