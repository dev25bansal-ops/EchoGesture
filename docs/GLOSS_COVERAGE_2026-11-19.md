# Can the app predict all 100 glossary signs?

Asked directly on 2026-11-19. Answered from the **installed APK**, not from
source alone, plus the full test-split results.

## Short answer: yes — all 100 are reachable, and all 100 are emitted

Three independent checks, each reproducible:

### 1. The shipped APK contains all 100 labels

Pulled the installed `base.apk` off the device and read the assets directly:

```
assets/landmark_labels.json      1,593 bytes, 100 entries
assets/landmark_model_meta.json  1,653 bytes, 9 ensemble members
class indices: 100 distinct, range 0..99
missing indices from 0..99: NONE
```

So the label map covers every output index the nine models can produce. There
is no class that the network can emit but the decoder cannot name, and none
that the decoder names but no member can reach.

### 2. The model emits all 100 distinct labels on test

On the 1,251-clip signer-disjoint test split:

```
classes in test set: 100
classes NEVER predicted correctly: 0
predicted-class coverage: 100/100 distinct labels ever emitted
```

Every one of the 100 glosses is the top-1 output on at least one clip.

### 3. Only one sign is below 50% recall

```
classes below 50% recall: 1
   CHEESEGRATER     4/10 = 40%
```

That is a weak sign, not an unreachable one. It is recognised 4 times out of 10
on unseen signers. The next weakest in the earlier per-class table are SINK and
SCARF2 at 50%, COMB2 and MYSELF at 55%.

## The honest caveat

"Reachable" and "reliable" are different claims, and only the first is settled
here. All 100 signs can be produced by the app; 99 of 100 recall above 50% on
unseen signers; the corpus holds only 17–24 clips per class, so per-class
estimates carry wide error bars. CHEESEGRATER's 40% is 4 of 10 clips — the true
rate could plausibly be anywhere from ~15% to ~70%.

**For a live demo, this is why the safe-sign list exists.** Pick from
`docs/WIN_PLAN_2026-11-19.md`'s table: APPLE 91%, ELEVATOR1 73–80%, and avoid
CHEESEGRATER, SINK, SCARF2, RAZOR2.

## Reproduce

```bash
# 1. labels in the installed APK
adb pull /data/app/.../org.echogesture.finalapp-.../base.apk .scratch/app_pulled.apk
python -c "import zipfile,json; z=zipfile.ZipFile('.scratch/app_pulled.apk'); \
  L=json.loads(z.read('assets/landmark_labels.json')); \
  print(len(L), sorted(int(v) for v in L.values())[:3])"

# 2-3. per-class recall on the test split
python - <<'PY'
import numpy as np, json
d = np.load('.scratch/thresh/probs_test.npz')
p, gloss = d['probs'], d['gloss'].astype(str)
L = json.load(open('models/landmark_labels.json'))
i2l = {int(v): k.upper() for k, v in L.items()}
pred = np.array([i2l[int(i)] for i in np.argmax(p, axis=1)])
print('never correct:', [c for c in set(gloss)
                         if (pred[gloss == c] == c).sum() == 0])
print('distinct emitted:', len(set(pred)))
PY
```