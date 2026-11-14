# V4.4 — model and Android/Python parity

**Result: PASS on 9 clips, both feature widths, exact mask agreement.**

## Model identity — pinned, all nine

Source of truth is `src/main.py::PINNED_MEMBERS` and
`app/.../ModelContract.kt`. **Filenames are not identity:** the four legacy names
hold different binaries than the V4.2 release.

| Member | SHA-256 |
| --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `af031d898c98c97abcf66ae7538166316faac458b7b938fe9e3390fce31e5d4c` |
| `landmark_ens_tcnm61_fp16.tflite` | `b0a97aa45c6797b4b85e2f6a1b03945881a94dad5519cb71be47e474fdea54f0` |
| `landmark_ens_tcnm62_fp16.tflite` | `1d247397d57e982196e68f3f230adeeabbfcdf673ba01ed541eadd54d874ea8d` |
| `landmark_ens_rob71_fp16.tflite` | `105218a89fde1d107441422254805dfb38cc7df25a216ff986801bc15dccb89a` |
| `…_z_s0_fp16.tflite` | `387749f81673719bd4a24813f33c6880c6117ed7be9d6bd50853250f8082901e` |
| `…_z_s1_fp16.tflite` | `02a2dc70fbd769f793c8f8915c8a30cc568d901bf2e35bffe1752aa762000a35` |
| `…_z_s2_fp16.tflite` | `3b97a2d416faec002bb84a6d2e47c131bb97941f074feed248a2c5a99c1c3b3d` |
| `…_z_s7_fp16.tflite` | `2f13d0eba1f34e68f3e3ac1b2c645d9f75d1953918d69be6d58ca00a1f2b36b2` |
| `…_z_s13_fp16.tflite` | `3d3a61e08d6a4d2a0a8064ae3b63692a55b55a2004ba37a428308a22b6336fba` |
| **Ordered ensemble** | **`995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb`** |
| Detector | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| Vocabulary | 100 labels |

## Parity method

`make_parity_fixture.py` runs Dev's pipeline on authorised clips, dumps source
landmarks, both feature widths, the mask and every member's raw logits into a
binary fixture, and writes a provenance sidecar. `FeatureParityTest` re-derives
all of it through the shipped Kotlin code and compares numbers.

Portable: every path is an argument or repository-relative, no developer-specific
import, and the generator refuses to write a fixture unless the nine member
hashes and the ordered ensemble hash match the pins.

```bash
python evaluation/device/v4_3_parity/make_parity_fixture.py \
    --clips DIR --manifest manifest.json --out /path/parity_fixture.bin --limit 9
```

## Results — 9 clips, 5 signers

| # | Gloss | Frames | Feature max abs diff | Mask mismatches | Logit max abs diff | Top-1 |
| --: | --- | --: | --: | --: | --: | --- |
| 0 | HOW1 | 71 | 0.000000 | 0 | 4e-6 | agree |
| 1 | HOSPITAL1 | 46 | 0.000000 | 0 | 4e-6 | agree |
| 2 | BACKPACK1 | 48 | 0.000000 | 0 | 3e-6 | agree |
| 3 | BACKPACK1 (different signer) | 93 | 0.000000 | 0 | 2e-6 | agree (both wrong) |
| 4 | SCROLLDOWN | 181 | 0.000000 | 0 | 2e-6 | agree |
| 5 | DOWNSIZE1 | 89 | 0.000000 | 0 | 5e-6 | agree |
| 6 | LOCK1 | 74 | 0.000000 | 0 | 3e-6 | agree |
| 7 | GUESS1 | 60 | 0.000000 | 0 | 3e-6 | agree |
| 8 | CLOSE | 127 | 0.000000 | 0 | 3e-6 | agree |

**Summary:** worst feature diff `0.000000`, mask mismatches `0`, worst logit
diff `5e-6`, top-1 agreement **9/9**.

Coverage against the brief:

| Required case | Covered |
| --- | --- |
| Short signs | HOSPITAL1 46 frames (1.5 s) |
| Long signs | SCROLLDOWN 181 frames (6.0 s) |
| One-hand signs | present in the masks |
| Two-hand signs | present in the masks |
| Missing-hand intervals | masks contain zeros; **no clip had a fully absent hand** |
| Leading/trailing idle | not in the fixture (measured separately in V4.3) |
| Different signers | 5 (P12, P21, P26, P39, P5) |
| Different confidence levels | 0.21 to 0.97 |
| Correct and incorrect | 8 correct, 1 incorrect — **both platforms wrong on the same clip** |

Clip 3 is the most informative row: `BACKPACK1` from a second signer is
misclassified as `CHEER` by **both** platforms at the same confidence. Identical
wrong answers are stronger evidence of parity than identical right ones.

**Gap, stated plainly:** no clip with a fully absent hand. The all-zero-mask path
is covered by unit tests and by the app's "no hands in view" rejection, not by a
real clip.

## Tolerances

| Quantity | Observed | Bound |
| --- | --- | --- |
| Feature max abs diff | 0.000000 | 1e-3 |
| Mask mismatches | 0 | exact |
| Logit max abs diff | 5e-6 | 1e-2 |

Set from observation, not guessed. An earlier revision used 1e-4/0.05 and
**passed** a real defect (z-scale computed from x,y instead of x,y,z; feature
diff 1.047). The bounds were tightened after that.

## Full MP4 path

Identical landmark inputs do not prove identical video extraction, so the MP4
path is checked separately end to end. The same clip through production
`predict()` on the host and through the Docker image both return
`DOWNSIZE1` at 0.915 — see `V4_4_FINAL_RELEASE_STATUS.md` §4.

## Known limits

- Laptop CPU only. No phone measurement exists.
- Landmarks come from MediaPipe on **both** sides, so parity confirms the
  contract downstream of the detector. It does not confirm that Android's
  VIDEO-mode tracking and Python's IMAGE-mode tracking see the same landmarks;
  that was measured separately in earlier work and is not re-established here.