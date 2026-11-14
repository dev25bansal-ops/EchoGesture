# Build artifacts — restricted, NOT redistributable

This orphan branch carries built artifacts for the two-person sprint. It is
**not** a release and **not** a licence grant.

| | |
| --- | --- |
| Branch | `dist/apk-v4.3` |
| Cut from | `origin/main` (`f81dd30`) |
| Purpose | Hand a verified APK between the two developers |

## Contents

| File | Bytes | SHA-256 | Source commit |
| --- | --: | --- | --- |
| `echogesture-v4.3-masked.apk` | 83,350,522 | `3db9d94ad761a76bdcf5dd9b3eaeff7e91dd14a51eaee2eca44acb98f1380896` | `5251bf8` |
| `echogesture-v4.2.apk` | 77,908,968 | `5490e02908159e24dac1b45dca7b21afbbe77909fb62ceab4348828e8d939dfc` | rollback |
| `echogesture-v4.4-provisional.zip` | 13,384,710 | `aee394d22db2dfd0e2070326eebae611e8834810bc640d9a822747af19f8c105` | `8dace91` |

(Byte counts and hashes as recorded at V4.4. Verify before installing —
`shasum -a 256`.)

## What is inside

`echogesture-v4.3-masked.apk` embeds the detector and all nine ensemble
members:

```text
assets/hand_landmarker.task                              7,819,105
assets/landmark_ens_tcn21_fp16.tflite                      942,160
assets/landmark_ens_rob71_fp16.tflite                      942,160
assets/landmark_ens_tcnm61_fp16.tflite                     835,896
assets/landmark_ens_tcnm62_fp16.tflite                     835,896
assets/landmark_ens_mirror_handdrop_tcn_z_s0_fp16.tflite   990,544
assets/landmark_ens_mirror_handdrop_tcn_z_s1_fp16.tflite   990,544
assets/landmark_ens_mirror_handdrop_tcn_z_s2_fp16.tflite   990,544
assets/landmark_ens_mirror_handdrop_tcn_z_s7_fp16.tflite   990,544
assets/landmark_ens_mirror_handdrop_tcn_z_s13_fp16.tflite  990,544
```

Ordered ensemble SHA-256:
`995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb`.

## Rights — read before sharing

The nine members are trained on ASL Citizen-derived data. **Redistribution is
not cleared.** See `NOTICE` and `models/README.md` on the source branches.

`NOTICE` records: *"Including an artifact in this repository is NOT a licence
grant for it."* That applies here too.

This branch exists because both developers need the same binary. **It is not
permission to publish, mirror or hand these artifacts to anyone else**, and it
does not satisfy the organizer's rights requirement.

**Owner approval is required before:** making this branch public-facing,
announcing it, mirroring it, or submitting an APK to the organizer.

## Why a separate branch

The source branches gitignore `*.apk` (`*.apk`, `*.aab` at `.gitignore:36-37`)
because the artifacts are large and embed restricted weights. Keeping them on
an isolated branch means:

- `git clone` of a source branch does not pull 160 MB of binaries
- the mainline history stays free of them
- deleting this branch later does not touch any source code

## Reproducing instead of downloading

`docs/release/BUILD_INSTRUCTIONS_SHARED.md` on
`shared/v4.4-dev-integration` gives the full procedure: supply authorised
weights, verify the eleven SHA-256 values, build with JDK 17 and the pinned
SDK, run `./gradlew clean testDebugUnitTest assembleDebug`.

An APK is not byte-reproducible across machines, but its **content** is: same
nine member hashes, same metadata, same source commit.