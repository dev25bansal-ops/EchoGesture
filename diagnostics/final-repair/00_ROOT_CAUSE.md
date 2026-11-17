# 00 — Root cause: the release branch cannot build a working app

## The finding

`release/samsung-final-app-test` at `57759b8` cannot produce a functioning
recogniser. Not "recognises poorly" — cannot load its models at all.

## Evidence chain

### 1. The branch ships 4 models; the contract requires 9

On-disk assets at `app/android_final/app/src/main/assets/`:

```
hand_landmarker.task
landmark_ens_rob71_fp16.tflite
landmark_ens_tcn21_fp16.tflite
landmark_ens_tcnm61_fp16.tflite
landmark_ens_tcnm62_fp16.tflite
landmark_labels.json
landmark_model_meta.json
```

The meta file lists exactly those 4 members and has **no `members_detail` key**:

```
members: 4 [...tcn21, tcnm61, tcnm62, rob71]
has members_detail: False
```

### 2. The contract rejects that

`ModelContract.kt` pins 9 members and requires their exact SHA-256:

| Member | Pinned hash |
| --- | --- |
| tcn21 | `af031d89…` |
| tcnm61 | `b0a97aa4…` |
| tcnm62 | `1d247397…` |
| rob71 | `105218a8…` |
| z_s0 | `387749f8…` |
| z_s1 | `02a2dc70…` |
| z_s2 | `3b97a2d4…` |
| z_s7 | `2f13d0eb…` |
| z_s13 | `3d3a61e0…` |

The branch's four files are the **old V4.2 binaries**:

```
app/.../assets/landmark_ens_tcn21_fp16.tflite -> 112279bc8609
```

`112279bc…` != `af031d89…`. `ModelContract.load()` verifies every member and
throws on the first mismatch, and also throws earlier when the meta list has no
`members_detail`. **Either way the app cannot initialise the classifier.**

### 3. The delivered APK is not buildable from this branch

The APK actually installed on the phone
(`EchoGesture-Test-Ready.apk`) contains **all nine** members:

```
model count: 9
assets/landmark_ens_tcn21_fp16.tflite -> af031d898c98
```

All nine match the pinned hashes exactly (verified member by member).

**`src/main.py`'s `PINNED_MEMBERS` agrees with the APK, not with `models/`.**
So there are two competing model sets on disk:

| Location | Members | tcn21 hash | Matches contract |
| --- | --- | --- | --- |
| delivered APK `/tmp/apk4/assets` | **9** | `af031d89…` | **YES** |
| `.claude/worktrees/v47/.../assets` | **9** | `af031d89…` | **YES** |
| `models/` (mixed dir) | 4 ens + 2 unrelated | `112279bc…` | NO |
| `app/android_final/.../assets` (branch) | **4** | `112279bc…` | NO |

## What this means for the user's phone test

The phone ran the correctly-built 9-model APK. So the user's "most signs
predicted incorrectly" is **not** explained by the 4-vs-9 defect — that defect
only bites anyone who rebuilds from this branch.

Two separate problems therefore exist and must not be conflated:

1. **Build integrity (this file):** the branch cannot reproduce the delivered
   APK. Anyone rebuilding from it gets a dead app. **Confirmed and fixable now.**
2. **Live recognition quality:** the correctly-built APK still misrecognises.
   Cause not yet established; needs the PHASE 3 four-path experiment, and
   path D needs a phone that is not attached.

## Consequence for the deliverable

A final APK must be built from a tree whose assets match the contract. The
authoritative source is the delivered APK itself, since it is the only copy
proven to run on the user's phone.
