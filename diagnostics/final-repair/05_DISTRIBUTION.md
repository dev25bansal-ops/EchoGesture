# 05 — Distribution: the branch could not build, and still cannot be rebuilt from a clone

Two separate problems. Neither is a recognition defect, and both block anyone
but the original operator.

## 5.1 The branch shipped the wrong models — FIXED

At `57759b8` the branch's asset directory held **four** model binaries and a meta
file listing four members with no `members_detail`:

```
members: 4 [...tcn21, tcnm61, tcnm62, rob71]
has members_detail: False
assets/landmark_ens_tcn21_fp16.tflite -> 112279bc8609   (old V4.2 bytes)
```

`ModelContract` pins **nine** members and rejects an unpinned or mismatched
binary, and rejects a meta list without `members_detail`. An APK built from that
tree throws during model load — it does not misrecognise, it fails to start.

The APK actually installed on the phone contains all nine
(`af031d89…` … `3d3a61e0…`), every one matching the pinned hashes.

**Fixed** by replacing the branch's `app/src/main/assets/` and `models/` model
files with the nine verified members taken from the delivered APK, and the meta
file with the nine-member version that carries `members_detail`.

Verified after the fix: the assembled APK contains nine members, the
meta-order-concatenated digest is `995cc8d0b6f6ac25…`, and the app launches to
the Camera tab showing `9-model ensemble · FP16 TFLite · SHA-256 995cc8d0b6f6…`.

## 5.2 No model is tracked in git — OPEN

This one has no safe fix inside this session, so it is reported rather than
acted on.

Rules that apply (verified with `git check-ignore -v`):

```
.gitignore:72          *.tflite                                   -> models/*.tflite
.gitignore:73          *.task
.gitignore:94          app/android_final/app/src/main/assets/*.task
.gitignore:95          app/android_final/app/src/main/assets/*.tflite
app/android_final/.gitignore:3   app/src/main/assets/hand_landmarker.task
app/android_final/.gitignore:5   app/src/main/assets/landmark_labels.json
app/android_final/.gitignore:6   app/src/main/assets/landmark_model_meta.json
```

So on this branch, `git ls-files` shows **zero** of:

- the nine ensemble members
- the MediaPipe `hand_landmarker.task` detector
- `landmark_labels.json`
- `landmark_model_meta.json`

The `.gitignore` explains the intent — *"Keep model binaries in the repository's
existing `models/` directory. The documented local copy step places them here
only for packaging."* — but `models/*.tflite` is ignored too, so the copy step
has nothing to copy from in a fresh clone.

**Consequence.** A fresh `git clone` of `release/samsung-final-app-test` cannot
produce a working APK, even after the 5.1 fix. It would need the model files
supplied out of band.

**Why I did not commit them.** The brief states: *"Do not commit the APK or
trained weights to a public repository until redistribution rights are
confirmed."* The repository is public:

```console
$ gh repo view dev25bansal-ops/EchoGesture --json visibility
{"isPrivate":false,"visibility":"PUBLIC"}
```

Committing the weights would publish them. That is a rights decision for the
owner, not one I should make to make a build reproducible.

**What the owner must decide.** Either:

1. confirm redistribution rights and commit the seven non-.tflite assets plus
   the nine members (with a documented provenance note), or
2. keep the repository private, or
3. leave it as is and ship the APK only, accepting that the branch is not
   self-reproducing.

Until one of those is chosen, this branch reproduces the **source** but not the
**artifact**.

## 5.3 Two model files are tracked, and they are not the shipped ones

`models/ptq_100_int8.tflite` and `models/s3d_conv1d_qat_int8.tflite` are the only
trained binaries in git. They are the older INT8 S3D path, not the nine-member
ensemble the app runs. They are unrelated to recognition here and were left
untouched.