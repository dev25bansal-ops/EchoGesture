# The demo fallback was broken, 2026-11-19

The demo script and the win plan both say: *if the phone misbehaves, run
`app/offline_demo.py` — it is honest and it works.* That sentence was wrong in a
way that would have cost the demo, because it is the fallback for the one
deliverable that does not exist yet.

## What was wrong

**1. The file is not on this branch.** `app/offline_demo.py` exists on
`hackathon/abhijeet-app-fix`, `origin/hackathon/abhijeet-integration` and nine
other branches. On `hackathon/dev-model` there is no `app/` directory at all.
Anyone following the script on this checkout gets "No such file or directory".

**2. Where it does exist, it does not run against this checkout's `src/`.**
The mirror copy calls `predict(path, clip_id=...)`. This branch's `src/main.py`
is `def predict(video_path: str)` — it derives `clip_id` from the filename itself,
and the keyword does not exist:

```
ERROR: TypeError: predict() got an unexpected keyword argument 'clip_id'
```

**3. Pairing it with Abhijeet's `main.py` fails differently.** His `predict()` does
take `clip_id`, but his loader expects 255-wide members and rejects the checkpoint:

```
ValueError: Ensemble member 'landmark_ens_tcn21_fp16.tflite'
  input shape [1, 24] != [1, 24, 255]
```

So there is no combination of the files as they stand that runs to a result.

**4. The HUD truncated the threshold off the end.** The subtitle is built as
`{state} | {model_label} | provisional retry threshold {t}` and cut at 86
characters. The nine-model label (`landmark:9-model ensemble sha 995cc8d0b6f6
8508832 B`) is long enough that the cut lands mid-word, and the screen reads:

```
CAPTION | landmark:9-model ensemble sha 995cc8d0b6f6 8508832 B | provisional retry th...
```

The threshold value — the one number on that line a judge might ask about — is
the part that falls off.

## What was done

Restored `app/offline_demo.py` into this branch and fixed two lines:

- the call is now `predict(str(video_path))`; `clip_id` is still passed to
  `sanitize_result` for labelling;
- the subtitle puts the threshold first: `{state} | threshold {t} | {model_label}`,
  so a truncation now costs the sha tail instead of the number.

## Verified, not assumed

Run from the repo root with `C:/Python314/python.exe` (the only interpreter here
that has cv2, mediapipe and the model runtime together):

| clip | prediction | confidence | rendered |
|---|---|---|---|
| `002273170951422543-CHEW 2.mp4` | **CHEW1** | 0.277 | `hud.png` |
| `002040595117913435-SLIDE 2.mp4` | **SLIDE2** | 0.9479 | `hud_slide.png` |

Both agree exactly with `src/main.py` run directly on the same files, which is
the authority on what the model actually predicts. The HUD now reads
`CAPTION | threshold 0.25 | landmark:9-model ensemble sha 995cc8d0b6f6 8508832 B`
with nothing cut mid-word.

## What this does not fix

The desktop fallback is honest — it runs the real predictor on a real file — but
it is **not** the phone. Beat 5's live-camera moment still needs a camera. If the
phone is unavailable entirely, the recording has to say so and show the desktop
path, which is what the script already instructs.

The CHEW1 confidence of 0.277 is just above the 0.25 threshold. That is a real
property of the model on that clip, not a bug, but a demo that shows it should
prefer a clip the model is confident about.
