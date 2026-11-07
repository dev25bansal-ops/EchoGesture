# Track 2 submission checklist

Abhijeet owns this checklist. Dev verifies model and evaluation claims and
reruns the release before Abhijeet submits. Use the organizer's latest starter
kit and final submission form to settle any differences in the slides.

[Official brief, PDF pages 9–12 and 24–27](../../XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf) ·
[Abhijeet's workflow](ABHIJEET.md) · [Dev's workflow](DEV.md)

## Registration and organizer inputs

- [ ] Register for Track 2 by **8 November 2026, 11:59 PM IST** with both
  teammates from the same college. Use `CollegeName_TeamName`. The slides
  allow up to four members, one theme and one submission per team.
- [ ] Save the official vocabulary and test-manifest instructions. Confirm
  the clip ID mapping, allowed data sources, target device/runtime,
  timing boundary and whether detector files count toward the size limit.
- [ ] Check for the final form. The brief says the link goes only to
  registered teams. Use **17 November** as the internal submission date.
  The published deadline is **18 November 2026, 11:59 PM IST**.

## Package and clean-room check

- [ ] Provide the public or shared GitHub link at the exact release commit.
  Check that a reviewer can access the repo and demo link.
- [ ] Use the `team_name.zip` layout from PDF page 12. Keep
  `src/main.py`, `src/requirements.txt` and `src/README.md`. Keep all
  executable evaluator source under `src/`. Include the model, vocabulary,
  conversion script, Docker setup and any required local assets. Test that
  relative paths still resolve in the actual ZIP.
- [ ] In a fresh Python 3.11 environment, call `predict(video_path) -> dict`
  on a permitted local clip without network access. Verify `clip_id`,
  uppercase `prediction` from the fixed vocabulary, `confidence` and
  `latency_ms`. If supplied, `top5` must use the same vocabulary and ranking.
- [ ] Generate one prediction object per organizer test clip, with no
  missing, duplicate or extra manifest IDs. Keep the repository's ASL
  Citizen research predictions separate from the submission.
- [ ] Fill `results/results.csv` with one row per official metric and the
  template's columns for metric, target, measured value, unit, device,
  runtime and pass status. Link each metric to its model SHA and evidence
  in the accompanying notes. Mark unavailable measurements explicitly.
- [ ] Check the stated targets for top-1 (≥70%), clip p95 (≤50 ms in the
  results table), robustness drop (≤5 percentage points), average power
  (≤500 mW) and 10-minute thermal drift (≤20%). Record the target device and
  runtime separately for clip evaluation and live app measurements.
- [ ] Share a **2–3 minute** YouTube or Drive demo. Show an external camera
  view of the signer, on-device captions and the offline run. Use the
  release model in the demo and identify its supported vocabulary scope.
- [ ] Submit a PPT or PDF named `CollegeName_TeamName` using the organizer
  template. Cover the theme ID, project title and team, problem in your own
  words, solution and architecture, tools, innovation, results and limits.
- [ ] Have Dev reproduce the run from the extracted ZIP. Abhijeet submits
  the reviewed package once and saves the receipt and release SHA.

## Slide differences and team defaults

| Item | Slides | Working choice until the organizer confirms |
| --- | --- | --- |
| Model size | The Track 2 rubric says **<20 MB**. The sample results sheet says **≤25 MB**. | Build under **20 MB**. Report bytes for each deployed model and the total. Use the official sheet's field value when filling it in. |
| Prediction filename | The prediction contract specifies `predictions.jsonl`. The bundle diagrams use `predictions.json`. | Generate JSON Lines internally. Set the final filename and serialization from the starter kit or final form. |
| Demo | The Track 2 slides say **2–3 minutes** or an app URL. The general timeline says five minutes, and the final submission rules cap video at three minutes. | Submit a **2–3 minute** video, with an app link only if requested or useful for the judges. |
| Latency | The results sheet uses **≤50 ms p95**. The rubric says **<50 ms** and organizers remeasure on their device. | Aim for **under 50 ms** end-to-end p95 on the clip path. Report camera-to-caption timing separately. |
