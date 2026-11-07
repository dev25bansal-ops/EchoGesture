# EchoGesture agent guide

Start with the [two-person sprint](docs/hackathon/README.md). Read the named
owner's workflow and the [handoff contract](docs/hackathon/HANDOFF.md) before
changing code. The active teammate is the one named in the current task.
If ownership is unclear, use the file and responsibility boundaries below and
ask only if the task crosses them.

| Teammate | Owns | Workflow |
| --- | --- | --- |
| Dev | Dataset, signer splits, shared model features, training, INT8 conversion and model artifacts | [Dev's workflow](docs/hackathon/DEV.md) |
| Abhijeet | Organizer requirements, evaluator entry point, device evaluation, live demo, packaging and submission | [Abhijeet's workflow](docs/hackathon/ABHIJEET.md) |

- Dev works on `hackathon/dev-model`. Abhijeet works on
  `hackathon/abhijeet-integration` and owns merges into
  `hackathon/two-person-sprint`. Open PRs against that shared branch and
  review each other's work. Follow the [GitHub workflow](docs/hackathon/GITHUB.md)
  and the file ownership in the [handoff](docs/hackathon/HANDOFF.md).
- Dev owns the feature and model interface. Abhijeet owns `src/main.py` and the
  public `predict(video_path) -> dict` contract. Coordinate changes across that
  boundary through the handoff notes.
- Keep evaluation on-device and compatible with Python 3.11. Use exact
  organizer clip IDs and vocabulary labels. Keep raw videos and feature caches
  outside Git. Report measured values with the model, device and runtime used.
- Start with the first unfinished task in the relevant workflow. Verify the
  actual entry point or artifact, then leave the commit SHA, reproduction
  commands, measurements and blockers in the handoff.
