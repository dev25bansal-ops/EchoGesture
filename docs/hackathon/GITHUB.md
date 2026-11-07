# GitHub workflow for Abhijeet and Dev

[Start here](README.md) · [File ownership and handoffs](HANDOFF.md)

## Publish the plan once

Abhijeet owns the shared integration branch. The repository owner must grant
Abhijeet write access, or publish the first branch and grant access for
Abhijeet's subsequent work. Accept any GitHub collaborator invitation and
authenticate Git as the account with access. Work can start locally before the
invitation arrives.

From a clean clone of EchoGesture, check the remote and whether the branch
already exists.

```sh
git remote -v
git fetch origin
git ls-remote --heads origin refs/heads/hackathon/two-person-sprint
```

If the last command prints a branch, use that branch as the sprint base.
If it prints nothing, the repository owner or a write-enabled Abhijeet can
publish the supplied patch from a fresh branch. Replace the example path with
the downloaded patch's absolute path. Apply it only to a checkout that does
not already contain the plan.

```sh
git switch -c hackathon/two-person-sprint origin/main
git am /absolute/path/to/EchoGesture-hackathon-plan.patch
git push -u origin hackathon/two-person-sprint
```

If the checkout already has the updated plan on its local
`hackathon/two-person-sprint` branch, run only the push command. If
`git am` stops with a conflict, run `git am --abort` and inspect the
existing files before trying to apply anything again.

## Each teammate starts from the shared branch

Abhijeet runs these commands once for the integration work.

```sh
git fetch origin
git switch -c hackathon/abhijeet-integration origin/hackathon/two-person-sprint
```

Dev runs these commands once for the model work.

```sh
git fetch origin
git switch -c hackathon/dev-model origin/hackathon/two-person-sprint
```

If your feature branch already exists locally, use `git switch <branch>`.
If it exists only on GitHub, use `git switch --track origin/<branch>`.
Use one working checkout per teammate for feature work. Agents working in
the same checkout should use the branch assigned to that teammate.

## Daily work and review

1. On your feature branch, run `git status --short --branch` before editing.
   Start with a clean working tree or finish and commit the current work
   before syncing. Run `git fetch origin` and
   `git merge origin/hackathon/two-person-sprint` to bring in merged work.
2. Work on one task ID at a time. Stage specific files, run
   `git diff --cached --check` and review `git diff --cached --stat`. Commit
   with the task ID and a concrete description.
3. Push your branch with `git push -u origin hackathon/abhijeet-integration`
   or `git push -u origin hackathon/dev-model`. For later commits use
   `git push` after confirming the current branch.
4. Open a PR with **base `hackathon/two-person-sprint`** and **compare your
   feature branch**. Include the task ID, verification command and result,
   any model or interface changes, and the next handoff. Dev reviews
   Abhijeet's PRs. Abhijeet reviews Dev's PRs and coordinates integration.
5. Abhijeet merges reviewed work after the relevant checks pass, including
   Dev's approval on Abhijeet's own work. Use **Create a merge commit** if
   the repository permits it and you plan to reuse the feature branches.
   If you squash, start the next task on a fresh branch from the updated
   sprint head so old commits stay out of the next PR.
6. Verify the combined work on the shared sprint branch. When the release
   bundle passes its clean-room check, Abhijeet can open the final PR from
   `hackathon/two-person-sprint` to `main`.

## If Git rejects a push or reports a conflict

- **Permission denied or 403.** Confirm the repository and account in use,
  accept any pending invitation, and verify write access. The local commits
  remain available. Keep working on your local branch while access is fixed.
- **Non-fast-forward.** Fetch and inspect the remote changes before deciding
  how to integrate. Use `git log --oneline --left-right HEAD...@{upstream}`
  if the branch has an upstream. Sync your feature branch by merging reviewed
  changes. Avoid force-pushing shared history.
- **Merge conflict.** Check `git status` and consult the file owner before
  resolving a model config, label map or evaluator contract. If you cannot
  safely resolve an in-progress merge, use `git merge --abort` from the
  clean-start workflow above and record the conflicting files.

Do not commit raw video, extracted landmark caches, private manifests, local
environment files, signing keys or large training checkpoints. Dev adds only
approved release weights and model metadata. Abhijeet regenerates the
submission results after the model is frozen.
