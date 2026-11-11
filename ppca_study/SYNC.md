# Working together on this branch

Three agents, on different clusters, work on this branch (`ppca/best-v1` of `ma-gilles/relax`, public) for the owner.
The branch is the only channel we share, so everything another agent needs must be in it. Read this file first.

## Roster

| agent | cluster | focus | own log |
|---|---|---|---|
| lead-della | Princeton Della (+ Polar) | PPCA ab initio lead: shift program, real-data tracker, best practice | `sync/lead-della.md` |
| flatiron-filament | Flatiron Rusty (H100) | cryo-ET attachment PPCA on filaments: realistic synthetic benchmark matched to real tilt-series data, real microtubule data, axoneme positive control | `sync/flatiron-filament.md` |
| (agent 3: add yourself) | | | `sync/<your-name>.md` |

## Files and who edits them

- `sync/<name>.md`: your own log. **Only you edit it** (so there are never merge conflicts in logs). Newest entry on top.
- `BEST_PRACTICE.md`: the standard. Changes only with an owner decision or a result that settles an OPEN item, each
  with a dated change-log line. Any agent may edit; keep edits small and commit them alone.
- `SHARED_RULES.md`: datasets, protocol, scoring and reporting rules. Same rule as BEST_PRACTICE.md.
- Code folders (`best_v1/`, `best_v2/`, ...): drivers are **never edited in place**. A change is a new file named with
  its sha256 prefix (`coupled_arm_<feature>_<sha8>.py`) plus a line in the folder's RECIPE with what changed and the
  off-path check (new flags off reproduce the parent within run-to-run noise, both numbers stated).
- This file: edit to add yourself to the roster or to improve the protocol; say what you changed in the commit message.

## Git protocol

1. Before you start, before each new experiment, before each commit, and at least every 2 hours while working:
   `git fetch origin && git rebase origin/ppca/best-v1` (or `git pull --rebase`). Then read every `sync/*.md` entry
   newer than your last read.
2. Commit small and push at once. If the push is rejected, rebase and push again. **Never force-push, never rewrite
   published history, never push to relax `main`.**
3. Commit messages start with `ppca_study:` and your agent name, e.g. `ppca_study: [lead-della] 10180 fixed-mean rows`.
4. Rebase conflicts: in someone else's log, you have edited a file you don't own; drop your change. In shared files,
   keep both sides; if they disagree in substance, keep both, mark `CONFLICT (owner to decide)`, and say so in your log.

## How to use your log

Each entry is dated (UTC) and uses these tags, so others can scan them:
- `CLAIM:` what you are starting (one line: set, method, seed, expected finish). Check the other logs for the same
  claim first; don't duplicate. `DONE:` / `DROPPED:` closes a claim.
- `RESULT:` numbers in the agreed form (SHARED_RULES.md section 4): map score next to the mean-only and true-model
  references, suite pose (direct fit only as a labelled diagnostic), seed, updates, driver sha and the exact flags.
  Enough that another agent can reproduce it without your cluster's paths.
- `ASK [name]:` a question or request for another agent. Answer in your own log with `RE [name] <date>:`.
  Write names in square brackets, never with an at-sign (GitHub would notify unrelated accounts).
- `NOTE:` a trap, bug or decision others should know about.
- `OWNER:` an owner decision you received (quote it), so the others apply it too.

## Rules everyone keeps

- This repository is public. No credentials, no private data, no personal details, and no mention of commercial
  cryo-EM packages other than RELION (state "an external consensus refinement" instead).
- RECOVAR code changes need the owner's OK. relax changes go on this branch or a feature branch, never `main`.
- Results from different clusters are only comparable with the same datasets, protocol and scorer (SHARED_RULES.md).
  If your cluster lacks a dataset or the scorer, say so with `ASK [lead-della]:` rather than rebuilding it differently.
- Predeclare the pass rule before running (in your log or a PREDECLARED file you reference), and report failures as
  plainly as successes.
- Keep the owner's standing preferences: shifted data first; map scores always next to mean-only and true-model
  references; raw PCs shown in the frame of their own mean, at raw scale (never aligned to a consensus); latent scatters
  in PC coordinates; no in-run detect-and-fix hacks.
