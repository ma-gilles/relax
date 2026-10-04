# Proposal: what to do with the large documents in docs/development

For the owner to decide, row by row. Written 2026-10-04 with the guide rewrite, measured on `main` at
`d2ebc5e`. None of it is done: the guide rewrite changed only a few lines here (output directories in
`della.md`, `agent_workflow.md`, `em_parity_runbook.md` and the two Polar pages now defer to the owner's global
instruction file). Sizes are bytes; "last" is the last commit that touched the file before the rewrite.

| Document | Size | Last | What an agent needs from it | Proposal |
| --- | --- | --- | --- | --- |
| `final_local_sampling_patch_review.md` | 440,586 | 10-04 | Nothing at work time. 75% is fenced code copied from the source; `em_status.md` (12), the algorithm map (4) and `codebase.md` (2) link to its anchors | **Generate, do not track.** Keep the prose that is not in the algorithm map by moving it there; produce excerpts per review from function names; remove the file from `main` |
| `em_status.md` | 127,304 | 10-04 (104 commits in 8 days) | Per workflow: what works, open gaps, next gate. The engine-removal list. The RELION-binding port table | **Split, keep and cap.** A status page of at most 150 lines, present tense; the two lists become plans; dated sections move to history |
| `refactor_extension_plan.md` | 74,982 | 10-04 | The current package and its acceptance test | **Keep and cap** at 200 lines: the live package only, edited in place; finished and superseded sections to history |
| `refactor_principles.md` | 64,695 | 10-04 | The rules: lines 1-423 and "Conditional complexity". Lines 424-898 are twenty lesson and example sections | **Keep and cap.** A rules file of about 10 KB; the lessons become an examples file opened on demand |
| `dense_single_volume_refactor_log.md` | 55,248 | 09-29 | Nothing; a record | **History** |
| `em_implementation.md` | 43,886 | 10-04 | Prose module descriptions; they drift with every move | **Merge, then delete**: what is true goes to the algorithm map and the per-directory guides |
| `codebase.md` | 38,758 | 10-04 | The workflow entry-point table | **Keep and cap** at 100 lines: the table without its two RECOVAR rows; drop the ownership paragraphs above it, which restate the review document |
| `final_search_patch_status.md` | 37,299 | 10-04 | Nothing current: the status of a merged package, with "historical source receipts" | **History** |
| `dense_gemm_experiment_plan_20260928.md`, `dense_gemm_coarse_engine_plan_20260928.md` | 25,637; 12,281 | 09-28; 09-30 | Concluded plans | **History** |
| `dense_single_volume_refactor_plan.md` | 19,755 | 10-04 | Superseded by the extension plan | **History** |
| `em_parity_runbook.md` | 19,661 | 09-29 | Validation ladder, oracle rules, investigation loop, benchmark design | **Keep.** Remove `./scripts/run_tests_parallel.sh` and `pixi run test-full`, which do not exist here (a test pins both strings) |
| `final_search_refactor_proposal.md`, `final_search_review_example.md` | 17,866; 6,279 | 10-03 | Both say they are superseded by the grid ownership proposal | **History** |
| `benchmarks.md`, `gpu_compatibility.md`, `relion_defaults.md` | 15,817; 13,223; 10,288 | 10-03; 10-04; 09-29 | Contracts and reference tables in use | **Keep** |
| `grid_ownership_proposal.md` | 10,681 | 10-03 | "Not yet accepted or implemented" | **Decide**: a plan with an owner, or history |
| `polar.md`, `polar_agents.md` | 10,322; 3,599 | 09-29 | Procedure in use | **Keep** |
| `della.md` | 8,373 | 09-23 | The job environment block | **Keep and cut**: remove what the owner's global file states and the RECOVAR paper-dataset half |
| `performance_design.md`, `resident_segments.md`, `em_half_texture_staging.md`, `gt_reporting.md` | 8,379; 6,510; 5,077; 3,631 | 09-23 to 09-27 | Design rationale | **Keep** |
| `retired_em_experiments.md`, `vdam_ppca_migration_handoff.md` | 5,755; 5,327 | 10-04 | Records | **History** |
| `agent_workflow.md` | 5,434 | 10-04 | Written for another tool and a coordination board that predates this repository | **Delete**; the `scripts/em_work_package.py` paragraph goes to `scripts/AGENTS.md` |

"History" means a directory whose files carry a first line saying they are records and not instructions, or
the existing private experiment archive. In total about 0.9 MB of the directory's 1.1 MB of Markdown would leave `main` or
the reading path.

## Reading path in bytes

"Before" is what `main` told an agent to read before the task, at `d2ebc5e`. "After" is the rewritten guides. The
owner's global file (15,913 bytes) loads in both and is not counted.

| Task | Before | After the guide rewrite | After this proposal |
| --- | --- | --- | --- |
| Loaded at session start | 13,675 (root guide) | 7,751 | 7,751 |
| One-line fix in a helper under `relax/helpers/` | 226,892: root, `relax/` guide, `em_status.md`, `codebase.md`, `agent_workflow.md`, `tests/` guide, `CONTRIBUTING.md` | 20,635: root, `relax/`, `tests/` | 20,635 |
| Engine change in `relax/sparse_pass2/` | 270,743: the above, the runbook, `benchmarks.md`, `della.md` | 37,343: root, `relax/`, `relax/sparse_pass2/`, `tests/`, `CONTRIBUTING.md`; 56,824 with the runbook for parity work | the same |
| Refactor step in `relax/refinement/` | 291,587: the helper path and `refactor_principles.md`; 807,155 with the extension plan and the review document, which `refactor-agent-prompt.txt` asks for | 90,834: root, `relax/`, `relax/refinement/`, `tests/`, `refactor_principles.md` | about 36,000 with a 10 KB rules file |

## Order

1. Decide the review document and `em_status.md` first: they are 52% of the directory, and `em_status.md`
   is its most edited file.
2. Cap `refactor_principles.md` next: it is the largest file the new guides still send an agent to.
3. Move the records in one commit at a quiet hour, with the link updates the checker requires.
