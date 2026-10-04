# docs/: rules for documents

The root guide applies.

## What this directory owns

| Path | Kind | Kept current by |
| --- | --- | --- |
| `user_guide.md`, `reference/api/em.md` | user documentation | hand |
| `math/*.md` | algorithm descriptions and design rationale; `math/relion_refinement_algorithm.md` is the algorithm-to-code map | hand, with the code change |
| `math/*scorecard*.md`, `benchmarks/relion_vs_relax*.md`, `benchmarks/masked_fsc.md` | pages rendered from the JSON beside them (or from `tests/baselines/relion_vs_relax_benchmarks.json`) | a script with a check mode |
| `math/*.json`, `benchmarks/*.json`, `benchmarks/initialmodel_scores/` | score ledgers, suite definitions, frozen masks | scripts; pinned records |
| `development/*.md` | procedures (`em_parity_runbook.md`, `della.md`, `polar*.md`, `benchmarks.md`), references (`codebase.md`, `em_implementation.md`, `relion_defaults.md`, `refactor_principles.md`), the status ledger (`em_status.md`), plans and refactor records | hand |
| `development/refinement_structure_metrics.json` | structure ceilings for `relax/refinement/` | `scripts/report_refinement_structure.py` |
| `patches/` | RELION capture patches and their README | reference |

## Rules

1. Run `python scripts/check_agent_guides.py` after editing any Markdown file. It checks every tracked `.md`.
2. Link a file in the repository with a relative link that resolves. A tracked file never links a scratch
   path or any other path outside the repository: write it as plain text in a code span. The checker reports
   an absolute-path link as a finding.
3. Do not edit a rendered page. Change its JSON, rerun the renderer, commit both, and confirm with the check:
   `python scripts/render_benchmark_table.py --check`, `python scripts/masked_fsc.py render --check`,
   `python scripts/summarize_em_k1_realdata_science_equivalence.py --check-markdown`, and `--check` on the
   three `scripts/summarize_*_scorecard.py`.
4. Link an algorithm step to the function that implements it, by file and name, not by line number. Keep the
   docstring pointing back. Update both in the commit that changes the behaviour.
5. Label superseded evidence as historical and keep it with its original source. An old next action in a
   historical note is not an instruction.
6. Put the numbers a conclusion rests on (commands, checksums, measurements, job IDs) in the tracked document.
   A scratch path is a pointer that can be deleted, not the evidence.

## Checks for a change here

- `python scripts/check_agent_guides.py` (exit 0, no finding).
- The renderer's check mode for a rendered page or its JSON.
- `pixi run python -m pytest -q tests/unit/test_em_agent_policy_docs.py` when you edit
  `development/em_parity_runbook.md`, `math/em_parity_best_metrics.md` or `relax/AGENTS.md`: it asserts that
  specific sentences are present.

## Pitfalls

- Tests pin document text. `tests/unit/test_em_agent_policy_docs.py` requires, among others, the strings
  `at most once every 3-4 hours` and `pixi run test-full` in the runbook; rewording them fails the test.
- Scratch evidence disappears. `development/em_status.md` cites a `HANDOFF.json` as "current integration,
  receipts, preserved jobs and required gates"; the file is gone and the line now says "no longer available".
- `development/final_local_sampling_patch_review.md` is 440 KB, three quarters of it fenced code excerpts.
  Other documents link to its anchors. Open the anchor you were sent to; do not read the file whole, and read
  the code, not the excerpt, for what a function does today.
- `development/em_status.md` is more than 100 KB and mixes current state with dated history under headings such as
  "Current readability refactor". Check a claim against the code or a receipt before acting on it.
- `pixi run -e docs docs-build` runs `mkdocs build --strict`, and no `mkdocs.yml` is tracked. The
  documentation site does not build from this checkout.
- A code span is not checked. A path or a command inside backticks can go stale silently; run a command you
  document, and `ls` a path you name.
