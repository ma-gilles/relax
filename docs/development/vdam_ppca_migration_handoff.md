# VDAM–PPCA migration status (2026-09-23)

RELAX `codex/vdam-ppca` is based on `origin/main` at
`f58b55e19f82466102ad0e1f4b280b9ac353a758`. It selectively ports the
frozen RECOVAR candidate-6 ab-initio q=2 PPCA controller and dense statistics,
including the independently qualified float32 block-normalization repair.
It also adds opt-in native VDAM controls and the float32 projector coordinate
correction. The large RECOVAR-era input and output artifacts remain at their
original paths. This is an implementation candidate, not an accepted RELAX
science or performance result.

The [source inventory and per-file hashes](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relax_ppca_migration_20260923/INVENTORY.md)
identify the old source snapshots and new owners. The
[block repair receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_ppca_blocks_20260923/README.md)
pins its candidate-6-relative patch and fixed-state tests. The shared
`AugmentedPPCAStats` optional fields are in isolated RECOVAR `dev2` feature
commit `0cb6a0ea263da53acfbc12c40006e3a413704ded`; that commit is local
pending review, so RELAX's published dependency pin and lock still point to
`a63df5a623abda24b87fefab23d2ef236e9c7a9c`. Cross-repository validation
must identify explicitly when it uses the local RECOVAR feature checkout. The
[repin sequence](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relax_ppca_migration_20260923/RECOVAR_REPIN_PENDING.md)
keeps the final lock reproducible after coordinated publication.

The [managed science gallery](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_ppca_managed_science_20260923/README.md)
shows the completed 2k paired panels and 5k native-K3-only panels. The
[historical pilot scorecard](../math/vdam_ppca_pilot_scorecard_v1.md) retains
its RECOVAR-source measurements. The [faster K3 block-size evaluation](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relax_ppca_migration_20260923/K3_BLOCKS500.md)
completed all-nine FSC fits and six-view panels: states 1–2 remain strong,
state 0 remains poor, so all-state quality equivalence is not accepted. The
5k PPCA run stopped cleanly at step 110
with checkpoint
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_ppca_rep_baseline_perf_20260923/ppca5k_prefix10/checkpoint_0110.npz`
(SHA256 `cd0639862c9cd5ebf645a925d26b0852f8e09b9ae0d4b4bbcc0fd59646eb3532`).
It has no final T200 maps or all-particle embeddings. Its continuation requires
checkpoint/schema and fixed-state math validation against the migrated source;
the checkpoint and existing outputs must remain immutable. The original
7200-second trajectory cap and shared 8 GPU-hour package budget still apply.
The [read-only schema probe](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relax_ppca_migration_20260923/checkpoint_schema_probe.json)
loaded every numeric step-110 field exactly into RELAX state types. The
[explicit import tool](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relax_ppca_migration_20260923/migrate_checkpoint110.py)
validated the original config, source and checkpoint in dry-run mode; it has
not created a migrated checkpoint. A source-frozen matched-state float32 check
is still required before import or continuation.

The first CPU checks used the local RECOVAR feature source and an isolated
RELION binding (SHA256 `5a2ab0397e72bbac87816fed2773d626f7472343f21832fa7897193b52664a7a`).
The first new PPCA controller, evaluation and normalization tests passed 37/37.
The initial affected VDAM/projector/dense run passed 200/201: the all-retained
local versus dense RHS comparison differed in 5/96 entries after the centered
dense posterior repair (maximum absolute difference `1.38e-4`, existing
tolerance `2e-5`). An in-memory fixed-input replay that centered only the local
posterior passed this exact case. The production local path now uses the same
centered-score helper as dense; after the RELAX main fast-forward to
`f58b55e19f82466102ad0e1f4b280b9ac353a758`, the complete affected CPU
selection passed **238/238**. No tolerance or baseline changed.

The next gate is a bounded matched-state float32 check of the migrated source,
then explicit checkpoint import and continuation of the same 5k trajectory toward its final update
and embeddings if the remaining budget and measured throughput allow it.
Assess per-state shellwise FSC, six-view shapes, learned mean and signed PCs,
and latent scatter before making a quality or runtime claim. Do not treat the
older 2k T60 radius-8 noise-0.01 pilot as the 5k T200 finer-grid noise-0.1 run.
