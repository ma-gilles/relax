Mixture PPCA InitialModel (experimental)

Public CLI: relax ppca_initial_model --K K --q q ...
K=1 selects Marc's unchanged single-model controller. K>1 selects this package.
Native ET uses --ios optimisation_set.star --particle-diameter DIAMETER_ANG.
SPA uses the existing training-manifest interface. No species labels or true poses
are training inputs. ET tilts share one class, pose and latent per physical particle.

Current CLI defaults (2026-10-09): K>1 tomography enables auto-sampling and the
zero mask. Translation sampling starts at the upstream --shift-step 2; the
accuracy estimator may adapt it. --shift-step 1 remains an explicit experiment,
not the default. Use --no-auto-sampling for a fixed sampling schedule (also when
using --full-grid or --local-search-start). SPA manifests do not support the
auto-sampling estimator and keep it off. K=1 still runs the upstream controller
with upstream numerical defaults.

The auto-sampling default is the owner's choice from the three-species K5/q5
single-seed comparison: majority-label purity 0.886 in about 2 hours, versus
0.903 in 4h33m with --shift-step 1. This is a runtime/class-separation tradeoff,
not a claim of universally better reconstruction or multi-seed qualification.
To resume an older run, retain its recorded sampling options; a run recorded
with auto_sampling=false now needs --no-auto-sampling explicitly.

Fork boundary: all multi-class additions, tests and adapters are in this
package. The only difference outside it is the seven-line argument/dispatch
hook in commands/ppca_initial_model.py. initialization.py preserves the
mixture's signed bootstrap without modifying the single-model initializer;
tomo_input.py resolves merged tilt-stack addresses only for this workflow.
The shared RELION loader and its repository tests remain identical to upstream.

The owner requires all new implementation, tests and validation scripts to stay
in this package. The standard repository pytest/tier discovery does NOT cover
these tests. Run them explicitly, with the pinned project environment:
    python -m relax.ppca_initial_class3d.tests.run_tests -q
No shell PYTHONPATH or -p flag is needed. This entry point reuses the unchanged
repository test fixtures and CPU/shared-GPU guard; it requires the source checkout.
The GPU launcher is tests/run_gpu_validation.sh (tests, simulate, or all mode).
It records source identities and builds matching native libraries in a unique run.
Optional second/third arguments reuse a completed run's source-verified natives
and require its CUDA architecture to match (for example, NATIVE_DIR 86).
GPU coverage focuses on K=2/q=1 and the K=2/q=2 independent Gaussian reference.
The legacy K=1 full-loop reduction check remains in the CPU suite, not the new
feature's GPU acceptance gate. No test tolerances were widened.

Every checkpoint stores means/loadings, noise, priors, optimizer state and RNG.
Default checkpoint_interval=1 is preserved. At D64, K=10, q=5, one checkpoint's
main arrays occupy about 208 MB (199 MiB), or about 42 GB for iterations 0-200.
At K=10/q=10 this is about 370 MB each, about 74 GB for iterations 0-200.
Metadata and diagnostic output add overhead. --checkpoint-interval can be set
explicitly to reduce storage; the last update and requested stops still save.

Intermediate class membership (K>1)

Each checkpoint now also stores the latest OBSERVED marginal class posterior
for every physical particle, using the existing training E-step. There is no
additional all-data inference and no change to the model/optimizer updates.
Easy-to-read copies are saved at the same cadence, including initialization,
requested stops and failure checkpoints:
    memberships/checkpoint_0001.npz
    memberships/checkpoint_0001.tsv
The NPZ can be loaded with numpy.load(..., allow_pickle=False). Arrays:
    particle_ids: zero-based native particle STAR row for ET (NOT the lexical
                  RECOVAR group index); original dataset image index for SPA.
    class_labels: argmax over class_probabilities, 0..K-1; -1 means unknown.
    class_probabilities: N x K float32; all NaN for an unvisited particle.
    last_evaluated_iteration: training E-step that last evaluated this row.
    model_iteration: parameter state used, last_evaluated_iteration minus one.
    evaluated: whether a row has ever been evaluated in the recorded history.
    evaluated_in_checkpoint_iteration: whether the saved update evaluated it.
TSV contains IDs, labels, both iteration fields, the current-update flag, and
one probability column per class. The NPZ metadata documents these semantics.

These are NOT fresh assignments of all particles under the saved post-update
model. Minibatch training leaves older rows unchanged; use the timestamp fields
to distinguish them. At update t, theta/noise/priors are from t-1 while the pose
grid/support belongs to stage(t), including at a stage boundary. Hard labels
use the marginal class posterior, not the joint class-and-pose MAP winner.
Final assignments.npz remains the separate full-data final-model inference.

Membership is stored atomically INSIDE checkpoint_*.npz under membership_*
keys; the small NPZ/TSV views are derived, not required for resume. Checkpoints
from before this extension still load, but their missing historical labels
are explicitly unknown until re-evaluated. New membership history survives
resume. A loaded state's membership.export(checkpoint_path, state.iteration)
can regenerate a missing sidecar without any image loading or inference.
Running jobs use their original code; this feature takes effect on a new run
or continuation with the updated package, not by changing a running snapshot.

Membership validation, 2026-10-07: package CPU suite 112 passed, one explicitly
GPU-only test skipped (87.88 s); repository CPU fast guard 103 passed (70.63 s);
ruff passed for every changed Python file. Coverage includes original-ID
scatter, unseen/stale rows, marginal versus joint MAP, invalid records,
checkpoint/NPZ/TSV roundtrip, legacy checkpoint loading, and tiny SPA/ET
stop/resume across a support-stage change. No production job was restarted
and no GPU performance claim is made. Doctor verified the pinned environment;
the separate repository fixture-dependent tiers are unavailable here because
their external fixture roots are not readable on this cluster.

iterations.jsonl includes class occupancy, per-class VDAM gates/signal/disagreement,
loading singular values/powers, and class/half posterior and tiling diagnostics.
Posterior class contributions use JOINT responsibilities per physical particle,
not class-conditional normalization. Inspect these with class_soft_counts to
detect collapse. No occupancy-dependent VDAM step multiplier is introduced;
that is an algorithmic policy change, not a logging or numerical bug fix.

Speed defaults since 2026-10-08 (neither changes the model a checkpoint holds, so a
run resumes across them; both are excluded from the checkpoint identity):
  Stages scale with --iterations: radii 4/8/16/32 start at 0%/30%/55%/80% of the
    updates (1, 61, 111, 161 for 200; 1, 16, 29, 41 for 50) with coarse HEALPix
    orders 1/2/2/2. --stages still overrides the schedule.
  Adaptive oversampling (on): from the second stage's first update, each class runs
    RELION's two passes. Pass 1 scores the stage's coarse grid at RELION's coarse
    image window; pass 2 scores only the order-1 children (3.75 degrees for order 2)
    of each particle's significant samples (--maxsig, adaptive fraction) and keeps
    every class's small pass-2 results, so no class is scored twice. Significance
    is decided per class under its conditional posterior; the moments use the joint
    class/pose posterior. Final assignments then index the child grid. The first
    stage scores its full grid (flat posteriors, cheap grid). --oversampling-start N
    moves the start; --full-grid scores every stage's full grid instead.
  Tomogram batches (on): each minibatch is drawn as whole tomograms (tilt groups)
    in a fresh random order, the last one cut at random; pseudo-halves still split
    by particle parity. The streamed engine projects every pose once per tile and a
    tile holds one tilt group, so with a random draw over hundreds of tomograms a
    tile held about one particle. --no-tomogram-batches draws random particles.
  Tests: tests/test_mixture_oversampled.py.

Extending a finished run (2026-10-09): --resume its last checkpoint with a larger
--iterations and the recorded --stages (run.json "stages"; the default schedule would be
rescaled to the new length and is refused as an identity mismatch), into a new --output.
As with RELION's --continue, VDAM's step and tau2-fudge schedule are those of the new
length for the added updates. Tests: test_mixture_relion_mechanisms.py (extension tests).

RELION mechanisms added 2026-10-08 (auto-sampling and the zero mask are now on
by default for multi-class tilt particles; explicit local-search starts remain off):
  --zero-mask / --no-zero-mask (images.py): RELION's --zero_mask, the tilt images
    zeroed outside the particle diameter under a 5-pixel raised-cosine edge before any
    Fourier operand; the start-up bootstrap and noise see the same masked images.
    Default on for --ios inputs (RELION's default) since 2026-10-09: on the D64
    three-species tomography set (K=5, q=5, 50 updates) it raised the majority-species
    accuracy in both seeds (0.829 -> 0.865, 0.686 -> 0.817) at no cost, while
    --maxsig 2000, local searches and auto-sampling gave no reliable gain. Not
    implemented for single-particle manifests (their images come from RECOVAR's
    dataset), so --zero-mask with a manifest is refused. The mask is part of the input
    identity: resuming a run started without it needs --no-zero-mask.
  Pose memory (membership.py): every evaluation stores each particle's marginal-MAP
    class's conditional MAP pose (RELION Euler degrees and the shift in pixels) in the
    membership record and its checkpoint/NPZ views; older checkpoints load with
    unknown poses.
  --local-search-start N [--local-search-sigma DEG] (local_search.py): from update N
    on, RELION's local angular searches: a particle's candidate coarse poses are the
    grid directions within 3 sigma of its stored direction and psi within 3 sigma of
    its stored psi (sigma: twice the children's step, RELION's updateAngularSampling);
    particles without a stored pose search the whole grid. Departures from RELION:
    a flat prior over the kept poses (the engine's pose prior is per row, not per
    particle) and a translation grid centred on zero rather than on the stored shift
    (every grid shift is a candidate without auto-sampling). The final inference pass
    searches locally too.
  --auto-sampling [--accuracy-interval 10 --local-order 3] (auto_sampling.py):
    on by default for K>1 --ios; --no-auto-sampling disables it.
    RELION's gradient auto-sampling. Every 10 updates, if the VDAM-gate resolution
    (gate < 0.5 is RELION's data_vs_prior < 1) stalled for two updates, RELION's
    calculateExpectedAngularErrors on 100 stored-pose particles per class (class
    means, P = 0.01 projection difference) sets the coarse order (up while the
    children's step is at least 90% of the angular accuracy, at most max_order 3),
    the offset step (90% of the shift accuracy, at least three quarters of the
    previous step) and range (three times the observed shift changes, bounded as
    RELION bounds it), and switches on local searches at local_order (coarse 3 with
    children is RELION's base order 4). The stages then only set the resolution
    support. Departures: the switch to local searches is immediate (RELION-MPI
    suspends it for two sampling updates) and the estimate uses the class means.
  Records: iterations.jsonl carries local_searches, candidate_rows_mean, the coarse
    order/step, the sampling state and each sampling decision (accuracies per class).
  Tests: tests/test_mixture_relion_mechanisms.py.

Model/statistics streams are still rebuilt for each noise group, and most class
scores are evaluated twice to bound memory. The first pass leaves class zero's
kept buffer for its pass 2. This costs 2K-1 score passes, NOT a measured >=2x
end-to-end slowdown. Performance and large-mixture recovery remain unqualified.
The mixture imports the existing _select_halves helper rather than modifying
the legacy module or maintaining a duplicate sampler. That coupling is explicit.

models.npz and class*_model_*.mrc preserve the fitted Fourier coefficients and
their exact inverse transforms. *_gridding_corrected.mrc display maps multiply
by the separable sinc-squared window (the inverse Fourier interpolation basis),
not divide by the radial window. Use the unmodified model for quantitative
reprojection; display samples are not independent halfmaps or unbiased truth.

See the external English LaTeX notes in the experiment folder for mathematics.
Small synthetic execution and checkpoint tests do not establish scientific
recovery of classes, poses or nonzero loading volumes.

Review validation, 2026-10-06

CPU suite: 74 passed, one explicitly GPU-only test skipped, 113.08 seconds.
  Experiment validation/cpu_review_ADhVlH/{tests.log,results.xml}
GPU numerical suite after test backend isolation: 73 passed, two deliberately
deselected checks (ten-PC inference and legacy K1 full-loop numerical reduction),
zero skipped/failed, 87.44 seconds. Job 37076522 COMPLETED 0:0, RTX A6000.
  Experiment validation/gpu_37076522/results.xml
The two CPU-specific serialization checks run in fresh CPU subprocesses even
inside the GPU suite. A default-device context alone was insufficient because
RECOVAR caches GPU availability when selecting its backprojection backend.
The production algorithm was not changed to work around that test setup.

Simulator job 37076280 completed all six-iteration SPA/ET training and output
checks, but its strict ET continuation comparison FAILED. Three direct branches
and one stopped/reloaded branch start from identical checkpoint 2. Same-path
direct controls also fail the strict numerical band. The largest run/resumed
second-moment difference was already 0.01908493042 at checkpoint 4 BEFORE the
extra reload; at checkpoint 6 it is 0.01904678345, matching the earlier value
times the existing 0.999^2 decay within 3.8e-9. Final hard labels, pose indices,
translations and physical IDs agree, and all outputs are finite. This supports
pre-reload GPU numerical variability; it does NOT prove complete restart parity,
since some resumed differences exceed the small measured control range.
No numerical tolerance was relaxed. The failed status remains in:
  Experiment validation/gpu_37076280/simulated_k2/validation.json
Job 37076280 also had two CPU-in-GPU-process setup errors; those are fixed and
covered by the successful numerical rerun 37076522 above.

Here "Experiment" is:
/home/ywang/three_species_particle_stacks/relax_tomo_abinitio_k10_D64
Source hashes and native-library identities are retained in each GPU run.
The only change outside this new package remains the approved seven-line CLI
hook. Existing repository test tiers/docs/ledger and the single-model algorithm
were not edited. Standard smoke/medium/long tiers and biological recovery are
not qualified by these package-local checks. Real K10/q5 training was not
resubmitted as part of this review.

Upstream synchronization validation, 2026-10-09

Base: ma-gilles/relax main a678fc16e5a0876c8009614753c1d8e5e148e5b8.
Package CPU suite: 188 passed, one GPU-only test skipped, 210.91 seconds.
Repository CPU fast guard: 98 passed, 94.14 seconds.
Ruff and git whitespace checks pass for the changed source. Differences from
upstream are confined to this package and the seven-line CLI hook; K=1 dispatches
to the unchanged upstream controller. No test tolerances were widened.
These results do not requalify GPU execution or scientific recovery. GPU tiers
were not rerun; the repository's external fixture roots are unavailable here.
