# Momentum SGD for the first pose-free PPCA trial

The opt-in `--optimizer momentum_sgd` updates one mean Fourier volume and two
loading Fourier volumes using the existing PPCA pose marginalization. The
default `--optimizer vdam` retains its coupled per-frequency solve, two
pseudo-half histories, disagreement adaptation and shell shrinkage. This note
describes the implemented optimizer boundary; a dense GEMM E-step is a separate
engine change and is not implied here.

For each particle and pose, the current E-step computes a posterior coordinate
mean \(m\in\mathbb R^2\), covariance \(C\in\mathbb R^{2\times2}\), and pose mass
\(\gamma\). The augmented first and second moments are

\[
\alpha=(1,m_1,m_2),\qquad
M=\begin{pmatrix}1&m^T\\m&C+mm^T\end{pmatrix}.
\]

The direct expected-residual image for channel \(a\) is the weighted sum of
\(\gamma[\alpha_aY_1-\sum_b M_{ab}(D A_b)]\), where \(A_b\) is the **projected**
current mean or loading and \(D\) is the CTF-squared/noise weight. The shared
adjoint backprojects that image to `residual_gradient[f,a]`. This is the
gradient used below. The separate gridded `lhs_tri[f,a,b]` is a local block
curvature surrogate; replacing the gradient by `rhs - lhs @ theta` after
backprojection would lose projection-interpolation cross terms. Both pseudo
halves contribute their **raw sums** once, before normalization. The E-step
owner is [`residual_statistics.py`](../../relax/ppca_refinement/residual_statistics.py)
and the accumulation owner is
[`dense_dataset.py`](../../relax/ppca_refinement/dense_dataset.py).

Let \(G_f=G_{0f}+G_{1f}\), \(H_f=H_{0f}+H_{1f}\), and let \(S\) be the current
Fourier band. In the existing unnormalized full-real FFT/total-coefficient
noise units, the optimizer uses one scalar

\[
c=\max\left(\frac{0.5}{N^4},\ \max_{f\in S}\operatorname{tr}H_f\right),\qquad
v_t=0.9v_{t-1}+0.1\,\eta\,G/c,\qquad \eta=\texttt{sgd_learning_rate}.
\]

Only active Fourier frequencies receive the new gradient. A zero-curvature
batch produces no velocity. The trace is a cheap conservative eigenvalue
bound for a positive-semidefinite 3×3 block and is invariant when the two
loading coordinates are rotated by an orthogonal matrix. It is **not** an
exact Hessian, and `c` is not a global Lipschitz constant for the projected
likelihood. A duplicated minibatch doubles both \(G\) and \(H\), leaving the
normalized new-gradient term unchanged. The zero-start first step is 0.1 of
the normalized learning-rate step, with no bias correction. There is no
split-half disagreement gate, second-moment adaptation, explicit map prior,
class occupancy factor, positivity clamp or posterior rewhitening.

Only \(\operatorname{tr}H_f\) is read, so the streamed full-row engine forms
just that channel for momentum SGD: each pose's LHS images are summed over the
diagonal of the packed block and backprojected once, instead of backprojecting
all \((q+1)(q+2)/2\) packed channels. The backprojection is linear per
channel, so this is the same trace up to float32 summation order. The result is
[`TracePPCAStats`](../../relax/ppca_refinement/full_row_stream.py)
(`metric_trace`, no `lhs_tri`); other statistics paths pass their packed metric
through [`metric_trace`](../../relax/ppca_initial_model/sgd_update.py).

The controller applies the existing common real-space solvent mask and
Fourier bandlimit once to the proposed velocity for storage, and once to the
old model plus the unprojected proposed velocity for the new model. The soft
solvent mask is not an orthogonal projection: repeated iterations can shrink
edge amplitudes, but the new step is not masked twice within an iteration.
Momentum outside the active band is cleared. The same operation
on every channel commutes with orthogonal loading rotations, preserving the
valid gauge of \(z\sim N(0,I_2)\). Scaling or translating latent coordinates
would change this fixed-prior model unless a separate prior transformation is
specified. Implementation: [`sgd_update.py`](../../relax/ppca_initial_model/sgd_update.py)
and [`iteration_loop.py`](../../relax/ppca_initial_model/iteration_loop.py).
The controller's opt-in `log_direction_prior=False` omits only the large
per-update JSON copy of the angular prior; the state and checkpoints still
retain it, and the optimizer update is unchanged.

This first boundary deliberately retains the current PPCA noise estimator
and observation-space convention. Its noise state is total coefficient
variance in raw FFT units, while the K-class Momentum SGD package stores a
different normalized variance. The current PPCA InitialModel scores and
reconstructs full, unmasked Fourier images. An older experimental PPCA branch
has different masked-scoring semantics; its noise change is not silently
imported here. A later noise adaptation needs a separate end-to-end unit and
mask derivation, including the posterior covariance term in expected residual
power. Checkpoints store exactly one optimizer's state and reject mismatched
configurations on resume. Older VDAM checkpoints lack the new config fields
and fail the strict source/config identity check; they are not migrated.

The planned comparison will pass its radius and angular grid explicitly to
both optimizers; this implementation does not encode a geometry choice. A
radius-31 trial retains 91.68% of the pinned truth triplet's between-state
Fourier power versus 50.25% at radius 24; these are input-spectrum diagnostics,
not a recovery guarantee. Grid adequacy is being measured separately.

## Opt-in coarse two-pass residency

The --stream-coarse-recompute flag requires --oversampling 0 and one device.
It routes both optimizers through the existing PPCA full-row score and moment
kernels with a single artificial coarse parent for every rotation and shift.
That parent means all coarse poses are supported, while keeping the support
mask only B×1×1; the padded sentinel row remains unsupported. The existing
--image-batch-size bounds resident images per tile and
--rotation-block-size bounds rotations per score kernel. The flag is off by
default, so the current coarse dense path and the fine streamed path retain
their existing behavior.

For each image tile, pass one keeps only the best pose, running score center
and centered partition. If block \(b\) has scores \(s_b\), the recurrence is

\[
c_b=\max(c_{b-1},\max s_b),\qquad
Z_b=e^{c_{b-1}-c_b}Z_{b-1}+\sum_{\phi\in b}e^{s_\phi-c_b}.
\]

An image with no supported pose in an early block carries \(c=-\infty\) and
\(Z=0\) until its first supported block. Pass two recomputes one block's
scores and augmented moments, normalizes by the final \(c+\log Z\), and
backprojects through the existing residual and moment owners. Its projected
mean/loadings are reused by the block backprojection. This retains no
all-rotation score/moment tensor, but it scores latent moments twice and
changes the float32 partition reduction order. Per-block Kahan
backprojection order remains; changing image tile size also changes the
within-block particle reduction order. It is a GEMM-inspired reuse of the
existing PPCA engine, not a new dense CUDA/GEMM implementation or a speed
qualification.

For a matched stochastic comparison, `--stochastic-batch-size 300
--stochastic-all-iterations` keeps the same 300-particle update policy through
iteration 600 for either optimizer; by default the final update still uses all
training particles. Final saved embeddings remain a separate all-particle
evaluation after the updates.
`--balanced-stochastic-halves` additionally draws 150 even and 150 odd local
particle IDs from one fresh shuffled order at each update, fixing both GPU
shapes for a batch of 300. This parity stratification is independent of
simulated states or poses and is shared by VDAM and Momentum SGD. The default
unstratified sampling remains unchanged. `--checkpoint-interval 10` limits
checkpoint writes, and `--skip-final-embeddings` omits the separate all-particle
E-step when evaluation is performed in a dedicated run.

An older, unmerged branch at `eb3f383f` introduced opt-in gridding-corrected
E-step projection (`cf4e9276`) and a separate native-noise mode
(`a5f747fb`, `775ab854`). Neither option is present in the current main PPCA
InitialModel. The old gridding option corrected the forward reference but kept
the original residual adjoint; therefore a mathematical gradient with respect
to the uncorrected map would also need the adjoint of that correction. This
package retains main's forward/backprojection convention and does not assert
that its gridded trace is the exact map-space Hessian. Reconciliation of those
old options requires matched-source, matched-observation validation rather
than copying their units or masking assumptions into this trial.
