"""Six-view map panels and per-state rigid-fit PPCA/K3 comparison.

Evaluation only. See docs/development/vdam_ppca_visual_comparison.md.
Reuse an existing evaluation's input identities and fitted transforms, fit the
remaining K3/GT pairs, then match classes after independent rigid registration.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from recovar.utils.helpers import load_mrc, load_relion_volume, write_mrc
from scipy.optimize import linear_sum_assignment

from relax.diagnostics.gt_metrics import relion_alignment_rotations
from relax.diagnostics.gt_registration import RigidVolumeTransform
from relax.ppca_initial_model.checkpoint import file_hash
from scripts.evaluate_vdam_ppca_pilot import (
    _bandlimit,
    _class_mean_coordinates,
    _fit,
    _fsc_report,
    _states_from_model,
)
from scripts.fsc_metrics import shell_fsc
from scripts.plot_vdam_ppca_model import save_model_diagnostics


def match_classes(pair_auc):
    """Return one class per GT; rows are estimated classes, columns are GT states."""
    matrix = np.asarray(pair_auc, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Class matching requires a finite square FSC matrix")
    classes, states = linear_sum_assignment(-matrix)
    return {int(state): int(cls) for cls, state in zip(classes, states)}


def final_class_masses(meta_path, map_paths, n_particles):
    """Read occupancy from the final all-particle update that wrote these maps."""
    meta_path = Path(meta_path).resolve()
    expected = [meta_path.with_name(meta_path.name.replace("_recovar_meta.json", f"_class{k:03d}.mrc"))
                for k in range(1, 4)]
    if not meta_path.name.endswith("_recovar_meta.json") or [Path(p).resolve() for p in map_paths] != expected:
        raise ValueError("K3 final metadata must belong to the supplied class maps in order")
    meta = json.loads(meta_path.read_text())
    ids = np.asarray(meta["selected_particle_ids"])
    if meta["subset_size"] != -1 or ids.ndim != 1 or ids.dtype.kind not in "iu" or not np.array_equal(np.sort(ids), np.arange(n_particles)):
        raise ValueError("K3 occupancy requires the final all-particle update")
    masses = np.asarray(meta["class_posterior_sums_full"], dtype=np.float64)
    if masses.shape != (3,) or not np.isfinite(masses).all() or np.any(masses < 0) or not np.isclose(masses.sum(), n_particles, rtol=1e-5, atol=1e-3):
        raise ValueError("Invalid final K3 class posterior masses")
    return masses


def six_views(volume):
    """Same projection/slice order as recovar.output.plot_utils.plot_summary_t."""
    return [volume.sum(axis=axis) for axis in range(3)] + [
        np.take(volume, volume.shape[axis] // 2, axis=axis) for axis in range(3)
    ]


def panel(rows, reference, title, *, voxel_size, radius):
    """Use shared per-column contrast limits and one positive gain per volume."""
    images = [six_views(volume) for _, volume in rows]
    ref_images = six_views(reference)
    limits = [tuple(float(v) for v in np.percentile(im, [0.2, 99.8])) for im in ref_images]
    if any(high <= low for low, high in limits):
        raise ValueError("Reference views have no display contrast")
    fig, axes = plt.subplots(len(rows), 6, figsize=(17.5, 2.5 * len(rows)), squeeze=False)
    for row, (label, _) in enumerate(rows):
        for col, im in enumerate(images[row]):
            ax = axes[row, col]
            ax.imshow(im, cmap="gray", vmin=limits[col][0], vmax=limits[col][1], origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Projection {col}" if col < 3 else f"Central slice {col - 3}", fontsize=12)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=12)
    n = reference.shape[0]
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.012,
        f"All maps low-pass radius {radius} ({n * voxel_size / radius:.0f} Å); "
        f"box {n}, pixel {voxel_size:g} Å; central index {n // 2}.\n"
        "One positive display gain per estimate, identical before/after alignment; "
        "GT-based gray limits shared down each column. No per-view normalization.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.065, top=0.955, wspace=0.025, hspace=0.12)
    return fig, limits


def compare(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; earlier evaluations are immutable")
    prior = json.loads(Path(args.prior_report).read_text())
    train = json.loads(Path(args.training_manifest).read_text())
    verification = json.loads(Path(args.fixture_verification).read_text())
    radius = int(prior["active_radius"])
    n, voxel = int(train["box"]), float(train["voxel_size"])
    if not 1 <= radius < n // 2 - 1:
        raise ValueError("An explicit valid active radius is required")
    evaluation = Path(args.evaluation_dir)
    checks = {
        "training_manifest_sha256": args.training_manifest,
        "fixture_verification_sha256": args.fixture_verification,
        "evaluation_manifest_sha256": evaluation / "manifest.json",
        "checkpoint_sha256": args.checkpoint,
        "embeddings_sha256": args.embeddings,
    }
    for key, path in checks.items():
        if file_hash(path) != prior[key]:
            raise ValueError(f"Prior evaluation input mismatch: {key}")
    if verification["identity"]["manifest_sha256"] != prior["training_manifest_sha256"]:
        raise ValueError("Fixture identity mismatch")
    for name in ("truth.npz", "manifest.json", "state0.mrc", "state1.mrc", "state2.mrc"):
        if file_hash(evaluation / name) != verification["files"][f"evaluation/{name}"]:
            raise ValueError(f"Evaluation input mismatch: {name}")
    if [file_hash(path) for path in args.vdam_maps] != prior["vdam_map_sha256"]:
        raise ValueError("K3 map identity/order mismatch")
    postcheck = json.loads(Path(args.postcheck).read_text())
    for key in ("checkpoint", "embeddings"):
        if postcheck[f"ppca_{key}_sha256"] != prior[f"{key}_sha256"]:
            raise ValueError("Trajectory postcheck mismatch")
    masses = final_class_masses(args.k3_final_meta, args.vdam_maps, int(train["n_images"]))
    with np.load(args.checkpoint, allow_pickle=False) as saved:
        theta = saved["theta"].copy()
        metadata = json.loads(str(saved["metadata"]))
    if metadata["iteration"] != metadata["config"]["iterations"]:
        raise ValueError("Expected final checkpoint")
    if metadata["identity"]["manifest_sha256"] != prior["training_manifest_sha256"]:
        raise ValueError("Checkpoint training identity mismatch")
    with np.load(args.embeddings, allow_pickle=False) as saved:
        ids, coords = saved["particle_ids"].copy(), saved["z"].copy()
    with np.load(evaluation / "truth.npz", allow_pickle=False) as saved:
        labels = saved["labels"].copy()
    means = _class_mean_coordinates(ids, coords, labels)
    ppca = _states_from_model(theta, means, n)
    gt = np.stack([load_mrc(str(evaluation / f"state{k}.mrc")) for k in range(3)])
    k3 = np.stack([load_relion_volume(path) for path in args.vdam_maps])
    if ppca.shape != gt.shape or k3.shape != gt.shape:
        raise ValueError("Volume geometry mismatch")
    mask_path = Path(args.prior_report).parent / "common_mask.mrc"
    if file_hash(mask_path) != prior["common_mask_sha256"]:
        raise ValueError("Common mask identity mismatch")
    mask = load_mrc(str(mask_path))
    rotations = relion_alignment_rotations(2)
    output.mkdir(parents=True)
    gt_band = [_bandlimit(volume, radius) for volume in gt]

    def fit_pair(moving, state, cached=None):
        gt_hash = file_hash(evaluation / f"state{state}.mrc")
        if cached is None:
            transform, receipt = _fit(_bandlimit(moving, radius), gt_band[state], rotations, voxel, gt_hash)
        else:
            transform = RigidVolumeTransform.from_dict(cached[0])
            receipt = cached[1]
        aligned = transform.apply(moving, voxel_size=voxel, gt_sha256=gt_hash)
        metrics = _fsc_report(aligned, gt[state], voxel, mask, radius)
        return aligned, {
            "transform": transform.to_dict(),
            "fit_receipt": receipt,
            "metrics": metrics,
            "reused_prior_fit": cached is not None,
            "active_curve": shell_fsc(_bandlimit(aligned, radius), gt_band[state])[: radius + 1].tolist(),
        }

    ppca_fits, ppca_aligned = [], []
    k3_fits, k3_aligned = {}, {}
    prior_by_state = {row["true_state"]: row for row in prior["states"]}
    for k in range(3):
        row = prior_by_state[k]
        aligned, result = fit_pair(ppca[k], k, (row["ppca_individual_transform"], row["ppca_individual_fit_receipt"]))
        ppca_fits.append(result)
        ppca_aligned.append(aligned)
        for cls in range(3):
            cached = None
            if row["matched_vdam_class_zero_based"] == cls:
                cached = row["vdam_individual_transform"], row["vdam_individual_fit_receipt"]
            aligned, result = fit_pair(k3[cls], k, cached)
            k3_fits[cls, k], k3_aligned[cls, k] = result, aligned
            print(
                f"Fitted K3 class {cls + 1} to GT {k}: AUC={result['metrics']['raw']['active_band']['fsc_auc']:.6f}",
                flush=True,
            )
    pair_auc = np.array(
        [[k3_fits[cls, k]["metrics"]["raw"]["active_band"]["fsc_auc"] for k in range(3)] for cls in range(3)]
    )
    matched = match_classes(pair_auc)
    states, overview_rows = [], []
    with PdfPages(output / "comparison_six_views.pdf") as pdf:
        model_diagnostics = save_model_diagnostics(
            theta, ids, coords, labels, n=n, voxel=voxel, radius=radius, output=output, pdf=pdf
        )
        for k in range(3):
            cls = matched[k]
            raw_p, raw_k = _bandlimit(ppca[k], radius), _bandlimit(k3[cls], radius)
            norm_p, norm_k = np.linalg.norm(raw_p), np.linalg.norm(raw_k)
            if norm_p == 0 or norm_k == 0:
                raise ValueError("Cannot normalize a zero-volume estimate for display")
            gain_p, gain_k = float(np.linalg.norm(gt_band[k]) / norm_p), float(np.linalg.norm(gt_band[k]) / norm_k)
            aligned_p = _bandlimit(ppca_aligned[k], radius)
            aligned_k = _bandlimit(k3_aligned[cls, k], radius)
            auc_p = ppca_fits[k]["metrics"]["raw"]["active_band"]["fsc_auc"]
            auc_k = k3_fits[cls, k]["metrics"]["raw"]["active_band"]["fsc_auc"]
            mirror_p = ppca_fits[k]["transform"]["mirror_x"]
            mirror_k = k3_fits[cls, k]["transform"]["mirror_x"]
            fraction = float(masses[cls] / masses.sum())
            rows = [
                (f"GT state {k}", gt_band[k]),
                (f"PPCA unaligned\ngain ×{gain_p:.3g}", raw_p * gain_p),
                (f"PPCA aligned\nFSC AUC {auc_p:.3f}\nreflection: {mirror_p}", aligned_p * gain_p),
                (f"K3 class {cls + 1} unaligned\nclass mass {100 * fraction:.1f}%\ngain ×{gain_k:.3g}", raw_k * gain_k),
                (f"K3 aligned\nFSC AUC {auc_k:.3f}\nreflection: {mirror_k}", aligned_k * gain_k),
            ]
            title = f"GT state {k} | {len(labels):,} particles | independent rigid fits, reflection allowed for both methods"
            fig, limits = panel(rows, gt_band[k], title, voxel_size=voxel, radius=radius)
            filename = f"state_{k}_six_views.png"
            fig.savefig(output / filename, dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            overview_rows.extend(
                [
                    (f"GT {k}", gt_band[k]),
                    (f"PPCA {k}\nAUC {auc_p:.3f}", aligned_p * gain_p),
                    (f"K3 → GT {k}\nAUC {auc_k:.3f}", aligned_k * gain_k),
                ]
            )
            for name, volume in (
                ("gt", gt[k]),
                ("ppca_unaligned", ppca[k]),
                ("ppca_aligned", ppca_aligned[k]),
                ("k3_unaligned", k3[cls]),
                ("k3_aligned", k3_aligned[cls, k]),
            ):
                write_mrc(str(output / f"state_{k}_{name}.mrc"), volume, voxel)
            states.append(
                {
                    "true_state": k,
                    "particle_count": int(np.sum(labels == k)),
                    "z_bar": means[k].tolist(),
                    "matched_k3_class_zero_based": cls,
                    "k3_class_fraction": fraction,
                    "ppca": ppca_fits[k],
                    "k3": k3_fits[cls, k],
                    "panel": filename,
                    "display_gains": {"ppca": gain_p, "k3": gain_k},
                    "display_limits_by_column": limits,
                }
            )
    fig, _ = panel(
        overview_rows,
        gt_band[0],
        "GT / independently aligned estimates | all three states",
        voxel_size=voxel,
        radius=radius,
    )
    fig.savefig(output / "aligned_overview.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), squeeze=False)
    for state, ax in zip(states, axes[0]):
        for name, color in (("ppca", "#1769aa"), ("k3", "#d67500")):
            curve = state[name]["active_curve"]
            ax.plot(np.arange(1, radius + 1), curve[1:], label=name.upper(), color=color, lw=2)
        ax.set(title=f"GT state {state['true_state']}", xlabel="Fourier shell", ylabel="FSC", ylim=(-0.3, 1.03))
        ax.axhline(0.143, color="gray", ls=":")
        ax.legend()
    fig.suptitle(f"Independent rigid alignment for both methods | trained cutoff {n * voxel / radius:.0f} Å")
    fig.tight_layout()
    fig.savefig(output / "individually_aligned_fsc.png", dpi=160)
    plt.close(fig)
    result = {
        "schema": "recovar-vdam-ppca-visual-comparison-v1",
        "policy": "Per-state rigid rotation/translation/reflection for both methods; K3 Hungarian matching after all nine class/GT fits. Shared frame is diagnostic, not a recovery requirement.",
        "precision": "Original float32/complex64 reconstructions; existing CPU float64 registration/FSC diagnostics; positive display gains never enter metrics.",
        "active_radius": radius,
        "voxel_size": voxel,
        "prior_report_sha256": file_hash(args.prior_report),
        "postcheck_sha256": file_hash(args.postcheck),
        "k3_final_meta_sha256": file_hash(args.k3_final_meta),
        "input_sha256": {key: prior[key] for key in checks},
        "vdam_map_sha256": prior["vdam_map_sha256"],
        "k3_class_by_gt_auc": pair_auc.tolist(),
        "k3_all_pair_fits": [[k3_fits[cls, k] for k in range(3)] for cls in range(3)],
        "states": states,
        "model_diagnostics": model_diagnostics,
        "limitations": "One synthetic seed; PPCA class averages use evaluation labels; K3 classes are learned. Fit convergence flags and class occupancy must be inspected; no scientific acceptance threshold is set.",
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# PPCA versus K3: independent alignment",
        "",
        result["policy"],
        "",
        "| GT state | PPCA FSC AUC | K3 FSC AUC | K3 class (1-based) | Class mass |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for state in states:
        p = state["ppca"]["metrics"]["raw"]["active_band"]["fsc_auc"]
        k = state["k3"]["metrics"]["raw"]["active_band"]["fsc_auc"]
        lines.append(
            f"| {state['true_state']} | {p:.4f} | {k:.4f} | {state['matched_k3_class_zero_based'] + 1} | {100 * state['k3_class_fraction']:.1f}% |"
        )
    lines += [
        "",
        f"FSC AUC: shells 1–{radius}; {n * voxel / radius:.0f} Å nominal cutoff.",
        "",
        result["limitations"],
        "",
        "[Model, latents and state panels PDF](comparison_six_views.pdf) · [Aligned overview](aligned_overview.png) · [FSC curves](individually_aligned_fsc.png)",
        "",
        "![Mean and signed PCs, unaligned](model_components_six_views.png)",
        "",
        "![Inferred latent coordinates](latent_coordinates.png)",
        "",
    ]
    for state in states:
        lines += [f"## State {state['true_state']}", "", f"![Six views]({state['panel']})", ""]
    (output / "README.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "embeddings", "training_manifest", "evaluation_dir", "fixture_verification"):
        parser.add_argument(name)
    parser.add_argument("--prior-report", required=True)
    parser.add_argument("--postcheck", required=True)
    parser.add_argument("--k3-final-meta", required=True)
    parser.add_argument("--vdam-maps", nargs=3, required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    compare(args)
    print(f"Saved comparison and six-column panels to {args.output}", flush=True)


if __name__ == "__main__":
    main()
