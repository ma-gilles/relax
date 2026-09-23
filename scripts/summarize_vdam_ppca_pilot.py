"""Render the standing VDAM/PPCA pilot scorecard without adding pass thresholds."""

import argparse
import json
from pathlib import Path


def render_small_pilot(pilot):
    """Keep the separate exploratory measurements reproducible from the JSON."""
    lines = [
        "## Separate 2,000-particle exploratory pilot",
        "",
        "The [small-pilot handoff](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_small_pilot_handoff.md) records "
        "the original pair and confirmed input-sign repair. These are separate from the 20k rows above.",
        "",
        f"Seed {pilot['seed']}; {pilot['iterations']} iterations; state counts {pilot['state_counts']}. "
        f"Training manifest SHA256: `{pilot['training_manifest_sha256']}`. "
        f"Subset-index SHA256: `{pilot['subset_indices_sha256']}`.",
        "",
        f"First-pair shared-frame active FSC AUCs: PPCA {pilot['first_pair']['ppca_active_raw_fsc_auc']}; "
        f"K3 {pilot['first_pair']['k3_active_raw_fsc_auc']}. "
        "The PPCA input sign was wrong; these are diagnostic only. The corrected pair completed "
        "all-particle final updates and PPCA embeddings.",
        "",
        "| State | Corrected PPCA shared-frame AUC | Corrected K3 shared-frame AUC |",
        "| --- | ---: | ---: |",
    ]
    for row in pilot["corrected_states"]:
        lines.append(
            f"| {row['state']} | {row['ppca_shared_active_raw_auc']:.4f} | {row['k3_shared_active_raw_auc']:.4f} |"
        )
    lines += [
        "",
        f"Original package allocation: {pilot['package_gpu_hours']:.3f} A100 GPU hours, "
        "including probes, tests and the repaired follow-up.",
        "",
    ]
    comparison = pilot.get("visual_comparison")
    if comparison:
        lines += [
            "### Current per-state shape comparison",
            "",
            "The user clarified that common frame/hand is diagnostic, not a recovery requirement. "
            "The [six-column panels and protocol](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_visual_comparison.md) "
            "use independent rigid rotation/translation/reflection for both methods and "
            "Hungarian matching after all nine K3/GT fits. Older shared-frame assignments above "
            "remain historical evidence.",
            "",
            "| State | PPCA active FSC AUC | K3 active FSC AUC | K3 class, 1-based | K3 mass |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in comparison["states"]:
            lines.append(
                f"| {row['state']} | {row['ppca_active_raw_auc']:.4f} | {row['k3_active_raw_auc']:.4f} | "
                f"{row['k3_class_one_based']} | {100 * row['k3_class_fraction']:.1f}% |"
            )
        lines += [
            "",
            f"CPU job {comparison['job_id']}: completed, no retraining or GPU allocation. "
            "FSC uses shells 1–8 (48 Å nominal cutoff). PPCA and K3 state-0 local fits exhausted "
            "their evaluation budgets; full fit flags and all-pair scores remain in the "
            f"[report]({comparison['report']}) (SHA256 `{comparison['report_sha256']}`). "
            "PPCA uses evaluation-label coordinate averages; K3 learns its classes. "
            "The comparison does not establish a general method ranking.",
            "",
        ]
    lines += [f"Scientific interpretation: {pilot['scientific_acceptance']}.", ""]
    return "\n".join(lines)


def render(scorecard):
    fixture = scorecard["fixture"]
    validation = scorecard["implementation_validation"]
    feasibility = scorecard["feasibility"]
    runs = scorecard["runs"]
    lines = [
        "# VDAM/PPCA three-state pilot scorecard",
        "",
        "> Historical RECOVAR-source evidence. No RELAX-source 5k PPCA final quality or paired runtime is yet measured.",
        "",
        "This pilot uses 20,000 particles at box 64 and does not assess the K1/K4 completion gates.",
        "No numerical recovery threshold has been selected.",
        "",
        f"Training manifest SHA256: `{fixture['training_manifest_sha256']}`. STAR SHA256: `{fixture['training_star_sha256']}`.",
        "",
        "| Evidence | Current result |",
        "| --- | --- |",
        f"| Affected CPU / EM guard / final delta | {validation['affected_cpu_passed']} / {validation['em_fast_guard_passed']} / {validation['final_delta_passed']} passed |",
        f"| PPCA GPU tests | {validation['gpu_ppca_passed']} passed |",
        f"| Corrected projector CPU tests | {validation['projector_cpu_passed']} passed |",
        f"| CPU resume | {validation['cpu_resume']} |",
        f"| GPU bitwise resume | {validation['gpu_bitwise_resume']} |",
        f"| Controlled GPU checkpoint replay | job {validation['controlled_gpu_resume_job']}: "
        f"{validation['controlled_gpu_resume_status']} |",
        f"| No-custom-CUDA replay | job {validation['gpu_no_custom_cuda_replay_job']}: "
        f"{validation['gpu_no_custom_cuda_replay_status']} |",
        f"| Late grid feasibility | job {feasibility['late_grid_job']}: {feasibility['late_grid_status']}; "
        f"{feasibility['late_grid_elapsed_seconds']:.1f} s for 12 particles, "
        f"{feasibility['late_grid_peak_gpu_mib']} MiB peak, "
        f"{feasibility['late_grid_fine_rotation_count']} fine rotations |",
        f"| Matched A100 continuation | job {feasibility['late_grid_warm_job']}: {feasibility['late_grid_warm_status']}; "
        f"iteration 2 {feasibility['late_grid_iteration_2_elapsed_seconds']:.1f} s / "
        f"{feasibility['late_grid_iteration_2_fine_rotation_count']} fine rotations; "
        f"iteration 3 {feasibility['late_grid_iteration_3_elapsed_seconds']:.1f} s / "
        f"{feasibility['late_grid_iteration_3_fine_rotation_count']} fine rotations |",
        f"| Coarse-only throughput | job {feasibility['coarse_throughput_job']}: "
        f"{feasibility['coarse_48_particles_seconds']:.1f} s for 48 pilot particles "
        f"({feasibility['coarse_48_support_total']} selected samples); fixed tiny-trained state |",
        f"| Representative box64 update | job {feasibility['representative_schedule_job']}: "
        f"2,000 late-stage particles in {feasibility['late_2000_update_seconds']:.1f} s; "
        f"{feasibility['late_2000_update_coarse_retained']} coarse retained, "
        f"{feasibility['late_2000_update_fine_rotations']} fine rotations; "
        f"{feasibility['representative_schedule_peak_gpu_mib']} MiB peak sampled |",
        f"| Final embeddings | job {feasibility['final_embedding_2000_job']}: "
        f"2,000 particles in {feasibility['final_embedding_2000_seconds_compile_inclusive']:.1f} s; "
        f"matched 8-particle warm complete/embedding-only "
        f"{feasibility['final_embedding_8_matched_full_warm_seconds']:.2f}/"
        f"{feasibility['final_embedding_8_matched_embedding_only_warm_seconds']:.2f} s, bitwise-equal embeddings |",
        f"| Final embedding cold/warm | job {feasibility['final_embedding_2000_repeat_job']}: "
        f"2,000 particles {feasibility['final_embedding_2000_repeat_cold_seconds']:.1f}/"
        f"{feasibility['final_embedding_2000_repeat_warm_seconds']:.1f} s; "
        f"identical embedding/ID hashes |",
        f"| First trajectory feasibility | about "
        f"{feasibility['projected_full_trajectory_gpu_hours_warm_embedding']:.1f} A100 GPU hours "
        f"including final update and embeddings; {feasibility['projection_status']} |",
        f"| K3 tiny one-iteration diagnostic | job {runs['k3_tiny64_one_iteration']['job']}: "
        f"{runs['k3_tiny64_one_iteration']['status']}; {runs['k3_tiny64_one_iteration']['precision']} |",
        f"| K3 tiny corrected float32 route | job {runs['k3_tiny64_float32_one_iteration']['job']}: "
        f"{runs['k3_tiny64_float32_one_iteration']['status']}; "
        f"seed maps exact, projector StableHLO f64 ops "
        f"{runs['k3_tiny64_float32_one_iteration']['projector_stablehlo_f64_operations']} |",
        f"| End-to-end evaluator on 12-particle fixture | job {validation['evaluation_pipeline_tiny_job']}: "
        f"{validation['evaluation_pipeline_tiny_status']} |",
        f"| Evaluator with corrected float32 K3 maps | job {validation['evaluation_pipeline_tiny_float32_job']}: "
        f"{validation['evaluation_pipeline_tiny_float32_status']} |",
        f"| PPCA seed 11, 20k | {'not measured' if runs['ppca_seed11_20k'] is None else runs['ppca_seed11_20k']} |",
        f"| K3 seeds 21–23, 20k | {'not measured' if runs['k3_seed21_23_20k'] is None else runs['k3_seed21_23_20k']} |",
        f"| PPCA seeds 21–23, 20k | {'not measured' if runs['ppca_seed21_23_20k'] is None else runs['ppca_seed21_23_20k']} |",
        f"| Per-state scientific quality | {len(scorecard['per_state_quality'])} measured rows |",
        "",
        f"Scientific acceptance: **{scorecard['scientific_acceptance']}**.",
        "",
    ]
    if "small_pilot_2k" in scorecard:
        lines.extend([render_small_pilot(scorecard["small_pilot_2k"]), ""])
    return "\n".join(lines).rstrip("\n") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/math/vdam_ppca_pilot_scorecard_v1.json"
    target = source.with_suffix(".md")
    content = render(json.loads(source.read_text()))
    if args.check:
        if not target.exists() or target.read_text() != content:
            raise SystemExit("Pilot scorecard Markdown is stale")
    else:
        target.write_text(content)


if __name__ == "__main__":
    main()
