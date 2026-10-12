"""Optional multi-class arguments and dispatch for the existing InitialModel CLI."""

import argparse

SINGLE_MODEL_MAXSIG = 100  # relax.commands.ppca_initial_model --maxsig default
RELION_MAXSIG_PER_CLASS = 100  # RELION gradient InitialModel: maximum_significants = 100 * nr_classes


def add_args(parser):
    parser.add_argument("--K", dest="n_classes", type=int, default=None,
                        help="Number of PPCA classes. Any --K, 1 included, runs the multi-class controller "
                        "(relax/ppca_initial_class3d); without --K the original single-model workflow runs")
    parser.add_argument("--class-pseudocount", type=float,
                        help="Positive class-occupancy pseudocount for K>1 (default 1)")
    parser.add_argument("--full-grid", dest="full_grid", action="store_true",
                        help="K>1: score every stage's full pose grid instead of RELION's two adaptive passes "
                        "(the K>1 default; --oversampling is the single model's switch and is not read for K>1)")
    parser.add_argument("--oversampling-start", dest="oversampling_start", type=int,
                        help="K>1: first update that runs the two adaptive passes "
                        "(default: the second stage's first update)")
    parser.add_argument("--tomogram-batches", dest="tomogram_batches", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="K>1: draw each minibatch as whole tomograms so a tile holds a tomogram's particles "
                        "(default on; --no-tomogram-batches draws random particles)")
    parser.add_argument("--zero-mask", dest="zero_mask", action=argparse.BooleanOptionalAction, default=None,
                        help="K>1, tilt particles: RELION's --zero_mask, zeros outside the particle diameter on "
                        "every tilt image (default on for tilt particles, as in RELION; --no-zero-mask turns "
                        "it off; not implemented for single-particle manifests)")
    parser.add_argument("--local-search-start", dest="local_search_start", type=int,
                        help="K>1: first update with RELION's local angular searches around each particle's "
                        "last pose (default: only when --auto-sampling switches them on)")
    parser.add_argument("--local-search-sigma", dest="local_search_sigma_deg", type=float,
                        help="K>1: width in degrees of the local searches (default: twice the children's step)")
    parser.add_argument("--auto-sampling", dest="auto_sampling", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="K>1: RELION's --auto_sampling: angular order, offset step/range and local searches "
                        "from the measured accuracy every --accuracy-interval updates (default on for "
                        "tilt particles; --no-auto-sampling disables it; unavailable for single-particle manifests)")
    parser.add_argument("--adaptive-radius", dest="adaptive_radius", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="K>1: RELION's resolution-driven Fourier radius, the last update's resolution shell plus "
                        "10 (default on for tilt particles; --no-adaptive-radius keeps the stages' radii)")
    parser.add_argument("--gradual-pcs", dest="gradual_pcs", action="store_true",
                        help="--K: use no PC at first and open them one at a time, from --pc-start-fraction of the "
                        "run to 80%% of it; a class stops at the first PC that adds little log-likelihood "
                        "(default: every PC from the first update)")
    parser.add_argument("--pc-start-fraction", dest="pc_start_fraction_arg", type=float, default=0.3,
                        help="--K with --gradual-pcs: fraction of the run at which the first PC opens (default 0.3)")
    parser.add_argument("--center-latents", dest="center_latents", action=argparse.BooleanOptionalAction,
                        default=False, help="--K: after each M-step move W zbar into each class mean")
    parser.add_argument("--project-rigid-motions", dest="project_rigid_motions",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="--K: after each M-step remove the mean's rigid motions from the loadings")
    parser.add_argument("--gradual-pc-band", dest="pc_band_test", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="--K: let the PCs follow each rise of the resolution radius only if the new shells add "
                        "log-likelihood at the end of the stage; otherwise they keep the previous radius")
    parser.add_argument("--accuracy-interval", dest="accuracy_interval", type=int, default=10,
                        help="K>1 with --auto-sampling: updates between accuracy estimates (default 10)")
    parser.add_argument("--local-order", dest="local_order", type=int, default=3,
                        help="K>1 with --auto-sampling: coarse HEALPix order from which searches are local "
                        "(default 3: 3.75 degrees with the children, RELION's --auto_local_healpix_order 4)")


def dispatch(args):
    """Return False for the original single model (no --K given); otherwise run the mixture, K=1 included."""
    n_classes = getattr(args, "n_classes", None)
    class_pseudocount = getattr(args, "class_pseudocount", None)
    if n_classes is None:
        if class_pseudocount is not None:
            raise ValueError("--class-pseudocount requires --K")
        return False
    if n_classes < 1:
        raise ValueError("K must be a positive integer")

    import dataclasses
    import json

    from relax.ppca_initial_class3d.config import Config
    from relax.ppca_initial_class3d.runner import run

    defaults = Config()
    for flag, value, default in (
        ("--fine-image-tile-size", args.fine_image_tile_size, defaults.fine_image_tile_size),
        ("--sgd-learning-rate", args.sgd_learning_rate, defaults.sgd_learning_rate),
    ):
        if value != default:
            raise ValueError(f"{flag} is a single-model control and is not supported with --K")
    names = {"max_significant": "maxsig", "gemm_precision": "ppca_gemm_precision",
             "preread_images": "ppca_preread_images", "pass2_mass_floor": "ppca_pass2_mass_floor"}
    options = {
        field.name: getattr(args, names.get(field.name, field.name))
        for field in dataclasses.fields(Config)
        if field.name not in ("stages", "class_pseudocount") and hasattr(args, names.get(field.name, field.name))
    }
    options["class_pseudocount"] = defaults.class_pseudocount if class_pseudocount is None else class_pseudocount
    # RELION's gradient InitialModel caps an image's significant samples at 100 * K over all classes
    # (ml_optimiser.cpp:3692-3700). The mixture caps each class separately, so each class gets RELION's whole
    # budget, as an image's mass usually sits in one class: --maxsig left at the single model's 100 becomes 100 * K.
    if options.get("max_significant") == SINGLE_MODEL_MAXSIG:
        options["max_significant"] = RELION_MAXSIG_PER_CLASS * n_classes
    if args.stages:
        options["stages"] = tuple(tuple(stage) for stage in json.loads(args.stages))
    options["pc_start_fraction"] = float(args.pc_start_fraction_arg) if getattr(args, "gradual_pcs", False) else None
    options["oversampling"] = 0 if args.full_grid else defaults.oversampling
    # RELION's default: masked tilt images unless --no-zero-mask; single-particle manifests have no mask.
    options["zero_mask"] = bool(args.ios) if args.zero_mask is None else bool(args.zero_mask)
    # Resolve this at the input boundary: the accuracy estimator supports tomography, not SPA manifests.
    options["auto_sampling"] = bool(args.ios) if args.auto_sampling is None else bool(args.auto_sampling)
    # RELION's current image size follows its resolution; the stages' radii only for single-particle manifests.
    options["adaptive_radius"] = bool(args.ios) if args.adaptive_radius is None else bool(args.adaptive_radius)
    run(args, Config(**options))
    return True
