"""Optional multi-class arguments and dispatch for the existing InitialModel CLI."""

import argparse


def add_args(parser):
    parser.add_argument("--K", dest="n_classes", type=int, default=1,
                        help="Number of PPCA classes; default 1 preserves the single-model workflow")
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
    parser.add_argument("--accuracy-interval", dest="accuracy_interval", type=int, default=10,
                        help="K>1 with --auto-sampling: updates between accuracy estimates (default 10)")
    parser.add_argument("--local-order", dest="local_order", type=int, default=3,
                        help="K>1 with --auto-sampling: coarse HEALPix order from which searches are local "
                        "(default 3: 3.75 degrees with the children, RELION's --auto_local_healpix_order 4)")


def dispatch(args):
    """Return False for the original single model; otherwise run the mixture."""
    n_classes = getattr(args, "n_classes", 1)
    class_pseudocount = getattr(args, "class_pseudocount", None)
    if n_classes < 1:
        raise ValueError("K must be a positive integer")
    if n_classes == 1:
        if class_pseudocount is not None:
            raise ValueError("--class-pseudocount requires K>1")
        return False

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
            raise ValueError(f"{flag} is a single-model control and is not supported with K>1")
    names = {"max_significant": "maxsig", "gemm_precision": "ppca_gemm_precision",
             "preread_images": "ppca_preread_images", "pass2_mass_floor": "ppca_pass2_mass_floor"}
    options = {
        field.name: getattr(args, names.get(field.name, field.name))
        for field in dataclasses.fields(Config)
        if field.name not in ("stages", "class_pseudocount") and hasattr(args, names.get(field.name, field.name))
    }
    options["class_pseudocount"] = defaults.class_pseudocount if class_pseudocount is None else class_pseudocount
    if args.stages:
        options["stages"] = tuple(tuple(stage) for stage in json.loads(args.stages))
    options["oversampling"] = 0 if args.full_grid else defaults.oversampling
    # RELION's default: masked tilt images unless --no-zero-mask; single-particle manifests have no mask.
    options["zero_mask"] = bool(args.ios) if args.zero_mask is None else bool(args.zero_mask)
    # Resolve this at the input boundary: the accuracy estimator supports tomography, not SPA manifests.
    options["auto_sampling"] = bool(args.ios) if args.auto_sampling is None else bool(args.auto_sampling)
    run(args, Config(**options))
    return True
