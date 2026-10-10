"""Full-data inference coverage, provenance guards and truthful completion artifacts."""

import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.ppca_initial_class3d import evaluate_membership as evaluator

pytestmark = pytest.mark.unit


def tile(ids, values):
    return 0, 0, SimpleNamespace(original_image_ids=np.asarray(ids),
                                 class_probabilities=np.asarray(values, np.float32))


def test_collect_scatters_all_particles_by_native_ids_and_marginal_class():
    probabilities = evaluator.collect(iter([tile([2], [[.4, .6]]), tile([0, 1], [[.9, .1], [.2, .8]])]),
                                      3, 2, lambda *_: pytest.fail("Premature partial save"))
    assert_matches(probabilities, np.asarray([[.9, .1], [.2, .8], [.4, .6]], np.float32))
    np.testing.assert_array_equal(probabilities.argmax(axis=1), [0, 1, 1])


@pytest.mark.parametrize("tiles", [
    [tile([0], [[.5, .5]])],
    [tile([0, 0], [[.5, .5], [.5, .5]])],
    [tile([0], [[.5, .5]]), tile([0, 1], [[.5, .5], [.5, .5]])],
    [tile([2], [[.5, .5]])],
    [tile([-1], [[.5, .5]])],
    [tile([0.5], [[.5, .5]])],
    [tile([0, 1], [[np.nan, .5], [.5, .5]])],
    [tile([0, 1], [[.4, .4], [.5, .5]])],
    [tile([0, 1], [[1.1, -.1], [.5, .5]])],
    [tile([0, 1], [[.5, .5]])],
])
def test_collect_rejects_missing_duplicate_out_of_range_and_invalid_posteriors(tiles):
    with pytest.raises(ValueError):
        evaluator.collect(iter(tiles), 2, 2, lambda *_: None)


def test_partial_diagnostics_preserve_nan_for_unvisited_rows():
    snapshots = []
    values = np.tile(np.asarray([[.3, .7]], np.float32), (500, 1))
    tiles = iter([tile(np.arange(500), values), tile([500], [[.8, .2]])])
    final = evaluator.collect(tiles, 501, 2, lambda p, v: snapshots.append((p.copy(), v.copy())))
    assert len(snapshots) == 1
    probabilities, visited = snapshots[0]
    assert visited.sum() == 500 and not visited[-1]
    assert np.isnan(probabilities[-1]).all()
    assert np.isfinite(final).all()


def test_species_names_are_known_provenance_only():
    names = ["tomo/thg_10", "another/betagal_2", "tomo/ribosome_1"]
    np.testing.assert_array_equal(evaluator.species_from_names(names), ["thg", "betagal", "ribosome"])
    with pytest.raises(ValueError, match="Unknown species"):
        evaluator.species_from_names(["tomo/mystery_1"])


@pytest.fixture
def mock_inference(tmp_path, monkeypatch):
    # Deferred imports let collection and pure-array tests avoid engine loading.
    # The integration case replaces every model/data computation with a stub.
    from relax.commands import ppca_initial_model as command
    from relax.ppca_initial_class3d import checkpoint, expectation, images
    from relax.ppca_initial_class3d import tomo_input as mixture_tomo_input
    from relax.ppca_initial_class3d.state import State
    from relax.ppca_initial_model import tomo as ppca_tomo
    from relax.relion import tomo_input

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = {name: tmp_path / f"{name}.star" for name in ("ios", "particles", "tomograms")}
    for name, path in paths.items():
        path.write_text(f"{name} identity\n")
    ckpt = run_dir / "checkpoint_0001.npz"
    ckpt.write_bytes(b"unchanged checkpoint")
    hashes = {f"{name}_sha256": hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    hashes["tilt_series_sha256"] = "nested-tilt-identity"
    config = dict(n_classes=2, q=1, iterations=2, stages=[[1, 2, 0]])
    (run_dir / "run.json").write_text(json.dumps({"config": config, "identity": hashes}))
    state = State(theta=None, moments=(), noise=None, class_prior=np.asarray([.4, .6], np.float32),
                  iteration=1, order=np.asarray([1, 2, 0]), rng_state={}, offset_variance=1.0,
                  radius=2, initialization={})
    dataset = SimpleNamespace(n_images=3, voxel_size=4.0, original_image_indices_from_local=lambda ids: ids)
    native = SimpleNamespace(particle_names=np.asarray(["tomo/thg_1", "tomo/betagal_1", "tomo/ribosome_1"]),
                             images=SimpleNamespace(process_images_half=object()))
    calls = {"load": 0, "inference": 0}
    settings = {}

    def load(*args, **kwargs):
        calls["load"] += 1
        assert kwargs["lazy"] is True
        args[2].write_text("derived flat metadata\n")
        return native

    def infer(data, model, configuration, groups, iteration, grid, *, moments, diameter_ang, candidates):
        calls["inference"] += 1
        assert data is dataset and model is state and moments is False
        assert iteration == state.iteration and configuration.n_classes == 2
        settings.update(config=configuration, diameter_ang=diameter_ang, candidates=candidates)
        assert len(groups) == 1
        np.testing.assert_array_equal(groups[0], np.arange(3))
        yield tile([2], [[.2, .8]])
        yield tile([0, 1], [[.1, .9], [.8, .2]])

    monkeypatch.setattr(command, "tilt_series_hash", lambda _: hashes["tilt_series_sha256"])
    monkeypatch.setattr(command, "refuse_unsupported_tilt_optics", lambda _: None)
    monkeypatch.setattr(tomo_input, "read_optimisation_set", lambda _: (paths["particles"], paths["tomograms"]))
    monkeypatch.setattr(checkpoint, "load", lambda *args: state)
    monkeypatch.setattr(mixture_tomo_input, "load_tomo_dataset", load)
    monkeypatch.setattr(ppca_tomo, "tilt_particles_from_tomo_dataset", lambda _: dataset)
    def grid(*args, sampling_state):
        settings["sampling_state"] = sampling_state
        return SimpleNamespace(rotations=np.zeros((2, 3, 3)), translations=np.zeros((1, 3)),
                               eulers=np.zeros((2, 3)), order=1)

    def mask(data, process_images_half, diameter_px):
        assert data is dataset and process_images_half is native.images.process_images_half
        settings["mask_diameter_px"] = diameter_px
        return data

    monkeypatch.setattr(expectation, "pose_grid", grid)
    monkeypatch.setattr(images, "zero_masked_tilt_particles", mask)
    monkeypatch.setattr(expectation, "mixture_tiles", infer)
    args = SimpleNamespace(run_dir=run_dir, checkpoint=ckpt.name, ios=paths["ios"],
                           output=tmp_path / "evaluation", expected_particles=3)
    return SimpleNamespace(args=args, paths=paths, checkpoint=ckpt, state=state, calls=calls, settings=settings,
                           dataset=dataset, expectation=expectation)


def test_evaluate_all_data_writes_tables_and_completion_last(mock_inference):
    case = mock_inference
    original = case.checkpoint.read_bytes()
    summary = evaluator.evaluate(case.args)
    assert summary["status"] == "COMPLETED" and summary["all_particles"] and summary["inference_only"]
    assert case.calls == {"load": 1, "inference": 1}
    assert case.checkpoint.read_bytes() == original
    assert case.settings["sampling_state"] is None
    assert not case.settings["config"].auto_sampling and not case.settings["config"].zero_mask
    assert case.settings["candidates"] is None and "mask_diameter_px" not in case.settings
    assert summary["checkpoint_sha256_before"] == summary["checkpoint_sha256_after"]
    with np.load(case.args.output / "memberships.npz", allow_pickle=False) as result:
        np.testing.assert_array_equal(result["particle_ids"], np.arange(3))
        np.testing.assert_array_equal(result["class_labels"], [1, 0, 1])
        np.testing.assert_array_equal(result["species_provenance"], ["thg", "betagal", "ribosome"])
        assert result["class_probabilities"].dtype == np.float32
    with (case.args.output / "class_species_composition.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [int(row["particles"]) for row in rows] == [1, 2]
    assert_matches(np.asarray([[row["THG_percent"], row["Beta-Gal_percent"], row["Ribosome_percent"]]
                               for row in rows], np.float32), np.asarray([[0, 100, 0], [50, 0, 50]], np.float32))
    assert json.loads((case.args.output / "summary.json").read_text()) == summary
    assert not list(case.args.output.rglob("*.tmp"))
    assert not list(case.args.output.glob("class*.mrc"))


def test_evaluate_uses_recorded_mask_sampling_and_local_search(mock_inference):
    from relax.ppca_initial_class3d.auto_sampling import SamplingState
    from relax.ppca_initial_class3d.membership import Membership

    case = mock_inference
    path = case.args.run_dir / "run.json"
    run = json.loads(path.read_text())
    run["config"].update(auto_sampling=True, zero_mask=True)
    run["identity"]["particle_diameter_ang"] = 120.0
    path.write_text(json.dumps(run))
    case.state.sampling = SamplingState(healpix_order=1, offset_step_px=0.7, offset_range_px=2.1,
                                        local_searches=True, sigma_deg=7.5)
    case.state.membership = Membership.empty(np.arange(3), 2)
    summary = evaluator.evaluate(case.args)
    assert case.settings["sampling_state"] is case.state.sampling
    assert case.settings["config"].auto_sampling and case.settings["config"].zero_mask
    assert case.settings["mask_diameter_px"] == 30.0 and case.settings["diameter_ang"] == 120.0
    # Without historical poses, the existing local-search helper searches all poses.
    assert case.settings["candidates"](np.arange(3)) == [None, None, None]
    assert summary["sampling"] == case.state.sampling.to_json() and summary["local_searches"]
    assert summary["zero_mask"] and case.state.iteration == 1


def test_local_search_starts_at_saved_iteration_not_next_iteration(mock_inference):
    case = mock_inference
    path = case.args.run_dir / "run.json"
    run = json.loads(path.read_text())
    run["config"]["local_search_start"] = 2
    path.write_text(json.dumps(run))
    summary = evaluator.evaluate(case.args)
    assert case.state.iteration == 1 and not summary["local_searches"]
    assert case.settings["candidates"] is None


@pytest.mark.parametrize("options,message", [
    ({"auto_sampling": True}, "recorded sampling state"),
    ({"zero_mask": True}, "training particle diameter"),
])
def test_inference_refuses_missing_recorded_search_or_mask_inputs(mock_inference, options, message):
    case = mock_inference
    path = case.args.run_dir / "run.json"
    run = json.loads(path.read_text())
    run["config"].update(options)
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate(case.args)
    assert case.calls == {"load": 0, "inference": 0}


def test_rejects_existing_output_without_overwrite(mock_inference):
    case = mock_inference
    case.args.output.mkdir()
    sentinel = case.args.output / "keep.txt"
    sentinel.write_text("user content")
    with pytest.raises(FileExistsError):
        evaluator.evaluate(case.args)
    assert sentinel.read_text() == "user content"
    assert case.calls == {"load": 0, "inference": 0}


def test_input_hash_mismatch_prevents_dataset_reads(mock_inference):
    case = mock_inference
    case.paths["particles"].write_text("modified input")
    with pytest.raises(ValueError, match="input hashes"):
        evaluator.evaluate(case.args)
    assert case.calls == {"load": 0, "inference": 0}
    assert not (case.args.output / "summary.json").exists()


def test_expected_count_mismatch_prevents_inference(mock_inference):
    case = mock_inference
    case.args.expected_particles = 13828
    with pytest.raises(ValueError, match="particle count"):
        evaluator.evaluate(case.args)
    assert case.calls == {"load": 1, "inference": 0}
    assert not (case.args.output / "summary.json").exists()


def test_changed_checkpoint_never_publishes_completion(mock_inference, monkeypatch):
    case = mock_inference
    original_collect = evaluator.collect

    def change_after_inference(*args, **kwargs):
        result = original_collect(*args, **kwargs)
        case.checkpoint.write_bytes(b"changed externally during inference")
        return result

    monkeypatch.setattr(evaluator, "collect", change_after_inference)
    with pytest.raises(ValueError, match="checkpoint changed"):
        evaluator.evaluate(case.args)
    assert not (case.args.output / "summary.json").exists()
    assert not (case.args.output / "memberships.npz").exists()


def test_cli_has_no_sampling_switch():
    with pytest.raises(SystemExit) as error:
        evaluator.main(["--run-dir", "run", "--checkpoint", "checkpoint", "--ios", "ios",
                        "--output", "output", "--expected-particles", "13828", "--sample", "900"])
    assert error.value.code == 2
