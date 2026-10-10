"""RECOVAR's image prefetch queue depth, as relax's loaders use it."""

import pytest

pytestmark = pytest.mark.unit


def test_prefetch_depth_defaults_to_four_and_validates(monkeypatch):
    """The loader's queue depth is selectable and fails closed on a bad token.

    The default is 4 since the P4-I merge; 2 restores the previous behaviour.
    """

    from recovar.data_io import image_backends

    monkeypatch.delenv(image_backends.PREFETCH_DEPTH_ENV, raising=False)
    assert image_backends.prefetch_depth() == image_backends.DEFAULT_PREFETCH_DEPTH == 4
    monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, "2")
    assert image_backends.prefetch_depth() == 2
    monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, "6")
    assert image_backends.prefetch_depth() == 6
    assert image_backends._PrefetchIterator(iter(()))._buffer_size == 6
    monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, "0")
    with pytest.raises(ValueError, match=image_backends.PREFETCH_DEPTH_ENV):
        image_backends.prefetch_depth()
    monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, "deep")
    with pytest.raises(ValueError, match=image_backends.PREFETCH_DEPTH_ENV):
        image_backends.prefetch_depth()


def test_prefetch_yields_the_same_items_at_every_depth(monkeypatch):
    """Depth changes the buffering, never the sequence."""

    from recovar.data_io import image_backends

    payload = [("batch", index) for index in range(17)]
    monkeypatch.delenv(image_backends.PREFETCH_DEPTH_ENV, raising=False)
    control = list(image_backends._PrefetchIterator(iter(payload)))
    for depth in ("1", "2", "4", "8"):
        monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, depth)
        assert list(image_backends._PrefetchIterator(iter(payload))) == control == payload
