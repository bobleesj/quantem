import numpy as np

from quantem.widget import Show3DSlices


def test_show3dslices_live_volume_bytes_are_float32():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    w = Show3DSlices(data)

    assert w.offline is False
    assert len(w.volume_bytes) == data.nbytes
    decoded = np.frombuffer(w.volume_bytes, dtype=np.float32).reshape(data.shape)
    np.testing.assert_array_equal(decoded, data)


def test_show3dslices_offline_volume_bytes_are_uint8():
    data = np.linspace(-2.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)

    w = Show3DSlices(data, offline=True)

    assert w.offline is True
    assert len(w.volume_bytes) == data.size
    assert w._offline_min == float(data.min())
    assert w._offline_max == float(data.max())

    packed = np.frombuffer(w.volume_bytes, dtype=np.uint8)
    assert int(packed.min()) == 0
    assert int(packed.max()) == 255

    scale = (w._offline_max - w._offline_min) / 255.0
    decoded = packed.astype(np.float32) * scale + w._offline_min
    np.testing.assert_allclose(decoded, data.ravel(), atol=scale / 2 + 1e-6)


def test_show3dslices_offline_constant_volume_uses_zero_bytes():
    data = np.full((2, 3, 4), 5.0, dtype=np.float32)

    w = Show3DSlices(data, offline=True)

    assert w.offline is True
    assert w._offline_min == 5.0
    assert w._offline_max == 5.0
    assert np.frombuffer(w.volume_bytes, dtype=np.uint8).tolist() == [0] * data.size
