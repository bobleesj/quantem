import numpy as np
import pytest
import traitlets

from quantem.widget import Show3DSlices


def test_show3dslices_pixel_size_none_and_anisotropic_sequence():
    vol = np.random.rand(4, 6, 8).astype(np.float32)

    no_scale = Show3DSlices(vol, pixel_size=None)
    assert no_scale.pixel_size == 0.0
    assert no_scale.pixel_size_axes == pytest.approx([0.0, 0.0, 0.0])

    anisotropic = Show3DSlices(vol, pixel_size=(3.0, 0.5, 0.25))
    assert anisotropic.pixel_size_axes == pytest.approx([3.0, 0.5, 0.25])
    assert anisotropic.pixel_size == pytest.approx(0.375)

    with pytest.raises(ValueError, match="3 elements"):
        Show3DSlices(vol, pixel_size=(1.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        Show3DSlices(vol, pixel_size=(float("nan"), 1.0, 1.0))
    with pytest.raises(ValueError, match=">= 0"):
        Show3DSlices(vol, pixel_size=(-1.0, 1.0, 1.0))


def test_show3dslices_accepts_dataset3d_like_object():
    class Dataset3dLike:
        array = np.random.rand(4, 6, 8).astype(np.float32)
        name = "ptycho phase"
        sampling = (2.0, 0.5, 0.25)
        units = ("A", "A", "A")

    w = Show3DSlices(Dataset3dLike())

    assert (w.nz, w.ny, w.nx) == (4, 6, 8)
    assert w.title == "ptycho phase"
    assert w.pixel_size_axes == pytest.approx([2.0, 0.5, 0.25])
    assert w.pixel_size == pytest.approx(0.375)


def test_show3dslices_dataset3d_like_sampling_without_units_assumes_angstrom():
    class Dataset3dLike:
        array = np.random.rand(4, 6, 8).astype(np.float32)
        name = "unitless sampling"
        sampling = (3.0, 0.8, 0.8)

    w = Show3DSlices(Dataset3dLike())

    assert w.title == "unitless sampling"
    assert w.pixel_size_axes == pytest.approx([3.0, 0.8, 0.8])


def test_show3dslices_dataset3d_like_nm_sampling_converts_to_angstrom():
    class Dataset3dLike:
        array = np.random.rand(4, 6, 8).astype(np.float32)
        sampling = (1.2, 0.04, 0.04)
        units = ("nm", "nm", "nm")

    w = Show3DSlices(Dataset3dLike(), title="explicit title")

    assert w.title == "explicit title"
    assert w.pixel_size_axes == pytest.approx([12.0, 0.4, 0.4])
    assert w.pixel_size == pytest.approx(0.4)


def test_show3dslices_dataset3d_like_mixed_units_convert_per_axis():
    class Dataset3dLike:
        array = np.random.rand(4, 6, 8).astype(np.float32)
        sampling = (1.2, 0.4, 0.4)
        units = ("nm", "A", "A")

    w = Show3DSlices(Dataset3dLike())

    assert w.pixel_size_axes == pytest.approx([12.0, 0.4, 0.4])
    assert w.pixel_size == pytest.approx(0.4)


def test_show3dslices_crosshair_state_is_slice_specific():
    vol = np.random.rand(4, 8, 8).astype(np.float32)
    w = Show3DSlices(vol)

    assert w.viewer_kind == "slices"
    assert w.show_crosshair is True
    assert w.fft_window is False
    assert "show_crosshair" in w.state_dict()
    assert "fft_window" in w.state_dict()


def test_show3dslices_fft_window_constructor_and_state():
    vol = np.random.rand(4, 8, 8).astype(np.float32)
    w = Show3DSlices(vol, show_fft=True, fft_window=True)

    assert w.show_fft is True
    assert w.fft_window is True
    assert w.state_dict()["fft_window"] is True


def test_show3dslices_log_scale_save_image_uses_signed_log():
    vol = np.array([
        [[-3.0, -1.0], [0.0, 3.0]],
        [[-2.0, -0.5], [0.5, 2.0]],
    ], dtype=np.float32)
    w = Show3DSlices(vol, log_scale=True, auto_contrast=False, vmin=-3.0, vmax=3.0)
    normalized = w._normalize_slice(vol[0])

    signed = np.sign(vol[0]) * np.log1p(np.abs(vol[0]))
    vmin = -np.log1p(3.0)
    vmax = np.log1p(3.0)
    expected = np.clip((signed - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    assert np.array_equal(normalized, expected)


def test_show3dslices_rejects_dual_inputs():
    vol = np.random.rand(4, 8, 8).astype(np.float32)

    with pytest.raises(ValueError, match="single-object"):
        Show3DSlices(vol, data_b=vol)
    with pytest.raises(ValueError, match="difference/dual"):
        Show3DSlices(vol, show_diff=True)
    with pytest.raises(ValueError, match="one title"):
        Show3DSlices(vol, title_b="B")
    with pytest.raises(ValueError, match="linked_contrast"):
        Show3DSlices(vol, linked_contrast=False)


def test_show3dslices_state_surface_is_single_object_only():
    vol = np.random.rand(4, 8, 8).astype(np.float32)
    w = Show3DSlices(vol)

    state = w.state_dict()
    for key in ("dual_mode", "show_diff", "title_b", "linked_contrast"):
        assert key not in state

    for attr in (
        "volume_bytes_b",
        "_gif_export_requested",
        "_gif_data",
        "_zip_export_requested",
        "_zip_data",
    ):
        assert not hasattr(w, attr)


def test_show3dslices_z_stretch_clamps_to_widget_range():
    vol = np.random.rand(4, 8, 8).astype(np.float32)

    high = Show3DSlices(vol, z_stretch=99)
    assert high.z_stretch == pytest.approx(30.0)

    low = Show3DSlices(vol, z_stretch=0.1)
    assert low.z_stretch == pytest.approx(1.0)

    with pytest.raises(traitlets.TraitError, match="finite"):
        Show3DSlices(vol, z_stretch=float("nan"))


def test_show3dslices_load_state_rejects_dual_saved_state():
    vol = np.random.rand(4, 8, 8).astype(np.float32)

    with pytest.raises(ValueError, match="single 3D object"):
        Show3DSlices(vol, state={"dual_mode": True})

    with pytest.raises(ValueError, match="single 3D object"):
        Show3DSlices(vol, state={"show_diff": True})


def test_show3dslices_free_clears_volume_buffer():
    vol = np.random.rand(4, 8, 8).astype(np.float32)
    w = Show3DSlices(vol)
    assert w._data is not None
    assert len(w.volume_bytes) == vol.nbytes

    w.free()

    assert w._data is None
    assert w.volume_bytes == b""
