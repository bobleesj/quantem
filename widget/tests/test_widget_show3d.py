import json

import numpy as np
import pytest
import torch

from quantem.widget import Show3D


def test_show3d_numpy():
    data = np.random.rand(10, 32, 32).astype(np.float32)
    w = Show3D(data)
    assert w.n_slices == 10
    assert w.height == 32
    assert w.width == 32
    assert len(w.frame_bytes) > 0


def test_show3d_torch():
    data = torch.rand(10, 32, 32)
    w = Show3D(data)
    assert w.n_slices == 10
    assert w.height == 32
    assert w.width == 32


def test_show3d_rejects_2d():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show3D(data)
    assert w.n_slices == 1


def test_show3d_rejects_4d():
    data = np.random.rand(2, 4, 8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="3D"):
        Show3D(data)


def test_show3d_initial_slice_middle():
    data = np.random.rand(11, 16, 16).astype(np.float32)
    w = Show3D(data)
    assert w.slice_idx == 5


def test_show3d_labels_default():
    data = np.random.rand(4, 8, 8).astype(np.float32)
    w = Show3D(data)
    assert w.labels == ["0", "1", "2", "3"]


def test_show3d_labels_explicit():
    data = np.random.rand(3, 8, 8).astype(np.float32)
    w = Show3D(data, labels=["A", "B", "C"])
    assert w.labels == ["A", "B", "C"]


def test_show3d_title_cmap():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, title="Stack", cmap="viridis")
    assert w.title == "Stack"
    assert w.cmap == "viridis"


def test_show3d_pixel_size():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, pixel_size=2.5)
    assert w.pixel_size == pytest.approx(2.5)


def test_show3d_log_scale_auto_contrast():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, log_scale=True, auto_contrast=True)
    assert w.log_scale is True
    assert w.auto_contrast is True


def test_show3d_vmin_vmax_default_none():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    assert w.vmin is None
    assert w.vmax is None


def test_show3d_vmin_vmax_constructor():
    data = np.random.rand(5, 8, 8).astype(np.float32) * 100
    w = Show3D(data, vmin=10, vmax=80)
    assert w.vmin == pytest.approx(10)
    assert w.vmax == pytest.approx(80)


def test_show3d_fps():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, fps=12.0)
    assert w.fps == pytest.approx(12.0)


def test_show3d_playback_defaults():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    assert w.playing is False
    assert w.reverse is False
    assert w.boomerang is False
    assert w.loop is True


def test_show3d_play_pause():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    w.playing = True
    assert w.playing is True
    w.playing = False
    assert w.playing is False


def test_show3d_timestamps():
    data = np.random.rand(4, 8, 8).astype(np.float32)
    ts = [0.0, 1.0, 2.5, 5.0]
    w = Show3D(data, timestamps=ts, timestamp_unit="ms")
    assert list(w.timestamps) == ts
    assert w.timestamp_unit == "ms"


def test_show3d_size_explicit():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, size=800)
    assert w.size == 800


def test_show3d_diff_mode_default():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    assert w.diff_mode == "off"


def test_show3d_diff_mode_set():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, diff_mode="previous")
    assert w.diff_mode == "previous"
    w.diff_mode = "first"
    assert w.diff_mode == "first"


def test_show3d_diff_mode_invalid_raises():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    import traitlets as _t
    with pytest.raises(_t.TraitError):
        Show3D(data, diff_mode="bad")


def test_show3d_show_fft_default():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    assert w.show_fft is False


def test_show3d_show_fft_constructor():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, show_fft=True)
    assert w.show_fft is True


def test_show3d_multi_panel():
    a = np.random.rand(5, 16, 16).astype(np.float32)
    b = np.random.rand(5, 16, 16).astype(np.float32)
    w = Show3D(a, b, panel_titles=["A", "B"])
    assert w.n_panels == 2
    assert list(w.panel_titles) == ["A", "B"]
    assert w.n_slices == 5


def test_show3d_multi_panel_mismatch():
    a = np.random.rand(5, 16, 16).astype(np.float32)
    b = np.random.rand(7, 16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="frames"):
        Show3D(a, b)


def test_show3d_multi_panel_shape_mismatch():
    a = np.random.rand(5, 16, 16).astype(np.float32)
    b = np.random.rand(5, 12, 16).astype(np.float32)
    with pytest.raises(ValueError, match="shape"):
        Show3D(a, b)


def test_show3d_state_dict_keys():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    sd = w.state_dict()
    for required in ("cmap", "log_scale", "show_fft", "fps"):
        assert required in sd


def test_show3d_state_dict_roundtrip():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, cmap="viridis", log_scale=True, fps=15.0)
    sd = w.state_dict()
    w2 = Show3D(data, state=sd)
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.fps == pytest.approx(15.0)


def test_show3d_save_load(tmp_path):
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, cmap="plasma")
    path = tmp_path / "show3d_state.json"
    w.save(str(path))
    payload = json.loads(path.read_text())
    assert payload["widget_name"] == "Show3D"
    assert "state" in payload
    w2 = Show3D(data, state=str(path))
    assert w2.cmap == "plasma"


def test_show3d_repr():
    data = np.random.rand(8, 16, 16).astype(np.float32)
    w = Show3D(data, cmap="inferno")
    r = repr(w)
    assert "Show3D" in r
    assert "inferno" in r


def test_show3d_summary(capsys):
    data = np.random.rand(8, 16, 16).astype(np.float32)
    w = Show3D(data, title="Nano", cmap="inferno")
    w.summary()
    out = capsys.readouterr().out
    assert "Nano" in out
    assert "inferno" in out


def test_show3d_unknown_kwarg_rejected():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    with pytest.raises(TypeError, match="unexpected keyword"):
        Show3D(data, disabled_tools=["display"])


def test_show3d_set_image():
    data = np.random.rand(10, 16, 16).astype(np.float32)
    w = Show3D(data, cmap="viridis")
    new = np.random.rand(20, 24, 32).astype(np.float32)
    w.set_image(new)
    assert w.n_slices == 20
    assert w.height == 24
    assert w.width == 32
    assert w.cmap == "viridis"


def test_show3d_set_image_accepts_2d():
    data = np.random.rand(10, 16, 16).astype(np.float32)
    w = Show3D(data)
    img = np.random.rand(24, 32).astype(np.float32)
    w.set_image(img)
    assert w.n_slices == 1
    assert w.height == 24
    assert w.width == 32


def test_show3d_set_image_resets_multi_panel_state():
    a = np.random.rand(5, 8, 8).astype(np.float32)
    b = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(a, b, panel_titles=["A", "B"])
    w.set_image(np.random.rand(3, 6, 7).astype(np.float32))
    assert w.n_panels == 1
    assert list(w.panel_titles) == []
    assert w.width == 7


def test_show3d_dim_label():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, dim_label="Defocus")
    assert w.dim_label == "Defocus"


def test_show3d_widget_version_set():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    assert w.widget_version != "unknown"


def test_show3d_rejects_complex():
    data = (np.random.rand(5, 8, 8) + 1j * np.random.rand(5, 8, 8)).astype(np.complex64)
    with pytest.raises(TypeError, match="complex"):
        Show3D(data)


def test_show3d_rejects_complex_in_panel():
    a = np.random.rand(5, 8, 8).astype(np.float32)
    b = (np.random.rand(5, 8, 8) + 1j * np.random.rand(5, 8, 8)).astype(np.complex64)
    with pytest.raises(TypeError, match="complex"):
        Show3D(a, b)


def test_show3d_rejects_nan_in_panel():
    a = np.random.rand(5, 8, 8).astype(np.float32)
    b = np.random.rand(5, 8, 8).astype(np.float32)
    b[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Panel 1 contains NaN or inf"):
        Show3D(a, b)


def test_show3d_rejects_float32_overflow_in_panel():
    a = np.random.rand(5, 8, 8).astype(np.float32)
    b = np.full((5, 8, 8), np.float64(np.finfo(np.float32).max) * 2.0, dtype=np.float64)
    with pytest.raises(ValueError, match="Panel 1 exceeds float32 range"):
        Show3D(a, b)


def test_show3d_set_image_rejects_complex():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    complex_data = (np.random.rand(5, 8, 8) + 1j * np.random.rand(5, 8, 8)).astype(np.complex64)
    with pytest.raises(TypeError, match="complex"):
        w.set_image(complex_data)


def test_show3d_state_includes_slice_idx():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    w.slice_idx = 3
    sd = w.state_dict()
    assert sd["slice_idx"] == 3


def test_show3d_rejects_empty_stack():
    with pytest.raises(ValueError, match="Empty stack"):
        Show3D(np.zeros((0, 100, 100), dtype=np.float32))
    with pytest.raises(ValueError, match="Empty stack"):
        Show3D(np.zeros((2, 0, 100), dtype=np.float32))
    w = Show3D(np.zeros((2, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Empty stack"):
        w.set_image(np.zeros((2, 0, 4), dtype=np.float32))


def test_show3d_single_frame_diff_modes_do_not_crash():
    w = Show3D(np.ones((1, 4, 4), dtype=np.float32))
    w.diff_mode = "previous"
    assert w.data_min == 0.0
    assert w.data_max == 0.0
    w.diff_mode = "first"
    assert w.data_min == 0.0
    assert w.data_max == 0.0


def test_show3d_slice_idx_clamps_on_oob():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    w.slice_idx = 999
    assert w.slice_idx == 4
    w.slice_idx = -10
    assert w.slice_idx == 0


def test_show3d_float_slice_idx_coerced():
    data = np.random.rand(10, 8, 8).astype(np.float32)
    w = Show3D(data)
    w.slice_idx = 3.7
    assert w.slice_idx == 3


def test_show3d_roi_unknown_shape_rejected():
    import traitlets
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    with pytest.raises(traitlets.TraitError, match="unknown shape"):
        w.roi_list = [{"shape": "hexagon", "row": 0, "col": 0, "radius": 5}]


def test_show3d_roi_negative_radius_rejected():
    import traitlets
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    with pytest.raises(traitlets.TraitError, match=">="):
        w.roi_list = [{"shape": "circle", "row": 0, "col": 0, "radius": -5}]


def test_show3d_roi_selected_idx_clamps():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    w.roi_list = [{"shape": "circle", "row": 0, "col": 0, "radius": 5}]
    w.roi_selected_idx = 99
    assert w.roi_selected_idx == 0
    w.roi_selected_idx = -99
    # Auto-selected to 0 by _on_roi_change when roi_active + non-empty list,
    # but here roi_active is False so validator -1 stays.
    assert w.roi_selected_idx == -1


def test_show3d_profile_width_validator():
    import traitlets
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    with pytest.raises(traitlets.TraitError, match=">="):
        w.profile_width = 0


def test_show3dvolume_rejects_empty():
    from quantem.widget import Show3DVolume
    with pytest.raises(ValueError, match="Empty volume"):
        Show3DVolume(np.zeros((0, 10, 10), dtype=np.float32))


def test_show3dvolume_float_slice_coerced():
    from quantem.widget import Show3DVolume
    data = np.random.rand(10, 20, 30).astype(np.float32)
    w = Show3DVolume(data)
    w.slice_z = 3.7
    w.slice_y = 5.9
    w.slice_x = -2.5
    assert w.slice_z == 3
    assert w.slice_y == 5
    assert w.slice_x == 0


def test_show3dvolume_slice_clamps():
    from quantem.widget import Show3DVolume
    data = np.random.rand(8, 12, 16).astype(np.float32)
    w = Show3DVolume(data)
    w.slice_z = 99; w.slice_y = 99; w.slice_x = 99
    assert w.slice_z == 7
    assert w.slice_y == 11
    assert w.slice_x == 15


def test_show3d_rejects_nan():
    data = np.random.rand(3, 8, 8).astype(np.float32)
    data[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        Show3D(data)


def test_show3d_rejects_inf():
    data = np.random.rand(3, 8, 8).astype(np.float32)
    data[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or inf"):
        Show3D(data)


def test_show3d_large_array_full_finite_scan():
    data = np.zeros((2, 1001, 500), dtype=np.float32)
    data.ravel()[123457] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        Show3D(data)


def test_show3dvolume_rejects_nan():
    from quantem.widget import Show3DVolume
    data = np.random.rand(3, 8, 8).astype(np.float32)
    data[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        Show3DVolume(data)


def test_show3dvolume_rejects_float32_overflow():
    from quantem.widget import Show3DVolume
    data = np.full((3, 8, 8), np.float64(np.finfo(np.float32).max) * 2.0, dtype=np.float64)
    with pytest.raises(ValueError, match="float32 range"):
        Show3DVolume(data)


def test_show3dvolume_dual_rejects_float32_overflow_b():
    from quantem.widget import Show3DVolume
    a = np.random.rand(3, 8, 8).astype(np.float32)
    b = np.full((3, 8, 8), np.float64(np.finfo(np.float32).max) * 2.0, dtype=np.float64)
    with pytest.raises(ValueError, match="data_b exceeds float32 range"):
        Show3DVolume(a, b)


def test_show3d_state_timestamps_roundtrip():
    data = np.random.rand(4, 8, 8).astype(np.float32)
    w = Show3D(data, timestamps=[0, 1.5, 3.0, 4.5], timestamp_unit="ms")
    sd = w.state_dict()
    assert sd["timestamps"] == [0, 1.5, 3.0, 4.5]
    w2 = Show3D(data, state=sd)
    assert list(w2.timestamps) == [0, 1.5, 3.0, 4.5]
    assert w2.timestamp_unit == "ms"


def test_show3d_percentile_state_roundtrip_below_default_low():
    data = np.random.rand(4, 8, 8).astype(np.float32)
    w = Show3D(data, percentile_low=0.5, percentile_high=0.8)
    w2 = Show3D(data, state=w.state_dict())
    assert w2.percentile_low == pytest.approx(0.5)
    assert w2.percentile_high == pytest.approx(0.8)


def test_show3d_partial_vmin_state_roundtrip_affects_export_range():
    data = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    w = Show3D(data, vmin=5.0)
    w2 = Show3D(data, state=w.state_dict())
    assert w2._get_color_range(data[0])[0] == pytest.approx(5.0)


def test_show3d_state_with_active_roi_loads_before_timer_exists():
    data = np.ones((2, 4, 4), dtype=np.float32)
    w = Show3D(data, state={"roi_active": True})
    try:
        assert w.roi_active is True
    finally:
        w.free()


def test_show3d_load_state_ignores_data_derived_traits():
    data = np.ones((2, 4, 4), dtype=np.float32)
    w = Show3D(data)
    with pytest.warns(UserWarning, match="n_slices"):
        w.load_state_dict({"n_slices": 999, "height": 999, "width": 999})
    assert w.n_slices == 2
    assert w.height == 4
    assert w.width == 4
