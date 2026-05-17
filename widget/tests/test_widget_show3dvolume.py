import json

import numpy as np
import pytest
import torch
import traitlets

from quantem.widget import Show3DVolume


def test_show3dvolume_numpy():
    vol = np.random.rand(16, 32, 24).astype(np.float32)
    w = Show3DVolume(vol)
    assert (w.nz, w.ny, w.nx) == (16, 32, 24)
    assert len(w.volume_bytes) == vol.nbytes


def test_show3dvolume_torch():
    vol = torch.rand(8, 16, 16)
    w = Show3DVolume(vol)
    assert (w.nz, w.ny, w.nx) == (8, 16, 16)


def test_show3dvolume_rejects_2d():
    with pytest.raises(ValueError, match="3D"):
        Show3DVolume(np.random.rand(16, 16).astype(np.float32))


def test_show3dvolume_rejects_4d():
    with pytest.raises(ValueError, match="3D"):
        Show3DVolume(np.random.rand(2, 4, 8, 8).astype(np.float32))


def test_show3dvolume_initial_slices_middle():
    vol = np.random.rand(11, 13, 15).astype(np.float32)
    w = Show3DVolume(vol)
    assert (w.slice_z, w.slice_y, w.slice_x) == (5, 6, 7)


def test_show3dvolume_title_cmap():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(vol, title="Vol", cmap="viridis")
    assert w.title == "Vol"
    assert w.cmap == "viridis"


def test_show3dvolume_pixel_size():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(vol, pixel_size=2.5)
    assert w.pixel_size == pytest.approx(2.5)
    assert w.pixel_size_axes == pytest.approx([2.5, 2.5, 2.5])


def test_show3dvolume_pixel_size_anisotropic():
    # Tuple input: full triple stored, scalar trait is lateral mean.
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(vol, pixel_size=(2.0, 0.5, 0.5))
    assert w.pixel_size_axes == pytest.approx([2.0, 0.5, 0.5])
    assert w.pixel_size == pytest.approx(0.5)  # (0.5 + 0.5) / 2
    # List, ndarray, and 3-tuple all accepted.
    Show3DVolume(vol, pixel_size=[1.0, 2.0, 3.0])
    Show3DVolume(vol, pixel_size=np.array([1.0, 2.0, 3.0]))
    # Wrong length rejected.
    with pytest.raises(ValueError, match="3 elements"):
        Show3DVolume(vol, pixel_size=(1.0, 2.0))
    # NaN/neg rejected.
    with pytest.raises(ValueError, match="finite"):
        Show3DVolume(vol, pixel_size=(float("nan"), 1.0, 1.0))
    with pytest.raises(ValueError, match=">= 0"):
        Show3DVolume(vol, pixel_size=(-1.0, 1.0, 1.0))


def test_show3dvolume_log_scale_auto_contrast():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(vol, log_scale=True, auto_contrast=True)
    assert w.log_scale is True
    assert w.auto_contrast is True


def test_show3dvolume_vmin_vmax_default_none():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(vol)
    assert w.vmin is None
    assert w.vmax is None


def test_show3dvolume_fps_validator():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    with pytest.raises(traitlets.TraitError):
        w.fps = 0
    with pytest.raises(traitlets.TraitError):
        w.fps = -1


def test_show3dvolume_cmap_validator():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    with pytest.raises(traitlets.TraitError):
        Show3DVolume(vol, cmap="not_real")
    w = Show3DVolume(vol)
    with pytest.raises(traitlets.TraitError):
        w.cmap = ""


def test_show3dvolume_cmap_matches_js_dropdown():
    """Every name in js/colormaps.ts COLORMAP_POINTS must be accepted by the
    Python validator. The JS dropdown shows all of them, so a user picking
    one from the dropdown must not crash the Python trait sync round-trip.
    """
    import pathlib
    import re

    js_path = (
        pathlib.Path(__file__).parent.parent
        / "js"
        / "colormaps.ts"
    )
    text = js_path.read_text()
    # COLORMAP_POINTS keys are everything between the const declaration and
    # the matching closing brace. Pull them via a simple regex over top-level
    # identifiers followed by `: [`.
    block_start = text.index("const COLORMAP_POINTS")
    block = text[block_start:]
    block = block[: block.index("};") + 1]
    js_names = set(re.findall(r"\n  (\w+):\s*\[", block))
    assert js_names, "failed to parse JS COLORMAP_POINTS keys"
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    for name in js_names:
        # Round-trip: any name shown in the dropdown must be a valid trait
        # value. If this raises, the dropdown is offering a cmap the
        # validator rejects (silent user-pain bug).
        w = Show3DVolume(vol, cmap=name)
        assert w.cmap == name


def test_show3dvolume_pixel_size_validator():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    with pytest.raises(traitlets.TraitError):
        Show3DVolume(vol, pixel_size=float("nan"))
    with pytest.raises(traitlets.TraitError):
        Show3DVolume(vol, pixel_size=-1.0)
    w = Show3DVolume(vol)
    with pytest.raises(traitlets.TraitError, match="length 3"):
        w.pixel_size_axes = [1.0, 2.0]
    with pytest.raises(traitlets.TraitError, match="finite"):
        w.pixel_size_axes = [1.0, float("nan"), 1.0]
    with pytest.raises(traitlets.TraitError, match=">= 0"):
        w.pixel_size_axes = [1.0, -1.0, 1.0]


def test_show3dvolume_dim_labels_validator():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    with pytest.raises(traitlets.TraitError):
        w.dim_labels = ["X", "Y"]


def test_show3dvolume_play_axis_validator():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    for v in (0, 1, 2, 3):
        w.play_axis = v
    with pytest.raises(traitlets.TraitError):
        w.play_axis = 5


def test_show3dvolume_dual_mode():
    a = np.random.rand(8, 8, 8).astype(np.float32)
    b = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a, b, title_b="B")
    assert w.dual_mode is True
    assert w.title_b == "B"
    assert len(w.volume_bytes_b) == b.nbytes


def test_show3dvolume_dual_mode_shape_mismatch():
    a = np.random.rand(8, 8, 8).astype(np.float32)
    b = np.random.rand(4, 4, 4).astype(np.float32)
    with pytest.raises(ValueError, match="match"):
        Show3DVolume(a, b)


def test_show3dvolume_play_pause():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    w.play()
    assert w.playing is True
    w.pause()
    assert w.playing is False
    w.stop()
    assert w.playing is False
    assert (w.slice_z, w.slice_y, w.slice_x) == (2, 2, 2)


def test_show3dvolume_state_dict_keys():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    sd = w.state_dict()
    for k in ("cmap", "log_scale", "fps", "slice_x", "slice_y", "slice_z"):
        assert k in sd


def test_show3dvolume_state_roundtrip():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol, cmap="viridis", log_scale=True, fps=15.0)
    w2 = Show3DVolume(vol, state=w.state_dict())
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.fps == pytest.approx(15.0)


def test_show3dvolume_state_wrong_widget_rejected():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    payload = {"widget_name": "Show2D", "state": {"cmap": "magma"}}
    with pytest.raises(ValueError, match="Show2D"):
        Show3DVolume(vol, state=payload)


def test_show3dvolume_save_load_file(tmp_path):
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol, cmap="plasma", log_scale=True)
    path = tmp_path / "v.json"
    w.save(str(path))
    payload = json.loads(path.read_text())
    assert payload["widget_name"] == "Show3DVolume"
    w2 = Show3DVolume(vol, state=str(path))
    assert w2.cmap == "plasma"


def test_show3dvolume_save_image_png(tmp_path):
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(vol)
    for plane in ("xy", "xz", "yz"):
        out = tmp_path / f"{plane}.png"
        w.save_image(out, plane=plane)
        assert out.exists() and out.stat().st_size > 100


def test_show3dvolume_save_image_bad_plane():
    import tempfile
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        with pytest.raises(ValueError, match="Unknown plane"):
            w.save_image(f.name, plane="zw")


def test_show3dvolume_repr():
    vol = np.random.rand(5, 6, 7).astype(np.float32)
    w = Show3DVolume(vol, cmap="inferno")
    r = repr(w)
    assert "Show3DVolume" in r
    assert "5×6×7" in r


def test_show3dvolume_summary(capsys):
    vol = np.random.rand(4, 5, 6).astype(np.float32)
    w = Show3DVolume(vol, title="V", cmap="plasma")
    w.summary()
    out = capsys.readouterr().out
    assert "V" in out and "plasma" in out


def test_show3dvolume_unknown_kwarg_rejected():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    with pytest.raises(TypeError, match="unexpected keyword"):
        Show3DVolume(vol, disabled_tools=["display"])


def test_show3dvolume_dim_labels_default():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    assert list(w.dim_labels) == ["Z", "Y", "X"]


def test_show3dvolume_dim_labels_custom():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol, dim_labels=["Q1", "Q2", "Q3"])
    assert list(w.dim_labels) == ["Q1", "Q2", "Q3"]


def test_show3dvolume_widget_version():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    assert w.widget_version != "unknown"


def test_show3dvolume_rejects_complex():
    vol = (np.random.rand(4, 4, 4) + 1j * np.random.rand(4, 4, 4)).astype(np.complex64)
    with pytest.raises(TypeError, match="complex"):
        Show3DVolume(vol)


def test_show3dvolume_dual_rejects_complex_b():
    a = np.random.rand(4, 4, 4).astype(np.float32)
    b = (np.random.rand(4, 4, 4) + 1j * np.random.rand(4, 4, 4)).astype(np.complex64)
    with pytest.raises(TypeError, match="complex"):
        Show3DVolume(a, b)


def test_show3dvolume_z_stretch_default():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    assert w.z_stretch == pytest.approx(1.0)


def test_show3dvolume_z_stretch_kwarg():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol, z_stretch=8.0)
    assert w.z_stretch == pytest.approx(8.0)


def test_show3dvolume_z_stretch_clamp_min():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    w.z_stretch = 0.5
    assert w.z_stretch == pytest.approx(1.0)


def test_show3dvolume_z_stretch_clamp_max():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    w.z_stretch = 100
    assert w.z_stretch == pytest.approx(30.0)


def test_show3dvolume_z_stretch_rejects_nan():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    with pytest.raises(traitlets.TraitError):
        w.z_stretch = float("nan")


def test_show3dvolume_z_stretch_rejects_inf():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    with pytest.raises(traitlets.TraitError):
        w.z_stretch = float("inf")


def test_show3dvolume_z_stretch_in_state_dict():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol, z_stretch=4.0)
    sd = w.state_dict()
    assert "z_stretch" in sd
    assert sd["z_stretch"] == pytest.approx(4.0)


def test_show3dvolume_z_stretch_roundtrip():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    w.load_state_dict({"z_stretch": 7.5})
    assert w.z_stretch == pytest.approx(7.5)


def test_show3dvolume_smooth_default():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    assert w.smooth is False


def test_show3dvolume_smooth_kwarg():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol, smooth=True)
    assert w.smooth is True


def test_show3dvolume_smooth_toggle():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    w.smooth = True
    assert w.smooth is True
    w.smooth = False
    assert w.smooth is False


def test_show3dvolume_smooth_in_state_dict():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol, smooth=True)
    sd = w.state_dict()
    assert "smooth" in sd
    assert sd["smooth"] is True


def test_show3dvolume_smooth_roundtrip():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    w.load_state_dict({"smooth": True})
    assert w.smooth is True


def test_show3dvolume_load_state_warns_unknown_keys():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        w.load_state_dict({"slise_z": 5, "smooth": True})
    assert any("slise_z" in str(c.message) for c in caught)
    assert w.smooth is True  # known key still applied


def test_show3dvolume_load_state_rejects_dual_without_data_b():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)  # single-volume
    with pytest.raises(ValueError, match="dual_mode"):
        w.load_state_dict({"dual_mode": True})


def test_show3dvolume_linked_contrast_default():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    assert w.linked_contrast is True


def test_show3dvolume_linked_contrast_roundtrip():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    w.load_state_dict({"linked_contrast": False})
    assert w.linked_contrast is False


def test_show3dvolume_vmin_rejects_nan():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    with pytest.raises(traitlets.TraitError, match="finite"):
        w.vmin = float("nan")


def test_show3dvolume_vmax_rejects_inf():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol)
    with pytest.raises(traitlets.TraitError, match="finite"):
        w.vmax = float("inf")


def test_show3dvolume_linked_contrast_kwarg():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol, linked_contrast=False)
    assert w.linked_contrast is False


def test_show3dvolume_reverse_kwarg():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(vol, reverse=True)
    assert w.reverse is True


def test_show3dvolume_auto_z_stretch_thin_z():
    # nz=14, nxy=730 ptycho - ratio 52, should auto-pick z_stretch>1 and compact=True
    vol = np.random.rand(14, 730, 730).astype(np.float32)
    w = Show3DVolume(vol)
    assert w.z_stretch > 1.0
    assert w.z_stretch <= 30.0
    assert w.compact is True


def test_show3dvolume_no_auto_for_cubic():
    # nz=ny=nx - ratio 1, should keep z_stretch neutral. Compact layout is now
    # the only supported widget layout.
    vol = np.random.rand(64, 64, 64).astype(np.float32)
    w = Show3DVolume(vol)
    assert w.z_stretch == 1.0
    assert w.compact is True


def test_show3dvolume_user_z_stretch_wins():
    vol = np.random.rand(14, 730, 730).astype(np.float32)
    w = Show3DVolume(vol, z_stretch=2.0, compact=False)
    assert w.z_stretch == 2.0
    assert w.compact is True


def test_show3dvolume_export_diff_in_dual_mode():
    a = np.random.rand(4, 4, 4).astype(np.float32)
    b = np.random.rand(4, 4, 4).astype(np.float32)
    w = Show3DVolume(a, b, show_diff=True)
    slices = w._get_export_slices()
    # In dual+show_diff, exported volume should be |A - B|, not A
    expected = np.abs(a - b)
    np.testing.assert_array_almost_equal(slices[0], expected[0, :, :])
