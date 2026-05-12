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


def test_show3dvolume_pixel_size_validator():
    vol = np.random.rand(4, 4, 4).astype(np.float32)
    with pytest.raises(traitlets.TraitError):
        Show3DVolume(vol, pixel_size=float("nan"))
    with pytest.raises(traitlets.TraitError):
        Show3DVolume(vol, pixel_size=-1.0)


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
    assert list(w.dim_labels) == ["X", "Y", "Z"]


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
