import json
import re
from pathlib import Path

import numpy as np
import pytest
import traitlets

from quantem.widget import Show3DSlices


def test_show3dslices_live_volume_bytes_are_float32():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    w = Show3DSlices(data)

    assert w.offline is False
    assert len(w.volume_bytes) == data.nbytes
    decoded = np.frombuffer(w.volume_bytes, dtype=np.float32).reshape(data.shape)
    np.testing.assert_array_equal(decoded, data)


def test_show3dslices_removed_compact_kwarg_is_accepted_for_compatibility():
    w = Show3DSlices(np.zeros((2, 3, 4), dtype=np.float32), compact=False)

    assert "compact" not in w.traits()


def test_show3dslices_visual_playback_defaults():
    w = Show3DSlices(np.zeros((2, 3, 4), dtype=np.float32))

    assert w.cmap == "plasma"
    assert w.smooth is True
    assert w.fps == 30
    assert w.boomerang is True


def test_show3dslices_fps_validation_caps_playback_at_thirty():
    w = Show3DSlices(np.zeros((2, 3, 4), dtype=np.float32), fps=120)

    assert w.fps == 30

    w.fps = 90
    assert w.fps == 30

    with pytest.raises(Exception, match="fps must be > 0"):
        w.fps = 0


def test_show3dslices_dataset3d_sampling_sets_scale_bar_axes():
    from quantem.core.datastructures import Dataset3d

    data = np.zeros((2, 3, 4), dtype=np.float32)
    dataset = Dataset3d.from_array(
        data,
        name="calibrated volume",
        sampling=[20.0, 0.1846, 0.1846],
        units=["A", "A", "A"],
    )

    w = Show3DSlices(dataset)

    assert w.title == "calibrated volume"
    assert list(w.pixel_size_axes) == [20.0, 0.1846, 0.1846]
    assert w.pixel_size == pytest.approx(0.1846)


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


def test_show3dslices_padding_expands_volume_with_median_border():
    data = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)

    w = Show3DSlices(data, padding=1)

    assert (w.nz, w.ny, w.nx) == (2, 4, 5)
    pad_value = float(np.median(data))
    assert np.all(w._data[:, 0, :] == pad_value)
    assert np.all(w._data[:, -1, :] == pad_value)
    assert np.all(w._data[:, :, 0] == pad_value)
    assert np.all(w._data[:, :, -1] == pad_value)
    np.testing.assert_array_equal(w._data[:, 1:-1, 1:-1], data)


def test_show3dslices_crop_applies_before_padding():
    data = np.arange(1 * 6 * 7, dtype=np.float32).reshape(1, 6, 7)

    w = Show3DSlices(data, crop=(1, 2, 3, 1), padding=1, pad_mode="constant")

    assert (w.nz, w.ny, w.nx) == (1, 5, 5)
    np.testing.assert_array_equal(w._data[:, 1:-1, 1:-1], data[:, 1:4, 3:6])
    assert np.all(w._data[:, 0, :] == 0)
    assert np.all(w._data[:, :, 0] == 0)


def test_show3dslices_crop_rejects_empty_volume():
    data = np.zeros((1, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="removes the entire image"):
        Show3DSlices(data, crop=(2, 2, 0, 0))


def test_show3dslices_post_crop_applies_after_rotation():
    data = np.arange(1 * 5 * 5, dtype=np.float32).reshape(1, 5, 5)

    w = Show3DSlices(data, rotation_deg=180, post_crop=(1, 0, 2, 1))

    rotated = np.rot90(data, k=2, axes=(1, 2))
    np.testing.assert_array_equal(w._data, rotated[:, 1:, 2:4])
    assert (w.nz, w.ny, w.nx) == (1, 4, 2)


def test_show3dslices_rotation_matches_square_quarter_turn():
    data = np.arange(1 * 3 * 3, dtype=np.float32).reshape(1, 3, 3)

    w = Show3DSlices(data, rotation_deg=90)
    w_neg = Show3DSlices(data, rotation_deg=-90)

    np.testing.assert_array_equal(w._data, np.rot90(data, k=1, axes=(1, 2)))
    np.testing.assert_array_equal(w_neg._data, np.rot90(data, k=-1, axes=(1, 2)))


def test_show3dslices_rotation_keeps_shape_and_float32():
    data = np.arange(2 * 5 * 7, dtype=np.float32).reshape(2, 5, 7)

    w = Show3DSlices(data, rotation_deg=13.5)

    assert w._data.shape == data.shape
    assert w._data.dtype == np.float32
    assert np.isfinite(w._data).all()
    assert not np.shares_memory(w._data, data)


def test_show3dslices_quantem_config_infers_sampling_rotation_and_crop():
    data = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    config = {
        "data": {"rotation_deg": 90},
        "object": {"cropped_shape": [2, 2]},
        "reconstruction": {
            "slice_thickness_A": 20.0,
            "obj_sampling_A_per_px": 0.1846,
        },
    }

    w = Show3DSlices(data, config=config)

    expected = np.rot90(data, k=1, axes=(1, 2))[:, 1:3, 1:3]
    np.testing.assert_array_equal(w._data, expected)
    assert list(w.pixel_size_axes) == [20.0, 0.1846, 0.1846]
    assert w.pixel_size == pytest.approx(0.1846)
    assert w._rotation_deg == 90
    assert w._post_crop == (1, 1, 1, 1)


def test_show3dslices_quantem_config_transforms_can_be_disabled():
    data = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    config = {
        "data": {"rotation_deg": 90},
        "object": {"cropped_shape": [2, 2]},
        "reconstruction": {
            "slice_thickness_A": 20.0,
            "obj_sampling_A_per_px": 0.1846,
        },
    }

    w = Show3DSlices(data, config=config, apply_config_transforms=False)

    np.testing.assert_array_equal(w._data, data)
    assert list(w.pixel_size_axes) == [20.0, 0.1846, 0.1846]
    assert w._rotation_deg == 0
    assert w._post_crop == (0, 0, 0, 0)


def test_show3dslices_explicit_zero_transform_overrides_quantem_config():
    data = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    config = {
        "data": {"rotation_deg": 90},
        "object": {"cropped_shape": [2, 2]},
        "reconstruction": {
            "slice_thickness_A": 20.0,
            "obj_sampling_A_per_px": 0.1846,
        },
    }

    w = Show3DSlices(data, config=config, rotation_deg=0, post_crop=0)

    np.testing.assert_array_equal(w._data, data)
    assert list(w.pixel_size_axes) == [20.0, 0.1846, 0.1846]
    assert w._rotation_deg == 0
    assert w._post_crop == (0, 0, 0, 0)


def test_show3dslices_gpu_slice_log_uses_signed_transform():
    colormaps_src = Path(__file__).parents[1] / "js" / "colormaps.ts"
    text = colormaps_src.read_text()
    volume_slice_shaders = text[
        text.index("const VOLUME_SLICE_SHADER"):
        text.index("const VOLUME_PARAMS_BYTES")
    ]

    assert "max(val, 0.0)" not in volume_slice_shaders
    assert volume_slice_shaders.count("fn signedLog1p") == 2
    assert volume_slice_shaders.count("val = signedLog1p(val)") == 2


def test_show3dslices_export_html_writes_exact_and_quantized(tmp_path):
    data = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    w = Show3DSlices(data, title="Export Probe", image_vmin_pct=12.5, image_vmax_pct=87.5)
    w.show_crosshair = False
    w.flip = True
    w.smooth = True
    w.show_colorbar = True
    w.fft_colormap = "gray"
    w.fft_log_scale = True
    w.fft_auto = False
    w.show_slice_planes = False
    w.volume_opacity = 0.75
    w.slice_plane_opacity = 0.25

    exact = w.export_html(tmp_path / "exact.html", quantized=False)
    quantized = w.export_html(tmp_path / "quantized.html", quantized=True)

    assert exact.exists()
    assert quantized.exists()
    exact_text = exact.read_text()
    quantized_text = quantized.read_text()
    assert '"offline": false' in exact_text
    assert '"offline": true' in quantized_text
    assert '"_esm"' in exact_text
    assert '"show_crosshair": false' in exact_text
    assert '"flip": true' in exact_text
    assert '"smooth": true' in exact_text
    assert '"show_colorbar": true' in exact_text
    assert '"image_vmin_pct": 12.5' in exact_text
    assert '"image_vmax_pct": 87.5' in exact_text
    assert '"fft_colormap": "gray"' in exact_text
    assert '"fft_log_scale": true' in exact_text
    assert '"fft_auto": false' in exact_text
    assert '"show_slice_planes": false' in exact_text
    assert re.search(r'"plane_visibility":\s*\[\s*false,\s*false,\s*false\s*\]', exact_text)
    assert '"volume_opacity": 0.75' in exact_text
    assert '"slice_plane_opacity": 0.25' in exact_text
    assert "quantized" in w.export_status

    state_json = re.search(
        r'<script type="application/vnd.jupyter.widget-state\+json">\n(.*?)\n</script>',
        quantized_text,
        flags=re.S,
    ).group(1)
    state = json.loads(state_json)["state"]
    volume_models = [
        model for model in state.values()
        if any(buffer["path"] == ["volume_bytes"] for buffer in model.get("buffers", []))
    ]
    assert len(volume_models) == 1
    assert volume_models[0]["state"]["offline"] is True
    assert volume_models[0]["state"]["export_enabled"] is False


def test_show3dslices_export_request_trait_writes_toolbar_html(tmp_path, monkeypatch):
    data = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    w = Show3DSlices(data, title="Toolbar Export")
    monkeypatch.chdir(tmp_path)

    w.export_request = json.dumps({"mode": "quantized"})

    exported = tmp_path / "toolbar_export_2x3x4_quantized.html"
    assert exported.exists()
    assert "Exported toolbar_export_2x3x4_quantized.html" in w.export_status
    assert "quantized" in w.export_status
    assert '"offline": true' in exported.read_text()


def test_show3dslices_export_request_can_return_download_payload():
    data = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    w = Show3DSlices(data, title="Toolbar Download")

    w.export_request = json.dumps({
        "mode": "quantized",
        "download": True,
        "filename": "picked-folder.html",
        "id": "req-1",
    })

    assert w.export_filename == "picked-folder.html"
    assert w.export_payload_id == "req-1"
    assert len(w.export_payload) > 0
    text = w.export_payload.decode()
    assert '"offline": true' in text
    assert "Ready picked-folder.html" in w.export_status

    w.export_request = json.dumps({"mode": "clear", "id": "req-1-clear"})

    assert w.export_payload == b""
    assert w.export_payload_id == ""
    assert w.export_filename == ""


def test_show3dslices_state_dict_round_trips_display_conditions():
    data = np.zeros((2, 3, 4), dtype=np.float32)
    w = Show3DSlices(data)
    w.show_crosshair = False
    w.flip = True
    w.smooth = True
    w.show_colorbar = True
    w.image_vmin_pct = 10
    w.image_vmax_pct = 90
    w.fft_colormap = "gray"
    w.fft_log_scale = True
    w.fft_auto = False
    w.plane_visibility = [True, False, True]
    w.volume_opacity = 0.7
    w.slice_plane_opacity = 0.3

    restored = Show3DSlices(data)
    restored.load_state_dict(w.state_dict())

    assert restored.show_crosshair is False
    assert restored.flip is True
    assert restored.smooth is True
    assert restored.show_colorbar is True
    assert restored.image_vmin_pct == 10
    assert restored.image_vmax_pct == 90
    assert restored.fft_colormap == "gray"
    assert restored.fft_log_scale is True
    assert restored.fft_auto is False
    assert restored.show_slice_planes is True
    assert list(restored.plane_visibility) == [True, False, True]
    assert restored.volume_opacity == 0.7
    assert restored.slice_plane_opacity == 0.3


def test_show3dslices_plane_visibility_mirrors_legacy_toggle():
    data = np.zeros((2, 3, 4), dtype=np.float32)
    w = Show3DSlices(data)

    assert w.show_slice_planes is True
    assert list(w.plane_visibility) == [True, True, True]

    w.plane_visibility = [True, False, False]
    assert w.show_slice_planes is True
    assert list(w.plane_visibility) == [True, False, False]

    w.plane_visibility = [False, False, False]
    assert w.show_slice_planes is False

    w.show_slice_planes = True
    assert list(w.plane_visibility) == [True, True, True]

    w.show_slice_planes = False
    assert list(w.plane_visibility) == [False, False, False]

    with pytest.raises(traitlets.TraitError, match="plane_visibility"):
        w.plane_visibility = [True, False]


def test_show3dslices_old_state_show_slice_planes_sets_plane_visibility():
    data = np.zeros((2, 3, 4), dtype=np.float32)
    w = Show3DSlices(data)
    state = w.state_dict()
    state.pop("plane_visibility")
    state["show_slice_planes"] = False

    restored = Show3DSlices(data)
    restored.load_state_dict(state)

    assert restored.show_slice_planes is False
    assert list(restored.plane_visibility) == [False, False, False]
