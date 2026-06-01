import json
import re

import numpy as np
import pytest

from quantem.core.datastructures import Dataset3d
from quantem.widget import Show3D


def _four_panel_widget() -> Show3D:
    panels = [
        np.full((3, 4, 5), fill_value=i, dtype=np.float32)
        for i in range(4)
    ]
    return Show3D(*panels, link_contrast=False)


def test_per_panel_contrast_traits_are_independent():
    w = _four_panel_widget()

    w.vmin_per_panel = [0.0, 10.0, 20.0, 30.0]
    w.vmax_per_panel = [1.0, 11.0, 21.0, 31.0]

    assert w.n_panels == 4
    assert w.link_contrast is False
    assert w.vmin_per_panel == [0.0, 10.0, 20.0, 30.0]
    assert w.vmax_per_panel == [1.0, 11.0, 21.0, 31.0]

    w.vmin_per_panel = [0.0, 10.5, 20.0, 30.0]
    assert w.vmin_per_panel == [0.0, 10.5, 20.0, 30.0]
    assert w.vmax_per_panel == [1.0, 11.0, 21.0, 31.0]


def test_per_panel_histogram_state_round_trip():
    w = _four_panel_widget()
    w.auto_contrast = False
    w.log_scale = True
    w.avg_window = 5
    w.link_panels = False
    w.percentile_high = 97.0
    w.percentile_low = 2.0
    w.vmin_per_panel = [0.0, 1.0, 2.0, 3.0]
    w.vmax_per_panel = [4.0, 5.0, 6.0, 7.0]

    state = w.state_dict()
    w2 = _four_panel_widget()
    w2.load_state_dict(state)

    assert w2.link_contrast is False
    assert w2.auto_contrast is False
    assert w2.log_scale is True
    assert w2.avg_window == 5
    assert w2.link_panels is False
    assert w2.percentile_low == 2.0
    assert w2.percentile_high == 97.0
    assert w2.vmin_per_panel == [0.0, 1.0, 2.0, 3.0]
    assert w2.vmax_per_panel == [4.0, 5.0, 6.0, 7.0]


def test_removed_noop_constructor_knobs_are_accepted_for_compatibility():
    data = np.zeros((3, 4, 4), dtype=np.float32)

    w = Show3D(data, show_playback=True)
    assert "show_playback" not in w.traits()

    w = Show3D(data, link_zoom=False)
    assert w.link_panels is False

    w = Show3D(data, link_pan=False)
    assert w.link_panels is False


def test_dataset3d_micron_sampling_and_time_axis():
    pixel_size_micron = 0.05
    ds = Dataset3d.from_array(
        array=np.zeros((4, 8, 8), dtype=np.float32),
        name="UEM scan",
        sampling=[15, pixel_size_micron, pixel_size_micron],
        units=["ps", "micron", "micron"],
    )

    w = Show3D(ds)

    assert w.title == "UEM scan"
    assert w.pixel_unit == "micron"
    assert w.pixel_size == pixel_size_micron
    assert w.dim_sampling == 15
    assert w.dim_unit == "ps"
    assert w.timestamp_unit == "ps"
    assert w.timestamps == [0.0, 15.0, 30.0, 45.0]


def test_dataset3d_angstrom_units_pass_through_by_default():
    ds = Dataset3d.from_array(
        array=np.zeros((2, 4, 4), dtype=np.float32),
        sampling=[1, 0.75, 0.75],
        units=["A", "A", "A"],
    )

    w = Show3D(ds)

    assert w.pixel_unit == "A"
    assert w.pixel_size == 0.75


def test_dataset3d_nm_sampling_defaults_to_angstrom_scale_bar():
    ds = Dataset3d.from_array(
        array=np.zeros((2, 4, 4), dtype=np.float32),
        sampling=[1, 0.2, 0.2],
        units=["s", "nm", "nm"],
    )

    w = Show3D(ds)

    assert w.pixel_unit == "A"
    assert w.pixel_size == 2.0


def test_avg_window_validation_default_and_state_round_trip():
    w = Show3D(np.zeros((3, 4, 4), dtype=np.float32))

    assert w.avg_window == 1

    with pytest.raises(Exception, match="avg_window must be >= 1"):
        w.avg_window = 0
    with pytest.raises(Exception, match="avg_window must be <= 15"):
        w.avg_window = 16

    w.avg_window = 15
    state = w.state_dict()
    w2 = Show3D(np.zeros((3, 4, 4), dtype=np.float32))
    w2.load_state_dict(state)

    assert w2.avg_window == 15


def test_show3d_visual_playback_defaults():
    w = Show3D(np.zeros((3, 4, 4), dtype=np.float32))

    assert w.cmap == "plasma"
    assert w.smooth is True
    assert w.fps == 30
    assert w.boomerang is True


def test_fps_validation_caps_playback_at_thirty():
    w = Show3D(np.zeros((3, 4, 4), dtype=np.float32), fps=120)

    assert w.fps == 30

    w.fps = 90
    assert w.fps == 30

    with pytest.raises(Exception, match="fps must be > 0"):
        w.fps = 0


def test_show_kymograph_default_and_state_round_trip():
    w = Show3D(np.zeros((3, 4, 4), dtype=np.float32))

    assert w.show_kymograph is False

    w.show_kymograph = True
    state = w.state_dict()
    w2 = Show3D(np.zeros((3, 4, 4), dtype=np.float32))
    w2.load_state_dict(state)

    assert w2.show_kymograph is True


def test_diff_mode_validation_and_state_round_trip():
    w = Show3D(np.zeros((3, 4, 4), dtype=np.float32))

    with pytest.raises(Exception, match="Invalid diff_mode"):
        w.diff_mode = "later"

    w.diff_mode = "previous"
    state = w.state_dict()
    w2 = Show3D(np.zeros((3, 4, 4), dtype=np.float32))
    w2.load_state_dict(state)

    assert w2.diff_mode == "previous"


def test_multi_panel_show_kymograph_trait_persists():
    w = _four_panel_widget()

    w.show_kymograph = True
    state = w.state_dict()
    w2 = _four_panel_widget()
    w2.load_state_dict(state)

    assert w.n_panels == 4
    assert w2.show_kymograph is True


def test_linking_contrast_keeps_per_panel_state():
    w = _four_panel_widget()
    w.auto_contrast = False
    w.log_scale = True
    w.percentile_high = 98.0
    w.percentile_low = 5.0
    w.vmin_per_panel = [0.0, 1.0, 2.0, 3.0]
    w.vmax_per_panel = [4.0, 5.0, 6.0, 7.0]

    w.link_contrast = True

    assert w.link_contrast is True
    assert w.auto_contrast is False
    assert w.log_scale is True
    assert w.percentile_low == 5.0
    assert w.percentile_high == 98.0
    assert w.vmin_per_panel == [0.0, 1.0, 2.0, 3.0]
    assert w.vmax_per_panel == [4.0, 5.0, 6.0, 7.0]


def test_auto_contrast_range_is_stack_level():
    data = np.stack(
        [
            np.full((4, 4), 0.0, dtype=np.float32),
            np.full((4, 4), 10.0, dtype=np.float32),
            np.full((4, 4), 20.0, dtype=np.float32),
        ]
    )

    w = Show3D(data, percentile_low=0.0, percentile_high=100.0)

    assert w.auto_vmins == [0.0, 0.0, 0.0]
    assert w.auto_vmaxs == [20.0, 20.0, 20.0]


def test_show3d_export_html_writes_exact_and_quantized(tmp_path):
    data = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    w = Show3D(data, title="Export Probe", cmap="gray", image_vmin_pct=12.5, image_vmax_pct=87.5)
    w.show_fft = True
    w.show_kymograph = True
    w.smooth = True
    w.diff_mode = "first"
    w.avg_window = 3
    w.set_profile((0, 0), (2, 3))

    exact = w.export_html(tmp_path / "exact.html", quantized=False)
    quantized = w.export_html(tmp_path / "quantized.html", quantized=True)

    assert exact.exists()
    assert quantized.exists()
    exact_text = exact.read_text()
    quantized_text = quantized.read_text()
    assert '"offline": true' in exact_text
    assert '"offline": true' in quantized_text
    assert '"_esm"' in exact_text
    assert '"show_fft": true' in exact_text
    assert '"show_kymograph": true' in exact_text
    assert '"smooth": true' in exact_text
    assert '"diff_mode": "first"' in exact_text
    assert '"avg_window": 3' in exact_text
    assert '"image_vmin_pct": 12.5' in exact_text
    assert '"image_vmax_pct": 87.5' in exact_text
    assert '"export_enabled": false' in exact_text
    assert '"_offline_float_stack"' in exact_text
    assert '"_offline_stack"' in quantized_text
    assert "quantized" in w.export_status

    state_json = re.search(
        r'<script type="application/vnd.jupyter.widget-state\+json">\n(.*?)\n</script>',
        exact_text,
        flags=re.S,
    ).group(1)
    state = json.loads(state_json)["state"]
    exact_models = [
        model for model in state.values()
        if any(buffer["path"] == ["_offline_float_stack"] for buffer in model.get("buffers", []))
    ]
    assert len(exact_models) == 1
    assert exact_models[0]["state"]["offline"] is True
    assert exact_models[0]["state"]["export_enabled"] is False


def test_show3d_histogram_percent_range_round_trips():
    data = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    w = Show3D(data, image_vmin_pct=10.0, image_vmax_pct=90.0)

    state = w.state_dict()
    restored = Show3D(data)
    restored.load_state_dict(state)

    assert w.vmin is None
    assert w.vmax is None
    assert restored.image_vmin_pct == 10.0
    assert restored.image_vmax_pct == 90.0
    assert restored.vmin is None
    assert restored.vmax is None


def test_show3d_export_request_trait_writes_toolbar_html(tmp_path, monkeypatch):
    data = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    w = Show3D(data, title="Toolbar Export")
    monkeypatch.chdir(tmp_path)

    w.export_request = json.dumps({"mode": "quantized"})

    exported = tmp_path / "toolbar_export_2x3x4_quantized.html"
    assert exported.exists()
    assert "Exported toolbar_export_2x3x4_quantized.html" in w.export_status
    assert "quantized" in w.export_status
    assert '"offline": true' in exported.read_text()


def test_show3d_export_request_can_return_download_payload():
    data = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    w = Show3D(data, title="Toolbar Download")

    w.export_request = json.dumps({
        "mode": "exact",
        "download": True,
        "filename": "picked-folder.html",
        "id": "req-1",
    })

    assert w.export_filename == "picked-folder.html"
    assert w.export_payload_id == "req-1"
    assert len(w.export_payload) > 0
    text = w.export_payload.decode()
    assert '"offline": true' in text
    assert '"_offline_float_stack"' in text
    assert "Ready picked-folder.html" in w.export_status

    w.export_request = json.dumps({"mode": "clear", "id": "req-1-clear"})

    assert w.export_payload == b""
    assert w.export_payload_id == ""
    assert w.export_filename == ""



def test_padding_expands_stack_with_median_border():
    data = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)

    w = Show3D(data, padding=1, percentile_low=0.0, percentile_high=100.0)

    assert w.height == 4
    assert w.width == 5
    assert w._data.shape == (2, 4, 5)
    pad_value = float(np.median(data))
    assert np.all(w._data[:, 0, :] == pad_value)
    assert np.all(w._data[:, -1, :] == pad_value)
    assert np.all(w._data[:, :, 0] == pad_value)
    assert np.all(w._data[:, :, -1] == pad_value)
    np.testing.assert_array_equal(w._data[:, 1:-1, 1:-1], data)


def test_padding_preserved_by_set_image():
    w = Show3D(np.ones((1, 2, 2), dtype=np.float32), padding=(2, 1))
    replacement = np.full((1, 1, 3), 7, dtype=np.float32)

    w.set_image(replacement)

    assert w.height == 5
    assert w.width == 5
    np.testing.assert_array_equal(w._data[:, 2:3, 1:4], replacement)


def test_crop_applies_before_padding():
    data = np.arange(1 * 6 * 7, dtype=np.float32).reshape(1, 6, 7)

    w = Show3D(data, crop=(1, 2, 3, 1), padding=1, pad_mode="constant")

    assert w.height == 5
    assert w.width == 5
    np.testing.assert_array_equal(w._data[:, 1:-1, 1:-1], data[:, 1:4, 3:6])
    assert np.all(w._data[:, 0, :] == 0)
    assert np.all(w._data[:, :, 0] == 0)


def test_crop_preserved_by_set_image():
    w = Show3D(np.arange(1 * 5 * 5, dtype=np.float32).reshape(1, 5, 5), crop=1)
    replacement = np.arange(1 * 6 * 6, dtype=np.float32).reshape(1, 6, 6)

    w.set_image(replacement)

    assert w.height == 4
    assert w.width == 4
    np.testing.assert_array_equal(w._data, replacement[:, 1:-1, 1:-1])


def test_crop_rejects_empty_image():
    data = np.zeros((1, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="removes the entire image"):
        Show3D(data, crop=(2, 2, 0, 0))



def test_multi_panel_roi_api_is_disabled():
    w = _four_panel_widget()

    with pytest.raises(ValueError, match="single-panel"):
        w.add_roi()

    w.roi_active = True
    w.roi_list = [{"shape": "square", "row": 1, "col": 1, "radius": 1}]
    w.roi_selected_idx = 0
    w._on_roi_change()

    assert w.roi_stats == {}
    assert w.roi_plot_data == b""


def test_presentation_chrome_constructor_flags():
    w = Show3D(
        np.zeros((1, 4, 4), dtype=np.float32),
        show_panel_titles=False,
        show_resize_handles=False,
        show_zoom_indicator=False,
        show_scale_bar=False,
    )

    assert w.show_panel_titles is False
    assert w.show_resize_handles is False
    assert w.show_zoom_indicator is False
    assert w.scale_bar_visible is False
