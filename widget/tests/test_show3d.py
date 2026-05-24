import numpy as np

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
    assert w2.percentile_low == 2.0
    assert w2.percentile_high == 97.0
    assert w2.vmin_per_panel == [0.0, 1.0, 2.0, 3.0]
    assert w2.vmax_per_panel == [4.0, 5.0, 6.0, 7.0]


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
