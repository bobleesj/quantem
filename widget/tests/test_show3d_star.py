"""Tests for Show3D's per-panel star feature (mark best frame per panel).

Two real workflows scientists actually use:

1. Star current frame on each panel of a multi-trial comparison, read back via
   `starred_frames` dict.
2. Save state with stars, load into a fresh widget — stars survive.
"""

import numpy as np

from quantem.widget import Show3D


def test_star_per_panel_independent():
    """4-trial comparison: scrub each panel's best iter, star it, verify
    `starred_frames` returns the per-trial picks."""
    a = np.random.default_rng(0).random((10, 32, 32), dtype=np.float32)
    b = np.random.default_rng(1).random((10, 32, 32), dtype=np.float32)
    c = np.random.default_rng(2).random((10, 32, 32), dtype=np.float32)
    w = Show3D(a, b, c, panel_titles=["iter 0-10", "iter 5-15", "iter 20-30"])

    # No stars at start.
    assert w.starred == [-1, -1, -1]
    assert w.starred_frames == {}

    # Pick a best-iter per panel.
    w.goto(3); w.star_panel(0)            # panel 0 best at frame 3
    w.star_panel(1, frame=7)              # panel 1 best at frame 7
    w.star_panel(2, frame=9)              # panel 2 best at frame 9
    assert w.starred_frames == {0: 3, 1: 7, 2: 9}

    # Moving the star on panel 0 doesn't disturb panels 1+2.
    w.star_panel(0, frame=5)
    assert w.starred_frames == {0: 5, 1: 7, 2: 9}

    # Unstarring panel 1 only removes its entry.
    w.unstar_panel(1)
    assert w.starred_frames == {0: 5, 2: 9}


def test_star_state_round_trip():
    """state_dict save/load preserves per-panel stars."""
    stacks = [np.zeros((10, 16, 16), dtype=np.float32) for _ in range(3)]
    w = Show3D(*stacks)
    w.star_panel(0, frame=2)
    w.star_panel(2, frame=8)

    # Round-trip via state_dict.
    state = w.state_dict()
    assert state["starred"] == [2, -1, 8]

    w2 = Show3D(*stacks)
    w2.load_state_dict(state)
    assert w2.starred == [2, -1, 8]
    assert w2.starred_frames == {0: 2, 2: 8}


def test_identical_panel_dedupe_keeps_full_res_source():
    base = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    panels = [base.copy() for _ in range(3)]

    w = Show3D(*panels, display_bin=1, dedupe_identical_panels=True)

    assert w.n_panels == 3
    assert w.shared_panel_source is True
    assert w.height == 8
    assert w.width == 8
    assert w.panel_width_px == 8
    assert w._display_bin == 1
    np.testing.assert_array_equal(w._data, panels[0])


def test_nonidentical_panels_stay_separate_full_res():
    panels = [
        (np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6) + offset)
        for offset in (0, 1000, 2000)
    ]

    w = Show3D(*panels, display_bin=1, dedupe_identical_panels=True)
    try:
        assert w.n_panels == 3
        assert w.shared_panel_source is False
        assert w.separate_panel_frames is True
        assert w.height == 5
        assert w.width == 18
        assert w.panel_width_px == 6
        assert w._display_bin == 1

        for panel_idx, panel in enumerate(panels):
            np.testing.assert_array_equal(w._get_display_panel_frame(panel_idx, 2), panel[2])

        expected_joined = np.concatenate([panel[2] for panel in panels], axis=1)
        np.testing.assert_array_equal(w._get_display_frame(2), expected_joined)

        status, frame = w._frame_for_http(2, w.frame_server_version, panel=1)
        assert status == 200
        assert frame.flags.c_contiguous
        np.testing.assert_array_equal(frame, panels[1][2])

        w.goto(3)
        assert w.frame_bytes == b""
    finally:
        w.free()
