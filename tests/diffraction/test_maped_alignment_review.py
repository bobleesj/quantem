"""Review fitted scan translations without modifying the scientific workflow."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from quantem.core.datastructures import Dataset4dstem
from quantem.diffraction import MAPED


def translated_tilts():
    """Seven native acquisitions with exact integer scan translations."""
    row, col = np.indices((64, 64))
    sample = np.exp(-((row - 25) ** 2 + (col - 31) ** 2) / 32)
    sample += 0.5 * np.exp(-((row - 39) ** 2 + (col - 19) ** 2) / 12)
    displacements = [(0, 0), (2, -1), (-2, 3), (1, 2), (-1, -2), (3, 1), (-3, 0)]
    tilts = [
        Dataset4dstem.from_array(
            np.broadcast_to(np.roll(sample, shift, axis=(0, 1))[..., None, None], (64, 64, 4, 4))
            .astype(np.float32)
            .copy()
        )
        for shift in displacements
    ]
    maped = MAPED.from_datasets(tilts)
    maped.real_space_shifts = -torch.tensor(displacements, dtype=torch.float32)
    return maped


def test_all_tilts_before_after_and_report_preserve_fitted_values():
    pd = pytest.importorskip("pandas")
    maped = translated_tilts()
    original_t = torch.stack(maped.im_bf).clone()
    shifts_t = maped.real_space_shifts.clone()
    report = maped.report()
    assert isinstance(report, pd.DataFrame)
    assert report.to_dict(orient="list") == maped.report(as_dict=True)
    np.testing.assert_array_equal(report["Scan-image shift row (px)"], shifts_t[:, 0])
    maped.tilts = []  # Alignment review uses retained 2D images after input release.
    figure = maped.show_alignment()
    assert len(figure.axes) == 22  # 21 images and one shared difference colorbar.
    assert figure.number not in plt.get_fignums()  # Bare notebook expression renders only once.
    reference = figure.axes[7].images[0].get_array()
    for index in range(7):
        np.testing.assert_allclose(
            figure.axes[7 + index].images[0].get_array(), reference, atol=1e-6
        )
        assert (
            figure.axes[index].images[0].get_clim() == figure.axes[7 + index].images[0].get_clim()
        )
    torch.testing.assert_close(torch.stack(maped.im_bf), original_t, atol=0, rtol=0)
    torch.testing.assert_close(maped.real_space_shifts, shifts_t, atol=0, rtol=0)
    tilts_figure = maped.show_tilts(num_cols=4, size="small")
    assert sum(bool(axis.images) for axis in tilts_figure.axes) == 14
    assert tilts_figure.number not in plt.get_fignums()


def test_explicit_positions_show_exact_signed_differences_on_one_scale():
    maped = translated_tilts()
    original_t = torch.stack(maped.im_bf).clone()
    images = [image.square() + index * 0.25 for index, image in enumerate(maped.im_bf)]
    figure = maped.show_alignment(
        positions=[(25, 31), (39, 19)],
        crop_size=12,
        reference=3,
        images=images,
    )
    assert len(figure.axes) == 43
    limits = []
    for group, (row, col) in enumerate([(25, 31), (39, 19)]):
        start = group * 21
        raw = figure.axes[start].images[0].get_array()
        np.testing.assert_array_equal(raw, images[0][row - 6 : row + 6, col - 6 : col + 6])
        reference = figure.axes[start + 7 + 3].images[0].get_array()
        np.testing.assert_array_equal(figure.axes[start + 14 + 3].images[0].get_array(), 0)
        for index in range(7):
            artist = figure.axes[start + 14 + index].images[0]
            aligned = figure.axes[start + 7 + index].images[0].get_array()
            np.testing.assert_array_equal(artist.get_array(), aligned - reference)
            limits.append(artist.get_clim())
    assert len(set(limits)) == 1
    assert limits[0][0] == -limits[0][1]
    torch.testing.assert_close(torch.stack(maped.im_bf), original_t, atol=0, rtol=0)
