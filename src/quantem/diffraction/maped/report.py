"""Every line MAPED prints, and the table of shifts.

Kept apart from the computation so a script that sets ``verbose=False`` runs
the same code, and so the wording lives in one place.
"""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def header(maped) -> str:
    """``MAPED: 7 tilts, 512 x 512 scan, 192 x 192 detector, on cuda:0``."""
    scan_rows, scan_cols, detector_rows, detector_cols = maped.shape
    return (
        f"MAPED: {len(maped.tilt_names)} tilts, {scan_rows} x {scan_cols} scan, "
        f"{detector_rows} x {detector_cols} detector, on {maped.device}"
    )


def aligned(maped, iterations: int, started: float) -> str:
    """The largest shifts found, and the time since ``started``.

    The shifts are read before the clock, since reading them waits for the
    alignment to finish.
    """
    detector = float(maped.diffraction_shifts.abs().max())
    scan = float(maped.real_space_shifts.abs().max())
    return (
        f"Aligned in {time.perf_counter() - started:.1f} s: largest detector shift {detector:.2f} px, "
        f"largest scan shift {scan:.2f} px after {iterations} iterations"
    )


def crop(maped, scan_box: tuple[int, int, int, int]) -> str:
    """The scan box every tilt covers, and how to keep the full scan instead."""
    row0, row1, col0, col1 = scan_box
    return (
        f"Cropped to rows {row0}:{row1}, columns {col0}:{col1} "
        f"({row1 - row0} x {col1 - col0} of {maped.shape[0]} x {maped.shape[1]}), "
        f"where all {len(maped.tilt_names)} tilts overlap. crop=False keeps the full scan."
    )


def merged(merged, seconds: float) -> str:
    """Time and rounding error of the complete merge, in counts."""
    precision = merged.metadata.get("precision")
    if precision is None:
        return f"Merged in {seconds:.1f} s: exact float32"
    clipping = "No clipping." if precision["clipped"] == 0 else f"Clipped values: {precision['clipped']}."
    return (
        f"Merged in {seconds:.1f} s: scaled uint16 (approximate). "
        f"RMS rounding error {precision['rmse']:#.3g} counts; "
        f"maximum absolute error {precision['max_abs_error']:#.3g} counts. {clipping}"
    )


def shift_table(maped, *, as_dict: bool = False) -> "pd.DataFrame | dict[str, list]":
    """Return beam centers and measured shifts in pixels, grouped by column.

    Diffraction shifts move patterns in detector pixels; scan-image shifts
    move real-space images in scan pixels. Positive row/column shifts move
    down/right. Beam centers are positions, not shifts.

    Parameters
    ----------
    as_dict : bool
        Return the original column dictionary instead of a pandas table.
        This also supports computation-only installations without pandas.

    Returns
    -------
    pandas.DataFrame or dict
        Tilt labels and available row/column shifts as lists in acquisition
        order. The default DataFrame renders directly in a notebook, without
        rounding the stored values. Use ``as_dict=True`` for the original
        plain-Python representation.

    Examples
    --------
    >>> maped.align_diffraction(show=False)
    >>> maped.align_images(show=False)
    >>> maped.report()
    """
    columns = {"Tilt": maped.tilt_names.copy()}
    for attribute, label in (
        ("beam_centers", "Beam center"),
        ("diffraction_shifts", "Diffraction shift"),
        ("real_space_shifts", "Scan-image shift"),
    ):
        shifts_t = getattr(maped, attribute, None)
        if shifts_t is not None:
            columns[f"{label} row (px)"] = shifts_t[:, 0].detach().cpu().tolist()
            columns[f"{label} col (px)"] = shifts_t[:, 1].detach().cpu().tolist()
    if as_dict:
        return columns
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        if error.name != "pandas":
            raise
        raise ModuleNotFoundError(
            "Install tables with pip install pandas, "
            "or use maped.report(as_dict=True) without pandas."
        ) from error
    return pd.DataFrame(columns)
