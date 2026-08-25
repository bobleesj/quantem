"""Structured evidence assembled from pure drift diagnostics.

These helpers intentionally remain internal during staged integration. The
final API change will decide which table becomes ``DriftCorrection.report``.
"""

from collections.abc import Mapping, Sequence

from . import diagnostics


def _rename_rows(
    rows: Sequence[Mapping[str, float | int | str]],
    columns: Mapping[str, str],
) -> list[dict[str, float | int | str]]:
    """Return independent records with stable, reader-facing labels."""
    return [{columns.get(name, name): value for name, value in row.items()} for row in rows]


def _registration_report(
    correction,
    *,
    stages: Sequence[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Return common-coverage registration records by correction stage."""
    _, _, _, rows = diagnostics._registration_data(correction, stages)
    return _rename_rows(
        rows,
        {
            "stage": "Correction stage",
            "common_ncc": "Common NCC",
            "top_ncc": "Top third NCC",
            "middle_ncc": "Middle third NCC",
            "bottom_ncc": "Bottom third NCC",
            "mean_absolute_difference": "Mean absolute difference",
            "root_mean_square_difference": "Root mean square difference",
            "coverage": "Coverage",
        },
    )


def _displacement_report(
    correction,
    *,
    stages: Sequence[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Return explicit scan-line-origin displacement records by image."""
    rows = diagnostics._displacement_rows(correction, stages)
    return _rename_rows(
        rows,
        {
            "stage": "Correction stage",
            "image": "Image",
            "endpoint_row_displacement_px": "Endpoint row displacement (px)",
            "endpoint_column_displacement_px": "Endpoint column displacement (px)",
            "endpoint_displacement_px": "Endpoint displacement (px)",
            "rms_displacement_px": "RMS displacement (px)",
            "max_displacement_px": "Maximum displacement (px)",
            "slow_roughness_px": "Slow-direction roughness (px)",
            "fast_roughness_px": "Fast-direction roughness (px)",
            "fast_curvature_px": "Fast-direction curvature (px)",
        },
    )
