from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch
from matplotlib.patches import Circle, Ellipse
from scipy.optimize import least_squares

from quantem.core.utils.filter import otsu_threshold

if TYPE_CHECKING:
    from quantem.core.datastructures.dataset2d import Dataset2d


def fit_probe_circle(
    img: np.ndarray | torch.Tensor | Dataset2d, threshold: float | None = None, show: bool = True
) -> tuple[float, float, float]:
    """
    Fit a circle to the probe shape in an image.

    Args:
        img (np.ndarray | Dataset2d): Input diffraction pattern containing the probe.
        threshold (Optional[float]): Threshold for binarization. If None, Otsu's method is used.
        show (bool): Whether to display the fitted circle. Default is True.

    Returns:
        Tuple[float, float, float]: Center coordinates (center_row, center_col) and radius radius of the fitted circle.
    """
    if not isinstance(img, np.ndarray) and hasattr(img, "array"):
        img = img.array
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D diffraction pattern, got shape {img.shape}.")

    if isinstance(img, torch.Tensor):
        center_row, center_col, radius = _fit_probe_circle_tensor(img, threshold)
        if show:
            _, ax = plt.subplots()
            ax.imshow(img.detach().cpu().numpy(), cmap="gray")
            ax.add_patch(Circle((center_col, center_row), radius, color="red", alpha=0.3))
            plt.show()
        return center_row, center_col, radius
    if threshold is None:
        threshold = otsu_threshold(img)
    binary = ndi.binary_closing(img > threshold, iterations=2)
    binary: np.ndarray = ndi.binary_fill_holes(binary)  # type: ignore

    smoothed = ndi.gaussian_filter(binary.astype(float), sigma=1)
    grad_mag = np.hypot(ndi.sobel(smoothed, axis=1), ndi.sobel(smoothed, axis=0))

    edge_points = np.argwhere(grad_mag > grad_mag.mean())
    row, col = edge_points[:, 0], edge_points[:, 1]

    def radial_distance(
        center_col: float, center_row: float, col: np.ndarray, row: np.ndarray
    ) -> np.ndarray:
        return np.sqrt((col - center_col) ** 2 + (row - center_row) ** 2)

    def f_2(c: Tuple[float, float], col: np.ndarray, row: np.ndarray) -> np.ndarray:
        radii = radial_distance(*c, col, row)
        return radii - radii.mean()

    center_estimate = np.mean(col), np.mean(row)
    result = least_squares(f_2, center_estimate, args=(col, row))
    center_col, center_row = result.x
    radius = radial_distance(center_col, center_row, col, row).mean()

    if show:
        _fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.add_patch(Circle((center_col, center_row), radius, color="red", alpha=0.3, linewidth=2))
        plt.show()

    return center_row, center_col, radius


def fit_probe_ellipse(
    img: np.ndarray, threshold: Optional[float] = None, show: bool = True
) -> Tuple[float, float, float, float, float]:
    """
    Fit an ellipse to the probe shape in an image using the direct least squares fitting method for
    ellipses, which solves the general conic equation Ax² + Bxy + Cy² + Dx + Ey + F = 0 subject to
    the constraint B² - 4AC < 0 (ellipse condition).

    Args:
        img (np.ndarray): Input image containing the probe.
        threshold (Optional[float]): Threshold for binarization. If None, Otsu's method is used.
        show (bool): Whether to display the fitted ellipse. Default is True.

    Returns:
        Tuple[float, float, float, float, float]: Center coordinates (xc, yc),
        semi-major axis (a_axis), semi-minor axis (b_axis), and rotation angle (theta) in radians.
    """
    # Threshold the image to create a binary mask
    if threshold is None:
        threshold = otsu_threshold(img)
    binary: np.ndarray = ndi.binary_fill_holes(img > threshold)  # type: ignore

    # Find edge points using gradient magnitude
    smoothed = ndi.gaussian_filter(binary.astype(float), sigma=1)
    grad_x = ndi.sobel(smoothed, axis=1)
    grad_y = ndi.sobel(smoothed, axis=0)
    grad_mag = np.hypot(grad_x, grad_y)
    edge_points = np.argwhere(grad_mag > grad_mag.mean())
    y, x = edge_points[:, 0], edge_points[:, 1]

    # Set up the design matrix for the conic equation Ax² + Bxy + Cy² + Dx + Ey + F = 0
    # Each row represents one edge point [x², xy, y², x, y, 1]
    D = np.vstack([x**2, x * y, y**2, x, y, np.ones_like(x)]).T
    S = np.dot(D.T, D)  # Scatter matrix

    # Constraint matrix to enforce ellipse condition (B² - 4AC < 0)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2  # Constraint: 4AC
    C[1, 1] = -1  # Constraint: -B²

    # Solve the generalized eigenvalue problem to find ellipse coefficients
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S).dot(C))
    cond = np.logical_and(np.isreal(eigvals), eigvals > 0)
    a = eigvecs[:, cond][:, 0].real  # type: ignore

    # Extract coefficients from the conic equation
    # General form: a0*x² + a1*xy + a2*y² + a3*x + a4*y + a5 = 0
    b, c, d, f, g, a0 = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]

    # Calculate ellipse center coordinates
    num = b * b - a0 * c
    xc = (c * d - b * f) / num
    yc = (a0 * f - b * d) / num

    # Calculate semi-major and semi-minor axes lengths
    up = 2 * (a0 * f * f + c * d * d + g * b * b - 2 * b * d * f - a0 * c * g)
    down1 = (b * b - a0 * c) * ((c - a0) * np.sqrt(1 + 4 * b * b / ((a0 - c) ** 2)) - (c + a0))
    down2 = (b * b - a0 * c) * ((a0 - c) * np.sqrt(1 + 4 * b * b / ((a0 - c) ** 2)) - (c + a0))
    a_axis = np.sqrt(up / down1)  # Semi-major axis
    b_axis = np.sqrt(up / down2)  # Semi-minor axis

    # Calculate rotation angle of the ellipse
    theta = 0.5 * np.arctan(2 * b / (a0 - c))

    if show:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ellipse = Ellipse(
            (xc, yc),
            2 * a_axis,
            2 * b_axis,
            angle=np.degrees(theta),
            color="red",
            alpha=0.3,
            linewidth=2,
        )
        ax.add_patch(ellipse)
        plt.show()

    return yc, xc, a_axis, b_axis, theta


@torch.no_grad()
def _fit_probe_circle_tensor(img_t: torch.Tensor, threshold: float | None):
    """Match the NumPy circle fit while retaining calibration arrays on device."""
    import torch.nn.functional as F

    values_t = img_t.double()
    if threshold is None:
        lower, upper = float(img_t.min()), float(img_t.max())
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        counts_t = torch.histc(img_t, bins=256, min=lower, max=upper).double()
        edges_t = torch.linspace(lower, upper, 257, device=img_t.device, dtype=img_t.dtype)[
            :-1
        ].double()
        background_t = counts_t.cumsum(0)
        foreground_t = img_t.numel() - background_t
        moment_t = (counts_t * edges_t).cumsum(0)
        difference_t = moment_t / background_t - (moment_t[-1] - moment_t) / foreground_t
        variance_t = background_t * foreground_t * difference_t.square()
        variance_t[(background_t == 0) | (foreground_t == 0)] = 0
        threshold = float(edges_t[variance_t.argmax()])
    cross_t = values_t.new_tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])[None, None]
    binary_t = values_t > threshold
    for _ in range(2):
        binary_t = F.conv2d(binary_t[None, None].double(), cross_t, padding=1)[0, 0] > 0
    for _ in range(2):
        binary_t = F.conv2d(binary_t[None, None].double(), cross_t, padding=1)[0, 0] == 5
    exterior_t = torch.zeros_like(binary_t)
    exterior_t[0] = ~binary_t[0]
    exterior_t[-1] = ~binary_t[-1]
    exterior_t[:, 0] = ~binary_t[:, 0]
    exterior_t[:, -1] = ~binary_t[:, -1]
    while True:
        expanded_t = (
            F.conv2d(exterior_t[None, None].double(), cross_t, padding=1)[0, 0] > 0
        ) & ~binary_t
        if torch.equal(expanded_t, exterior_t):
            break
        exterior_t = expanded_t
    binary_t = ~exterior_t

    def reflect_filter(array_t, kernel_t):
        radius = kernel_t.shape[-1] // 2
        row_t = torch.arange(-radius, array_t.shape[0] + radius, device=img_t.device) % (
            2 * array_t.shape[0]
        )
        col_t = torch.arange(-radius, array_t.shape[1] + radius, device=img_t.device) % (
            2 * array_t.shape[1]
        )
        row_t = torch.minimum(row_t, 2 * array_t.shape[0] - 1 - row_t)
        col_t = torch.minimum(col_t, 2 * array_t.shape[1] - 1 - col_t)
        padded_t = array_t[row_t[:, None], col_t[None, :]]
        return F.conv2d(padded_t[None, None], kernel_t[None, None])[0, 0]

    support_t = torch.arange(-4, 5, device=img_t.device, dtype=torch.float64)
    gaussian_t = torch.exp(-support_t.square() / 2)
    gaussian_t /= gaussian_t.sum()
    smooth_t = reflect_filter(binary_t.double(), gaussian_t[:, None] * gaussian_t[None, :])
    sobel_t = values_t.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_t = torch.hypot(
        reflect_filter(smooth_t, sobel_t), reflect_filter(smooth_t, sobel_t.T)
    )
    points_t = (gradient_t > gradient_t.mean()).nonzero().double()
    # Keep the original (column, row) parameter order for the two-column SVD.
    points_t = points_t.flip(1)
    center_t = points_t.mean(0)

    def residual(center_t):
        offsets_t = points_t - center_t
        radii_t = torch.sqrt(offsets_t[:, 0].square() + offsets_t[:, 1].square())
        return radii_t - radii_t.mean()

    # The reference uses a forward-difference Jacobian and stops on relative
    # cost change. An analytic Jacobian converged further and changed calibration.
    epsilon = torch.finfo(torch.float64).eps**0.5
    residual_t = residual(center_t)
    for _ in range(199):
        columns_t = []
        for axis in range(2):
            shifted_t = center_t.clone()
            step_t = epsilon * torch.maximum(center_t[axis].abs(), center_t.new_tensor(1))
            step_t = torch.where(center_t[axis] < 0, -step_t, step_t)
            shifted_t[axis] += step_t
            step_t = shifted_t[axis] - center_t[axis]
            columns_t.append((residual(shifted_t) - residual_t) / step_t)
        jacobian_t = torch.stack(columns_t, dim=1)
        gradient_t = jacobian_t.T @ residual_t
        if float(gradient_t.abs().max()) < 1e-8:
            break
        left_t, singular_t, right_t = torch.linalg.svd(jacobian_t, full_matrices=False)
        step_t = -(right_t.T @ ((left_t.T @ residual_t) / singular_t))
        next_center_t = center_t + step_t
        next_residual_t = residual(next_center_t)
        cost_t = 0.5 * torch.dot(residual_t, residual_t)
        next_cost_t = 0.5 * torch.dot(next_residual_t, next_residual_t)
        reduction_t = cost_t - next_cost_t
        projected_t = jacobian_t @ step_t
        predicted_t = -(0.5 * torch.dot(projected_t, projected_t) + gradient_t @ step_t)
        ratio_t = reduction_t / predicted_t
        converged = bool((reduction_t < 1e-8 * cost_t) & (ratio_t > 0.25)) or bool(
            torch.linalg.vector_norm(step_t)
            < 1e-8 * (1e-8 + torch.linalg.vector_norm(center_t))
        )
        if bool(reduction_t > 0):
            center_t, residual_t = next_center_t, next_residual_t
        if converged:
            break
    offsets_t = points_t - center_t
    radius = float(torch.sqrt(offsets_t[:, 0].square() + offsets_t[:, 1].square()).mean())
    return float(center_t[1]), float(center_t[0]), radius
