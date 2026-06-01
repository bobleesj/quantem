#!/usr/bin/env python3
"""Automated Show3D PR #230 smoke test.

This script builds the widget bundle, executes a tiny notebook to HTML, opens the
HTML with Playwright, and checks the three user-facing behaviors from the PR
feedback:

- spatial padding expands the displayed FOV,
- multi-panel ROI controls/state are hidden,
- multi-panel line profile follows panel zoom/pan and remains clipped to its slot.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(tempfile.gettempdir()) / "quantem_show3d_pr230_smoke"
NOTEBOOK = ARTIFACT_DIR / "show3d_pr230_smoke.ipynb"
HTML = ARTIFACT_DIR / "show3d_pr230_smoke.html"


def run(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int = 120) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout, check=True)


def write_notebook() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    code = r"""
import numpy as np
from IPython.display import display
from quantem.widget import Show3D

rng = np.random.default_rng(230)
base = np.zeros((5, 48, 64), dtype=np.float32)
for i in range(base.shape[0]):
    base[i, 8 + i:28 + i, 12:36] = 1.0 + i
    base[i, 26:40, 34 + i:50 + i] = 3.0 + i
base += rng.normal(0, 0.03, base.shape).astype(np.float32)

single = Show3D(
    base,
    title="PR230 padding single",
    padding=12,
    size=260,
    offline=True,
    show_controls=True,
    show_stats=False,
    percentile_low=0.0,
    percentile_high=100.0,
)
display(single)

panels = [base + j * 2.0 for j in range(4)]
multi = Show3D(
    *panels,
    title="PR230 profile multi",
    panel_titles=["A", "B", "C", "D"],
    max_cols=2,
    panel_gap=12,
    size=260,
    offline=True,
    show_controls=True,
    show_stats=False,
    percentile_low=0.0,
    percentile_high=100.0,
)
display(multi)
"""
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(code)]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.write(nb, NOTEBOOK)


def browser_bbox(page: Page, root_index: int) -> dict[str, float]:
    bbox = page.evaluate(
        """
        (rootIndex) => {
          const root = document.querySelectorAll('.show3d-root')[rootIndex];
          if (!root) throw new Error(`missing root ${rootIndex}`);
          const canvases = Array.from(root.querySelectorAll('canvas'));
          const overlay = canvases[2];
          if (!overlay) throw new Error('missing overlay canvas');
          const ctx = overlay.getContext('2d', { willReadFrequently: true });
          const img = ctx.getImageData(0, 0, overlay.width, overlay.height).data;
          let minX = overlay.width, minY = overlay.height, maxX = -1, maxY = -1, count = 0;
          for (let y = 0; y < overlay.height; y++) {
            for (let x = 0; x < overlay.width; x++) {
              const a = img[(y * overlay.width + x) * 4 + 3];
              if (a > 0) {
                count++;
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
              }
            }
          }
          if (count === 0) return { count: 0 };
          const rect = overlay.getBoundingClientRect();
          const sx = overlay.width / rect.width;
          const sy = overlay.height / rect.height;
          return {
            count,
            minX: minX / sx,
            minY: minY / sy,
            maxX: maxX / sx,
            maxY: maxY / sy,
            width: (maxX - minX + 1) / sx,
            height: (maxY - minY + 1) / sy,
            cx: ((minX + maxX) / 2) / sx,
            cy: ((minY + maxY) / 2) / sy,
          };
        }
        """,
        root_index,
    )
    if not isinstance(bbox, dict) or bbox.get("count", 0) == 0:
        raise AssertionError("overlay has no non-transparent profile pixels")
    return bbox


def wait_for_widgets(page: Page) -> None:
    page.goto(HTML.as_uri(), wait_until="domcontentloaded")
    page.wait_for_function(
        """
        () => {
          const roots = Array.from(document.querySelectorAll('.show3d-root'));
          return roots.length >= 2 && roots.every(root => {
            const canvas = root.querySelector('canvas[role="img"]');
            return canvas && canvas.width > 0 && canvas.height > 0;
          });
        }
        """,
        timeout=30_000,
    )
    page.wait_for_timeout(500)


def assert_padding(page: Page) -> None:
    label = page.locator(".show3d-root").nth(0).locator('canvas[role="img"]').first.get_attribute("aria-label") or ""
    if "(88 by 72 pixels)" not in label:
        raise AssertionError(f"single-panel padded dimensions missing from aria-label: {label!r}")


def assert_multi_panel_roi_hidden(page: Page) -> None:
    multi = page.locator(".show3d-root").nth(1)
    if multi.locator('[aria-label="Toggle ROI selection tool"]').count() != 0:
        raise AssertionError("multi-panel widget still exposes ROI toggle")
    text = multi.text_content() or ""
    if "ROI:" in text or "ROI FFT" in text:
        raise AssertionError(f"multi-panel widget still shows ROI UI text: {text[:300]!r}")




def assert_cursor_coordinates_are_panel_local(page: Page) -> None:
    multi = page.locator(".show3d-root").nth(1)
    canvas = multi.locator('canvas[role="img"]').first
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    if not box:
        raise AssertionError("missing multi-panel main canvas bounding box")
    gap = 12
    panel_w = (box["width"] - gap) / 2
    panel_h = (box["height"] - gap) / 2
    # Top-right panel, near its right edge. The displayed column should still
    # be local to a 64-px panel, not global across the 4 concatenated panels.
    page.mouse.move(box["x"] + panel_w + gap + panel_w * 0.90, box["y"] + panel_h * 0.50)
    page.wait_for_timeout(150)
    text = multi.text_content() or ""
    import re
    matches = re.findall(r"\((\d+),\s*(\d+)\)\s*[-+0-9.eE]+", text)
    if not matches:
        raise AssertionError(f"cursor readout did not appear in multi-panel widget: {text[:500]!r}")
    row, col = map(int, matches[-1])
    if not (0 <= row < 48 and 0 <= col < 64):
        raise AssertionError(f"cursor readout is not panel-local: row={row}, col={col}, text={text[:500]!r}")


def assert_unlinked_histograms_do_not_overlap_playback(page: Page) -> None:
    multi = page.locator(".show3d-root").nth(1)
    link = multi.locator('[aria-label="Link contrast across panels"]')
    if link.is_checked():
        link.click()
    page.wait_for_timeout(450)
    result = page.evaluate(
        """
        () => {
          const root = document.querySelectorAll('.show3d-root')[1];
          if (!root) throw new Error('missing multi-panel root');
          const histCanvases = Array.from(root.querySelectorAll('canvas[aria-label="Histogram of intensity values with min and max clip handles"]'));
          const playbackSlider = Array.from(root.querySelectorAll('input[aria-label], [role="slider"]')).find((el) => {
            const label = el.getAttribute('aria-label') || '';
            return label.startsWith('Current ') || label.startsWith('Loop range');
          });
          const rect = (el) => {
            const r = el.getBoundingClientRect();
            return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height };
          };
          const playback = playbackSlider ? rect(playbackSlider) : null;
          const hist = histCanvases.map(rect);
          const overlap = playback ? hist.some((h) => !(h.right <= playback.left || h.left >= playback.right || h.bottom <= playback.top || h.top >= playback.bottom)) : true;
          return { histCount: hist.length, playback, hist, overlap };
        }
        """
    )
    if result["histCount"] < 4:
        raise AssertionError(f"expected four independent histograms after unlinking contrast: {result}")
    if result["overlap"]:
        raise AssertionError(f"independent histograms overlap the playback slider: {result}")


def assert_profile_tracks_zoom_and_pan(page: Page) -> None:
    multi = page.locator(".show3d-root").nth(1)
    multi.locator('[aria-label="Toggle line intensity profile tool"]').click()
    profile_plot = multi.locator('[aria-label="Line intensity profile along the drawn line"]')
    profile_plot.wait_for(timeout=5_000)
    page.wait_for_timeout(100)

    canvas = multi.locator('canvas[role="img"]').first
    gap = 12

    def panel_geometry() -> tuple[dict[str, float], float, float, float]:
        box = canvas.bounding_box()
        if not box:
            raise AssertionError("missing multi-panel main canvas bounding box")
        panel_w = (box["width"] - gap) / 2
        panel_h = (box["height"] - gap) / 2
        panel_x = panel_w + gap
        return box, panel_w, panel_h, panel_x

    box, panel_w, panel_h, panel_x = panel_geometry()
    canvas.click(position={"x": panel_x + panel_w * 0.25, "y": panel_h * 0.35})
    canvas.click(position={"x": panel_x + panel_w * 0.75, "y": panel_h * 0.65})
    page.wait_for_timeout(250)
    before = browser_bbox(page, 1)
    if before["cx"] < panel_x or before["cy"] > panel_h:
        raise AssertionError(f"profile was not drawn in top-right panel before zoom: {before}")

    box, panel_w, panel_h, panel_x = panel_geometry()
    page.mouse.move(box["x"] + panel_x + panel_w * 0.5, box["y"] + panel_h * 0.5)
    page.mouse.wheel(0, -600)
    page.wait_for_timeout(250)
    zoomed = browser_bbox(page, 1)
    if zoomed["width"] <= before["width"] * 1.04:
        raise AssertionError(f"profile bbox did not scale after panel zoom: before={before}, zoomed={zoomed}")
    if zoomed["cx"] < panel_x or zoomed["cy"] > panel_h:
        raise AssertionError(f"profile left its owning panel after zoom: {zoomed}")

    box, panel_w, panel_h, panel_x = panel_geometry()
    drag_x = box["x"] + panel_x + panel_w * 0.1
    drag_y = box["y"] + panel_h * 0.9
    page.mouse.move(drag_x, drag_y)
    page.mouse.down()
    page.mouse.move(drag_x + 28, drag_y + 14, steps=4)
    page.mouse.up()
    page.wait_for_timeout(250)
    panned = browser_bbox(page, 1)
    if panned["cx"] <= zoomed["cx"] + 8 or panned["cy"] <= zoomed["cy"] + 4:
        raise AssertionError(f"profile bbox did not follow panel pan: zoomed={zoomed}, panned={panned}")
    if panned["maxX"] > box["width"] + 1 or panned["maxY"] > box["height"] + 1:
        raise AssertionError(f"profile was not clipped to the canvas: {panned}")

    profile_switch = multi.locator('[aria-label="Toggle line intensity profile tool"]')
    profile_switch.click()
    page.wait_for_timeout(100)
    multi.locator('[aria-label="Play"]').click()
    page.wait_for_timeout(700)
    if profile_switch.is_checked():
        raise AssertionError("profile switch reactivated during playback after being turned off")
    if multi.locator('[aria-label="Line intensity profile along the drawn line"]').count() != 0:
        raise AssertionError("profile plot reappeared during playback after being turned off")
    overlay_display = page.evaluate(
        """
        () => {
          const root = document.querySelectorAll('.show3d-root')[1];
          const overlay = root?.querySelectorAll('canvas')[2];
          return overlay ? getComputedStyle(overlay).display : null;
        }
        """
    )
    if overlay_display != "none":
        raise AssertionError(f"profile overlay canvas stayed visible after playback: {overlay_display!r}")


def run_browser_checks() -> None:
    errors: list[str] = []

    def on_console_error(msg) -> None:
        if msg.type != "error":
            return
        location = msg.location.get("url", "") if msg.location else ""
        errors.append(f"console error: {msg.text} ({location})")

    def expected_browser_noise(message: str) -> bool:
        lower = message.lower()
        if "webgpu" in lower or "gpu" in lower:
            return True
        # The Jupyter widget HTML manager probes for a sibling anywidget.js
        # before falling back to the CDN. Keep this narrow so missing Show3D
        # resources or dead frame-server URLs still fail the smoke test.
        return "err_file_not_found" in lower and "anywidget.js" in lower

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on("console", on_console_error)
        wait_for_widgets(page)
        assert_padding(page)
        assert_multi_panel_roi_hidden(page)
        assert_cursor_coordinates_are_panel_local(page)
        assert_unlinked_histograms_do_not_overlap_playback(page)
        assert_profile_tracks_zoom_and_pan(page)
        browser.close()
    if errors:
        unexpected = [e for e in errors if not expected_browser_noise(e)]
        if unexpected:
            raise AssertionError("Browser errors during smoke test:\n" + "\n".join(unexpected[:10]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="do not run npm run build first")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("MPLBACKEND", "Agg")

    if not args.skip_build:
        run(["npm", "run", "build"], env=env)
    write_notebook()
    run([
        "jupyter", "nbconvert",
        "--to", "html",
        "--execute",
        "--output", HTML.stem,
        "--output-dir", str(ARTIFACT_DIR),
        str(NOTEBOOK),
    ], env=env, timeout=180)
    run_browser_checks()
    print(f"OK: Show3D PR #230 smoke passed: {HTML}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
