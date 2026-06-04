// Colormap LUT utilities for client-side canvas rendering.
// Mirrors ../../js/colormaps.ts - kept local so tsconfig "include": ["src"] resolves it.

const COLORMAP_POINTS: Record<string, number[][]> = {
  inferno: [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [252, 193, 57], [252, 255, 164],
  ],
  viridis: [
    [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141],
    [42, 120, 142], [33, 145, 140], [34, 168, 132], [68, 191, 112],
    [122, 209, 81], [189, 223, 38], [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150],
    [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40],
    [240, 249, 33],
  ],
  magma: [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
    [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135],
    [252, 253, 191],
  ],
  cividis: [
    [0, 34, 78], [0, 42, 102], [0, 52, 110], [39, 68, 108],
    [65, 84, 106], [89, 99, 111], [112, 115, 115], [137, 131, 120],
    [163, 148, 120], [189, 165, 113], [216, 184, 95], [243, 205, 39],
    [255, 233, 69],
  ],
  twilight: [
    [226, 217, 226], [198, 196, 221], [161, 176, 211], [123, 156, 200],
    [88, 135, 188], [59, 113, 174], [44, 88, 154], [47, 62, 131],
    [62, 38, 104], [82, 25, 83], [103, 25, 63], [123, 36, 49],
    [141, 57, 45], [158, 82, 53], [174, 109, 72], [188, 137, 100],
    [202, 166, 137], [215, 193, 181], [226, 217, 226],
  ],
  hot: [
    [0, 0, 0], [128, 0, 0], [255, 0, 0], [255, 128, 0],
    [255, 255, 0], [255, 255, 255],
  ],
  gray: [[0, 0, 0], [255, 255, 255]],
  // Divergent — matplotlib's `RdBu_r`. Centered hue should sit at the
  // midpoint of the LUT, so when the data is recentered around the
  // median (see contrast-tools.divergentRecenter) the median renders as
  // neutral white.
  rdbu: [
    [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222],
    [209, 229, 240], [247, 247, 247], [253, 219, 199], [244, 165, 130],
    [214, 96, 77], [178, 24, 43], [103, 0, 31],
  ],
};

function createColormapLUT(points: number[][]): Uint8Array {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (points.length - 1);
    const idx = Math.floor(t);
    const frac = t - idx;
    const p0 = points[Math.min(idx, points.length - 1)];
    const p1 = points[Math.min(idx + 1, points.length - 1)];
    lut[i * 3] = Math.round(p0[0] + frac * (p1[0] - p0[0]));
    lut[i * 3 + 1] = Math.round(p0[1] + frac * (p1[1] - p0[1]));
    lut[i * 3 + 2] = Math.round(p0[2] + frac * (p1[2] - p0[2]));
  }
  return lut;
}

export const COLORMAPS: Record<string, Uint8Array> = Object.fromEntries(
  Object.entries(COLORMAP_POINTS).map(([name, points]) => [name, createColormapLUT(points)])
);

/** Apply colormap LUT to float data, writing into an RGBA Uint8ClampedArray. */
export function applyColormap(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): void {
  const range = vmax > vmin ? vmax - vmin : 1;
  const uniformData = !(vmax > vmin);
  for (let i = 0; i < data.length; i++) {
    const clipped = Math.max(vmin, Math.min(vmax, data[i]));
    const v = uniformData ? 128 : Math.min(255, Math.floor(((clipped - vmin) / range) * 255));
    const j = i * 4;
    const lutIdx = v * 3;
    rgba[j] = lut[lutIdx];
    rgba[j + 1] = lut[lutIdx + 1];
    rgba[j + 2] = lut[lutIdx + 2];
    rgba[j + 3] = 255;
  }
}
