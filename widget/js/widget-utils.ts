import { sliderRange } from "./stats";

export const signedLog1p = (x: number): number => x >= 0 ? Math.log1p(x) : -Math.log1p(-x);

export function shouldIgnoreWidgetShortcut(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return target.closest([
    "input",
    "textarea",
    "button",
    "select",
    "[contenteditable='true']",
    "[role='button']",
    "[role='slider']",
    "[role='switch']",
    "[role='textbox']",
    "[role='combobox']",
    "[role='menuitem']",
    ".MuiSlider-root",
    ".MuiSelect-select",
  ].join(",")) !== null;
}

export function extractXY(vol: Float32Array, nx: number, ny: number, nz: number, z: number): Float32Array {
  if (z < 0 || z >= nz) return new Float32Array(ny * nx);
  const start = z * ny * nx;
  return vol.subarray(start, start + ny * nx);
}

export function extractXZ(vol: Float32Array, nx: number, ny: number, nz: number, y: number): Float32Array {
  const out = new Float32Array(nz * nx);
  if (y < 0 || y >= ny) return out;
  for (let z = 0; z < nz; z++) {
    const srcOffset = z * ny * nx + y * nx;
    for (let x = 0; x < nx; x++) out[z * nx + x] = vol[srcOffset + x];
  }
  return out;
}

export function extractYZ(vol: Float32Array, nx: number, ny: number, nz: number, x: number): Float32Array {
  const out = new Float32Array(nz * ny);
  if (x < 0 || x >= nx) return out;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) out[z * ny + y] = vol[z * ny * nx + y * nx + x];
  }
  return out;
}

export function reverseLut(lut: Uint8Array): Uint8Array {
  const out = new Uint8Array(lut.length);
  const n = lut.length / 3;
  for (let i = 0; i < n; i++) {
    const src = (n - 1 - i) * 3;
    const dst = i * 3;
    out[dst + 0] = lut[src + 0];
    out[dst + 1] = lut[src + 1];
    out[dst + 2] = lut[src + 2];
  }
  return out;
}

export function maybeFlip(data: Float32Array, flip: boolean): Float32Array {
  if (!flip) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = -data[i];
  return out;
}

export function findFFTPeak(
  mag: Float32Array,
  width: number,
  height: number,
  col: number,
  row: number,
  radius: number,
): { row: number; col: number } {
  const c0 = Math.max(0, Math.floor(col) - radius);
  const r0 = Math.max(0, Math.floor(row) - radius);
  const c1 = Math.min(width - 1, Math.floor(col) + radius);
  const r1 = Math.min(height - 1, Math.floor(row) + radius);
  let bestCol = Math.round(col);
  let bestRow = Math.round(row);
  let bestVal = -Infinity;

  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = mag[ir * width + ic];
      if (val > bestVal) {
        bestVal = val;
        bestCol = ic;
        bestRow = ir;
      }
    }
  }

  const wc0 = Math.max(0, bestCol - 1);
  const wc1 = Math.min(width - 1, bestCol + 1);
  const wr0 = Math.max(0, bestRow - 1);
  const wr1 = Math.min(height - 1, bestRow + 1);
  let sumW = 0;
  let sumWC = 0;
  let sumWR = 0;
  for (let ir = wr0; ir <= wr1; ir++) {
    for (let ic = wc0; ic <= wc1; ic++) {
      const w = mag[ir * width + ic];
      sumW += w;
      sumWC += w * ic;
      sumWR += w * ir;
    }
  }
  if (sumW > 0) return { row: sumWR / sumW, col: sumWC / sumW };
  return { row: bestRow, col: bestCol };
}

export function resolveDisplayRange(
  dataMin: number,
  dataMax: number,
  traitVmin: number | null | undefined,
  traitVmax: number | null | undefined,
  logScale: boolean,
  vminPct: number,
  vmaxPct: number,
): { vmin: number; vmax: number } {
  const baseMin = logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin);
  const baseMax = logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax);
  return sliderRange(baseMin, baseMax, vminPct, vmaxPct);
}

export function resolveDisplayBounds(
  dataMin: number,
  dataMax: number,
  traitVmin: number | null | undefined,
  traitVmax: number | null | undefined,
  logScale: boolean,
): { min: number; max: number } {
  return {
    min: logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin),
    max: logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax),
  };
}
