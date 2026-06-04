import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

// Standalone WebGPU 4D-STEM browser. No backend: the GUI is the quantem.live
// Browse page, the data layer reads a locally-picked folder of Arina .h5 files
// and decodes them on the GPU via the shared js/engine WGSL engine.
const here = dirname(fileURLToPath(import.meta.url));
const widgetRoot = resolve(here, "..");

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // jsfive's bundled dist drops BTreeV1RawDataChunks; the engine needs the
      // loose esm sources, so point the bare deep-imports at them.
      "jsfive/esm/high-level.js": resolve(widgetRoot, "node_modules/jsfive/esm/high-level.js"),
      "jsfive/esm/btree.js": resolve(widgetRoot, "node_modules/jsfive/esm/btree.js"),
    },
  },
  server: {
    // allow importing the engine symlink target (widget/js/engine) + jsfive from
    // the parent node_modules, both outside the web/ root.
    fs: { allow: [here, widgetRoot] },
  },
  build: { target: "es2022", chunkSizeWarningLimit: 4000 },
});
