import { useState, useEffect } from "react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import { Routes, Route, Navigate } from "react-router-dom";
import { ShortcutRegistryProvider } from "./components/ShortcutRegistry";
import Browse from "./pages/browse/Browse";
import { colors, fontSizes } from "./theme";
import { scanFolder, type LocalFile } from "./local/store";

// Reuse the quantem.live dashboard theme verbatim (same MUI palette / type / custom
// nav700 breakpoint) so the Browse GUI renders exactly as it does in the live app.
const theme = createTheme({
  palette: { mode: "light", background: { default: colors.text.white, paper: colors.text.white } },
  typography: {
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    fontSize: fontSizes.lg,
  },
  breakpoints: { values: { xs: 0, sm: 600, md: 900, lg: 1200, xl: 1536, nav700: 700 } },
});
declare module "@mui/material/styles" { interface BreakpointOverrides { nav700: true } }

function filesFromInput(list: FileList): LocalFile[] {
  return Array.from(list)
    .filter((f) => /\.h5$/i.test(f.name))
    .map((f) => ({ name: f.name, relPath: (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name, bytes: () => f.arrayBuffer(), source: f }));
}
async function filesFromDirHandle(dir: FileSystemDirectoryHandle, prefix = ""): Promise<LocalFile[]> {
  const out: LocalFile[] = [];
  // @ts-expect-error - entries() is part of the File System Access API
  for await (const [name, handle] of dir.entries()) {
    const rel = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === "file" && /\.h5$/i.test(name)) {
      out.push({ name, relPath: rel, bytes: async () => (await handle.getFile()).arrayBuffer(), source: handle as FileSystemFileHandle });
    } else if (handle.kind === "directory") {
      out.push(...await filesFromDirHandle(handle, rel));
    }
  }
  return out;
}

function FolderGate({ onReady }: { onReady: () => void }) {
  const [busy, setBusy] = useState("");
  async function load(files: LocalFile[]) {
    if (!files.length) { setBusy("No .h5 files in that folder."); return; }
    setBusy(`Scanning ${files.length} files...`);
    await scanFolder(files);
    onReady();
  }
  const pickInput = () => {
    const input = document.createElement("input");
    input.type = "file"; input.webkitdirectory = true; input.multiple = true;
    input.onchange = () => input.files && load(filesFromInput(input.files));
    input.click();
  };
  const pick = async () => {
    if ("showDirectoryPicker" in window) {
      try {
        // @ts-expect-error - File System Access API
        const dir = await window.showDirectoryPicker();
        return load(await filesFromDirHandle(dir));
      } catch { /* cancelled / unsupported -> fall through */ }
    }
    pickInput();
  };
  return (
    <Box sx={{ height: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 2.5, px: 3, textAlign: "center" }}>
      <Typography sx={{ fontSize: 28, fontWeight: 600, letterSpacing: "-0.02em" }}>quantem 4D-STEM browser</Typography>
      <Typography sx={{ color: colors.text.tertiary, maxWidth: 560, lineHeight: 1.5 }}>
        Open a folder of Arina .h5 datasets. Everything decodes on your GPU - no install, no server, no upload.
      </Typography>
      <Button variant="contained" size="large" onClick={pick} sx={{ textTransform: "none", fontSize: 15, px: 3, py: 1.2 }}>Choose folder</Button>
      <Typography sx={{ color: colors.text.muted, fontSize: 13 }}>{busy || "Your data never leaves this machine."}</Typography>
    </Box>
  );
}

export default function App() {
  const [ready, setReady] = useState(false);
  // Served-folder hook for CDP/dev verification (the OS folder picker can't be driven
  // headlessly): __loadServed("/gold04/", ["a_master.h5", "a_data_000001.h5", ...]).
  useEffect(() => {
    (window as unknown as { __loadServed: (base: string, paths: string[]) => Promise<void> }).__loadServed =
      async (base: string, paths: string[]) => {
        const files: LocalFile[] = paths.map((p) => ({
          name: p.split("/").pop()!, relPath: p, bytes: async () => (await fetch(base + p)).arrayBuffer(),
        }));
        await scanFolder(files);
        setReady(true);
      };
    // Real disk-path hook for measurement: reads File objects (getFile().arrayBuffer()),
    // exactly what the picker yields - no http. Driven via CDP DOM.setFileInputFiles.
    (window as unknown as { __loadFileList: (list: FileList) => Promise<void> }).__loadFileList =
      async (list: FileList) => {
        const files: LocalFile[] = Array.from(list).filter((f) => /\.h5$/i.test(f.name)).map((f) => ({
          name: f.name, relPath: (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name, bytes: () => f.arrayBuffer(), source: f,
        }));
        await scanFolder(files);
        setReady(true);
      };
  }, []);
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ShortcutRegistryProvider>
        {ready ? (
          <Routes>
            <Route path="/browse/*" element={<Browse />} />
            <Route path="*" element={<Navigate to="/browse" replace />} />
          </Routes>
        ) : (
          <FolderGate onReady={() => setReady(true)} />
        )}
      </ShortcutRegistryProvider>
    </ThemeProvider>
  );
}
