// Folder picker for the in-page "Choose folder" affordance (no separate gate page). Opens the OS
// directory picker (File System Access API, or a webkitdirectory <input> fallback), scans the
// chosen folder(s), and fires `quantem-folder-loaded` so the Browse page refreshes its dataset tree.
import { scanFolder, type LocalFile } from "./store";

let pickedFiles = new Map<string, LocalFile>();

function mergePickedFiles(files: LocalFile[]): LocalFile[] {
  for (const file of files) pickedFiles.set(file.relPath, file);
  return Array.from(pickedFiles.values());
}

function uniqueRootName(root: string): string {
  const clean = root.trim() || "folder";
  const used = new Set(Array.from(pickedFiles.keys()).map((path) => path.split("/").filter(Boolean)[0]));
  if (!used.has(clean)) return clean;
  let i = 2;
  while (used.has(`${clean}-${i}`)) i++;
  return `${clean}-${i}`;
}

function filesUnderRoot(files: LocalFile[], root: string, replaceExistingRoot: boolean): LocalFile[] {
  const nextRoot = uniqueRootName(root);
  return files.map((file) => {
    const parts = file.relPath.split("/").filter(Boolean);
    const relInsideRoot = replaceExistingRoot && parts.length > 1 ? parts.slice(1).join("/") : file.relPath;
    return { ...file, relPath: `${nextRoot}/${relInsideRoot}` };
  });
}

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

function pickViaInput(): Promise<LocalFile[]> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file"; input.webkitdirectory = true; input.multiple = true;
    // Attach to the DOM (hidden) so the change event fires reliably - a detached input can drop
    // it in some browsers + breaks automation. Removed once the files are read.
    input.style.display = "none";
    document.body.appendChild(input);
    input.onchange = () => { resolve(input.files ? filesFromInput(input.files) : []); input.remove(); };
    input.click();
  });
}

/** Open the picker, scan the chosen folder. Returns the # of .h5 files scanned (0 = cancelled /
 *  none). Notifies the Browse page via the `quantem-folder-loaded` event so it re-reads sessions. */
export async function pickFolderAndScan(): Promise<number> {
  let files: LocalFile[] = [];
  let inputFallback = false;
  if ("showDirectoryPicker" in window) {
    try {
      // @ts-expect-error - File System Access API
      const dir = await window.showDirectoryPicker();
      files = filesUnderRoot(await filesFromDirHandle(dir), dir.name, false);
    } catch { inputFallback = true; files = await pickViaInput(); }   // unsupported (file://) / cancelled -> input fallback
  } else {
    inputFallback = true;
    files = await pickViaInput();
  }
  if (inputFallback && files.length && files[0].relPath.includes("/")) files = filesUnderRoot(files, files[0].relPath.split("/")[0], true);
  if (!files.length) return 0;
  const allFiles = mergePickedFiles(files);
  await scanFolder(allFiles);
  window.dispatchEvent(new Event("quantem-folder-loaded"));
  return allFiles.length;
}
