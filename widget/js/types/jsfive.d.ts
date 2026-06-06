declare module "jsfive/esm/high-level.js" {
  export class File {
    constructor(buffer: ArrayBuffer | Uint8Array | DataView | unknown, filename?: string);
    get(path: string): unknown;
    keys(): string[];
  }
}
