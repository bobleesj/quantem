# QuantEM tutorials

## MAPED: seven-tilt merging

Use these three pieces together:

1. [Interactive MAPED tutorial](maped_interactive.html) explains the complete
   seven-tilt flow. Its controls show what the alignment parameters change, how
   region size affects peak memory, and how globally scaled uint16 affects
   precision. Download or clone the repository and open the HTML file directly
   in a browser; it is self-contained.
2. [MAPED on a 24 GiB GPU](maped_24gb_workflow.md) gives the practical memory,
   output, precision, and timing contract.
3. [CUDA MAPED notebook](../../notebooks/maped/cuda_maped_merge%20\(1\).ipynb)
   runs the same public API on all seven tilts and opens the saved result with
   Show4DSTEM.

The tutorial files and their figures live in this repository. There are no
separate workspace copies to keep synchronized.
