"""
Drift correction report — EDS Au (Gold), one page per trial.
Usage: mamba run -n cuda-env python save_report.py
"""
import os, sys, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, str(Path('/home/owner/repos/quantem/src')))

from quantem.imaging import DriftCorrection
import torch.nn.functional as F
from rsciio.emd import file_reader

DEVICE    = torch.device('cuda')
DATA_DIR  = Path('/home/owner/ssd/data/bob/20260324_drift_colin_caitlyn_eds_4dstem')
CACHE_DIR = DATA_DIR / '.cache'
OUT_PDF   = Path(__file__).parent / 'drift_report_au_0041.pdf'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ── helpers ───────────────────────────────────────────────────────────────────
def rcrop(img, lo=0.1, hi=0.9):
    h, w = img.shape[:2]
    return img[int(h*lo):int(h*hi), int(w*lo):int(w*hi)]

def znorm(img):
    return (img - img.mean()) / (img.std() + 1e-8)

def ncc(a, b):
    return float(np.corrcoef(a.ravel().astype(np.float64), b.ravel().astype(np.float64))[0, 1])

def mae(a, b):
    return float(np.abs(znorm(a.astype(np.float64)) - znorm(b.astype(np.float64))).mean())

def rmse(a, b):
    return float(np.sqrt(np.mean((znorm(a.astype(np.float64)) - znorm(b.astype(np.float64)))**2)))

# ── load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
haadf_0042_0043 = np.load(CACHE_DIR / 'au_0042_0043_haadf_native.npy')
datasets = file_reader(str(DATA_DIR / '0041-CaSIO3_134hr_exsitu_SI_1.85_Mx_53.9_nm_EDS_HAADF_Diffraction_Nano.emd'))
haadf_0041 = au_0041 = eds_cube = None
for ds in datasets:
    d = ds['data']
    title = ds.get('metadata', {}).get('General', {}).get('title', '')
    if d.ndim == 3 and eds_cube is None:
        eds_cube = np.asarray(d)
    elif d.ndim == 2 and 'HAADF' in title and haadf_0041 is None:
        haadf_0041 = d.astype(np.float32)
    elif d.ndim == 2 and title == 'Au' and au_0041 is None:
        au_0041 = d.astype(np.float32)

scan_h, scan_w = haadf_0041.shape
haadf_ref = np.load(CACHE_DIR / f'au_haadf_ref_{scan_h}x{scan_w}.npy')
eds_total_raw = eds_cube.sum(axis=-1, dtype=np.uint64).astype(np.float32)

# ── run affine once (shared baseline for all trials) ─────────────────────────
print('Running affine alignment...')
dc_affine = DriftCorrection.from_data([haadf_ref, haadf_0041], [0.0, 0.0])
dc_affine._device = DEVICE
dc_affine.preprocess(pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5,
                     number_knots=1, normalize=False, show_merged=False, show_images=False)
dc_affine.align_affine(step=0.02, num_tests=11, refine=True, upsample_factor=8,
                       max_image_shift=64, fixed_indices=[0], verbose=True,
                       show_merged=False, show_images=False)

def apply(dc, img_np):
    return dc.apply_correction(
        images=torch.tensor(img_np, device=DEVICE, dtype=torch.float32), mode='bicubic'
    ).cpu().numpy()

haadf_affine = apply(dc_affine, haadf_0041)
print(f'  affine: NCC={ncc(rcrop(haadf_affine), rcrop(haadf_ref)):.4f}  '
      f'MAE={mae(rcrop(haadf_affine), rcrop(haadf_ref)):.4f}')

# ── trial configs ─────────────────────────────────────────────────────────────
TRIALS = [
    dict(opt='lbfgs', sigma=8.0, step=0.8, iters=12, lr=None,
         label='LBFGS  σ=8  step=0.8  iters=12  (original baseline)'),
    dict(opt='lbfgs', sigma=2.0, step=0.8, iters=12, lr=None,
         label='LBFGS  σ=2  step=0.8  iters=12'),
    dict(opt='adam',  sigma=2.0, step=0.8, iters=12, lr=None,
         label='Adam   σ=2  lr=auto   iters=12'),
    dict(opt='adam',  sigma=2.0, step=0.8, iters=12, lr=0.1,
         label='Adam   σ=2  lr=0.1    iters=12'),
    dict(opt='adam',  sigma=2.0, step=0.8, iters=12, lr=0.5,
         label='Adam   σ=2  lr=0.5    iters=12  ← best'),
    dict(opt='adam',  sigma=2.0, step=0.8, iters=12, lr=1.0,
         label='Adam   σ=2  lr=1.0    iters=12'),
    dict(opt='adam',  sigma=2.0, step=0.8, iters=20, lr=0.5,
         label='Adam   σ=2  lr=0.5    iters=20'),
]

# ── run all trials ────────────────────────────────────────────────────────────
import pickle, time

print('\nRunning trials...')
results = []  # list of dicts: label, haadf_corr, metrics
for i, cfg in enumerate(TRIALS):
    dc = pickle.loads(pickle.dumps(dc_affine))
    dc._device = DEVICE
    dc.images_t = [t.to(DEVICE) for t in dc.images_t]
    t0 = time.perf_counter()
    dc.align_nonrigid(optimizer_name=cfg['opt'], lr=cfg['lr'],
                      max_image_shift=64, num_iterations=cfg['iters'],
                      regularization_sigma_px=cfg['sigma'],
                      regularization_update_step_size=cfg['step'],
                      fixed_indices=[0], verbose=False,
                      show_merged=False, show_images=False)
    elapsed = time.perf_counter() - t0
    corr = apply(dc, haadf_0041)
    idx = -1 % len(dc.knots)
    delta = (dc.knots[idx] - dc._initial_knots[idx])[:, :, 0]
    max_shift = float(delta.abs().max())
    results.append(dict(
        label=cfg['label'], corr=corr, elapsed=elapsed, max_shift=max_shift,
        ncc=ncc(rcrop(corr), rcrop(haadf_ref)),
        mae=mae(rcrop(corr), rcrop(haadf_ref)),
        rmse=rmse(rcrop(corr), rcrop(haadf_ref)),
    ))
    print(f'  [{i+1}/{len(TRIALS)}] {cfg["label"][:45]:<45} '
          f'NCC={results[-1]["ncc"]:.4f}  MAE={results[-1]["mae"]:.4f}  {elapsed:.1f}s')

# ── fixed display ranges (computed once, same across all pages) ───────────────
ref_c    = rcrop(haadf_ref)
raw_c    = rcrop(haadf_0041)
raw_diff = znorm(ref_c.astype(np.float64)) - znorm(raw_c.astype(np.float64))
DIFF_VMAX  = float(np.abs(raw_diff).max())             # same scale on every diff page
# Full min/max range — no quantile clipping — so all noise is visible and
# brightness is directly comparable across all HAADF panels in the report.
all_haadf = np.stack([raw_c, rcrop(haadf_affine), ref_c] + [rcrop(r['corr']) for r in results])
IMG_VLIM = [float(all_haadf.min()), float(all_haadf.max())]
del all_haadf

def show_img(ax, img, title, cmap='gray', vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=13, pad=6, fontweight='bold')
    ax.axis('off')

def metric_label(r, ref):
    return f'NCC {ncc(r,ref):.4f}   MAE {mae(r,ref):.4f}   RMSE {rmse(r,ref):.4f}'

# ── build PDF ─────────────────────────────────────────────────────────────────
print(f'\nSaving {OUT_PDF} ...')
with PdfPages(OUT_PDF) as pdf:

    # ── cover page ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.15)

    show_img(fig.add_subplot(gs[0, 0]), rcrop(haadf_0041), 'HAADF 0041 — raw',
             vmin=IMG_VLIM[0], vmax=IMG_VLIM[1])
    show_img(fig.add_subplot(gs[0, 1]), rcrop(haadf_affine), 'HAADF 0041 — after affine',
             vmin=IMG_VLIM[0], vmax=IMG_VLIM[1])
    show_img(fig.add_subplot(gs[0, 2]), rcrop(haadf_ref),  'HAADF ref (0042+0043)',
             vmin=IMG_VLIM[0], vmax=IMG_VLIM[1])

    ax_diff = fig.add_subplot(gs[1, 0])
    show_img(ax_diff, raw_diff, 'Difference: ref − raw', cmap='seismic',
             vmin=-DIFF_VMAX, vmax=DIFF_VMAX)

    ax_diff2 = fig.add_subplot(gs[1, 1])
    aff_diff = znorm(ref_c.astype(np.float64)) - znorm(rcrop(haadf_affine).astype(np.float64))
    show_img(ax_diff2, aff_diff, 'Difference: ref − after affine', cmap='seismic',
             vmin=-DIFF_VMAX, vmax=DIFF_VMAX)

    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis('off')
    info = (
        'Dataset: EDS Au (Gold)\n'
        f'File 0041: HAADF {haadf_0041.shape[0]}×{haadf_0041.shape[1]}\n'
        f'EDS cube: {eds_cube.shape}\n'
        'Reference: HAADF 0042+0043\n'
        '   (drift-corrected, downsampled)\n\n'
        f'Affine drift: +0.020 / +0.016 px/line\n'
        f'  Total: ~26 px over 1024 rows\n\n'
        f'Baseline (raw):\n'
        f'  NCC  {ncc(raw_c, ref_c):.4f}\n'
        f'  MAE  {mae(raw_c, ref_c):.4f}\n'
        f'  RMSE {rmse(raw_c, ref_c):.4f}\n\n'
        f'After affine:\n'
        f'  NCC  {ncc(rcrop(haadf_affine), ref_c):.4f}\n'
        f'  MAE  {mae(rcrop(haadf_affine), ref_c):.4f}\n'
        f'  RMSE {rmse(rcrop(haadf_affine), ref_c):.4f}'
    )
    ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                 fontsize=11, va='top', ha='left', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5', edgecolor='#cccccc'))

    fig.suptitle('EDS Au (Gold) — Drift Correction Report\nFile 0041 vs Reference 0042+0043',
                 fontsize=16, fontweight='bold', y=1.02)
    pdf.savefig(fig, bbox_inches='tight', dpi=150)
    plt.close(fig)

    # ── one page per trial ────────────────────────────────────────────────────
    best_mae = min(r['mae'] for r in results)

    for r in results:
        corr_c = rcrop(r['corr'])
        diff_c = znorm(ref_c.astype(np.float64)) - znorm(corr_c.astype(np.float64))
        is_best = abs(r['mae'] - best_mae) < 1e-6

        fig = plt.figure(figsize=(18, 7))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.08)

        show_img(fig.add_subplot(gs[0]), corr_c, 'HAADF 0041 — drift-corrected',
                 vmin=IMG_VLIM[0], vmax=IMG_VLIM[1])
        show_img(fig.add_subplot(gs[1]), ref_c,  'HAADF ref (0042+0043)',
                 vmin=IMG_VLIM[0], vmax=IMG_VLIM[1])
        show_img(fig.add_subplot(gs[2]), diff_c, 'Difference: ref − corrected',
                 cmap='seismic', vmin=-DIFF_VMAX, vmax=DIFF_VMAX)

        badge = '  ★ BEST' if is_best else ''
        fig.suptitle(
            f'{r["label"]}{badge}\n'
            f'NCC {r["ncc"]:.4f}   MAE {r["mae"]:.4f}   RMSE {r["rmse"]:.4f}'
            f'   |   nonrigid max shift {r["max_shift"]:.1f} px   |   {r["elapsed"]:.1f}s',
            fontsize=13, fontweight='bold' if is_best else 'normal', y=1.03,
        )
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)

    # ── summary comparison page ───────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    ax  = fig.add_subplot(111)
    ax.axis('off')

    rows = [['raw',    f'{ncc(raw_c, ref_c):.4f}',
                       f'{mae(raw_c, ref_c):.4f}',
                       f'{rmse(raw_c, ref_c):.4f}', '—', '—']]
    rows += [['affine', f'{ncc(rcrop(haadf_affine), ref_c):.4f}',
                        f'{mae(rcrop(haadf_affine), ref_c):.4f}',
                        f'{rmse(rcrop(haadf_affine), ref_c):.4f}', '—', '—']]
    for r in results:
        rows.append([r['label'].replace('← best', '').strip(),
                     f'{r["ncc"]:.4f}', f'{r["mae"]:.4f}', f'{r["rmse"]:.4f}',
                     f'{r["max_shift"]:.1f} px', f'{r["elapsed"]:.1f}s'])

    tbl = ax.table(
        cellText=rows,
        colLabels=['Config', 'NCC ↑', 'MAE ↓', 'RMSE ↓', 'Max shift', 'Time'],
        cellLoc='center', loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.2)

    # highlight best nonrigid row
    best_row = min(range(len(results)), key=lambda i: results[i]['mae'])
    for col in range(6):
        tbl[best_row + 3, col].set_facecolor('#d4edda')  # +3: header + raw + affine rows

    fig.suptitle('Summary — All Trials\nReference: HAADF 0042+0043 (drift-corrected)',
                 fontsize=15, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight', dpi=150)
    plt.close(fig)

print(f'Done → {OUT_PDF}  ({len(TRIALS)+2} pages)')
