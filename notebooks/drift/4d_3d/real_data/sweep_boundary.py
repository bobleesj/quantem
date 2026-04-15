"""
Boundary overcorrection sweep — measures metrics separately for
full image, top half, and bottom rows to find parameters that
don't degrade the boundary while still correcting the center.

Usage: mamba run -n cuda-env python sweep_boundary.py
"""
import os, sys, numpy as np, torch, pickle, time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, str(Path('/home/owner/repos/quantem/src')))

from quantem.imaging import DriftCorrection
from rsciio.emd import file_reader

DEVICE   = torch.device('cuda')
DATA_DIR = Path('/home/owner/ssd/data/bob/20260324_drift_colin_caitlyn_eds_4dstem')
CACHE_DIR = DATA_DIR / '.cache'

# --- load ---
haadf_0042_0043 = np.load(CACHE_DIR / 'au_0042_0043_haadf_native.npy')
datasets = file_reader(str(DATA_DIR / '0041-CaSIO3_134hr_exsitu_SI_1.85_Mx_53.9_nm_EDS_HAADF_Diffraction_Nano.emd'))
haadf_0041 = None
for ds in datasets:
    d, title = ds['data'], ds.get('metadata', {}).get('General', {}).get('title', '')
    if d.ndim == 2 and 'HAADF' in title and haadf_0041 is None:
        haadf_0041 = d.astype(np.float32)

scan_h, scan_w = haadf_0041.shape
haadf_ref = np.load(CACHE_DIR / f'au_haadf_ref_{scan_h}x{scan_w}.npy')

def znorm(img):
    return (img - img.mean()) / (img.std() + 1e-8)

def mae(a, b):
    return float(np.abs(znorm(a.astype(np.float64)) - znorm(b.astype(np.float64))).mean())

def ncc(a, b):
    return float(np.corrcoef(a.ravel().astype(np.float64), b.ravel().astype(np.float64))[0, 1])

def region_crop(img, row_lo, row_hi, col_lo=0.1, col_hi=0.9):
    """Crop by row fraction and column fraction."""
    h, w = img.shape
    return img[int(h*row_lo):int(h*row_hi), int(w*col_lo):int(w*col_hi)]

# Regions: center (10-80%) and bottom boundary (80-95%)
# Column margins always 10-90% to avoid edge artifacts
def center_crop(img):   return region_crop(img, 0.10, 0.80)
def bottom_crop(img):   return region_crop(img, 0.80, 0.95)
def top_crop(img):      return region_crop(img, 0.10, 0.40)
def full_crop(img):     return region_crop(img, 0.10, 0.90)

# --- affine baseline ---
print('Running affine alignment...')
dc_base = DriftCorrection.from_data([haadf_ref, haadf_0041], [0.0, 0.0])
dc_base._device = DEVICE
dc_base.preprocess(pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5,
                   number_knots=1, normalize=False, show_merged=False, show_images=False)
dc_base.align_affine(step=0.02, num_tests=11, refine=True, upsample_factor=8,
                     max_image_shift=64, fixed_indices=[0], verbose=False,
                     show_merged=False, show_images=False)

def apply(dc, img_np):
    return dc.apply_correction(
        images=torch.tensor(img_np, device=DEVICE, dtype=torch.float32), mode='bicubic'
    ).cpu().numpy()

haadf_affine = apply(dc_base, haadf_0041)
ref = haadf_ref

print('\n  Baselines:')
print(f'    raw    — full MAE={mae(full_crop(haadf_0041), full_crop(ref)):.4f}  '
      f'top MAE={mae(top_crop(haadf_0041), top_crop(ref)):.4f}  '
      f'bottom MAE={mae(bottom_crop(haadf_0041), bottom_crop(ref)):.4f}')
print(f'    affine — full MAE={mae(full_crop(haadf_affine), full_crop(ref)):.4f}  '
      f'top MAE={mae(top_crop(haadf_affine), top_crop(ref)):.4f}  '
      f'bottom MAE={mae(bottom_crop(haadf_affine), bottom_crop(ref)):.4f}')

# --- sweep ---
configs = [
    # vary sigma (main suspect for boundary overcorrection)
    dict(opt='adam', sigma=2.0, lr=0.5, iters=12, label='Adam σ=2  lr=0.5  it=12  (current best)'),
    dict(opt='adam', sigma=3.0, lr=0.5, iters=12, label='Adam σ=3  lr=0.5  it=12'),
    dict(opt='adam', sigma=4.0, lr=0.5, iters=12, label='Adam σ=4  lr=0.5  it=12'),
    dict(opt='adam', sigma=6.0, lr=0.5, iters=12, label='Adam σ=6  lr=0.5  it=12'),
    dict(opt='adam', sigma=8.0, lr=0.5, iters=12, label='Adam σ=8  lr=0.5  it=12'),
    # vary lr at sigma=4 (likely sweet spot)
    dict(opt='adam', sigma=4.0, lr=0.3, iters=12, label='Adam σ=4  lr=0.3  it=12'),
    dict(opt='adam', sigma=4.0, lr=0.5, iters=20, label='Adam σ=4  lr=0.5  it=20'),
    # LBFGS comparison
    dict(opt='lbfgs', sigma=4.0, lr=None, iters=12, label='LBFGS σ=4        it=12'),
    dict(opt='lbfgs', sigma=2.0, lr=None, iters=12, label='LBFGS σ=2        it=12'),
]

print(f'\n  {"Config":<42}  {"full MAE":>8}  {"top MAE":>8}  {"bot MAE":>8}  {"bot delta":>10}  {"time":>6}')
print(f'  {"-"*42}  {"--------":>8}  {"--------":>8}  {"--------":>8}  {"----------":>10}  {"------":>6}')

ref_bot_affine = mae(bottom_crop(haadf_affine), bottom_crop(ref))

for cfg in configs:
    dc = pickle.loads(pickle.dumps(dc_base))
    dc._device = DEVICE
    dc.images_t = [t.to(DEVICE) for t in dc.images_t]

    t0 = time.perf_counter()
    dc.align_nonrigid(
        optimizer_name=cfg['opt'], lr=cfg.get('lr'),
        max_image_shift=64, num_iterations=cfg['iters'],
        regularization_sigma_px=cfg['sigma'],
        regularization_update_step_size=0.8,
        fixed_indices=[0], verbose=False,
        show_merged=False, show_images=False,
    )
    elapsed = time.perf_counter() - t0

    corr = apply(dc, haadf_0041)
    full_mae  = mae(full_crop(corr),   full_crop(ref))
    top_mae   = mae(top_crop(corr),    top_crop(ref))
    bot_mae   = mae(bottom_crop(corr), bottom_crop(ref))
    bot_delta = bot_mae - ref_bot_affine   # positive = got WORSE vs affine

    flag = '  <-- WORSE boundary' if bot_delta > 0.005 else ('  <-- OK' if bot_delta < -0.002 else '')
    print(f'  {cfg["label"]:<42}  {full_mae:8.4f}  {top_mae:8.4f}  {bot_mae:8.4f}  '
          f'{bot_delta:+10.4f}  {elapsed:6.1f}s{flag}')
