"""
Parameter sweep for nonrigid drift correction on EDS Au (Gold) data.
Runs entirely from cache — does not modify the notebook.
"""
import os, sys, numpy as np, torch
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, str(Path('/home/owner/repos/quantem/src')))

from quantem.imaging import DriftCorrection
import torch.nn.functional as F

DEVICE = torch.device('cuda')
DATA_DIR = Path('/home/owner/ssd/data/bob/20260324_drift_colin_caitlyn_eds_4dstem')
CACHE_DIR = DATA_DIR / '.cache'

# --- load from cache ---
haadf_0042_0043 = np.load(CACHE_DIR / 'au_0042_0043_haadf_native.npy')

from rsciio.emd import file_reader
eds_path = DATA_DIR / '0041-CaSIO3_134hr_exsitu_SI_1.85_Mx_53.9_nm_EDS_HAADF_Diffraction_Nano.emd'
datasets = file_reader(str(eds_path))
haadf_0041 = au_0041 = None
for ds in datasets:
    d, title = ds['data'], ds.get('metadata', {}).get('General', {}).get('title', '')
    if d.ndim == 2 and 'HAADF' in title and haadf_0041 is None:
        haadf_0041 = d.astype(np.float32)
    elif d.ndim == 2 and title == 'Au' and au_0041 is None:
        au_0041 = d.astype(np.float32)

scan_h, scan_w = haadf_0041.shape
haadf_ref = np.load(CACHE_DIR / f'au_haadf_ref_{scan_h}x{scan_w}.npy')

def rcrop(img, lo=0.1, hi=0.9):
    h, w = img.shape[:2]
    return img[int(h*lo):int(h*hi), int(w*lo):int(w*hi)]

def znorm(img):
    return (img - img.mean()) / (img.std() + 1e-8)

def ncc(a, b):
    return float(np.corrcoef(a.ravel().astype(np.float64), b.ravel().astype(np.float64))[0, 1])

def mae(a, b):
    return float(np.abs(znorm(a.astype(np.float64)) - znorm(b.astype(np.float64))).mean())

ref_c = rcrop(haadf_ref)

# --- run affine once (shared across all nonrigid runs) ---
print('Running affine alignment...')
dc_base = DriftCorrection.from_data([haadf_ref, haadf_0041], [0.0, 0.0])
dc_base._device = DEVICE
dc_base.preprocess(pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5,
                   number_knots=1, normalize=False, show_merged=False, show_images=False)
dc_base.align_affine(step=0.02, num_tests=11, refine=True, upsample_factor=8,
                     max_image_shift=64, fixed_indices=[0], verbose=False,
                     show_merged=False, show_images=False)

affine_corr = dc_base.apply_correction(
    images=torch.tensor(haadf_0041, device=DEVICE, dtype=torch.float32), mode='bicubic'
).cpu().numpy()
print(f'  raw:          NCC={ncc(rcrop(haadf_0041), ref_c):.4f}  MAE={mae(rcrop(haadf_0041), ref_c):.4f}')
print(f'  after affine: NCC={ncc(rcrop(affine_corr), ref_c):.4f}  MAE={mae(rcrop(affine_corr), ref_c):.4f}')
print()

# --- sweep nonrigid parameters ---
import copy, time

configs = [
    # LBFGS sweep (best sigma from previous run)
    dict(opt='lbfgs', sigma=8.0, step=0.8, iters=12, lr=None, label='lbfgs  sigma=8  (baseline)'),
    dict(opt='lbfgs', sigma=2.0, step=0.8, iters=12, lr=None, label='lbfgs  sigma=2  (best from prev sweep)'),
    # Adam sweep at sigma=2
    dict(opt='adam',  sigma=2.0, step=0.8, iters=12, lr=None, label='adam   sigma=2  lr=auto'),
    dict(opt='adam',  sigma=2.0, step=0.8, iters=12, lr=0.1,  label='adam   sigma=2  lr=0.1'),
    dict(opt='adam',  sigma=2.0, step=0.8, iters=12, lr=0.5,  label='adam   sigma=2  lr=0.5'),
    dict(opt='adam',  sigma=2.0, step=0.8, iters=12, lr=1.0,  label='adam   sigma=2  lr=1.0'),
    dict(opt='adam',  sigma=2.0, step=0.8, iters=20, lr=0.5,  label='adam   sigma=2  lr=0.5  iters=20'),
]

print(f'  {"Config":<45}  {"NCC":>8}  {"MAE":>8}  {"max_px":>8}  {"time":>6}')
print(f'  {"-"*45}  {"------":>8}  {"------":>8}  {"------":>8}  {"------":>6}')

for cfg in configs:
    # deep-copy the dc state after affine so each run starts from the same point
    import pickle
    dc = pickle.loads(pickle.dumps(dc_base))
    dc._device = DEVICE
    # restore torch tensors (pickle doesn't carry GPU tensors)
    dc.images_t = [t.to(DEVICE) for t in dc.images_t]

    t0 = time.perf_counter()
    dc.align_nonrigid(
        optimizer_name=cfg['opt'],
        max_image_shift=64,
        num_iterations=cfg['iters'],
        regularization_sigma_px=cfg['sigma'],
        regularization_update_step_size=cfg['step'],
        lr=cfg['lr'],
        fixed_indices=[0], verbose=False,
        show_merged=False, show_images=False,
    )
    elapsed = time.perf_counter() - t0

    corr = dc.apply_correction(
        images=torch.tensor(haadf_0041, device=DEVICE, dtype=torch.float32), mode='bicubic'
    ).cpu().numpy()

    # nonrigid max correction (px)
    idx = -1 % len(dc.knots)
    delta = (dc.knots[idx] - dc._initial_knots[idx])[:, :, 0]
    max_px = float(delta.abs().max())

    print(f'  {cfg["label"]:<45}  {ncc(rcrop(corr), ref_c):8.4f}  {mae(rcrop(corr), ref_c):8.4f}  {max_px:8.2f}  {elapsed:6.1f}s')
