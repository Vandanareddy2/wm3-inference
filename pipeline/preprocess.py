"""
Turns physical GFS fields into the normalized tensors the two WM-3 encoders expect.

Both encoders are fed from GFS. The GFS encoder sees GFS's native 25 levels. The HRES encoder was trained on
ECMWF HRES's 20 levels, so we build that 20-level view from GFS: 19 levels are shared, and 20 hPa (which GFS
pgrb2 doesn't carry) is interpolated in log-pressure between 10 and 30 hPa. Each encoder input is then
normalized at its native levels and interpolated to the model's 28 levels with utils.interp_levels, matching
how the encoders were fed during training.

One known GFS/HRES difference is patched: GFS specific humidity at <=100 hPa is 2-6 standard deviations drier than
ECMWF's (checked against the provided HRES sample), which would put the HRES encoder far outside its training
distribution. For the HRES view only, humidity at those levels is replaced by an HRES zonal-mean profile
(constants/hres_strat_q_zonal.npy, built from the sample). Stratospheric water vapour varies little, so this is a
small approximation compared with feeding out-of-distribution values.
"""
import json
import numpy as np
import torch
from utils import levels_full, levels_gfs, levels_hres, levels_medium, interp_levels, core_pressure_vars

with open("constants/normalization.json") as f:
    NORMS = json.load(f)

HRES_STRAT_Q = np.load("constants/hres_strat_q_zonal.npy")  # (6 levels: 10-100 hPa, 721 lats), normalized

def to_hres_levels(pr_gfs):
    """(..., 5, 25 GFS levels) physical -> (..., 5, 20 HRES levels) physical."""
    out = []
    for level in levels_hres:
        if level in levels_gfs:
            out.append(pr_gfs[..., levels_gfs.index(level)])
        else:
            lo, hi = max(l for l in levels_gfs if l < level), min(l for l in levels_gfs if l > level)
            w = (np.log(level) - np.log(lo)) / (np.log(hi) - np.log(lo))
            out.append(pr_gfs[..., levels_gfs.index(lo)] * (1 - w) + pr_gfs[..., levels_gfs.index(hi)] * w)
    return np.stack(out, axis=-1)

def normalize(pr, sfc, levels, sfc_vars):
    """Applies WM normalization: (x - mean) / std, per variable and pressure level."""
    idx = [levels_full.index(l) for l in levels]
    mean = np.array([np.array(NORMS[v]["mean"])[idx] for v in core_pressure_vars], dtype=np.float32)
    std = np.array([np.array(NORMS[v]["std"])[idx] for v in core_pressure_vars], dtype=np.float32)
    pr_n = (pr - mean) / std

    smean = np.array([NORMS[v]["mean"] for v in sfc_vars], dtype=np.float32)
    sstd = np.array([NORMS[v]["std"] for v in sfc_vars], dtype=np.float32)
    sfc_n = (sfc - smean) / sstd
    return pr_n, sfc_n

def encoder_input(pr_n, sfc_n, mesh):
    """
    Packs normalized native-level fields into the (1, 720, 1440, mesh.n_vars) tensor one encoder expects:
    [pressure vars x 28 levels | 8 surface vars | zero padding].
    """
    H, W, C, L = pr_n.shape
    x = np.concatenate([pr_n.reshape(H, W, C * L), sfc_n, np.zeros((H, W, mesh.extra_sfc_pad), np.float32)], axis=-1)
    x = torch.from_numpy(x[:720].astype(np.float32))  # the model's grid excludes the south pole row
    x = interp_levels(x, mesh, list(mesh.input_levels), levels_medium)
    assert x.shape == mesh.shape(), f"{x.shape} vs {mesh.shape()}"
    return x.unsqueeze(0)

def build_inputs(pr_gfs, sfc, gfs_mesh, hres_mesh, sfc_vars):
    """Physical GFS fields -> [gfs_tensor, hres_tensor], both (1, 720, 1440, 157)."""
    gfs = encoder_input(*normalize(pr_gfs, sfc, levels_gfs, sfc_vars), gfs_mesh)
    pr_hres_n, sfc_n = normalize(to_hres_levels(pr_gfs), sfc, levels_hres, sfc_vars)
    q = core_pressure_vars.index("133_q")
    for j, profile in enumerate(HRES_STRAT_Q):
        pr_hres_n[:, :, q, j] = profile[:, None]  # the top len(HRES_STRAT_Q) HRES levels are <= 100 hPa
    hres = encoder_input(pr_hres_n, sfc_n, hres_mesh)
    return [gfs, hres]
