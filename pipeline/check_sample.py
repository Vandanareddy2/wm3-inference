"""
One-off check of our preprocessing: rebuild WindBorne's sample input (2025-03-07 00Z) from the archived GFS
analysis and compare it with the provided sample .npz files, variable by variable.

Usage: python -m pipeline.check_sample /path/to/sample/root
"""
import sys
import numpy as np
from datetime import datetime, timezone
from utils import levels_gfs, levels_hres, core_pressure_vars
from pipeline import gfs, preprocess

SFC_VARS = list(gfs.SURFACE_VARS)

def load_sample(root, source):
    base = f"{root}/huge/proc/haoxing_data/wm3/data/{source}"
    f = np.load(f"{base}/f000/202503/1741305600.npz")
    extras = [np.load(f"{base}/extra/{v}/202503/1741305600.npz")["x"] for v in SFC_VARS[4:]]
    return f["pr"].astype(np.float32), np.concatenate([f["sfc"].astype(np.float32), np.stack(extras, -1).astype(np.float32)], -1)

def report(name, ours, theirs):
    diff = ours - theirs
    print(f"{name:>10}  ours mean {ours.mean():8.3f} std {ours.std():7.3f} | sample mean {theirs.mean():8.3f} std {theirs.std():7.3f} | "
          f"rmse {np.sqrt((diff ** 2).mean()):6.3f}  corr {np.corrcoef(ours.ravel()[::97], theirs.ravel()[::97])[0, 1]:.4f}")

if __name__ == "__main__":
    t0 = datetime.fromtimestamp(1741305600, timezone.utc)
    pr, sfc = gfs.fetch_analysis(t0, levels_gfs)

    for source, levels, pr_phys in [("neogfs", levels_gfs, pr), ("neohres", levels_hres, preprocess.to_hres_levels(pr))]:
        print(f"--- {source} (normalized units; rmse 0 = identical) ---")
        ours_pr, ours_sfc = preprocess.normalize(pr_phys, sfc, levels, SFC_VARS)
        s_pr, s_sfc = load_sample(sys.argv[1], source)
        for i, v in enumerate(core_pressure_vars):
            for l in (50, 500, 850):
                j = levels.index(l)
                report(f"{v}@{l}", ours_pr[:, :, i, j], s_pr[:, :, i, j])
        for i, v in enumerate(SFC_VARS):
            report(v, ours_sfc[:, :, i], s_sfc[:, :, i])
