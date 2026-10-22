"""
Benchmarks WM-3 inference on the current GPU and checks how much bfloat16 changes the forecast vs float32.

Times model loading, then runs the full 10-day forecast (40 x 6h steps) several times with outputs left on the GPU
and discarded, so the numbers are model compute only (no file writing or upload).

Usage: python -m pipeline.benchmark [YYYYMMDDHH]
"""
import sys
import time
import numpy as np
import torch
from datetime import datetime, timedelta, timezone
from utils import levels_gfs, levels_medium
from pipeline import gfs, preprocess, infer, validate

LEADS = list(range(6, 241, 6))
COMPARE_LEADS = (24, 120, 240)
RUNS = 3

def timed(fn):
    torch.cuda.synchronize()
    start = time.time()
    result = fn()
    torch.cuda.synchronize()
    return time.time() - start, result

def forecast(model, x, dtype, keep=()):
    """Runs all LEADS; returns {lead: normalized output on CPU} for the leads in keep."""
    kept = {}
    def on_step(dt, y):
        if dt in keep:
            kept[dt] = y[0][0].float().cpu()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        model(x, LEADS, send_to_cpu=False, callback=on_step)
    return kept

if __name__ == "__main__":
    init = datetime.strptime(sys.argv[1], "%Y%m%d%H").replace(tzinfo=timezone.utc) if len(sys.argv) > 1 else gfs.latest_cycle()
    while not gfs.is_available(init):
        init -= timedelta(hours=6)
    print(f"GPU: {torch.cuda.get_device_name()}, init {init:%Y-%m-%d %HZ}, {len(LEADS)} steps (10 days)")

    t_load, model = timed(lambda: infer.load_model("WeatherMesh3.pt"))
    print(f"model load (disk -> GPU): {t_load:.1f} s")

    pr, sfc = gfs.fetch_analysis(init, levels_gfs)
    gi, hi = model.config.inputs
    out_mesh = model.config.outputs[0]
    x = [t.cuda() for t in preprocess.build_inputs(pr, sfc, gi, hi, list(gfs.SURFACE_VARS))]
    x.append(torch.tensor([int(init.timestamp())], device="cuda"))

    results = {}
    for name, dtype in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
        forecast(model, x, dtype)  # warm-up (CUDA kernels, allocator)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(RUNS):
            t, kept = timed(lambda: forecast(model, x, dtype, keep=COMPARE_LEADS))
            times.append(t)
        results[name] = kept
        print(f"{name:>9}: 10-day forecast {np.mean(times):.1f} s avg / {min(times):.1f} s best over {RUNS} runs "
              f"= {np.mean(times) / len(LEADS):.2f} s per 6h step, peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    print("bfloat16 vs float32 difference (lat-weighted RMSE):")
    for lead in COMPARE_LEADS:
        a = infer.to_physical(results["bfloat16"][lead], out_mesh)
        b = infer.to_physical(results["float32"][lead], out_mesh)
        z = levels_medium.index(500)
        z500 = np.sqrt(validate.wmean((a["geopotential"][1][z] - b["geopotential"][1][z]) ** 2)) / gfs.GRAVITY
        t2m = np.sqrt(validate.wmean((a["temperature_2m"][1] - b["temperature_2m"][1]) ** 2))
        print(f"  +{lead:3d}h: Z500 {z500:5.1f} m, T2m {t2m:.2f} K")
