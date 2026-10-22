"""
Runs WM-3 from preprocessed inputs and converts the normalized output into named physical fields.
"""
import numpy as np
import torch
from model import get_WeatherMesh3

# WM-3 output pressure variable -> (WindBorne API name, units)
PRESSURE_NAMES = {
    "129_z": ("geopotential", "m2/s2"),
    "130_t": ("temperature", "K"),
    "131_u": ("wind_u", "m/s"),
    "132_v": ("wind_v", "m/s"),
    "133_q": ("specific_humidity", "kg/kg"),
}
# WM-3 output surface variable -> (WindBorne API name, units)
SURFACE_NAMES = {
    "165_10u": ("wind_u_10m", "m/s"),
    "166_10v": ("wind_v_10m", "m/s"),
    "167_2t": ("temperature_2m", "K"),
    "151_msl": ("pressure_msl", "Pa"),
    "15_msnswrf": ("short_wave_radiation", "W/m2"),
    "45_tcc": ("total_cloud_cover", "0-1"),
    "168_2d": ("dewpoint_2m", "K"),
    "246_100u": ("wind_u_100m", "m/s"),
    "247_100v": ("wind_v_100m", "m/s"),
    "201_mx2t-6h": ("max_temperature_2m_6h", "K"),
    "202_mn2t-6h": ("min_temperature_2m_6h", "K"),
}
# Precipitation is predicted as log(metres) per component; we report their sum in mm like the API does
PRECIP_COMPONENTS = ["142_lsp-6h", "143_cp-6h"]
POLE_ROWS = 2

def load_model(weights_path, device="cuda"):
    model = get_WeatherMesh3(weights_path)
    return model.to(device).eval()

def run_forecast(model, inputs, init_unix, lead_hours, on_step, device="cuda"):
    """
    Runs the model once for all lead times; on_step(lead_hour, normalized (720, 1440, 157) CPU tensor) is called
    as each lead time finishes, so outputs never accumulate on the GPU.
    """
    x = [t.to(device) for t in inputs] + [torch.tensor([init_unix], device=device)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(x, sorted(lead_hours), send_to_cpu=True, callback=lambda dt, y: on_step(dt, y[0][0].float()))

def to_physical(y, mesh):
    """Normalized model output -> {api_name: (units, array)}; pressure fields are (n_levels, 720, 1440)."""
    y = y.numpy() * mesh.normalization_matrix_std + mesh.normalization_matrix_mean
    out = {}
    for i, v in enumerate(mesh.pressure_vars):
        name, units = PRESSURE_NAMES[v]
        out[name] = (units, np.moveaxis(y[:, :, i * mesh.n_levels:(i + 1) * mesh.n_levels], -1, 0))

    sfc = {v: y[:, :, mesh.n_pr + i] for i, v in enumerate(mesh.sfc_vars)}
    for v, (name, units) in SURFACE_NAMES.items():
        out[name] = (units, sfc[v])
    out["total_precipitation_6h"] = ("mm", sum(np.exp(sfc[v]) for v in PRECIP_COMPONENTS) * 1000.0)

    # Known output artifacts, corrected rather than shipped:
    #  - precipitation and cloud cover blow up in the two northernmost rows (90N, 89.75N); copy them from 89.5N
    #  - the model doesn't bound humidity or cloud cover, so clip them to their physical ranges
    for name in ("total_precipitation_6h", "total_cloud_cover"):
        out[name][1][:POLE_ROWS] = out[name][1][POLE_ROWS]
    np.clip(out["specific_humidity"][1], 0, None, out=out["specific_humidity"][1])
    np.clip(out["total_cloud_cover"][1], 0, 1, out=out["total_cloud_cover"][1])
    return out
