"""
Automatic sanity checks run on every forecast step:
  1. physical bounds: every field is finite and inside a broad physically possible range
  2. global means: area-weighted means that should stay near climatology (and are tracked over lead time to spot drift)
  3. agreement with GFS: lat-weighted RMSE against GFS's own forecast at the same lead time. The two models start
     from the same analysis, so at short leads they should agree closely, with differences growing over time.
"""
import numpy as np
import xarray as xr
from utils import levels_medium
from pipeline import gfs

WEIGHTS = np.cos(np.radians(np.arange(90, -90, -0.25)))[:, None] * np.ones((1, 1440))
WEIGHTS /= WEIGHTS.sum()

# api_name -> (min, max) physically possible values; anything outside means the model or preprocessing is broken
BOUNDS = {
    "temperature_2m": (170, 340),
    "dewpoint_2m": (160, 320),
    "pressure_msl": (85000, 110000),
    "total_cloud_cover": (-0.1, 1.1),
    "wind_u_10m": (-80, 80),
    "wind_v_10m": (-80, 80),
    "total_precipitation_6h": (0, 500),
    "temperature": (160, 340),
    "specific_humidity": (-1e-4, 0.04),
    "wind_u": (-150, 200),
    "wind_v": (-150, 150),
}

# (api_name, level or None, GFS grib key, unit conversion) used for the GFS comparison
GFS_COMPARE = [
    ("geopotential", 500, ("HGT", "500 mb"), gfs.GRAVITY),
    ("temperature", 850, ("TMP", "850 mb"), 1.0),
    ("temperature_2m", None, ("TMP", "2 m above ground"), 1.0),
    ("pressure_msl", None, ("PRMSL", "mean sea level"), 1.0),
]

def wmean(a):
    return float((a * WEIGHTS).sum())

def _field(fields, name, level):
    arr = fields[name][1]
    return arr[levels_medium.index(level)] if level is not None else arr

def check_step(fields):
    """Returns (summary stats, list of failure messages) for one lead time."""
    failures = []
    for name, (units, arr) in fields.items():
        if not np.isfinite(arr).all():
            failures.append(f"{name}: {np.size(arr) - np.isfinite(arr).sum()} non-finite values")
        if name in BOUNDS:
            lo, hi = BOUNDS[name]
            if arr.min() < lo or arr.max() > hi:
                failures.append(f"{name}: range [{arr.min():.4g}, {arr.max():.4g}] outside [{lo}, {hi}]")

    stats = {
        "t2m_global_mean_K": wmean(fields["temperature_2m"][1]),
        "msl_global_mean_hPa": wmean(fields["pressure_msl"][1]) / 100,
        "z500_global_mean_m": wmean(_field(fields, "geopotential", 500)) / gfs.GRAVITY,
        "precip_6h_global_mean_mm": wmean(fields["total_precipitation_6h"][1]),
        "t2m_min_K": float(fields["temperature_2m"][1].min()),
        "t2m_max_K": float(fields["temperature_2m"][1].max()),
        "max_wind_10m_ms": float(np.hypot(fields["wind_u_10m"][1], fields["wind_v_10m"][1]).max()),
    }
    if not 280 < stats["t2m_global_mean_K"] < 295:
        failures.append(f"global mean 2m temperature {stats['t2m_global_mean_K']:.1f} K is implausible")
    if not 1005 < stats["msl_global_mean_hPa"] < 1020:
        failures.append(f"global mean MSL pressure {stats['msl_global_mean_hPa']:.1f} hPa is implausible")
    return stats, failures

def compare_with_gfs(fields, init_time, lead_hour):
    """Lat-weighted RMSE and bias of WM-3 vs GFS's forecast for the same init and lead time."""
    return _compare(lambda name, level: _field(fields, name, level), init_time, lead_hour)

def compare_netcdf_with_gfs(path, init_time, lead_hour):
    """Same as compare_with_gfs, for a forecast step already saved as NetCDF. GFS publishes its later lead times
    after our run has finished, so those comparisons are filled in afterwards from the saved files."""
    with xr.open_dataset(path) as ds:
        return _compare(lambda name, level: (ds[name].sel(level=level) if level else ds[name])[0].values, init_time, lead_hour)

def _compare(get_field, init_time, lead_hour):
    gfs_fields = gfs.fetch_fields(init_time, [k for _, _, k, _ in GFS_COMPARE], fhour=lead_hour)
    out = {}
    for name, level, key, conv in GFS_COMPARE:
        diff = get_field(name, level) - gfs_fields[key][:720] * conv
        label = f"{name}{level or ''}"
        out[f"{label}_rmse"] = float(np.sqrt(wmean(diff ** 2)))
        out[f"{label}_bias"] = wmean(diff)
    return out
