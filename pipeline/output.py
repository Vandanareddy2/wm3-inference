"""
Writes one forecast lead time to a compressed CF-style NetCDF file, using WindBorne API variable names and units.
"""
import threading
import numpy as np
import xarray as xr
from datetime import timedelta
from utils import levels_gfs, levels_medium

# Upper-air levels written to disk: the 25 levels the WindBorne gridded-forecast API offers (identical to the GFS
# levels), all of which are among the model's 28 output levels
SAVED_LEVELS = levels_gfs

LATS = np.arange(90, -90, -0.25, dtype=np.float32)
LONS = np.arange(0, 360, 0.25, dtype=np.float32)

_HDF5_LOCK = threading.Lock()  # libhdf5 (under netCDF4) is not thread-safe and segfaults on concurrent writes

def _packed_int16(arr):
    """CF int16 packing: ~4x smaller than float32 with a precision of (range / 65000)."""
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    scale = (hi - lo) / 65000 or 1.0
    return {"dtype": "int16", "scale_factor": scale, "add_offset": (hi + lo) / 2, "_FillValue": -32768, "zlib": True, "complevel": 1}

def write_netcdf(path, fields, init_time, lead_hour, model_name="WeatherMesh-3"):
    level_idx = [levels_medium.index(l) for l in SAVED_LEVELS]
    data_vars, encoding = {}, {}
    for name, (units, arr) in fields.items():
        if arr.ndim == 3:
            arr = arr[level_idx]
            dims = ("time", "level", "latitude", "longitude")
        else:
            dims = ("time", "latitude", "longitude")
        data_vars[name] = (dims, arr[None].astype(np.float32), {"units": units})
        encoding[name] = _packed_int16(arr)

    valid_time = init_time + timedelta(hours=lead_hour)
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": ("time", [np.datetime64(valid_time.replace(tzinfo=None), "ns")]),
            "level": ("level", np.array(SAVED_LEVELS, dtype=np.int32), {"units": "hPa"}),
            "latitude": ("latitude", LATS, {"units": "degrees_north"}),
            "longitude": ("longitude", LONS, {"units": "degrees_east"}),
        },
        attrs={
            "model": model_name,
            "initialization_time": init_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_hour": lead_hour,
            "valid_time": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "initial_conditions": "NCEP GFS 0.25 deg analysis (f000), used for both the GFS and HRES encoders",
            "grid": "0.25 deg regular lat/lon, 90N to 89.75S, 0E to 359.75E",
        },
    )
    with _HDF5_LOCK:
        ds.to_netcdf(path, encoding=encoding)
