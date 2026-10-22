"""
Fetches the GFS 0.25 degree analysis (f000) fields that WeatherMesh-3 needs from NOAA's open-data bucket.

Only the GRIB messages we need are downloaded, using the .idx sidecar file to issue HTTP byte-range
requests, which brings a ~500MB file down to ~150MB.
"""
import threading
import time
import numpy as np
import requests
import eccodes
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

GFS_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GRAVITY = 9.80665

# WeatherMesh variable name -> GFS GRIB short name
PRESSURE_VARS = {"129_z": "HGT", "130_t": "TMP", "131_u": "UGRD", "132_v": "VGRD", "133_q": "SPFH"}

# WeatherMesh variable name -> (GFS GRIB short name, GFS level string), in the order the model expects
SURFACE_VARS = {
    "165_10u": ("UGRD", "10 m above ground"),
    "166_10v": ("VGRD", "10 m above ground"),
    "167_2t": ("TMP", "2 m above ground"),
    "151_msl": ("PRMSL", "mean sea level"),
    "45_tcc": ("TCDC", "entire atmosphere"),
    "168_2d": ("DPT", "2 m above ground"),
    "246_100u": ("UGRD", "100 m above ground"),
    "247_100v": ("VGRD", "100 m above ground"),
}

def grib_url(init_time, fhour=0):
    return f"{GFS_URL}/gfs.{init_time:%Y%m%d}/{init_time:%H}/atmos/gfs.t{init_time:%H}z.pgrb2.0p25.f{fhour:03d}"

def latest_cycle(now=None):
    """Most recent 00/06/12/18Z cycle time at or before now (UTC)."""
    now = now or datetime.now(timezone.utc)
    return now.replace(hour=now.hour - now.hour % 6, minute=0, second=0, microsecond=0)

def is_available(init_time, fhour=0):
    """NOAA writes the .idx file after the GRIB file is complete, so its presence means the file is ready."""
    r = requests.head(grib_url(init_time, fhour) + ".idx", timeout=30)
    return r.status_code == 200

def _parse_idx(text):
    """
    Parses a GRIB .idx file into {(short_name, level): (start_byte, end_byte_or_None)}.
    Lines look like '585:405701179:d=2026102206:UGRD:10 m above ground:anl:'
    """
    lines = [l.split(":") for l in text.strip().splitlines()]
    offsets = [int(l[1]) for l in lines]
    out = {}
    for i, l in enumerate(lines):
        end = offsets[i + 1] - 1 if i + 1 < len(lines) else None
        out.setdefault((l[3], l[4]), (offsets[i], end))
    return out

def _get(url, headers=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

_ECCODES_LOCK = threading.Lock()  # downloads run in parallel, but eccodes decoding is not guaranteed thread-safe

def _decode(message):
    with _ECCODES_LOCK:
        return _decode_unlocked(message)

def _decode_unlocked(message):
    h = eccodes.codes_new_from_message(message)
    try:
        nj, ni = eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni")
        assert eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees") == 90, "expected GFS grid to start at the north pole"
        return eccodes.codes_get_values(h).reshape(nj, ni).astype(np.float32)
    finally:
        eccodes.codes_release(h)

def fetch_fields(init_time, keys, fhour=0):
    """Downloads and decodes the given (short_name, level) GRIB messages. Returns {key: (721, 1440) array}."""
    url = grib_url(init_time, fhour)
    index = _parse_idx(_get(url + ".idx").decode())
    missing = [k for k in keys if k not in index]
    assert not missing, f"GFS {init_time} f{fhour:03d} is missing fields: {missing}"

    def fetch_one(key):
        start, end = index[key]
        rng = f"bytes={start}-{'' if end is None else end}"
        return key, _decode(_get(url, headers={"Range": rng}))

    with ThreadPoolExecutor(max_workers=16) as pool:
        return dict(pool.map(fetch_one, keys))

def fetch_analysis(init_time, levels):
    """
    Downloads the GFS analysis in physical units, in WeatherMesh conventions.

    Returns:
        pr (np.ndarray): (721, 1440, 5, len(levels)) pressure variables, ordered as PRESSURE_VARS
        sfc (np.ndarray): (721, 1440, 8) surface variables, ordered as SURFACE_VARS
    """
    pr_keys = [(g, f"{l} mb") for g in PRESSURE_VARS.values() for l in levels]
    fields = fetch_fields(init_time, pr_keys + list(SURFACE_VARS.values()))

    pr = np.stack([np.stack([fields[(g, f"{l} mb")] for l in levels], axis=-1) for g in PRESSURE_VARS.values()], axis=-2)
    pr[:, :, 0] *= GRAVITY  # GFS gives geopotential height (m); WM-3 uses geopotential (m^2/s^2)

    sfc = np.stack([fields[k] for k in SURFACE_VARS.values()], axis=-1)
    sfc[:, :, list(SURFACE_VARS).index("45_tcc")] /= 100.0  # GFS cloud cover is %, WM-3 uses 0-1
    return pr, sfc
