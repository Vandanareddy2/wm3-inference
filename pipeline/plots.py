"""Quick-look PNG maps for each forecast, uploaded next to the NetCDF files."""
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utils import levels_medium

_PYPLOT_LOCK = threading.Lock()  # pyplot keeps global state and is not thread-safe

def plot_step(fields, path, title):
    with _PYPLOT_LOCK:
        _plot_step(fields, path, title)

def _plot_step(fields, path, title):
    t2m = fields["temperature_2m"][1] - 273.15
    msl = fields["pressure_msl"][1] / 100
    z500 = fields["geopotential"][1][levels_medium.index(500)] / 9.80665 / 10  # decametres
    precip = fields["total_precipitation_6h"][1]
    extent = [0, 360, -90, 90]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), constrained_layout=True)
    im = axes[0].imshow(t2m, extent=extent, cmap="RdYlBu_r", vmin=-40, vmax=40)
    cs = axes[0].contour(np.linspace(0, 360, 1440), np.linspace(90, -90, 720), msl, levels=np.arange(960, 1050, 8), colors="k", linewidths=0.5)
    axes[0].clabel(cs, fontsize=6)
    axes[0].set_title("2m temperature (°C) + MSL pressure (hPa)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    im = axes[1].imshow(np.where(precip > 0.1, precip, np.nan), extent=extent, cmap="Blues", vmin=0, vmax=30)
    cs = axes[1].contour(np.linspace(0, 360, 1440), np.linspace(90, -90, 720), z500, levels=np.arange(480, 600, 6), colors="purple", linewidths=0.5)
    axes[1].clabel(cs, fontsize=6)
    axes[1].set_title("6h precipitation (mm) + 500 hPa height (dam)")
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    fig.suptitle(title)
    fig.savefig(path, dpi=90)
    plt.close(fig)
