# WeatherMesh-3 real-time inference pipeline

A real-time forecasting pipeline built on top of WindBorne's open-source [WeatherMesh-3](https://github.com/windborne/WeatherMesh-3) architecture ([paper](https://arxiv.org/abs/2503.22235)).

Every 6 hours it downloads NOAA's latest GFS analysis, converts it into WM-3's input format, runs a 14-day global forecast on one GPU, checks the output, and publishes NetCDF files to S3, with a public status page and phone alerts on failure.

- **Live status page:** https://windborne-wm3-vandana-131548386286-us-east-2-an.s3.us-east-2.amazonaws.com/forecasts/index.html
- **Outputs:** `https://windborne-wm3-vandana-131548386286-us-east-2-an.s3.us-east-2.amazonaws.com/forecasts/{YYYYMMDDHH}/wm3_{YYYYMMDDHH}_f{HHH}.nc`

## How it works

```
NOAA GFS (public S3)  ->  download only the needed fields (~13 s)
                      ->  preprocess: units, normalization, 25 -> 28 levels, GFS also feeds the HRES encoder (~8 s)
                      ->  WM-3 on GPU, 56 x 6h steps = 14 days (~36 s on an RTX 4090)
                      ->  per step: physical units -> validation -> NetCDF (+ PNG maps) -> S3
                      ->  status.json / manifest.json, ntfy alerts
```

WM-3 has two encoders (GFS and ECMWF HRES). ECMWF's open data doesn't have all the required levels, so GFS feeds both. For the HRES encoder, GFS is put on HRES's 20 levels (20 hPa is interpolated in log-pressure), and humidity at 10–100 hPa is replaced by an HRES-like profile, because GFS is much drier up there than what that encoder saw in training. See `pipeline/preprocess.py` for details.

## Repo layout

| Path | What it does |
|---|---|
| `model.py`, `model_latlon/`, `meshes.py`, `utils.py`, `constants/` | WindBorne's original model code and constants (unchanged) |
| `pipeline/gfs.py` | Downloads the GFS fields WM-3 needs, using HTTP byte-range requests |
| `pipeline/preprocess.py` | Turns GFS fields into the two encoder inputs |
| `pipeline/infer.py` | Loads the model, runs the forecast, converts outputs to physical units |
| `pipeline/output.py` | Writes NetCDF using WindBorne API variable names, units, attributes and the API's 25 pressure levels |
| `pipeline/validate.py` | Per-step checks: physical bounds, global means, comparison with GFS's own forecast |
| `pipeline/run_cycle.py` | The pipeline: one cycle, or `--loop` to poll for new GFS cycles |
| `pipeline/benchmark.py` | Inference speed (bf16 vs fp32) and accuracy impact |
| `pipeline/check_sample.py` | Checks preprocessing against WindBorne's provided sample input |
| `run.sh` | Supervisor: restarts the loop if it crashes and sends an alert |
| `monitoring/index.html` | Status page (uploaded to S3) |
| `.github/workflows/watchdog.yml` | External watchdog: alerts if the pipeline stops reporting, even if the GPU machine is dead |

## Setup

**Requirements:** Linux, an NVIDIA GPU with at least 16 GB of memory (it uses ~8–9 GB; tested on an RTX 4090), CUDA 12 drivers, and an S3 bucket.

The model weights (`WeatherMesh3.pt`, ~1.2 GB, provided by WindBorne) are not in this repo. Put them in your bucket, e.g. at `private/WeatherMesh3.pt`.

### Option A: Docker

```bash
docker build -t wm3-pipeline .
docker run --gpus all \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_DEFAULT_REGION=us-east-2 \
  -e S3_BUCKET=your-bucket -e NTFY_TOPIC=your-topic \
  wm3-pipeline
```

The container downloads the weights from `s3://$S3_BUCKET/private/WeatherMesh3.pt` and starts the polling loop.

*Testing note:* the live pipeline runs on a RunPod pod, which is itself a container and can't build Docker images, so the image hasn't been built end to end. Instead, the Dockerfile's install steps were run in a fresh, empty Python environment on the pod (same commands; the cp312 wheel instead of cp311). All imports worked and the model produced a 24-hour forecast that passed every validation check.

### Option B: directly on a GPU machine (how the live pipeline runs, on RunPod)

```bash
git clone https://github.com/Vandanareddy2/wm3-inference.git && cd wm3-inference
python3 -m venv ../venv && . ../venv/bin/activate
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# natten only ships prebuilt wheels for torch 2.4 + CUDA 12.1. Its server's TLS certificate was expired
# at the time of writing, hence -k. Pick the wheel matching your Python version (cp310/cp311/cp312).
curl -sk -O "https://shi-labs.com/natten/wheels/cu121/torch2.4.0/natten-0.17.3%2Btorch240cu121-cp312-cp312-linux_x86_64.whl"
pip install ./natten-0.17.3+torch240cu121-cp312-cp312-linux_x86_64.whl
pip install --index-url https://test.pypi.org/simple/ --no-deps matepoint
pip install numpy==2.2.2 boto3 xarray netCDF4 eccodes requests matplotlib
aws configure   # or export AWS_* environment variables
python -c "import boto3; boto3.client('s3').download_file('your-bucket', 'private/WeatherMesh3.pt', 'WeatherMesh3.pt')"
```

Commands must be run from the repo root, because the model code loads `constants/` with relative paths.

## Running

```bash
python -m pipeline.run_cycle                    # newest available GFS cycle, once
python -m pipeline.run_cycle --init 2026102212  # a specific cycle
nohup ./run.sh > pipeline.log 2>&1 &            # production: poll every 10 min, restart on crash
python -m pipeline.benchmark                    # speed benchmark
python -m pipeline.check_sample /path/to/sample # compare preprocessing with WindBorne's sample
```

**Configuration (environment variables):** `S3_BUCKET`, `S3_PREFIX` (default `forecasts`), `NTFY_TOPIC`, `WEIGHTS_PATH` (default `WeatherMesh3.pt`), `WORK_DIR` (default `/workspace/runs`), `MAX_LEAD_HOURS` (default `336`).

## Outputs

For each GFS cycle, in `s3://$S3_BUCKET/forecasts/{YYYYMMDDHH}/`:
- `wm3_{YYYYMMDDHH}_f{HHH}.nc`: one file per 6-hour step, `f006` to `f336`. Variables use WindBorne API names and units (`temperature_2m` K, `pressure_msl` Pa, `geopotential` m²/s², `wind_u_10m` m/s, `total_precipitation_6h` mm, …) on the API's 25 pressure levels, with global attributes `model`, `initialization_time` and `forecast_hour`. Grid: 0.25°, 90N→89.75S, 0E→359.75E.
- `wm3_..._f{024,072,120,240,336}.png`: quick-look maps.
- `manifest.json`: file list, timings, and validation results for every step.

`forecasts/status.json` holds the pipeline heartbeat and recent runs; `forecasts/index.html` displays it.

## Monitoring and alerts

- **Failures** (download, model, validation, upload) send a high-priority ntfy push alert, and the cycle is retried up to 3 times.
- **Crashes:** `run.sh` restarts the loop and sends an alert.
- **Dead machine:** the GitHub Actions watchdog checks `status.json` every 30 minutes, from outside the GPU machine, and alerts if there's no heartbeat or no fresh forecast.
- **Success:** a silent (low-priority) notification after each cycle.
- **GFS comparison catch-up:** a cycle runs as soon as GFS's analysis is out, before GFS publishes its later forecast hours, so the 5/10/14-day comparisons are skipped at run time. Every 10 minutes the loop fills them in for the last 4 cycles once GFS has published them, and updates each cycle's `manifest.json`.

## Known limitations

- Both encoders are fed GFS; the HRES encoder was trained on ECMWF data, so its input is an approximation.
- Output fixes, applied and documented: precipitation and cloud cover in the two northernmost rows are copied from 89.5°N (the model blows up there), and humidity and cloud cover are clipped to physical ranges.
- Precipitation and max/min temperature are 6-hour windows (what WM-3 predicts); the WindBorne API uses 3-hour windows.
- A 14-day forecast takes ~36 s on an RTX 4090 vs 12 s in the paper. About half the gap is because every step is decoded and saved; the rest is unexplained (the released weights have 8 transformer blocks per processor, while the paper describes 10).

## Original WeatherMesh-3 code

`model.py` and `model_latlon/` are WindBorne's open-source architecture code. `python3 model.py` still loads and prints the model.
