# WeatherMesh-3 Real-Time Inference

A real-time forecasting pipeline for WindBorne's [WeatherMesh-3](https://github.com/windborne/WeatherMesh-3) ([paper](https://arxiv.org/abs/2503.22235)).

Every 6 hours, the pipeline pulls the latest NOAA GFS analysis, runs a 14-day global forecast on a single GPU, validates the output, and publishes it to S3.

**Live status:** [status page](https://windborne-wm3-vandana-131548386286-us-east-2-an.s3.us-east-2.amazonaws.com/forecasts/index.html)

## How it works

| Step | What happens | Time (RTX 4090) |
|---|---|---|
| Download | Fetch only the ~130 GFS fields the model needs | ~13 s |
| Preprocess | Convert units, normalize, map GFS onto both encoders' levels | ~8 s |
| Inference | 56 six-hour steps (14 days) | ~36 s |
| Validate | Physical-range checks and comparison with GFS's own forecast | per step |
| Publish | NetCDF + quick-look maps to S3, status update, alerts | ~6 min |

WM-3 has two encoders, one for GFS and one for ECMWF HRES. Since ECMWF's open data doesn't include every required level, both are fed from GFS. The HRES encoder gets GFS remapped to HRES's 20 levels, with upper-air humidity adjusted to match what that encoder saw in training. See `pipeline/preprocess.py`.

## Installation

### Docker

```
docker build -t wm3-pipeline .
docker run --gpus all \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_DEFAULT_REGION=us-east-2 \
  -e S3_BUCKET=your-bucket -e NTFY_TOPIC=your-topic \
  wm3-pipeline
```

The container downloads the weights from `s3://$S3_BUCKET/private/WeatherMesh3.pt` and starts the pipeline. The image builds and imports cleanly, and the same install steps were run end to end on the GPU machine.

### Manual

Requires Linux, an NVIDIA GPU with 16 GB+ of memory (~8 GB used), and Python 3.10–3.12.

```
python3 -m venv venv && . venv/bin/activate
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
curl -sk -O "https://shi-labs.com/natten/wheels/cu121/torch2.4.0/natten-0.17.3%2Btorch240cu121-cp312-cp312-linux_x86_64.whl"
pip install ./natten-0.17.3+torch240cu121-cp312-cp312-linux_x86_64.whl
pip install --index-url https://test.pypi.org/simple/ --no-deps matepoint
pip install numpy==2.2.2 boto3 xarray netCDF4 eccodes requests matplotlib
```

natten only ships wheels for torch 2.4 + CUDA 12.1; pick the one matching your Python version. Its server's certificate was expired at the time of writing, hence `-k`.

Place `WeatherMesh3.pt` (provided by WindBorne, not in this repo) in the repo root, and run everything from there.

## Usage

```
python -m pipeline.run_cycle                    # run the latest GFS cycle once
python -m pipeline.run_cycle --init 2026102212  # run a specific cycle
./run.sh                                        # run continuously, restarting on failure
python -m pipeline.benchmark                    # benchmark inference
```

Configuration is through environment variables: `S3_BUCKET`, `NTFY_TOPIC`, `MAX_LEAD_HOURS` (default `336`), `WEIGHTS_PATH`, `WORK_DIR`.

## Outputs

Each cycle is written to `forecasts/{YYYYMMDDHH}/`:

```
wm3_{YYYYMMDDHH}_f006.nc ... wm3_{YYYYMMDDHH}_f336.nc   one file per 6-hour step
wm3_{YYYYMMDDHH}_f{024,072,120,240,336}.png             quick-look maps
manifest.json                                            files, timings, validation results
```

Files follow WindBorne's gridded-forecast API: same variable names and units (`temperature_2m`, `pressure_msl`, `geopotential`, …), the API's 25 pressure levels, and `model` / `initialization_time` / `forecast_hour` attributes. Runs before 2026-10-22 18Z use a 10-day horizon.

## Monitoring

- **Failure alerts:** push notification (ntfy) on any failed or invalid cycle, with up to 3 retries.
- **Crash recovery:** `run.sh` restarts the pipeline and alerts.
- **External watchdog:** a GitHub Actions job checks the heartbeat every 30 minutes, so a dead machine still alerts.
- **Status:** `forecasts/status.json` and the status page show recent cycles, timings and validation.

## Limitations

- The HRES encoder receives GFS rather than ECMWF data, so its input is an approximation.
- Precipitation and cloud cover in the two northernmost grid rows are replaced with the adjacent row's values, and humidity and cloud cover are clipped to physical ranges.
- Precipitation and min/max temperature are 6-hour totals, as predicted by WM-3; the API uses 3-hour windows.
- Inference takes ~36 s for 14 days versus the paper's 12 s. About half of this is from decoding every step for output.

## Repository layout

```
pipeline/                 download, preprocessing, inference, output, validation, main loop
run.sh                    supervisor
monitoring/index.html     status page
.github/workflows/        external watchdog
model.py, model_latlon/   original WeatherMesh-3 architecture (unchanged)
```
