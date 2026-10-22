# WeatherMesh-3: real-time forecasting pipeline

This repo takes WindBorne's open-source [WeatherMesh-3](https://github.com/windborne/WeatherMesh-3) model ([paper](https://arxiv.org/abs/2503.22235)) and turns it into something that runs on its own. Every 6 hours it grabs NOAA's newest GFS weather data, runs a 14-day global forecast on a GPU, checks that the result makes sense, and uploads it to S3.

You can see it running here: [status page](https://windborne-wm3-vandana-131548386286-us-east-2-an.s3.us-east-2.amazonaws.com/forecasts/index.html)

## What happens in each run

1. **Download.** When NOAA publishes a new GFS cycle (00, 06, 12 and 18 UTC), the pipeline downloads only the ~130 fields the model needs instead of the full 500 MB file. This takes about 13 seconds.
2. **Prepare the input.** The data is converted to the units and format WM-3 expects. I checked this step against the sample input WindBorne provided, and it matches exactly.
3. **Run the model.** WM-3 forecasts 14 days ahead in 6-hour steps. This takes about 36 seconds on an RTX 4090.
4. **Check the output.** Every step is checked for impossible values, and the forecast is compared with NOAA's own GFS forecast from the same starting point.
5. **Save and upload.** Each step is saved as a NetCDF file, using the same variable names, units and pressure levels as WindBorne's forecast API, and uploaded to S3 along with a few quick-look maps.

A full run takes about 7 minutes from start to finish.

### A note on the two encoders

WM-3 has two encoders, one built for GFS data and one for ECMWF's HRES data. ECMWF's free data doesn't include all the pressure levels the model needs, so both encoders get GFS data here. For the HRES encoder, I rearrange GFS onto HRES's 20 pressure levels. I also replace humidity high in the atmosphere (10–100 hPa), because GFS is much drier there than the ECMWF data that encoder was trained on. The details are in `pipeline/preprocess.py`.

## Setting it up

You need a Linux machine with an NVIDIA GPU (16 GB or more; the model uses about 8 GB) and an S3 bucket. The model weights (`WeatherMesh3.pt`) aren't in this repo. Put them in your bucket at `private/WeatherMesh3.pt`.

### With Docker

```bash
docker build -t wm3-pipeline .
docker run --gpus all \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_DEFAULT_REGION=us-east-2 \
  -e S3_BUCKET=your-bucket -e NTFY_TOPIC=your-topic \
  wm3-pipeline
```

The container downloads the weights from your bucket and starts the pipeline.

I tested this two ways. The image builds cleanly, and everything imports inside the container (I built it on my laptop, which has no suitable GPU, so it couldn't run a forecast there). The live pipeline runs on a RunPod machine, which can't build Docker images itself, so I also ran the Dockerfile's install steps in a fresh environment on that machine. That worked, and the model produced a valid forecast.

### Without Docker (how the live pipeline runs)

```bash
git clone https://github.com/Vandanareddy2/wm3-inference.git && cd wm3-inference
python3 -m venv ../venv && . ../venv/bin/activate

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# natten only has prebuilt versions for torch 2.4 + CUDA 12.1. Its download site had an expired
# certificate when I set this up, hence the -k. Use the file matching your Python version (cp310/311/312).
curl -sk -O "https://shi-labs.com/natten/wheels/cu121/torch2.4.0/natten-0.17.3%2Btorch240cu121-cp312-cp312-linux_x86_64.whl"
pip install ./natten-0.17.3+torch240cu121-cp312-cp312-linux_x86_64.whl

pip install --index-url https://test.pypi.org/simple/ --no-deps matepoint
pip install numpy==2.2.2 boto3 xarray netCDF4 eccodes requests matplotlib
```

Then set up your AWS credentials and download the weights into the repo folder:

```bash
python -c "import boto3; boto3.client('s3').download_file('your-bucket', 'private/WeatherMesh3.pt', 'WeatherMesh3.pt')"
```

Always run things from the repo folder, because the model code looks for `constants/` there.

## Running it

```bash
python -m pipeline.run_cycle                    # run the newest GFS cycle once
python -m pipeline.run_cycle --init 2026102212  # run a specific cycle
nohup ./run.sh > pipeline.log 2>&1 &            # keep it running (checks every 10 minutes, restarts if it crashes)
python -m pipeline.benchmark                    # time the model
```

Settings come from environment variables: `S3_BUCKET`, `NTFY_TOPIC` (for phone alerts), `MAX_LEAD_HOURS` (default 336, i.e. 14 days), `WEIGHTS_PATH` and `WORK_DIR`.

## Where the forecasts go

Each run gets its own folder in the bucket, named after its start time:

```
forecasts/2026102218/
    wm3_2026102218_f006.nc    6 hours ahead
    ...
    wm3_2026102218_f336.nc    14 days ahead
    wm3_2026102218_f120.png   quick-look map (also for 24h, 72h, 240h, 336h)
    manifest.json             file list, timings and check results
```

The first two runs (22 Oct 06Z and 12Z) were made before I switched to 14 days, so they stop at `f240`.

## Knowing when something breaks

- If a run fails (download, model, checks or upload), my phone gets an alert and the run is retried up to 3 times.
- If the pipeline process crashes, `run.sh` restarts it and sends an alert.
- If the whole machine dies, it can't alert anyone. So a small GitHub Actions job checks the pipeline's heartbeat every 30 minutes from outside, and alerts if it's gone quiet.
- After each successful run, I get a silent "done" notification.
- NOAA publishes the later GFS forecast hours after our run has finished, so the 5-, 10- and 14-day comparisons are filled in later, once that data is out.

## Things to know

- The HRES encoder gets GFS data instead of the ECMWF data it was trained on, so its input is an approximation.
- The model produces bad rain and cloud values in the two northernmost rows of the map, so I copy those rows from the row next to them. Humidity and cloud cover are also clipped to sensible ranges.
- Rain and max/min temperature cover 6-hour windows, because that's what WM-3 predicts. WindBorne's API uses 3-hour windows.
- The paper says a 14-day forecast takes 12 seconds on an RTX 4090. Mine takes about 36 seconds. Roughly half of the difference is because I save every 6-hour step; I couldn't pin down the rest.

## Files

- `pipeline/`: everything I added (download, preprocessing, model run, output, checks, the main loop, benchmark)
- `run.sh`: keeps the pipeline running
- `monitoring/index.html`: the status page
- `.github/workflows/watchdog.yml`: the outside watchdog
- `model.py`, `model_latlon/`, `constants/`: WindBorne's original model code (unchanged, except for one small humidity file I added to `constants/`)
