"""
Real-time WM-3 pipeline: newest GFS cycle -> preprocess -> WM-3 -> validate -> NetCDF + PNG -> S3, with a public
status.json for monitoring and ntfy push alerts on failure.

Usage (from the repo root):
    python -m pipeline.run_cycle                    # run the newest available cycle once (skips if already done)
    python -m pipeline.run_cycle --init 2026102206  # run a specific cycle
    python -m pipeline.run_cycle --loop             # poll forever; this is what runs in production
"""
import argparse
import json
import os
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import boto3
import requests

from utils import levels_gfs
from pipeline import gfs, preprocess, infer, output, validate, plots

BUCKET = os.environ.get("S3_BUCKET", "windborne-wm3-vandana-131548386286-us-east-2-an")
PREFIX = os.environ.get("S3_PREFIX", "forecasts")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "wm3-vandana-alerts-4821")
WEIGHTS = os.environ.get("WEIGHTS_PATH", "WeatherMesh3.pt")
WORK_DIR = os.environ.get("WORK_DIR", "/workspace/runs")
MAX_LEAD = int(os.environ.get("MAX_LEAD_HOURS", "240"))
POLL_SECONDS = 600
MAX_PENDING = 6  # forecast steps allowed to wait for postprocessing at once
MAX_ATTEMPTS = 3
STALE_HOURS = 9  # alert if no successful cycle within this long (GFS cycles are 6h apart, ~4h publication delay)

LEADS = list(range(6, MAX_LEAD + 1, 6))
PLOT_LEADS = {24, 72, 120, 240}
GFS_COMPARE_LEADS = {6, 24, 120, 240}

s3 = boto3.client("s3")
_model = None

def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}", flush=True)

def alert(title, msg, priority="high"):
    log(f"ALERT {title}: {msg}")
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode()[:3500], headers={"Title": title, "Priority": priority}, timeout=15)
    except requests.RequestException as e:
        log(f"could not send alert: {e}")

def upload(local_path, key, content_type):
    s3.upload_file(local_path, BUCKET, f"{PREFIX}/{key}", ExtraArgs={"ContentType": content_type})

def put_json(key, obj):
    s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{key}", Body=json.dumps(obj, indent=2, default=str).encode(),
                  ContentType="application/json", CacheControl="no-cache")

def get_json(key, default):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{key}")["Body"].read())
    except s3.exceptions.NoSuchKey:
        return default

def get_model():
    global _model
    if _model is None:
        _model = infer.load_model(WEIGHTS)
    return _model

def run_cycle(init_time):
    """Runs one forecast cycle end to end and returns its manifest. Raises on any failure."""
    tag = f"{init_time:%Y%m%d%H}"
    run_dir = os.path.join(WORK_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)
    timings, steps = {}, {}
    t = time.time()

    pr, sfc = gfs.fetch_analysis(init_time, levels_gfs)
    timings["download_s"] = time.time() - t; t = time.time()

    model = get_model()
    gfs_mesh, hres_mesh = model.config.inputs
    out_mesh = model.config.outputs[0]
    inputs = preprocess.build_inputs(pr, sfc, gfs_mesh, hres_mesh, list(gfs.SURFACE_VARS))
    timings["preprocess_s"] = time.time() - t

    def postprocess(lead, y):
        fields = infer.to_physical(y, out_mesh)
        stats, failures = validate.check_step(fields)
        if lead in GFS_COMPARE_LEADS:
            stats["vs_gfs"] = validate.compare_with_gfs(fields, init_time, lead)
        nc = os.path.join(run_dir, f"wm3_{tag}_f{lead:03d}.nc")
        output.write_netcdf(nc, fields, init_time, lead)
        upload(nc, f"{tag}/{os.path.basename(nc)}", "application/x-netcdf")
        if lead in PLOT_LEADS:
            png = nc.replace(".nc", ".png")
            plots.plot_step(fields, png, f"WeatherMesh-3  init {init_time:%Y-%m-%d %HZ}  +{lead}h  (valid {init_time + timedelta(hours=lead):%Y-%m-%d %HZ})")
            upload(png, f"{tag}/{os.path.basename(png)}", "image/png")
        os.remove(nc)
        steps[lead] = {"stats": stats, "failures": failures}

    # The GPU produces lead times faster than they can be written, so postprocessing runs on a thread pool. Each
    # step is ~650MB in memory and the container is capped at ~83GB, so the GPU waits once MAX_PENDING are queued.
    t = time.time()
    slots = threading.BoundedSemaphore(MAX_PENDING)

    def postprocess_and_release(lead, y):
        try:
            postprocess(lead, y)
        finally:
            slots.release()

    def on_step(lead, y):
        slots.acquire()
        futures.append(pool.submit(postprocess_and_release, lead, y))

    with ThreadPoolExecutor(max_workers=MAX_PENDING) as pool:
        futures = []
        infer.run_forecast(model, inputs, int(init_time.timestamp()), LEADS, on_step)
        timings["inference_s"] = time.time() - t
        for f in futures:
            f.result()  # re-raises any postprocessing error
    timings["postprocess_and_upload_s"] = time.time() - t - timings["inference_s"]
    shutil.rmtree(run_dir, ignore_errors=True)

    failures = {lead: s["failures"] for lead, s in steps.items() if s["failures"]}
    manifest = {
        "init_time": init_time.isoformat(),
        "status": "ok" if not failures else "validation_failed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "lead_hours": LEADS,
        "files": [f"{PREFIX}/{tag}/wm3_{tag}_f{l:03d}.nc" for l in LEADS],
        "timings": {k: round(v, 1) for k, v in timings.items()},
        "validation_failures": failures,
        "steps": {str(k): steps[k]["stats"] for k in sorted(steps)},
    }
    put_json(f"{tag}/manifest.json", manifest)
    if failures:
        alert(f"WM-3 {tag}: validation failed", json.dumps(failures)[:1500])
    return manifest

def update_status(**changes):
    status = get_json("status.json", {"runs": []})
    status.update(changes)
    status["heartbeat"] = datetime.now(timezone.utc).isoformat()
    put_json("status.json", status)
    return status

def record_run(tag, result):
    status = get_json("status.json", {"runs": []})
    status["runs"] = ([result] + [r for r in status["runs"] if r["cycle"] != tag])[:40]
    if result["status"] == "ok":
        status["latest_successful_cycle"] = tag
    put_json("status.json", status | {"heartbeat": datetime.now(timezone.utc).isoformat()})

def process(init_time, attempts):
    tag = f"{init_time:%Y%m%d%H}"
    attempts[tag] = attempts.get(tag, 0) + 1
    log(f"running cycle {tag} (attempt {attempts[tag]})")
    update_status(state="running", current_cycle=tag)
    started = time.time()
    try:
        m = run_cycle(init_time)
        result = {"cycle": tag, "status": m["status"], "timings": m["timings"], "wall_s": round(time.time() - started, 1), "finished": m["completed_at"]}
        log(f"cycle {tag} {m['status']} {m['timings']}")
    except Exception as e:
        result = {"cycle": tag, "status": "error", "error": f"{type(e).__name__}: {e}", "attempt": attempts[tag], "finished": datetime.now(timezone.utc).isoformat()}
        alert(f"WM-3 {tag} failed (attempt {attempts[tag]}/{MAX_ATTEMPTS})", traceback.format_exc()[-1500:])
    record_run(tag, result)
    update_status(state="idle", current_cycle=None)
    return result["status"] != "error"

def loop():
    attempts = {}
    upload("monitoring/index.html", "index.html", "text/html")
    alert("WM-3 pipeline started", f"polling every {POLL_SECONDS}s", priority="default")
    while True:
        try:
            status = update_status(state="idle")
            done = {r["cycle"] for r in status["runs"] if r["status"] != "error"}
            # Newest cycle first; look back a day so a restart catches up on the latest missed cycle
            for back in range(0, 24, 6):
                init_time = gfs.latest_cycle() - timedelta(hours=back)
                tag = f"{init_time:%Y%m%d%H}"
                if tag in done:
                    break
                if attempts.get(tag, 0) >= MAX_ATTEMPTS or not gfs.is_available(init_time):
                    continue
                process(init_time, attempts)
                break

            last_ok = status.get("latest_successful_cycle")
            if last_ok:
                age_h = (datetime.now(timezone.utc) - datetime.strptime(last_ok, "%Y%m%d%H").replace(tzinfo=timezone.utc)).total_seconds() / 3600
                if age_h > STALE_HOURS + 6 and not status.get("stale_alerted") == last_ok:
                    alert("WM-3 pipeline stale", f"newest successful cycle {last_ok} is {age_h:.0f}h old")
                    update_status(stale_alerted=last_ok)
        except Exception:
            alert("WM-3 pipeline loop error", traceback.format_exc()[-1500:])
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--init", help="cycle to run, YYYYMMDDHH")
    args = ap.parse_args()
    if args.loop:
        loop()
    else:
        init = datetime.strptime(args.init, "%Y%m%d%H").replace(tzinfo=timezone.utc) if args.init else gfs.latest_cycle()
        while not args.init and not gfs.is_available(init):
            init -= timedelta(hours=6)
        process(init, {})
