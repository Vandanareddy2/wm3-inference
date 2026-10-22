#!/bin/bash
# Supervisor for the real-time pipeline: restarts the polling loop if the process dies (segfault, OOM kill, ...)
# and sends an alert each time, since a hard crash can't report itself.
cd "$(dirname "$0")"
NTFY_TOPIC="${NTFY_TOPIC:-wm3-vandana-alerts-4821}"
while true; do
    python -m pipeline.run_cycle --loop
    code=$?
    curl -s -H "Title: WM-3 pipeline process died" -H "Priority: high" \
        -d "exit code $code on $(hostname); restarting in 60s" "https://ntfy.sh/$NTFY_TOPIC" > /dev/null
    sleep 60
done
