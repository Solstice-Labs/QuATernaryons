import subprocess
import time
import os
import json
import re

kernel_id = "solsticeaiofficial/quaternaryons-spark-benchmark"
out_dir = "/Users/shreyangondaliya/Project Indexes/QuATernaryons/results_spark"
os.makedirs(out_dir, exist_ok=True)

print(f"Starting lightweight monitoring daemon for {kernel_id}...", flush=True)

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

start_time = time.time()
last_step_logged = ""

while True:
    try:
        status_res = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
        status_line = status_res.stdout.strip()
        elapsed = int(time.time() - start_time)

        # Pull recent logs
        try:
            raw_log = api.kernels_logs(kernel_id)
            if isinstance(raw_log, str):
                # Search for latest Step log
                matches = re.findall(r"Step\s+(\d+/\d+)\s+\|\s+Forward KL\s+=\s+([\d\.]+)\s+nats/token", raw_log)
                if matches:
                    latest = f"Step {matches[-1][0]} | KL = {matches[-1][1]} nats/token"
                    if latest != last_step_logged:
                        print(f"[{elapsed}s] [PROGRESS] {latest}", flush=True)
                        last_step_logged = latest
        except Exception:
            pass

        print(f"[{elapsed}s] {status_line}", flush=True)

        if "complete" in status_line.lower():
            print(f"Kernel completed! Fetching log & results to {out_dir}...", flush=True)
            # Fetch log directly
            raw_log = api.kernels_logs(kernel_id)
            with open(os.path.join(out_dir, "quaternaryons-spark-benchmark.log"), "w") as f:
                f.write(raw_log)
            # Fetch files
            subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", out_dir])
            break
        elif "error" in status_line.lower() or "cancel" in status_line.lower():
            print(f"Kernel terminated with status: {status_line}. Fetching log...", flush=True)
            raw_log = api.kernels_logs(kernel_id)
            with open(os.path.join(out_dir, "quaternaryons-spark-benchmark.log"), "w") as f:
                f.write(raw_log)
            break

    except Exception as e:
        print(f"Monitor exception: {e}", flush=True)

    time.sleep(30)
