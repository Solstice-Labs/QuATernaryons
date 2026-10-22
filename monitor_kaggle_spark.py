import subprocess
import time
import os
import json

kernel_id = "solsticeaiofficial/quaternaryons-spark-benchmark"
out_dir = "/Users/shreyangondaliya/Project Indexes/QuATernaryons/results_spark"
os.makedirs(out_dir, exist_ok=True)

print(f"Starting monitoring daemon for {kernel_id}...", flush=True)

start_time = time.time()
while True:
    res = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
    status_line = res.stdout.strip()
    elapsed = int(time.time() - start_time)
    print(f"[{elapsed}s] {status_line}", flush=True)

    if "complete" in status_line.lower():
        print(f"Kernel completed! Downloading outputs to {out_dir}...", flush=True)
        subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", out_dir])
        break
    elif "error" in status_line.lower() or "cancel" in status_line.lower():
        print(f"Kernel failed/canceled! Fetching logs...", flush=True)
        subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", out_dir])
        break

    time.sleep(30)
