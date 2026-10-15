"""
make_data.py
------------
Generates synthetic data mirroring the schema of the Alibaba Cluster Trace
dataset (cluster-trace-v2017 / v2018), specifically the machine_usage table.

Official schema reference:
  https://github.com/alibaba/clusterdata

Columns match machine_usage.csv:
  machine_id   - machine identifier
  time_stamp   - seconds from trace start (60s intervals)
  cpu_util     - CPU utilization % (0-100, normalized)
  mem_util     - Memory utilization % (0-100, normalized)
  net_in       - Normalized inbound network (0-100)
  net_out      - Normalized outbound network (0-100)
  disk_io      - Disk I/O utilization % (0-100)

IMPORTANT DISCLOSURE:
  The Alibaba Cluster Trace (https://github.com/alibaba/clusterdata) requires
  a registration/request process to download. The full trace is also very large
  (117 GB+ uncompressed for the Azure trace; Alibaba trace is similarly large).
  This script generates synthetic data that mirrors the Alibaba schema exactly,
  using realistic sinusoidal patterns + Gaussian noise to approximate real
  production cluster behaviour. This must be disclosed in the project report.

  Dataset source: Synthetic, schema-compatible with Alibaba Cluster Trace 2017.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Simulate 5 machines across 600 time steps (10 hours @ 60s intervals)
N_MACHINES = 5
N_STEPS    = 600          # 600 * 60s = 10 hours per machine -> 3000 rows total
INTERVAL   = 60           # seconds

machines = [f"m_{1000 + i}" for i in range(N_MACHINES)]
rows = []

for idx, machine_id in enumerate(machines):
    t = np.arange(N_STEPS)
    offset = idx * 7  # phase offset per machine

    # CPU: base load varies per machine, with daily-ish oscillation
    base_cpu = 40 + idx * 5
    cpu = np.clip(
        base_cpu + 20 * np.sin((t + offset) / 50)
        + 5 * np.sin((t + offset) / 20)
        + np.random.normal(0, 4, N_STEPS),
        1, 99
    )

    # Memory: slower drift, generally higher than CPU
    base_mem = 55 + idx * 3
    mem = np.clip(
        base_mem + 8 * np.sin((t + offset) / 80)
        + np.random.normal(0, 3, N_STEPS),
        1, 99
    )

    # Network in/out: faster oscillation (burst patterns)
    net_in = np.clip(
        30 + 15 * np.sin((t + offset) / 30)
        + np.random.normal(0, 5, N_STEPS),
        0, 99
    )
    net_out = np.clip(
        25 + 12 * np.sin((t + offset + 5) / 30)
        + np.random.normal(0, 4, N_STEPS),
        0, 99
    )

    # Disk I/O: generally low with occasional spikes
    disk_io = np.clip(
        15 + 8 * np.sin((t + offset) / 100)
        + np.abs(np.random.normal(0, 6, N_STEPS)),
        0, 80
    )

    for i in range(N_STEPS):
        rows.append({
            "machine_id": machine_id,
            "time_stamp": int(i * INTERVAL),
            "cpu_util":   round(cpu[i], 2),
            "mem_util":   round(mem[i], 2),
            "net_in":     round(net_in[i], 2),
            "net_out":    round(net_out[i], 2),
            "disk_io":    round(disk_io[i], 2),
        })

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)
df.to_csv("data/cloud_data.csv", index=False)

print("=" * 55)
print("  Alibaba-Schema Synthetic Dataset Generated")
print("=" * 55)
print(f"  Rows      : {len(df)}")
print(f"  Machines  : {N_MACHINES}")
print(f"  Time steps: {N_STEPS} per machine ({N_STEPS * INTERVAL // 3600}h window)")
print(f"  Saved to  : data/cloud_data.csv")
print()
print(df[["cpu_util", "mem_util", "net_in", "net_out", "disk_io"]].describe().round(2))
print()
print("NOTE: Synthetic data - schema mirrors Alibaba Cluster Trace 2017.")
print("      Disclose this in the project report.")
