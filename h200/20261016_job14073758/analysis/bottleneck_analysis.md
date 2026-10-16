# H200 decode bottleneck analysis

## Measured kernel evidence

Counters below are bounded kernel replays of explicitly identified programs. Percentages use the exact hardware metric denominators in the machine-readable data. These samples cannot supply full-step DRAM totals or a removable compute fraction.

| Model and shape | Layer | Work | DRAM read KiB | DRAM sustained % | L2 sector hit % | Active occupancy % |
|---|---:|---|---:|---:|---:|---:|
| bridge B1/C1024 | 0 | kv_update | 23.50 | 0.15 | NA | 9.78 |
| bridge B1/C1024 | 0 | attention_forward | 1109.25 | 1.89 | 59.27 | 11.32 |
| bridge B1/C1024 | 0 | attention_combine | 26.25 | 0.10 | NA | 11.51 |
| bridge B1/C1024 | 14 | kv_update | 23.50 | 0.15 | NA | 9.90 |
| bridge B1/C1024 | 14 | attention_forward | 1109.00 | 1.89 | 59.04 | 11.46 |
| bridge B1/C1024 | 14 | attention_combine | 23.25 | 0.08 | NA | 11.64 |
| bridge B1/C1024 | 27 | kv_update | 24.75 | 0.25 | NA | 7.58 |
| bridge B1/C1024 | 27 | attention_forward | 1109.25 | 1.91 | 59.80 | 11.50 |
| bridge B1/C1024 | 27 | attention_combine | 25.75 | 0.09 | NA | 11.65 |
| bridge B8/C4096 | 0 | linear | 6178.00 | 17.97 | 10.11 | 14.24 |
| bridge B8/C4096 | 0 | kv_update | 23.50 | 0.14 | NA | 10.01 |
| bridge B8/C4096 | 0 | attention_forward | 32870.50 | 37.13 | 6.23 | 12.33 |
| bridge B8/C4096 | 0 | attention_combine | 354.00 | 1.20 | 95.54 | 11.98 |
| bridge B8/C4096 | 0 | linear | 4632.75 | 15.02 | 10.82 | 14.49 |
| bridge B8/C4096 | 0 | linear | 53806.00 | 64.33 | 3.36 | 14.22 |
| bridge B8/C4096 | 0 | linear | 27038.25 | 47.34 | 5.63 | 14.43 |
| bridge B8/C4096 | 14 | linear | 6174.00 | 18.30 | 10.54 | 14.16 |
| bridge B8/C4096 | 14 | kv_update | 23.50 | 0.14 | NA | 9.86 |
| bridge B8/C4096 | 14 | attention_forward | 32873.00 | 38.14 | 6.16 | 12.33 |
| bridge B8/C4096 | 14 | attention_combine | 353.50 | 1.16 | 95.15 | 12.09 |
| bridge B8/C4096 | 14 | linear | 4632.25 | 14.62 | 11.87 | 14.53 |
| bridge B8/C4096 | 14 | linear | 53802.50 | 64.37 | 3.18 | 14.24 |
| bridge B8/C4096 | 14 | linear | 27038.50 | 45.90 | 6.10 | 14.49 |
| bridge B8/C4096 | 27 | linear | 6184.50 | 17.31 | 10.79 | 14.02 |
| bridge B8/C4096 | 27 | kv_update | 32.00 | 0.27 | NA | 8.96 |
| bridge B8/C4096 | 27 | attention_forward | 32873.00 | 38.60 | 6.26 | 12.33 |
| bridge B8/C4096 | 27 | attention_combine | 353.75 | 1.19 | 95.35 | 12.00 |
| bridge B8/C4096 | 27 | linear | 4634.25 | 14.63 | 11.67 | 14.58 |
| bridge B8/C4096 | 27 | linear | 53801.25 | 60.55 | 3.12 | 14.09 |
| bridge B8/C4096 | 27 | linear | 27038.25 | 48.83 | 5.54 | 14.47 |
| bridge B1/C1024 | None | attention_scheduler | 7.25 | 0.04 | NA | 1.56 |
| bridge B1/C1024 | None | kv_update | 24.75 | 0.13 | NA | 7.03 |
| bridge B1/C1024 | None | attention_forward | 1107.25 | 2.07 | 63.34 | 11.53 |
| bridge B1/C1024 | None | attention_scheduler | 7.50 | 0.04 | NA | 1.56 |
| bridge B1/C1024 | None | kv_update | 25.00 | 0.11 | NA | 7.98 |
| bridge B1/C1024 | None | attention_forward | 1109.75 | 2.08 | 36.14 | 11.51 |
| bridge B1/C1024 | None | linear | 6161.25 | 18.84 | 9.68 | 14.33 |
| bridge B1/C1024 | None | linear | 4625.25 | 13.01 | 49.58 | 14.87 |
| bridge B1/C1024 | None | linear | 53782.50 | 63.52 | 3.04 | 14.17 |
| bridge B1/C1024 | None | linear | 26916.00 | 45.59 | 5.01 | 14.54 |
| bridge B1/C1024 | None | linear | 6168.50 | 18.60 | 6.55 | 14.13 |
| bridge B1/C1024 | None | linear | 4632.00 | 12.79 | 43.52 | 14.84 |
| bridge B1/C1024 | None | linear | 53784.00 | 64.14 | 3.87 | 14.23 |
| bridge B1/C1024 | None | linear | 26914.50 | 47.24 | 4.50 | 14.39 |
| bridge B8/C4096 | None | attention_scheduler | 7.25 | 0.04 | NA | 1.56 |
| bridge B8/C4096 | None | kv_update | 31.75 | 0.16 | NA | 6.75 |
| bridge B8/C4096 | None | attention_forward | 32870.00 | 38.46 | 4.71 | 12.34 |
| bridge B8/C4096 | None | linear | 6161.25 | 18.46 | 10.37 | 14.45 |
| bridge B8/C4096 | None | linear | 4625.25 | 14.98 | 11.84 | 14.49 |
| bridge B8/C4096 | None | linear | 53783.00 | 60.55 | 3.34 | 14.16 |
| bridge B8/C4096 | None | linear | 27038.50 | 48.15 | 5.40 | 14.40 |
| mistral7b B1/C1024 | 16 | linear | 49179.75 | 58.31 | 3.57 | 14.33 |
| mistral7b B1/C1024 | 16 | kv_update | 23.50 | 0.14 | NA | 19.36 |
| mistral7b B1/C1024 | 16 | attention_forward | 4185.75 | 6.94 | 22.03 | 11.96 |
| mistral7b B1/C1024 | 16 | attention_combine | 26.75 | 0.11 | NA | 12.21 |
| mistral7b B1/C1024 | 16 | linear | 32785.75 | 55.61 | 9.03 | 14.58 |
| mistral7b B1/C1024 | 16 | linear | 229416.75 | 82.92 | 3.27 | 14.49 |
| mistral7b B1/C1024 | 16 | linear | 114722.25 | 76.66 | 3.54 | 14.24 |
| mistral7b B1/C1024 | None | linear | 49171.75 | 58.28 | 3.35 | 14.22 |
| mistral7b B1/C1024 | None | linear | 32796.50 | 53.55 | 7.67 | 14.48 |
| mistral7b B1/C1024 | None | linear | 229409.75 | 82.79 | 3.14 | 14.47 |
| mistral7b B1/C1024 | None | linear | 114749.75 | 75.51 | 3.41 | 14.23 |

## Interpretation

FlashAttention 3, paged BF16 KV, and NVJET Hopper linear kernels are observed in the bridge. They differ from the historical RTX GEMV inventory. Low tensor activity alone cannot establish removable arithmetic area.

Short-context attention samples with low achieved DRAM percentage and low occupancy do not establish HBM saturation. Bandwidth, latency/MLP, launch/issue, and resource limitations must be separated by kernel and workload. Linears require their own sampled counters.

A raw L2 sector-hit percentage outside 0–100 is excluded from interpretation and retained with an invalid-value reason. Uncontrolled-cache multi-pass replay can produce inconsistent derived values; [NVIDIA documents this limitation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html). Valid-looking ratios remain replay-scoped. Separately named cold-cache controls characterize this measurement sensitivity; they are not ordinary application cache behavior.

The Nsight Systems inventories retain the sum of instrumented kernel durations and their shares. These are GPU busy-time shares, not synchronized wall-time Amdahl fractions. Graph/eager full-step comparisons use separate uninstrumented runs.

Full-step counter coverage, complete tensor-specific traffic attribution, productive/empty attention-work counts, and simulator timing/power calibration retain explicit pending gates. Dynamic split values are interpreted only after native fixed-split schema probes pass.

## Kernel resources

| Model and shape | Engine | Cache control | Work | Grid | Registers/thread | Dynamic shared KiB/block | Local-load sectors |
|---|---|---|---|---|---:|---:|---:|
| bridge B1/C1024 | vllm_eager | none | kv_update | (1, 1, 1) | 56.00 | 0.00 | 0.00 |
| bridge B1/C1024 | vllm_eager | none | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| bridge B1/C1024 | vllm_eager | none | attention_combine | (2, 1, 1) | 46.00 | 17.12 | 0.00 |
| bridge B1/C1024 | vllm_eager | none | kv_update | (1, 1, 1) | 56.00 | 0.00 | 0.00 |
| bridge B1/C1024 | vllm_eager | none | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| bridge B1/C1024 | vllm_eager | none | attention_combine | (2, 1, 1) | 46.00 | 17.12 | 0.00 |
| bridge B1/C1024 | vllm_eager | none | kv_update | (1, 1, 1) | 56.00 | 0.00 | 0.00 |
| bridge B1/C1024 | vllm_eager | none | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| bridge B1/C1024 | vllm_eager | none | attention_combine | (2, 1, 1) | 46.00 | 17.12 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 8, 1) | 168.00 | 160.46 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | kv_update | (8, 1, 1) | 56.00 | 0.00 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | attention_combine | (20, 1, 1) | 46.00 | 17.12 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 6, 1) | 168.00 | 160.46 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 24, 1) | 168.00 | 216.27 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 24, 1) | 168.00 | 160.44 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 8, 1) | 168.00 | 160.46 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | kv_update | (8, 1, 1) | 56.00 | 0.00 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | attention_combine | (20, 1, 1) | 46.00 | 17.12 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 6, 1) | 168.00 | 160.46 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 24, 1) | 168.00 | 216.27 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 24, 1) | 168.00 | 160.44 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 8, 1) | 168.00 | 160.46 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | kv_update | (8, 1, 1) | 56.00 | 0.00 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | attention_combine | (20, 1, 1) | 46.00 | 17.12 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 6, 1) | 168.00 | 160.46 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 24, 1) | 168.00 | 216.27 | 0.00 |
| bridge B8/C4096 | vllm_eager | none | linear | (4, 24, 1) | 168.00 | 160.44 | 0.00 |
| bridge B1/C1024 | vllm_graph | none | attention_scheduler | (1, 1, 1) | 16.00 | 0.00 | 0.00 |
| bridge B1/C1024 | vllm_graph | none | kv_update | (1, 1, 1) | 56.00 | 0.00 | 0.00 |
| bridge B1/C1024 | vllm_graph | none | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| bridge B1/C1024 | vllm_graph | all | attention_scheduler | (1, 1, 1) | 16.00 | 0.00 | 0.00 |
| bridge B1/C1024 | vllm_graph | all | kv_update | (1, 1, 1) | 56.00 | 0.00 | 0.00 |
| bridge B1/C1024 | vllm_graph | all | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| bridge B1/C1024 | vllm_graph | none | linear | (4, 8, 1) | 168.00 | 160.46 | 0.00 |
| bridge B1/C1024 | vllm_graph | none | linear | (4, 24, 1) | 168.00 | 176.46 | 0.00 |
| bridge B1/C1024 | vllm_graph | none | linear | (4, 24, 1) | 168.00 | 216.27 | 0.00 |
| bridge B1/C1024 | vllm_graph | none | linear | (4, 24, 1) | 168.00 | 160.44 | 0.00 |
| bridge B1/C1024 | vllm_graph | all | linear | (4, 8, 1) | 168.00 | 160.46 | 0.00 |
| bridge B1/C1024 | vllm_graph | all | linear | (4, 24, 1) | 168.00 | 176.46 | 0.00 |
| bridge B1/C1024 | vllm_graph | all | linear | (4, 24, 1) | 168.00 | 216.27 | 0.00 |
| bridge B1/C1024 | vllm_graph | all | linear | (4, 24, 1) | 168.00 | 160.44 | 0.00 |
| bridge B8/C4096 | vllm_graph | none | attention_scheduler | (1, 1, 1) | 16.00 | 0.00 | 0.00 |
| bridge B8/C4096 | vllm_graph | none | kv_update | (8, 1, 1) | 56.00 | 0.00 | 0.00 |
| bridge B8/C4096 | vllm_graph | none | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| bridge B8/C4096 | vllm_graph | none | linear | (4, 8, 1) | 168.00 | 160.46 | 0.00 |
| bridge B8/C4096 | vllm_graph | none | linear | (4, 6, 1) | 168.00 | 160.46 | 0.00 |
| bridge B8/C4096 | vllm_graph | none | linear | (4, 24, 1) | 168.00 | 216.27 | 0.00 |
| bridge B8/C4096 | vllm_graph | none | linear | (4, 24, 1) | 168.00 | 160.44 | 0.00 |
| mistral7b B1/C1024 | vllm_eager | none | linear | (4, 24, 1) | 168.00 | 160.46 | 0.00 |
| mistral7b B1/C1024 | vllm_eager | none | kv_update | (1, 1, 1) | 56.00 | 0.00 | 0.00 |
| mistral7b B1/C1024 | vllm_eager | none | attention_forward | (132, 1, 1) | 255.00 | 147.00 | 0.00 |
| mistral7b B1/C1024 | vllm_eager | none | attention_combine | (4, 1, 1) | 46.00 | 17.12 | 0.00 |
| mistral7b B1/C1024 | vllm_eager | none | linear | (2, 64, 1) | 168.00 | 160.44 | 0.00 |
| mistral7b B1/C1024 | vllm_eager | none | linear | (2, 66, 1) | 168.00 | 220.36 | 0.00 |
| mistral7b B1/C1024 | vllm_eager | none | linear | (2, 64, 1) | 168.00 | 220.35 | 0.00 |
| mistral7b B1/C1024 | vllm_graph | none | linear | (4, 24, 1) | 168.00 | 160.46 | 0.00 |
| mistral7b B1/C1024 | vllm_graph | none | linear | (2, 64, 1) | 168.00 | 160.44 | 0.00 |
| mistral7b B1/C1024 | vllm_graph | none | linear | (2, 66, 1) | 168.00 | 220.36 | 0.00 |
| mistral7b B1/C1024 | vllm_graph | none | linear | (2, 64, 1) | 168.00 | 220.35 | 0.00 |

Allocated registers/shared memory and the observed grid jointly constrain concurrency. A high register count is not evidence of spilling; actual local-memory traffic is retained. More storage does not automatically increase productive work in an unchanged underfilled or persistent grid.

## Instrumented decode windows

| Timeline | Decode wall ms | Kernel union ms | CUDA kernels |
|---|---:|---:|---:|
| bridge_profile_b1_c1024_eager | 39.186 | 5.876 | 1065 |
| bridge_profile_b8_c4096_eager | 41.020 | 6.447 | 1062 |
| bridge_profile_graph_b1_c1024 | 12.362 | 5.585 | 1062 |
| bridge_profile_graph_b8_c4096 | 14.540 | 6.287 | 1059 |
| mistral7b_profile_b1_c1024_eager | 44.148 | 13.980 | 1305 |
| mistral7b_profile_graph_b1_c1024 | 21.355 | 13.921 | 1398 |

These windows include profiling overhead. Kernel unions exclude overlapping kernel time but also exclude GPU copies. The difference from wall time cannot be assigned entirely to CPU scheduling, nor transferred as an uninstrumented Amdahl fraction.
