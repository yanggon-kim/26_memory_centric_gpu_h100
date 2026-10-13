# H100 environment inventory — 2026-10-13 UTC

Classification: measured machine inventory. The initial machine-readable snapshot is `results/environment.json`; `results/environment_cu128.json` records the separately installed runtime check. Subsequent H100 experiments are indexed in `results/case_manifest.json`.

| Item | Observed value |
|---|---|
| Allocated physical GPU | NVIDIA H100 80GB HBM3, UUID `GPU-18558721-7ba2-20a7-f883-b2c4e2593aff`; user confirms exclusive allocation |
| HBM / compute capability | 81,559 MiB reported; sm_90 |
| Driver | 570.195.03; `nvidia-smi` reports CUDA 12.8 driver capability |
| Installed toolkit | Default `/usr/local/cuda-12.4`, `nvcc` 12.4.131; explicit `/usr/local/cuda-12.8` toolkit installed for native tracer build; default restored to 12.4 |
| Profilers | Nsight Compute 2024.1.1 available but metric collection returns `ERR_NVGPUCTRPERM`; `RmProfilingAdminOnly:1`, and container lacks `CAP_SYS_ADMIN`/`CAP_PERFMON`; `nsys` unavailable |
| Host | Linux 6.8.0-90-generic x86_64; Python 3.10.12 available |
| Storage | 25 GiB overlay filesystem, approximately 23 GiB free at inventory; no additional mounted workspace storage |
| GPU contention | No compute processes at initial snapshot; cooperating lock cannot exclude external processes later |
| Runtime at initial snapshot | Pinned torch/vLLM/transformers environment absent |
| Separate bridge runtime | Python 3.10 virtual environment with torch 2.11.0+cu128 and transformers 4.57.6; lock in `configs/runtime_cu128.lock.txt`; no vLLM |
| Selected CUDA identity | NVML/CUDA UUID crosscheck passed; 132 SMs; 50 MiB L2 (`52,428,800` bytes) queried from CUDA device properties; runtime 12.8 |

NVIDIA's CUDA compatibility documentation places ordinary CUDA 13.x minor compatibility at R580 or newer. The installed R570 driver and CUDA 12.4 toolkit do not satisfy the package's pinned CUDA 13 path. A separately recorded CUDA 12.8 PyTorch bridge route is being investigated without changing the driver or GPU settings. No result from that route may be called the pinned vLLM baseline. The hard storage limit also prevents assuming that a raw, grouped and archived large trace will fit.

The queried device is the 80GB HBM3 H100 with 132 SMs. Qualified internal board-energy sensor agreement was observed in one eager bridge configuration, but external absolute calibration remains absent. Profiler permissions prevent H100 hardware-counter bottleneck classification; model fit and correctness were tested only for the selected bridge shapes.
