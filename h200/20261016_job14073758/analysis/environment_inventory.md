# H200 environment inventory

The allocated H200 passed CUDA execution, hardware-counter collection, pinned runtime imports, native tracing, postprocessing, and one-kernel sm_90 replay. These checks establish Phase 0 tool feasibility. H200 workload timing, Hopper async semantics, and simulator power remain separate calibration gates.

| Resource | Measured or recorded value |
|---|---|
| Node | atl1-1-03-020-18-0.pace.gatech.edu |
| Allocation | Slurm 14073758; gpu-h200 partition; embers QOS |
| Allocated host resources | 8 CPU cores; 196 GiB RAM |
| Physical GPU | NVIDIA H200; GPU-a613c806-89b1-05d1-f71c-3d52ab46d246 |
| CUDA architecture | 9.0; 132 SMs |
| Queried L2 | 62,914,560 bytes, 60 MiB |
| Shared memory | 233,472 bytes per SM; 232,448 bytes maximum opt-in per block |
| Registers | 65,536 32-bit registers per SM |
| GPU memory | nvidia-smi reports 143,771 MiB total; CUDA reports 150,156,607,488 usable bytes |
| Driver | 615.71.09 |
| Selected toolkit | CUDA 13.0.1, nvcc 13.0.88; explicit installed path in runtime_environment.json |
| Runtime | Python 3.10.13; torch 2.11.0+cu130; vLLM 0.22.1; transformers 4.57.6 |
| Initial device state | No compute processes; MIG disabled; 700 W configured limit, unchanged |
| Profiling permission | RmProfilingAdminOnly=0; actual DRAM read/write micro counters collected |
| Energy sensors | Cumulative counter and board power accessible; decode qualification performed separately |
| Scratch quota | 15 TiB byte quota; approximately 25,800 remaining file entries at initial inspection |
| Runtime and build placement | Owned /tmp/h200_research_jkim4112_job14073758 directory |
| Scheduled job end | October 16 2026, 15:33:58 EDT; scheduler reports preemption eligibility |

The native dynamic-input micro captured 640 warp records, grouped one actual sm_90 kernel, and replayed with 4,121 cycles and 19,456 active-thread instructions. Replay used the pinned stock H100 configuration as a parser smoke. Its memory/cache configuration does not establish H200 timing accuracy.

Raw inventory, CUDA properties, runtime freeze, CPU contracts, micro counter logs, native export manifest, and replay log are retained in `../environment/`. Node-wide RAM and disk availability are not treated as the job allocation or user quota.
