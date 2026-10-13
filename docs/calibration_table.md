# H100 calibration and evidence table — 2026-10-13

| Quantity or route | Current evidence | Calibration status and consequence |
|---|---|---|
| GPU identity and L2 | CUDA/NVML selected UUID agree; sm_90, 132 SMs, queried 50 MiB L2 | Measured environment fact; no multi-GPU data. |
| Compiled StaticCache B1 bridge | BF16 model.generate/eager/compiled IDs pass B1/C128 and B1/C1024; three warmups/trials | Valid measured software control for those shapes only. Torch compile B8/C4096 fails the strict output gate. |
| Optimized vLLM baseline | Original pinned CUDA 13 route incompatible with R570; no validated cu128 optimized vLLM route | Missing; cannot compare optimized Hopper kernels or claim baseline speedup. |
| Nsight H100 traffic and utilization | Nsight Systems 2025.6.3 CUDA timeline works on a BF16 control; Nsight Compute and Nsight Systems GPU metrics return `ERR_NVGPUCTRPERM` | HBM bytes, L2 hit rate, tensor/LSU/issue utilization NA; no quantitative bottleneck label. |
| Roofline controls | BF16 GEMM 718.40 TFLOP/s; device copy 3032.09 useful GB/s; three bridge timing/analytical proxy points | Measured control ceilings plus analytical decode intensity only; no actual HBM read or achieved decode FLOP counter. |
| Eager kernel inventory | PyTorch profiler: 2,626 CUDA events, 36 unique kernels for two decode steps | Instrumented eager inventory only; no counters or optimized-kernel attribution. |
| Board decode energy | Three approximately 33-second eager B1/C1024 windows; raw counter and sampled power agree within 0.32% | Internally qualified whole-board J/token; absolute calibration and HBM/dynamic/static split absent. |
| Native H100 trace parser | Real sm_90 dynamic micro and first-layer capture/group/replay with positive cycles, instructions and kernel stats | Parser/completion smoke only. Partial layer cannot establish full-model working-set reuse. `mbarrier` postprocess warning is unresolved. |
| Whole-model HF eager trace | Correct two-step capture, 2,336 grouped sm_90 kernels, 316,425,467 warp records; positive one-kernel replay after a recorded/recovered postprocess timeout | Full weight-stream coverage exists; 1,568 `mbarrier` TRYWAIT-drop warnings and no exact-SKU/same-program timing or optimized-vLLM calibration. |
| Logical reuse model | 50 MiB L2 and explicit 4 KiB/64 KiB/2 MiB synthetic full-block fills | Analytical classification; no GPU instruction trace, HBM byte counter, area or energy proof. |
| Source/runtime/config pins | Immutable model SHA, exact pip freeze, Accel-Sim source/backend pins and one-line patch hashes | Reproducible code identity; does not repair missing hardware or semantic calibration. |
| Archive transfer | Whole-model 5.14 GB HF eager archive uploaded to private Hugging Face dataset; local, Hub LFS and full streamed readback SHA256 agree | Existing eager archive transfer passes; optimized vLLM traces still pending. |

The V100 legacy fixture and historical RTX results are not used for any H100 calibration row. Positive simulator exit status is accepted only together with a real sm_90 kernel list and positive cycle/instruction/kernel statistics.
