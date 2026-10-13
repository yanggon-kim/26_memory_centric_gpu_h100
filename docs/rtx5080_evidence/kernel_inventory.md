# Optimized decode kernel evidence

Qwen2.5-1.5B BF16, vLLM0.22.1/torch2.11, FA2, RTX5080. Profiler captures are separate from unprofiled timing. Raw delayed traces include prefill; only GPU execute_context_0 ranges are eligible as pure decode. NCU replay/cache effects preclude substituting sample timing for ordinary end-to-end latency.

B1/C1024: one pure decode GPU annotation spans 4.471 ms. 348 enclosed kernels sum to 4.301 ms: 113 cuBLAS gemvx calls 3.777 ms; 28 flash split-KV calls .178 ms; 28 combine calls .132 ms. This establishes weight GEMV dominance for that profiled step. It also shows that the launch gaps in the original eager HF baseline are greatly reduced by the optimized backend.

| Sampled B1 kernel | Duration us (replay) | DRAM reads MB | DRAM % sustained | L2 hit % | Long scoreboard % | Tensor % |
|---|---:|---:|---:|---:|---:|---:|
| QKV GEMV | 11.360 | 6.323 | 59.06 | 7.93 | 22.18 | 0 |
| O GEMV | 9.408 | 4.754 | 53.54 | 10.00 | 20.23 | 0 |
| gate/up GEMV | 66.080 | 55.091 | 88.15 | 1.66 | 55.85 | 0 |
| down GEMV | 37.024 | 27.562 | 78.76 | 6.12 | 47.68 | 0 |
| split-KV attention C4096 | 10.016 | 4.250 | 45.03 | 16.16 | 15.43 | 22.74 |

Percent metrics use NCU's named sustained elapsed/active denominators; long scoreboard uses per-warp-active denominator. These representative first-layer counters are not full-model bytes/token. Initial linear capture used default profiler clock boost; attention capture used clock-control none. Both used cache-control none and approximately ten replay passes. Scope and intervention are retained in raw commands/logs.

A 256 MiB FP32 copy control reaches approximately 819 GB/s counting useful read plus write bytes. It is a workload-specific reference, not a universal read-only roofline. Qwen1.5B unique weight bytes are 3,087,428,608: streaming all weights once at this copy rate would take approximately 3.77 ms before KV, activations, launch/control and differing read-only bandwidth. This diagnostic estimate is consistent with the measured millisecond scale, but is not a rigorous lower bound derived from a measured peak read ceiling.

## SASS evidence

The observed BF16 gemvx function was extracted from the runtime cu13 libcublasLt.so.13 for sm_120. The saved static inventory contains 1,719 instructions, 524 LSU instructions (521 loads, three stores), 571 FMA and 545 ALU. Static counts are not dynamic execution counts or cycle fractions. The loop includes LDG.E/U16 weight loads, unpack/shift operations and HFMA2 accumulation, with backward branches and subsequent reduction/epilogue. Load-derived values feed arithmetic, consistent with the observed long-scoreboard stalls; this does not assign an unsupported absolute memory latency or prove every stall originates from weights. No Tensor MMA instruction was identified in this selected GEMV function.

Raw artifacts: phase1_gemvx_sm120.sass, phase1_gemvx_instruction_inventory.csv, phase1_ncu_vllm_linears.csv, phase1_ncu_vllm_attention_long.csv and annotation-filtered decode_kernel_summary.json. Vendor disassembly is retained locally; small counter and summary artifacts are versioned.

## Batch/context migration

Extended B8/C4096 capture contains two pure decode annotations (6.017/6.037 ms), 752 enclosed kernels totaling11.930 ms. CUTLASS BF16 WMMA GEMM contributes8.588 ms (72.0% busy time), split-KV attention2.524 ms (21.2%), combine.213 ms. Unlike the B1 GEMV case, this larger batch selects a Tensor-operation kernel and attention becomes a material fraction. Kernel names establish the implementation change; they do not by themselves measure Tensor Pipe saturation. The initial shorter B8 capture contains only prefill/mixed iterations and is explicitly ineligible for decode analysis.

Bounded B8/C4096 NCU capture, clock-controlnone/cache-controlnone, confirms one pure-decode first-layer QKV GEMM sample and one attention sample. GEMM:11.904us,6.308MB reads,56.17% DRAM,26.97% L2 hits,7.06% Tensor activity,32.68% long scoreboard,5.18% occupancy. Split-KV:45.344us,33.626MB reads,78.43% DRAM,3.18% L2 hits,40.13% Tensor activity,8.43% long scoreboard,12.00% occupancy. The attention sample is close to its expected eight-request KV footprint and has substantial bandwidth pressure. The selected small GEMM is not compute-saturated despite using a Tensor kernel. Two samples do not characterize every layer or establish a removable compute fraction.
