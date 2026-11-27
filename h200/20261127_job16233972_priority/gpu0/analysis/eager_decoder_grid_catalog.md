# Saved eager decoder launch geometry

Single three-tick placement capture per SM setting, complete eager decoder launches. Raw grids/register/shared allocations are launch metadata, not measured occupancy, issue rate, clean timing or physical HBM traffic. CPU NVTX operation associations do not establish dynamic instruction semantics or complete attribution. No three-acquisition timing interval is derived.

| Path | Actual SMs | Captured launches | Distinct symbol/resource signatures |
|---|---:|---:|---:|
| Green | 40 | 2553 | 28 |
| Green | 56 | 2553 | 28 |
| Green | 80 | 2553 | 28 |
| Green | 96 | 2553 | 28 |
| Green | 112 | 2553 | 28 |
| Green | 132 | 2553 | 28 |
| stock | 132 | 2553 | 28 |

Exact Green launch-signature equality across all six SM settings: **True**.

The comparison includes demangled symbol, grid/block dimensions, registers/thread, static/dynamic shared memory, local-memory/thread and launch counts; durations and Green IDs are excluded from that equality check. Equality applies only to these captured ranges and settings. It does not replace an explicit kernel-grid retuning experiment.

Limitations: one placement acquisition per setting and instrumented durations; ordinary timing qualification is recorded separately in the workload curve. Only uniquely correlated operation ranges are attributed; remaining attention/norm/head work, achieved occupancy/issue/stalls, dynamic instruction semantics and optimized-path traffic joins remain separate. Raw launch metadata is retained with exact SQLite and case checksums in `../results/eager_decoder_grid_catalog.json`.

## Observed varying launch groups

| Actual SMs | Symbol | Grid | Block | Launches | CPU-correlated NVTX operations |
|---|---|---|---|---:|---|

Attribution uses the kernel CUDA correlation ID, the unique runtime API interval on its actual host thread, and containing versioned operation/tick NVTX ranges. Kernel-name substrings are not used to assign operation semantics. Missing or ambiguous joins remain explicitly unattributed.

Actual CUDA cluster dimensions are not provided by this SQLite schema. Do not infer cluster semantics from symbol names. Launch differences, where observed, do not establish causality, generality or latency thresholds. Resident-model protocols may establish different library/algorithm state from fresh-model-per-point initialization; retain their full setting-specific proofs and separate labels. Unavailable operation NVTX ranges remain unattributed.
