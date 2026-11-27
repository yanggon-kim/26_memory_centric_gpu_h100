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

Exact Green launch-signature equality across all six SM settings: **False**.

The comparison includes demangled symbol, grid/block dimensions, registers/thread, static/dynamic shared memory, local-memory/thread and launch counts; durations and Green IDs are excluded from that equality check. Equality applies only to these captured ranges and settings. It does not replace an explicit kernel-grid retuning experiment.

Limitations: one placement acquisition per setting, instrumented durations and clock-unqualified ordinary timing. Only uniquely correlated operation ranges are attributed; remaining attention/norm/head work, achieved occupancy/issue/stalls, dynamic instruction semantics and optimized-path traffic joins remain separate. Raw launch metadata is retained with exact SQLite and case checksums in `../results/eager_decoder_grid_catalog.json`.

## Observed varying launch groups

| Actual SMs | Symbol | Grid | Block | Launches | CPU-correlated NVTX operations |
|---|---|---|---|---:|---|
| 40 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 2×40×1 | 384×1×1 | 192 | self_attn_o_proj |
| 56 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 2×40×1 | 384×1×1 | 192 | self_attn_o_proj |
| 80 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 2×40×1 | 384×1×1 | 192 | self_attn_o_proj |
| 96 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 2×40×1 | 384×1×1 | 192 | self_attn_o_proj |
| 112 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 4×20×1 | 384×1×1 | 192 | self_attn_o_proj |
| 132 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 4×20×1 | 384×1×1 | 192 | self_attn_o_proj |

Attribution uses the kernel CUDA correlation ID, the unique runtime API interval on its actual host thread, and containing versioned operation/tick NVTX ranges. Kernel-name substrings are not used to assign operation semantics. Missing or ambiguous joins remain explicitly unattributed.

The differing symbols contain4x1/2x1 text, but actual CUDA cluster dimensions are not provided by this SQLite schema. Do not infer cluster semantics from those names. One captured kernel path changes between112and96SM; causality, generality and a latency transition are unresolved. A resident-model protocol may establish different library/algorithm state from fresh-model-per-point initialization, so its full setting-specific proofs and separate labels remain required.
