# goal_v1 — mi350x (pilot)

## Gates

- placement_ok: **False**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok_v4: **True**
- leak_ok_v2: **False**
- clock_ok: **False**
- release_g1: **False**
- release_g5: **False**
- isolation route: {'route': 'leak_v4', 'ok': True}

## G1 restriction curve (Qwen32B graph B1/C4096; R vs stock full device, same block)

- drift stock last/first: n/a
- clock sensitivity: {'elasticity_dlogT_dlogf': -0.07811932818215743, 'trials': 20, 'within_cell_gfx_log_var': 1.648069732860519e-05, 'note': 'pre-registered sensitivity only; not a qualification route'}

| mask | grid | R [95% CI] | gfx MHz min-max | clock_ok | outputs = stock | disposition | clock-adjusted (sensitivity) |
|---:|---|---|---|---|---|---|---|
| 192 | 192 | 0.9866 [0.9854, 0.9876] (1 blocks) | 1598-2182 | False | True | observational_clock_confounded | 0.9829 |

## G2 isolated phases (ratio vs A = full mask, stock kernel)

| case | B full/reduced grid | C reduced/grid | D reduced/stock | Cb boundary |
|---|---|---|---|---|
| qwen32b_prefill | n/a | n/a | n/a | n/a |

## G2 kernel routes (does the CU-count override reach a kernel?)

- qwen32b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.80788952173861, prefill 0.0006470628327679098
- qwen7b_graph_b8_c4096 tokens=None: override reaches a kernel: **False**; skinny share decode 0.0, prefill 0.0

## Mixed serving (ratio vs A)

### g2_serve:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 314 | 571 | 1.11e+03 | 14.4 | 52.2 | 5.71e+03 | — | 1.0 |

### g3_serve_energy:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| C | 250 | 678 | 1.31e+03 | 19.7 | 69.6 | 7.54e+03 | 3.3 | None |

## G3 decode energy (board accumulator, decode window)

- qwen32b_graph_b1_c4096 C: J/token 11.74795885327483, P 954.0919598219211 W, window 6.29206369384192 s; J/tok ratio n/a, P ratio n/a, T ratio n/a

## G4 counter audit

- status: measured; physical-HBM read counter: []
- FETCH_SIZE: unclassified
- WRITE_SIZE: write-traffic counter, streaming-consistent (KiB); MALL behaviour of writes not separable here

## G5 tuned decoder x memory service (ratio vs absent, same block)

- n=224: mem100 n/a (1212 GB/s)

## G6 kernel-shape sweep (best wvSplitK grid per mask)

- mask 192: qkv wvsplitk_c192 13.8 us; o_proj wvsplitk_n 11.1 us; gate_up wvsplitk_n 91.2 us; down wvsplitk_c192 49.4 us; lm_head wvsplitk_c192 244.5 us

## G7

- logical: {'weights_bytes_per_token': 63968378880, 'kv_read_bytes_per_token_at_4096': 1073741824, 'kv_write_bytes_per_token': 262144, 'compulsory_read_bytes_per_token': 65042120704, 'scope': 'logical demand, BF16; activations and norms excluded (< 1 MB); no reuse within a token is assumed or excluded'}
- qwen32b_graph_b1_c4096__n256__absent__stock__eager__FETCH_SIZE: {'counter': 'FETCH_SIZE', 'n': 256, 'grid': None, 'returncode': 0, 'decode_total': 24041451099.0, 'dispatch_rows': 247909, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [63616392.97123016, 63528380.8640873, 63616389.13888889, 63616393.00793651, 63616389.37003968, 63616389.552579366], 'per_tick': 63616389.46130952, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}
- qwen32b_graph_b1_c4096__n192__absent__g192__eager__FETCH_SIZE: {'counter': 'FETCH_SIZE', 'n': 192, 'grid': 192, 'returncode': 0, 'decode_total': 24047282319.3125, 'dispatch_rows': 247968, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [63617148.490079366, 63617149.94246032, 63617148.86805555, 63617149.39880952, 63617147.884920634, 63617149.37301587], 'per_tick': 63617149.12053572, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}

Observational unless a G1 row says qualified. AMD acknowledgement: "This work was supported in part by Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
