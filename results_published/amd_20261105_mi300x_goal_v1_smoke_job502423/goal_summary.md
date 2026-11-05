# goal_v1 — mi300x (pilot)

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
- clock sensitivity: {'elasticity_dlogT_dlogf': -0.19314652966228374, 'trials': 20, 'within_cell_gfx_log_var': 5.383514228369641e-05, 'note': 'pre-registered sensitivity only; not a qualification route'}

| mask | grid | R [95% CI] | gfx MHz min-max | clock_ok | outputs = stock | disposition | clock-adjusted (sensitivity) |
|---:|---|---|---|---|---|---|---|
| 128 | 128 | 0.9902 [0.9881, 0.9924] (1 blocks) | 1316-1898 | False | True | observational_clock_confounded | 0.9817 |

## G2 isolated phases (ratio vs A = full mask, stock kernel)

| case | B full/reduced grid | C reduced/grid | D reduced/stock | Cb boundary |
|---|---|---|---|---|
| qwen32b_prefill | n/a | n/a | n/a | n/a |

## G2 kernel routes (does the CU-count override reach a kernel?)

- qwen32b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.8312535074865249, prefill 0.00032687228926099533
- qwen7b_graph_b8_c4096 tokens=None: override reaches a kernel: **False**; skinny share decode 0.0, prefill 0.0

## Mixed serving (ratio vs A)

### g2_serve:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 182 | 1.15e+03 | 2.22e+03 | 23.5 | 112 | 1.02e+04 | — | 1.0 |

### g3_serve_energy:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| C | 132 | 2.03e+03 | 4.01e+03 | 28.4 | 187 | 1.5e+04 | 5.12 | None |

## G3 decode energy (board accumulator, decode window)

- qwen32b_graph_b1_c4096 C: J/token 13.589255364984263, P 753.5770727551308 W, window 9.214860643516294 s; J/tok ratio n/a, P ratio n/a, T ratio n/a

## G4 counter audit

- status: measured; physical-HBM read counter: []
- FETCH_SIZE: pre-MALL request-side read counter (KiB)
- WRITE_SIZE: write-traffic counter, streaming-consistent (KiB); MALL behaviour of writes not separable here

## G5 tuned decoder x memory service (ratio vs absent, same block)

- n=256: mem100 n/a (866 GB/s)

## G6 kernel-shape sweep (best wvSplitK grid per mask)

- mask 128: qkv wvsplitk_c64 23.5 us; o_proj wvsplitk_c256 16.3 us; gate_up wvsplitk_c256 132.3 us; down wvsplitk_c128 72.9 us; lm_head wvsplitk_c256 354.4 us

## G7

- logical: {'weights_bytes_per_token': 63968378880, 'kv_read_bytes_per_token_at_4096': 1073741824, 'kv_write_bytes_per_token': 262144, 'compulsory_read_bytes_per_token': 65042120704, 'scope': 'logical demand, BF16; activations and norms excluded (< 1 MB); no reuse within a token is assumed or excluded'}
- qwen32b_graph_b1_c4096__n304__absent__stock__eager__FETCH_SIZE: {'counter': 'FETCH_SIZE', 'n': 304, 'grid': None, 'returncode': 0, 'decode_total': 24017702762.0625, 'dispatch_rows': 247665, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [63616819.72718254, 63534277.752976194, 63616819.87301587, 63430948.59126984, 63417691.36309524, 63616819.86805555], 'per_tick': 63575548.740079366, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}
- qwen32b_graph_b1_c4096__n128__absent__g128__eager__FETCH_SIZE: {'counter': 'FETCH_SIZE', 'n': 128, 'grid': 128, 'returncode': 0, 'decode_total': 24021326779.0, 'dispatch_rows': 247690, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [63618287.71626984, 63570969.48611111, 63618301.89781746, 63432706.09424603, 63432334.0327381, 63618302.02678572], 'per_tick': 63594628.60119048, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}

Observational unless a G1 row says qualified. AMD acknowledgement: "This work was supported in part by Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
