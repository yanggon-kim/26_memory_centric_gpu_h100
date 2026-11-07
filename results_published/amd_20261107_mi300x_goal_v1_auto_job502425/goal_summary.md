# goal_v1 — mi300x (auto)

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok_v4: **False**
- leak_ok_v2: **False**
- clock_ok: **False**
- release_g1: **True**
- release_g5: **False**
- isolation route: {'route': 'leak_v4', 'ok': False}

## G1 restriction curve (Qwen32B graph B1/C4096; R vs stock full device, same block)

- drift stock last/first: 0.9993 [0.9981, 1.0007] (8 blocks)
- clock sensitivity: {'elasticity_dlogT_dlogf': -0.27361626048936727, 'trials': 960, 'within_cell_gfx_log_var': 0.0007428554318173272, 'note': 'pre-registered sensitivity only; not a qualification route'}

| mask | grid | R [95% CI] | gfx MHz min-max | clock_ok | outputs = stock | disposition | clock-adjusted (sensitivity) |
|---:|---|---|---|---|---|---|---|
| 304 | 288 | 1.0003 [0.9988, 1.0018] (8 blocks) | 1299-1669 | False | True | observational_clock_confounded | 1.0 |
| 304 | 128 | 0.9953 [0.9936, 0.9972] (8 blocks) | 1299-1669 | False | True | observational_clock_confounded | 0.9798 |
| 256 | 256 | 0.9983 [0.9967, 1.0000] (8 blocks) | 1289-1679 | False | True | observational_clock_confounded | 0.9971 |
| 224 | 224 | 0.9985 [0.9972, 0.9997] (8 blocks) | 1299-1750 | False | True | observational_clock_confounded | 0.9916 |
| 192 | 192 | 0.9983 [0.9971, 0.9994] (8 blocks) | 1299-1818 | False | True | observational_clock_confounded | 0.9883 |
| 160 | 160 | 0.9971 [0.9956, 0.9988] (8 blocks) | 1299-1879 | False | True | observational_clock_confounded | 0.9832 |
| 128 | stock | 1.0681 [1.0641, 1.0715] (8 blocks) | 1299-1937 | False | True | observational_clock_confounded | 1.0129 |
| 128 | 128 | 0.9943 [0.9923, 0.9963] (8 blocks) | 1299-1947 | False | True | observational_clock_confounded | 0.9774 |
| 96 | stock | 1.0906 [1.0866, 1.0935] (8 blocks) | 1299-2014 | False | True | observational_clock_confounded | 1.0286 |
| 96 | 96 | 1.0091 [1.0064, 1.0115] (8 blocks) | 1299-2024 | False | True | observational_clock_confounded | 0.9798 |

## G2 isolated phases (ratio vs A = full mask, stock kernel)

| case | B full/reduced grid | C reduced/grid | D reduced/stock | Cb boundary |
|---|---|---|---|---|
| mistral7b_graph_b1_c4096 | 1.0347 [1.0282, 1.0393] (4 blocks) | 1.0395 [1.0330, 1.0455] (4 blocks) | 1.1157 [1.1091, 1.1213] (4 blocks) | 1.0126 [1.0090, 1.0151] (4 blocks) |
| qwen14b_graph_b1_c4096 | 0.9972 [0.9947, 0.9998] (4 blocks) | 0.9957 [0.9945, 0.9969] (4 blocks) | 1.0851 [1.0836, 1.0868] (4 blocks) | 1.0126 [1.0101, 1.0155] (4 blocks) |
| qwen32b_graph_b1_c16384 | 0.9957 [0.9908, 1.0008] (4 blocks) | 0.9991 [0.9946, 1.0045] (4 blocks) | 1.0726 [1.0680, 1.0778] (4 blocks) | 1.0079 [1.0029, 1.0133] (4 blocks) |
| qwen32b_prefill | 1.0016 [1.0002, 1.0031] (4 blocks) | 1.7843 [1.7794, 1.7885] (4 blocks) | 1.7840 [1.7786, 1.7890] (4 blocks) | 2.2586 [2.2511, 2.2696] (4 blocks) |
| qwen7b_graph_b4_c1024 | 1.0773 [1.0698, 1.0842] (4 blocks) | 1.0911 [1.0875, 1.0962] (4 blocks) | 1.2901 [1.2833, 1.2989] (4 blocks) | 1.1872 [1.1798, 1.1956] (4 blocks) |
| qwen7b_graph_b8_c4096 | 1.0025 [0.9994, 1.0058] (4 blocks) | 1.4446 [1.4424, 1.4469] (4 blocks) | 1.4449 [1.4393, 1.4506] (4 blocks) | 1.6435 [1.6372, 1.6501] (4 blocks) |

## G2 kernel routes (does the CU-count override reach a kernel?)

- qwen32b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.8392086908498716, prefill 0.0003329092510539914
- qwen32b_graph_b1_c16384 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7595131046139204, prefill 6.780740438093288e-05
- qwen7b_graph_b8_c4096 tokens=None: override reaches a kernel: **False**; skinny share decode 0.0, prefill 0.0
- qwen7b_graph_b4_c1024 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7745436671128905, prefill 0.0014842675471568628
- qwen14b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7698494638850703, prefill 0.0007278653693184938
- mistral7b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7358577964611445, prefill 0.0002690233722411307
- qwen32b_graph_b1_c4096 tokens=2: override reaches a kernel: **True**; skinny share decode 0.8351787430429822, prefill 0.00033382156735132167

## Mixed serving (ratio vs A)

### g2_serve:c2

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 90.3 | 232 | 758 | 18.6 | 19.4 | 5.86e+03 | — | 1.0 |
| B | 0.989 [0.985, 0.994] | 1.000 [0.998, 1.001] | 1.001 [0.998, 1.003] | 1.014 [1.008, 1.019] | 1.025 [1.019, 1.030] | 1.012 [1.008, 1.016] | — | 0.9921875 |
| C | 0.895 [0.892, 0.898] | 1.821 [1.815, 1.828] | 1.704 [1.693, 1.715] | 1.018 [1.013, 1.021] | 1.015 [1.010, 1.020] | 1.157 [1.152, 1.161] | — | 0.96875 |
| Cb | 0.848 [0.843, 0.853] | 2.078 [2.067, 2.088] | 2.069 [2.048, 2.091] | 1.028 [1.021, 1.034] | 1.029 [1.022, 1.035] | 1.241 [1.234, 1.248] | — | 0.96875 |
| D | 0.850 [0.845, 0.855] | 1.820 [1.812, 1.828] | 1.701 [1.689, 1.713] | 1.086 [1.080, 1.093] | 1.079 [1.071, 1.091] | 1.211 [1.205, 1.219] | — | 0.96875 |

### g2_serve:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 188 | 1.04e+03 | 2.19e+03 | 23 | 440 | 1.1e+04 | — | 1.0 |
| B | 0.994 [0.992, 0.996] | 1.001 [0.999, 1.003] | 1.001 [0.999, 1.004] | 1.000 [0.997, 1.002] | 1.002 [0.996, 1.011] | 1.000 [0.998, 1.003] | — | 1.0 |
| C | 0.719 [0.718, 0.721] | 1.870 [1.859, 1.881] | 1.796 [1.784, 1.807] | 1.198 [1.196, 1.200] | 1.739 [1.725, 1.753] | 1.490 [1.485, 1.494] | — | 0.9375 |
| Cb | 0.625 [0.623, 0.627] | 2.339 [2.321, 2.357] | 2.259 [2.240, 2.279] | 1.285 [1.280, 1.289] | 2.189 [2.168, 2.206] | 1.757 [1.751, 1.764] | — | 0.9375 |
| D | 0.712 [0.710, 0.714] | 1.872 [1.861, 1.882] | 1.797 [1.785, 1.808] | 1.201 [1.196, 1.204] | 1.740 [1.725, 1.756] | 1.492 [1.487, 1.496] | — | 0.9375 |

### g3_serve_energy:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 188 | 1.04e+03 | 2.2e+03 | 23 | 440 | 1.1e+04 | 3.84 | 1.0 |
| C | 0.720 [0.717, 0.723] | 1.866 [1.853, 1.879] | 1.792 [1.779, 1.805] | 1.198 [1.196, 1.200] | 1.739 [1.725, 1.752] | 1.489 [1.483, 1.495] | 1.279 [1.273, 1.284] | 0.9375 |

## G3 decode energy (board accumulator, decode window)

- qwen32b_graph_b1_c4096 A: J/token 13.469366699747512, P 752.2852880808155 W, window 18.316392723792525 s; J/tok ratio n/a, P ratio n/a, T ratio n/a
- qwen32b_graph_b1_c4096 B: J/token 13.377846089990491, P 752.2755444167773 W, window 18.192169485356136 s; J/tok ratio 0.9932 [0.9914, 0.9950] (4 blocks), P ratio 1.0000 [0.9999, 1.0001] (4 blocks), T ratio 0.9932 [0.9914, 0.9950] (4 blocks)
- qwen32b_graph_b1_c4096 C: J/token 13.32682641995487, P 752.267670214346 W, window 18.12298890580132 s; J/tok ratio 0.9894 [0.9865, 0.9923] (4 blocks), P ratio 1.0000 [0.9999, 1.0001] (4 blocks), T ratio 0.9894 [0.9865, 0.9924] (4 blocks)
- qwen32b_graph_b1_c4096 Cb: J/token 13.489600913355615, P 752.2327061566028 W, window 18.345185229647903 s; J/tok ratio 1.0015 [0.9981, 1.0052] (4 blocks), P ratio 0.9999 [0.9998, 1.0000] (4 blocks), T ratio 1.0016 [0.9982, 1.0053] (4 blocks)

## G4 counter audit

- status: measured; physical-HBM read counter: []
- FETCH_SIZE: pre-MALL request-side read counter (KiB)
- WRITE_SIZE: write-traffic counter, streaming-consistent (KiB); MALL behaviour of writes not separable here
- TCC_EA0_RDREQ: pre-MALL request-side read counter (req128)
- TCC_EA0_RDREQ_DRAM: pre-MALL request-side read counter (req128)
- TCC_EA0_RDREQ_LEVEL: unclassified
- TCC_EA0_RD_UNCACHED_32B: unclassified
- TCC_EA0_WRREQ: write-traffic counter, streaming-consistent (req64); MALL behaviour of writes not separable here
- TCC_EA0_WRREQ_64B: write-traffic counter, streaming-consistent (req64); MALL behaviour of writes not separable here
- TCC_EA0_WRREQ_DRAM: write-traffic counter, streaming-consistent (req64); MALL behaviour of writes not separable here

## G5 tuned decoder x memory service (ratio vs absent, same block)

- n=256: mem100 n/a; mem25 n/a; mem50 n/a; inactive n/a; arith100 n/a; absent n/a; mem100_over_arith100 n/a
- n=192: mem100 n/a; mem50 n/a; absent n/a; arith100 n/a; inactive n/a; mem25 n/a; mem100_over_arith100 n/a
- n=128: inactive n/a; mem100 n/a; arith100 n/a; mem25 n/a; mem50 n/a; absent n/a; mem100_over_arith100 n/a

## G6 kernel-shape sweep (best wvSplitK grid per mask)

- mask 304: qkv wvsplitk_dev 17.4 us; o_proj wvsplitk_c224 13.4 us; gate_up wvsplitk_c96 128.6 us; down wvsplitk_c128 66.0 us; lm_head wvsplitk_c224 362.0 us
- mask 272: qkv wvsplitk_c192 18.0 us; o_proj wvsplitk_c224 13.5 us; gate_up wvsplitk_c96 128.6 us; down wvsplitk_c224 65.9 us; lm_head wvsplitk_c160 362.1 us
- mask 256: qkv wvsplitk_c160 17.9 us; o_proj wvsplitk_c224 13.4 us; gate_up wvsplitk_c96 126.7 us; down wvsplitk_c256 66.3 us; lm_head wvsplitk_c160 361.6 us
- mask 232: qkv wvsplitk_c160 18.0 us; o_proj wvsplitk_c224 13.3 us; gate_up wvsplitk_c224 129.9 us; down wvsplitk_c128 67.6 us; lm_head wvsplitk_c160 361.1 us
- mask 192: qkv wvsplitk_c160 17.8 us; o_proj wvsplitk_c128 13.7 us; gate_up wvsplitk_c160 126.9 us; down wvsplitk_c128 65.7 us; lm_head wvsplitk_c160 359.1 us
- mask 152: qkv wvsplitk_c128 18.1 us; o_proj wvsplitk_c128 14.1 us; gate_up wvsplitk_c96 128.8 us; down wvsplitk_c224 68.9 us; lm_head wvsplitk_c128 362.7 us
- mask 128: qkv wvsplitk_c128 18.1 us; o_proj wvsplitk_c128 13.8 us; gate_up wvsplitk_c96 128.8 us; down wvsplitk_c128 66.0 us; lm_head wvsplitk_c96 360.0 us
- mask 96: qkv wvsplitk_c96 18.3 us; o_proj wvsplitk_c96 14.0 us; gate_up wvsplitk_n 128.1 us; down wvsplitk_c160 68.9 us; lm_head wvsplitk_c96 361.8 us

## G7

- logical: {'weights_bytes_per_token': 63968378880, 'kv_read_bytes_per_token_at_4096': 1073741824, 'kv_write_bytes_per_token': 262144, 'compulsory_read_bytes_per_token': 65042120704, 'scope': 'logical demand, BF16; activations and norms excluded (< 1 MB); no reuse within a token is assumed or excluded'}
- qwen32b_graph_b1_c4096__n304__absent__stock__eager__FETCH_SIZE: {'counter': 'FETCH_SIZE', 'n': 304, 'grid': None, 'returncode': -15}
- qwen32b_graph_b1_c4096__n128__absent__g128__eager__FETCH_SIZE: {'counter': 'FETCH_SIZE', 'n': 128, 'grid': 128, 'returncode': 0, 'decode_total': 24047724439.625, 'dispatch_rows': 247968, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [63618313.52876984, 63618322.01984127, 63618314.604166664, 63618330.66964286, 63618304.12698413, 63618326.790674604], 'per_tick': 63618318.31200397, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}
- qwen32b_graph_b1_c4096__n304__absent__stock__eager__WRITE_SIZE: {'counter': 'WRITE_SIZE', 'n': 304, 'grid': None, 'returncode': 0, 'decode_total': 47913615.375, 'dispatch_rows': 247845, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [126818.18948412698, 126442.73214285714, 126818.24404761905, 126818.15376984127, 126818.1121031746, 126818.14583333333], 'per_tick': 126818.14980158731, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}
- qwen32b_graph_b1_c4096__n128__absent__g128__eager__WRITE_SIZE: {'counter': 'WRITE_SIZE', 'n': 128, 'grid': 128, 'returncode': 0, 'decode_total': 47882465.25, 'dispatch_rows': 247968, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [126674.25297619047, 126673.59821428571, 126672.25843253969, 126672.75793650794, 126673.43303571429, 126672.83035714286], 'per_tick': 126673.13169642858, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}

Observational unless a G1 row says qualified. AMD acknowledgement: "This work was supported in part by Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
