# goal_v1 — mi325x (auto)

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

- drift stock last/first: 1.0018 [0.9964, 1.0099] (8 blocks)
- clock sensitivity: {'elasticity_dlogT_dlogf': -0.08811232499257643, 'trials': 960, 'within_cell_gfx_log_var': 0.0002664254844296929, 'note': 'pre-registered sensitivity only; not a qualification route'}

| mask | grid | R [95% CI] | gfx MHz min-max | clock_ok | outputs = stock | disposition | clock-adjusted (sensitivity) |
|---:|---|---|---|---|---|---|---|
| 304 | 288 | 0.9970 [0.9931, 0.9996] (8 blocks) | 1619-1872 | False | True | observational_clock_confounded | 0.9973 |
| 304 | 160 | 0.9982 [0.9943, 1.0013] (8 blocks) | 1619-1914 | False | True | observational_clock_confounded | 0.9962 |
| 256 | 256 | 0.9981 [0.9951, 1.0001] (8 blocks) | 1619-1872 | False | True | observational_clock_confounded | 0.9978 |
| 224 | 224 | 1.0025 [0.9991, 1.0049] (8 blocks) | 1619-1908 | False | True | observational_clock_confounded | 1.0009 |
| 192 | 192 | 1.0024 [0.9985, 1.0052] (8 blocks) | 1619-1909 | False | True | observational_clock_confounded | 1.0005 |
| 160 | stock | 1.0207 [1.0163, 1.0240] (8 blocks) | 1619-1976 | False | True | observational_clock_confounded | 1.0165 |
| 160 | 160 | 1.0009 [0.9974, 1.0039] (8 blocks) | 1619-1977 | False | True | observational_clock_confounded | 0.9983 |
| 128 | stock | 1.1362 [1.1320, 1.1392] (8 blocks) | 1619-2064 | False | True | observational_clock_confounded | 1.1251 |
| 128 | 128 | 1.0051 [1.0015, 1.0083] (8 blocks) | 1582-2018 | False | True | observational_clock_confounded | 1.0016 |
| 96 | 96 | 1.0339 [1.0296, 1.0373] (8 blocks) | 1619-2068 | False | True | observational_clock_confounded | 1.0283 |

## G2 isolated phases (ratio vs A = full mask, stock kernel)

| case | B full/reduced grid | C reduced/grid | D reduced/stock | Cb boundary |
|---|---|---|---|---|
| mistral7b_graph_b1_c4096 | 0.9930 [0.9748, 1.0112] (4 blocks) | 0.9873 [0.9800, 0.9943] (4 blocks) | 1.0438 [1.0326, 1.0585] (4 blocks) | 1.0538 [1.0334, 1.0746] (4 blocks) |
| qwen14b_graph_b1_c4096 | 0.9868 [0.9607, 1.0066] (4 blocks) | 1.0094 [0.9925, 1.0289] (4 blocks) | 1.0190 [0.9966, 1.0379] (4 blocks) | 0.9878 [0.9623, 1.0186] (4 blocks) |
| qwen32b_graph_b1_c16384 | 0.9917 [0.9749, 1.0092] (4 blocks) | 1.0073 [0.9899, 1.0224] (4 blocks) | 1.0100 [0.9963, 1.0248] (4 blocks) | 1.0031 [0.9927, 1.0143] (4 blocks) |
| qwen32b_prefill | 1.0169 [0.9599, 1.0923] (4 blocks) | 1.6503 [1.6307, 1.6660] (4 blocks) | 1.6580 [1.5946, 1.7150] (4 blocks) | 2.1498 [2.1233, 2.1653] (4 blocks) |
| qwen7b_graph_b4_c1024 | 1.0213 [0.9931, 1.0488] (4 blocks) | 1.0061 [0.9846, 1.0325] (4 blocks) | 1.1084 [1.0832, 1.1340] (4 blocks) | 1.0672 [1.0450, 1.0927] (4 blocks) |
| qwen7b_graph_b8_c4096 | 1.0030 [0.9902, 1.0155] (4 blocks) | 1.2392 [1.2188, 1.2634] (4 blocks) | 1.2451 [1.2087, 1.2730] (4 blocks) | 1.5934 [1.5720, 1.6174] (4 blocks) |

## G2 kernel routes (does the CU-count override reach a kernel?)

- qwen32b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.8353898718631426, prefill 0.00036229392217140805
- qwen32b_graph_b1_c16384 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7677891796133187, prefill 6.816365348524848e-05
- qwen7b_graph_b8_c4096 tokens=None: override reaches a kernel: **False**; skinny share decode 0.0, prefill 0.0
- qwen7b_graph_b4_c1024 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7495948712716863, prefill 0.001383153160968514
- qwen14b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.756981952509496, prefill 0.0007778906640792121
- mistral7b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7065159729577513, prefill 0.0002686849286662681
- qwen32b_graph_b1_c4096 tokens=2: override reaches a kernel: **True**; skinny share decode 0.8348077075023534, prefill 0.0003633603515645856

## Mixed serving (ratio vs A)

### g2_serve:c2

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 103 | 216 | 695 | 15.8 | 23.5 | 5.26e+03 | — | 1.0 |
| B | 0.996 [0.972, 1.020] | 0.998 [0.953, 1.046] | 0.986 [0.943, 1.031] | 1.011 [1.005, 1.018] | 0.904 [0.742, 1.110] | 0.994 [0.954, 1.036] | — | 1.0 |
| C | 0.912 [0.892, 0.930] | 1.708 [1.654, 1.766] | 1.659 [1.591, 1.726] | 1.008 [1.001, 1.014] | 0.909 [0.781, 1.103] | 1.119 [1.084, 1.162] | — | 0.96875 |
| Cb | 0.854 [0.836, 0.870] | 2.109 [2.052, 2.176] | 2.052 [1.968, 2.136] | 1.038 [1.028, 1.046] | 0.821 [0.726, 0.959] | 1.212 [1.181, 1.252] | — | 0.96875 |
| D | 0.913 [0.896, 0.927] | 1.673 [1.628, 1.727] | 1.631 [1.570, 1.692] | 1.027 [1.022, 1.032] | 0.753 [0.684, 0.879] | 1.112 [1.083, 1.148] | — | 0.96875 |

### g2_serve:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 205 | 1e+03 | 2.05e+03 | 20.7 | 397 | 1.02e+04 | — | 1.0 |
| B | 1.009 [0.999, 1.024] | 0.989 [0.974, 1.003] | 0.983 [0.960, 1.001] | 0.998 [0.994, 1.000] | 0.989 [0.969, 1.003] | 0.989 [0.974, 1.001] | — | 1.0 |
| C | 0.782 [0.765, 0.797] | 1.624 [1.582, 1.668] | 1.658 [1.604, 1.718] | 1.112 [1.110, 1.115] | 1.611 [1.573, 1.640] | 1.359 [1.332, 1.390] | — | 0.9375 |
| Cb | 0.662 [0.651, 0.674] | 2.062 [2.014, 2.110] | 2.032 [1.974, 2.086] | 1.282 [1.280, 1.285] | 2.008 [1.970, 2.038] | 1.633 [1.603, 1.664] | — | 0.9375 |
| D | 0.776 [0.761, 0.788] | 1.637 [1.595, 1.674] | 1.671 [1.620, 1.730] | 1.115 [1.111, 1.119] | 1.615 [1.584, 1.639] | 1.367 [1.344, 1.393] | — | 0.9375 |

### g3_serve_energy:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 202 | 1.01e+03 | 2.11e+03 | 20.8 | 403 | 1.04e+04 | 4.37 | 1.0 |
| C | 0.792 [0.780, 0.803] | 1.613 [1.581, 1.650] | 1.609 [1.563, 1.653] | 1.109 [1.106, 1.114] | 1.586 [1.557, 1.615] | 1.338 [1.318, 1.360] | 1.164 [1.158, 1.170] | 0.9375 |

## G3 decode energy (board accumulator, decode window)

- qwen32b_graph_b1_c4096 A: J/token 15.47779358570923, P 978.8215442609586 W, window 16.186619123994024 s; J/tok ratio n/a, P ratio n/a, T ratio n/a
- qwen32b_graph_b1_c4096 B: J/token 15.284808033484705, P 983.3466506887155 W, window 15.903241631299897 s; J/tok ratio 0.9875 [0.9831, 0.9914] (4 blocks), P ratio 1.0046 [0.9965, 1.0130] (4 blocks), T ratio 0.9825 [0.9706, 0.9939] (4 blocks)
- qwen32b_graph_b1_c4096 C: J/token 15.236113810931343, P 970.3306193032855 W, window 16.067780110998136 s; J/tok ratio 0.9844 [0.9804, 0.9886] (4 blocks), P ratio 0.9913 [0.9770, 1.0050] (4 blocks), T ratio 0.9927 [0.9770, 1.0112] (4 blocks)
- qwen32b_graph_b1_c4096 Cb: J/token 15.255106279963458, P 972.1859174730012 W, window 16.057050438672015 s; J/tok ratio 0.9856 [0.9826, 0.9884] (4 blocks), P ratio 0.9932 [0.9798, 1.0085] (4 blocks), T ratio 0.9920 [0.9756, 1.0076] (4 blocks)

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

- n=256: mem100 n/a; inactive n/a; absent n/a; mem25 n/a; mem50 n/a; arith100 n/a; mem100_over_arith100 n/a
- n=192: inactive n/a; mem50 n/a; mem25 n/a; arith100 n/a; absent n/a; mem100 n/a; mem100_over_arith100 n/a
- n=128: absent n/a; mem25 n/a; arith100 n/a; inactive n/a; mem100 n/a; mem50 n/a; mem100_over_arith100 n/a

## G6 kernel-shape sweep (best wvSplitK grid per mask)

- mask 304: qkv wvsplitk_c224 16.0 us; o_proj wvsplitk_c288 12.1 us; gate_up wvsplitk_c160 110.9 us; down wvsplitk_c128 56.6 us; lm_head wvsplitk_c128 310.3 us
- mask 272: qkv wvsplitk_c160 16.0 us; o_proj wvsplitk_c256 12.1 us; gate_up wvsplitk_c96 110.7 us; down wvsplitk_c128 56.6 us; lm_head wvsplitk_c160 308.3 us
- mask 256: qkv wvsplitk_c160 16.0 us; o_proj wvsplitk_c256 12.1 us; gate_up wvsplitk_c96 110.6 us; down wvsplitk_c128 56.6 us; lm_head wvsplitk_c160 308.1 us
- mask 232: qkv wvsplitk_c224 16.1 us; o_proj wvsplitk_c224 12.1 us; gate_up wvsplitk_c96 110.7 us; down wvsplitk_c128 56.6 us; lm_head wvsplitk_c128 312.1 us
- mask 192: qkv wvsplitk_c160 16.0 us; o_proj wvsplitk_c160 12.4 us; gate_up wvsplitk_c160 110.7 us; down wvsplitk_c128 56.9 us; lm_head wvsplitk_c288 308.3 us
- mask 152: qkv wvsplitk_c128 16.2 us; o_proj wvsplitk_c128 12.5 us; gate_up wvsplitk_c96 110.2 us; down wvsplitk_c128 56.6 us; lm_head wvsplitk_c96 309.0 us
- mask 128: qkv wvsplitk_c128 16.1 us; o_proj wvsplitk_c128 13.1 us; gate_up wvsplitk_n 111.2 us; down wvsplitk_c128 56.4 us; lm_head wvsplitk_c256 307.5 us
- mask 96: qkv wvsplitk_c96 16.8 us; o_proj wvsplitk_c192 13.9 us; gate_up wvsplitk_c96 109.8 us; down wvsplitk_c96 59.9 us; lm_head wvsplitk_c288 316.6 us

## G7

- logical: {'weights_bytes_per_token': 63968378880, 'kv_read_bytes_per_token_at_4096': 1073741824, 'kv_write_bytes_per_token': 262144, 'compulsory_read_bytes_per_token': 65042120704, 'scope': 'logical demand, BF16; activations and norms excluded (< 1 MB); no reuse within a token is assumed or excluded'}

Observational unless a G1 row says qualified. AMD acknowledgement: "This work was supported in part by Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
