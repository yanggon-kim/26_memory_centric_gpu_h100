# goal_v1 — mi350x (auto)

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok_v4: **True**
- leak_ok_v2: **False**
- clock_ok: **False**
- release_g1: **True**
- release_g5: **True**
- isolation route: {'route': 'leak_v4', 'ok': True}

## G1 restriction curve (Qwen32B graph B1/C4096; R vs stock full device, same block)

- drift stock last/first: 1.0278 [1.0182, 1.0350] (8 blocks)
- clock sensitivity: {'elasticity_dlogT_dlogf': -0.2255999511344087, 'trials': 800, 'within_cell_gfx_log_var': 0.0001576355975067109, 'note': 'pre-registered sensitivity only; not a qualification route'}

| mask | grid | R [95% CI] | gfx MHz min-max | clock_ok | outputs = stock | disposition | clock-adjusted (sensitivity) |
|---:|---|---|---|---|---|---|---|
| 256 | 192 | 0.9892 [0.9845, 0.9946] (8 blocks) | 1318-2190 | False | True | observational_clock_confounded | 0.9856 |
| 224 | 224 | 0.9909 [0.9864, 0.9961] (8 blocks) | 1318-2186 | False | True | observational_clock_confounded | 0.9879 |
| 192 | stock | 1.1869 [1.1815, 1.1927] (8 blocks) | 1318-2199 | False | True | observational_clock_confounded | 1.1761 |
| 192 | 192 | 0.9963 [0.9912, 1.0025] (8 blocks) | 1318-2192 | False | True | observational_clock_confounded | 0.9908 |
| 160 | 160 | 1.0554 [1.0496, 1.0613] (8 blocks) | 1318-2197 | False | True | observational_clock_confounded | 1.0479 |
| 128 | stock | 1.1709 [1.1655, 1.1763] (8 blocks) | 1318-2198 | False | True | observational_clock_confounded | 1.1602 |
| 128 | 128 | 1.1546 [1.1493, 1.1603] (8 blocks) | 1318-2198 | False | True | observational_clock_confounded | 1.1446 |
| 96 | 96 | 1.3920 [1.3863, 1.3979] (8 blocks) | 1318-2204 | False | True | observational_clock_confounded | 1.3785 |

## G2 isolated phases (ratio vs A = full mask, stock kernel)

| case | B full/reduced grid | C reduced/grid | D reduced/stock | Cb boundary |
|---|---|---|---|---|
| mistral7b_graph_b1_c4096 | 1.0229 [0.9998, 1.0450] (4 blocks) | 1.0388 [1.0280, 1.0500] (4 blocks) | 1.1706 [1.1347, 1.2129] (4 blocks) | 1.1607 [1.1509, 1.1733] (4 blocks) |
| qwen14b_graph_b1_c4096 | 1.0164 [0.9917, 1.0384] (4 blocks) | 1.0224 [0.9950, 1.0590] (4 blocks) | 1.1903 [1.1695, 1.2121] (4 blocks) | 1.1462 [1.1179, 1.1749] (4 blocks) |
| qwen32b_graph_b1_c16384 | 1.0101 [0.9869, 1.0303] (4 blocks) | 0.9923 [0.9686, 1.0178] (4 blocks) | 1.1697 [1.1522, 1.1891] (4 blocks) | 1.1577 [1.1322, 1.1916] (4 blocks) |
| qwen32b_prefill | 0.9317 [0.8750, 0.9923] (4 blocks) | 1.2142 [1.0968, 1.3466] (4 blocks) | 1.1755 [1.0904, 1.2753] (4 blocks) | 1.2873 [1.1987, 1.4176] (4 blocks) |
| qwen7b_graph_b4_c1024 | 1.0510 [1.0159, 1.0900] (4 blocks) | 1.0236 [1.0160, 1.0315] (4 blocks) | 1.2696 [1.2515, 1.2878] (4 blocks) | 1.1506 [1.1437, 1.1566] (4 blocks) |
| qwen7b_graph_b8_c4096 | 0.9961 [0.9729, 1.0191] (4 blocks) | 1.1427 [1.1261, 1.1590] (4 blocks) | 1.1326 [1.1224, 1.1435] (4 blocks) | 1.2624 [1.2302, 1.2836] (4 blocks) |

## G2 kernel routes (does the CU-count override reach a kernel?)

- qwen32b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.8123845720874475, prefill 0.0016769570282816647
- qwen32b_graph_b1_c16384 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7426765230349431, prefill 0.0001341051790532283
- qwen7b_graph_b8_c4096 tokens=None: override reaches a kernel: **False**; skinny share decode 0.0, prefill 0.0
- qwen7b_graph_b4_c1024 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7009522012673558, prefill 0.007131146163747104
- qwen14b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.7184064212896251, prefill 0.001331511649674976
- mistral7b_graph_b1_c4096 tokens=None: override reaches a kernel: **True**; skinny share decode 0.6460883026405129, prefill 0.00047058433592038154
- qwen32b_graph_b1_c4096 tokens=2: override reaches a kernel: **True**; skinny share decode 0.8011187453486125, prefill 0.0006753549308521733

## Mixed serving (ratio vs A)

### g2_serve:c2

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 133 | 126 | 497 | 12.7 | 18.7 | 4.04e+03 | — | 1.0 |
| B | 0.981 [0.964, 0.998] | 1.008 [0.977, 1.040] | 1.010 [0.750, 1.372] | 1.017 [1.014, 1.019] | 1.009 [0.896, 1.168] | 1.021 [0.978, 1.066] | — | 1.0 |
| C | 0.946 [0.928, 0.964] | 1.295 [1.253, 1.337] | 1.178 [0.890, 1.580] | 1.026 [1.023, 1.028] | 1.033 [0.920, 1.186] | 1.064 [1.018, 1.112] | — | 1.0 |
| Cb | 0.802 [0.787, 0.816] | 1.485 [1.433, 1.543] | 1.344 [1.044, 1.757] | 1.223 [1.221, 1.225] | 1.133 [1.016, 1.312] | 1.247 [1.196, 1.300] | — | 1.0 |
| D | 0.806 [0.792, 0.820] | 1.289 [1.243, 1.338] | 1.173 [0.895, 1.556] | 1.246 [1.244, 1.249] | 1.157 [1.040, 1.327] | 1.237 [1.188, 1.286] | — | 1.0 |

### g2_serve:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 310 | 546 | 1.16e+03 | 14.4 | 196 | 6.42e+03 | — | 1.0 |
| B | 1.003 [0.982, 1.024] | 0.991 [0.961, 1.022] | 0.993 [0.953, 1.033] | 1.001 [0.997, 1.005] | 1.003 [0.978, 1.029] | 0.994 [0.970, 1.018] | — | 1.0 |
| C | 0.801 [0.786, 0.819] | 1.180 [1.144, 1.212] | 1.161 [1.106, 1.209] | 1.352 [1.347, 1.359] | 1.342 [1.316, 1.367] | 1.282 [1.252, 1.309] | — | 1.0 |
| Cb | 0.706 [0.697, 0.714] | 1.520 [1.487, 1.551] | 1.483 [1.449, 1.518] | 1.435 [1.427, 1.442] | 1.638 [1.608, 1.670] | 1.460 [1.441, 1.479] | — | 1.0 |
| D | 0.772 [0.762, 0.783] | 1.193 [1.167, 1.216] | 1.184 [1.152, 1.215] | 1.353 [1.345, 1.362] | 1.355 [1.334, 1.377] | 1.297 [1.277, 1.318] | — | 1.0 |

### g3_serve_energy:c8

| cond | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | J/tok | outputs agree with A |
|---|---|---|---|---|---|---|---|---|
| A | 311 | 548 | 1.14e+03 | 14.4 | 195 | 6.4e+03 | 2.81 | 1.0 |
| C | 0.802 [0.778, 0.823] | 1.170 [1.129, 1.217] | 1.162 [1.103, 1.230] | 1.346 [1.339, 1.354] | 1.344 [1.310, 1.382] | 1.280 [1.246, 1.323] | 1.121 [1.107, 1.142] | 1.0 |

## G3 decode energy (board accumulator, decode window)

- qwen32b_graph_b1_c4096 A: J/token 11.648595362246933, P 910.7009527449276 W, window 13.096830759570002 s; J/tok ratio n/a, P ratio n/a, T ratio n/a
- qwen32b_graph_b1_c4096 B: J/token 11.564735144019103, P 913.3029479800748 W, window 12.960889428295195 s; J/tok ratio 0.9928 [0.9871, 0.9986] (4 blocks), P ratio 1.0029 [0.9902, 1.0155] (4 blocks), T ratio 0.9896 [0.9716, 1.0080] (4 blocks)
- qwen32b_graph_b1_c4096 C: J/token 11.665102059802281, P 898.7654740996771 W, window 13.283103307662532 s; J/tok ratio 1.0014 [0.9965, 1.0068] (4 blocks), P ratio 0.9869 [0.9775, 0.9968] (4 blocks), T ratio 1.0142 [0.9995, 1.0290] (4 blocks)
- qwen32b_graph_b1_c4096 Cb: J/token 12.387420921252597, P 833.9181489670218 W, window 15.200918384175747 s; J/tok ratio 1.0634 [1.0566, 1.0702] (4 blocks), P ratio 0.9157 [0.9074, 0.9244] (4 blocks), T ratio 1.1607 [1.1434, 1.1771] (4 blocks)

## G4 counter audit

- status: measured; physical-HBM read counter: []
- FETCH_SIZE: unclassified
- WRITE_SIZE: write-traffic counter, streaming-consistent (KiB); MALL behaviour of writes not separable here
- TCC_EA0_RDREQ: pre-MALL request-side read counter (req128)
- TCC_EA0_RDREQ_128B: pre-MALL request-side read counter (req128)
- TCC_EA0_RDREQ_64B: unclassified
- TCC_EA0_RDREQ_DRAM: pre-MALL request-side read counter (req128)
- TCC_EA0_RDREQ_DRAM_32B: pre-MALL request-side read counter (req32)
- TCC_EA0_RDREQ_DRAM_CREDIT_STALL: unclassified
- TCC_EA0_RDREQ_LEVEL: unclassified

## G5 tuned decoder x memory service (ratio vs absent, same block)

- n=224: inactive 1.0110 [0.9844, 1.0409] (4 blocks); absent n/a; mem50 1.1112 [1.0711, 1.1515] (4 blocks) (559 GB/s); mem100 1.2577 [1.2377, 1.2773] (4 blocks) (1162 GB/s); mem25 1.0380 [1.0257, 1.0477] (4 blocks) (295 GB/s); arith100 1.0062 [0.9772, 1.0439] (4 blocks); mem100_over_arith100 1.2499 [1.2122, 1.2775] (4 blocks)
- n=192: mem100 1.2099 [1.1673, 1.2529] (4 blocks) (1298 GB/s); inactive 1.0213 [0.9922, 1.0678] (4 blocks); mem25 1.0583 [1.0260, 1.0960] (4 blocks) (304 GB/s); absent n/a; arith100 1.0306 [0.9794, 1.0886] (4 blocks); mem50 1.1230 [1.0819, 1.1601] (4 blocks) (591 GB/s); mem100_over_arith100 1.1740 [1.1228, 1.2159] (4 blocks)
- n=128: mem50 1.0567 [1.0386, 1.0790] (4 blocks) (737 GB/s); absent n/a; mem25 1.0600 [1.0377, 1.0838] (4 blocks) (347 GB/s); arith100 1.0287 [1.0055, 1.0493] (4 blocks); mem100 1.1419 [1.1294, 1.1549] (4 blocks) (1497 GB/s); inactive 1.0102 [1.0006, 1.0216] (4 blocks); mem100_over_arith100 1.1100 [1.0871, 1.1407] (4 blocks)

## G6 kernel-shape sweep (best wvSplitK grid per mask)

- mask 256: qkv wvsplitk_c256 12.9 us; o_proj wvsplitk_c256 9.8 us; gate_up wvsplitk_c224 82.7 us; down wvsplitk_c224 42.5 us; lm_head wvsplitk_c224 222.5 us
- mask 232: qkv wvsplitk_c224 13.0 us; o_proj wvsplitk_c224 9.8 us; gate_up wvsplitk_c224 82.8 us; down wvsplitk_c224 42.6 us; lm_head wvsplitk_c224 222.7 us
- mask 224: qkv wvsplitk_c224 13.0 us; o_proj wvsplitk_c224 9.9 us; gate_up wvsplitk_c224 82.8 us; down wvsplitk_c224 42.4 us; lm_head wvsplitk_n 222.4 us
- mask 208: qkv wvsplitk_c192 13.0 us; o_proj wvsplitk_c192 10.1 us; gate_up wvsplitk_c192 83.7 us; down wvsplitk_c192 42.7 us; lm_head wvsplitk_c160 236.5 us
- mask 192: qkv wvsplitk_c192 13.0 us; o_proj wvsplitk_c192 10.1 us; gate_up wvsplitk_c192 83.0 us; down wvsplitk_c192 42.8 us; lm_head wvsplitk_c192 224.8 us
- mask 160: qkv wvsplitk_c160 13.9 us; o_proj wvsplitk_c160 10.8 us; gate_up wvsplitk_c160 89.4 us; down wvsplitk_c160 46.1 us; lm_head wvsplitk_n 236.5 us
- mask 128: qkv wvsplitk_n 15.4 us; o_proj wvsplitk_c128 11.6 us; gate_up wvsplitk_c128 102.3 us; down wvsplitk_c128 49.7 us; lm_head wvsplitk_c128 269.1 us
- mask 96: qkv wvsplitk_n 19.0 us; o_proj wvsplitk_n 14.6 us; gate_up wvsplitk_n 123.6 us; down wvsplitk_c96 64.2 us; lm_head wvsplitk_c192 322.1 us

## G7

- logical: {'weights_bytes_per_token': 63968378880, 'kv_read_bytes_per_token_at_4096': 1073741824, 'kv_write_bytes_per_token': 262144, 'compulsory_read_bytes_per_token': 65042120704, 'scope': 'logical demand, BF16; activations and norms excluded (< 1 MB); no reuse within a token is assumed or excluded'}
- qwen32b_graph_b1_c4096__n256__absent__stock__eager__FETCH_SIZE: {'counter': 'FETCH_SIZE', 'n': 256, 'grid': None, 'returncode': 0, 'decode_total': 24049204816.25, 'dispatch_rows': 247968, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [63637415.82638889, 63621186.784722224, 63617939.16269841, 63616590.66269841, 63618467.190476194, 63621810.1547619], 'per_tick': 63619826.98759921, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}
- qwen32b_graph_b1_c4096__n192__absent__g192__eager__FETCH_SIZE: {'counter': 'FETCH_SIZE', 'n': 192, 'grid': 192, 'returncode': 0, 'decode_total': 24047607627.125, 'dispatch_rows': 247968, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [63622181.53373016, 63617212.95138889, 63617148.31150793, 63617151.03571428, 63617182.508928575, 63617181.23214286], 'per_tick': 63617181.87053572, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}
- qwen32b_graph_b1_c4096__n256__absent__stock__eager__WRITE_SIZE: {'counter': 'WRITE_SIZE', 'n': 256, 'grid': None, 'returncode': 0, 'decode_total': 48761835.09375, 'dispatch_rows': 247968, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [126873.88293650794, 126873.84027777778, 139626.82539682538, 126873.63640873016, 126875.51884920635, 126873.67857142857], 'per_tick': 126873.86160714287, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}
- qwen32b_graph_b1_c4096__n192__absent__g192__eager__WRITE_SIZE: {'counter': 'WRITE_SIZE', 'n': 192, 'grid': 192, 'returncode': 0, 'decode_total': 47874779.46875, 'dispatch_rows': 247968, 'decode_ranges': 6, 'ticks_per_range': 63, 'per_tick_by_range': [126652.85714285714, 126652.8566468254, 126652.53869047618, 126652.96428571429, 126653.22222222222, 126652.69543650794], 'per_tick': 126652.85689484127, 'note': 'kernels matching ^(_|triton) excluded from collection; eager mode; per_tick = median over decode marker ranges (warmups and trials) of range total / decode iterations'}

Observational unless a G1 row says qualified. AMD acknowledgement: "This work was supported in part by Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
