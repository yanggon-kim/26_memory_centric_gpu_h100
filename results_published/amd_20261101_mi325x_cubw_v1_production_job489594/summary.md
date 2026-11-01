# cubw_v1 production — mi325x

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok: **False**
- leak_ok_v11: **False**
- clock_ok: **False**
- release_e2: **True**
- release_e3_minimal: **False**
- release_e3_full: **False**
- release_e5e8: **True**
- outputs: {'runs': 66, 'complete': 66, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {}
- leak v1.1 (relative to placement-proven native co-runners): {}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1630, 'gfx_max': 2083, 'gfx_ratio': 1.2779141104294478, 'mem_min': 1227, 'mem_max': 1500, 'samples': 1639, 'coverage_min': 0.6651267049713867}

## Placement

| probe | distinct CUs | expected | per-XCD | balanced | rc |
|---|---:|---:|---|---|---:|
| L0_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L0_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L0_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L0_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L0_n96 | 96 | 96 | {'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12} | True | 0 |
| L0_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L1_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L1_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L1_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L1_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L1_n96 | 96 | 96 | {'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12} | True | 0 |
| L1_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L2_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L2_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L2_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L2_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L2_n96 | 96 | 96 | {'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12} | True | 0 |
| L2_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L3_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L3_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L3_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L3_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L3_n96 | 96 | 96 | {'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12} | True | 0 |
| L3_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L4_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L4_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L4_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L4_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L4_n96 | 96 | 96 | {'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12} | True | 0 |
| L4_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L5_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L5_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L5_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L5_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L5_n96 | 96 | 96 | {'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12} | True | 0 |
| L5_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L6_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L6_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L6_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L6_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L6_n96 | 96 | 96 | {'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12} | True | 0 |
| L6_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L7_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L7_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L7_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L7_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L7_n96 | 96 | 96 | {'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12} | True | 0 |
| L7_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

### e2:qwen14b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 3/3 | 7.7869 | n/a | None (gfx xNone) |   |
| 232 | 3/3 | 8.824 | 1.1332 [1.0931, 1.2097] (3 blocks) | False (gfx x1.237) | observational_clock_confounded pooled clock envelope failed |
| 200 | 3/3 | 8.1491 | 1.0465 [1.0436, 1.0493] (3 blocks) | False (gfx x1.171) | observational_clock_confounded pooled clock envelope failed |
| 152 | 3/3 | 9.5144 | 1.2218 [1.1670, 1.2542] (3 blocks) | False (gfx x1.261) | observational_clock_confounded pooled clock envelope failed |
| 96 | 3/3 | 9.1077 | 1.1696 [1.1685, 1.1712] (3 blocks) | False (gfx x1.188) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

### e2:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 5/5 | 15.3394 | n/a | None (gfx xNone) |   |
| 232 | 5/5 | 16.4733 | 1.0739 [1.0710, 1.0767] (5 blocks) | False (gfx x1.226) | observational_clock_confounded pooled clock envelope failed |
| 200 | 5/5 | 16.1975 | 1.0559 [1.0400, 1.0847] (5 blocks) | False (gfx x1.211) | observational_clock_confounded pooled clock envelope failed |
| 152 | 5/5 | 18.2054 | 1.1868 [1.1653, 1.2110] (5 blocks) | False (gfx x1.273) | observational_clock_confounded pooled clock envelope failed |
| 96 | 5/5 | 18.1855 | 1.1855 [1.1747, 1.2036] (5 blocks) | False (gfx x1.278) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

## E3 T(n, z)

### e3:qwen14b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n96:absent | 0/3 | None | n/a | None | None | None |  |
| n96:arith100 | 0/3 | None | n/a | None | None | None |  |
| n96:mem100 | 0/3 | None | n/a | None | None | None |  |
| n96:mem25 | 0/3 | None | n/a | None | None | None |  |
| n96:mem50 | 0/3 | None | n/a | None | None | None |  |
| n96:mem75 | 0/3 | None | n/a | None | None | None |  |
| n152:absent | 0/3 | None | n/a | None | None | None |  |
| n152:arith100 | 0/3 | None | n/a | None | None | None |  |
| n152:mem100 | 0/3 | None | n/a | None | None | None |  |
| n152:mem25 | 0/3 | None | n/a | None | None | None |  |
| n152:mem50 | 0/3 | None | n/a | None | None | None |  |
| n152:mem75 | 0/3 | None | n/a | None | None | None |  |
| n200:absent | 0/3 | None | n/a | None | None | None |  |
| n200:arith100 | 0/3 | None | n/a | None | None | None |  |
| n200:mem100 | 0/3 | None | n/a | None | None | None |  |
| n200:mem25 | 0/3 | None | n/a | None | None | None |  |
| n200:mem50 | 0/3 | None | n/a | None | None | None |  |
| n200:mem75 | 0/3 | None | n/a | None | None | None |  |
| n232:absent | 0/3 | None | n/a | None | None | None |  |
| n232:arith100 | 0/3 | None | n/a | None | None | None |  |
| n232:mem100 | 0/3 | None | n/a | None | None | None |  |
| n232:mem25 | 0/3 | None | n/a | None | None | None |  |
| n232:mem50 | 0/3 | None | n/a | None | None | None |  |
| n232:mem75 | 0/3 | None | n/a | None | None | None |  |

### e3:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n96:absent | 0/5 | None | n/a | None | None | None |  |
| n96:arith100 | 0/5 | None | n/a | None | None | None |  |
| n96:mem100 | 0/5 | None | n/a | None | None | None |  |
| n96:mem25 | 0/5 | None | n/a | None | None | None |  |
| n96:mem50 | 0/5 | None | n/a | None | None | None |  |
| n96:mem75 | 0/5 | None | n/a | None | None | None |  |
| n152:absent | 5/5 | 17.8487 | n/a | None | None | None |  |
| n152:arith100 | 5/5 | 18.1755 | 1.0183 [1.0143, 1.0218] (5 blocks) | 0.0 | 14.8 | 1.0 | observational_clock_confounded |
| n152:mem100 | 5/5 | 21.1012 | 1.1822 [1.1640, 1.2002] (5 blocks) | 1331 | 0.0 | 1.0 | observational_clock_confounded |
| n152:mem25 | 0/5 | None | n/a | None | None | None |  |
| n152:mem50 | 5/5 | 19.4453 | 1.0895 [1.0778, 1.1044] (5 blocks) | 652 | 0.0 | 1.0 | observational_clock_confounded |
| n152:mem75 | 0/5 | None | n/a | None | None | None |  |
| n200:absent | 0/5 | None | n/a | None | None | None |  |
| n200:arith100 | 0/5 | None | n/a | None | None | None |  |
| n200:mem100 | 0/5 | None | n/a | None | None | None |  |
| n200:mem25 | 0/5 | None | n/a | None | None | None |  |
| n200:mem50 | 0/5 | None | n/a | None | None | None |  |
| n200:mem75 | 0/5 | None | n/a | None | None | None |  |
| n232:absent | 5/5 | 16.8614 | n/a | None | None | None |  |
| n232:arith100 | 5/5 | 16.9192 | 1.0034 [0.9768, 1.0244] (5 blocks) | 0.0 | 14.4 | 1.0 | observational_clock_confounded |
| n232:inactive | 1/1 | 17.0998 | 1.0209 [0.9974, 1.0473] (1 blocks) | 0.0 | 0.0 | 1.0 | unavailable |
| n232:mem100 | 5/5 | 19.3188 | 1.1457 [1.1038, 1.1862] (5 blocks) | 1175 | 0.0 | 1.0 | observational_clock_confounded |
| n232:mem25 | 0/5 | None | n/a | None | None | None |  |
| n232:mem50 | 5/5 | 17.7453 | 1.0524 [1.0156, 1.0826] (5 blocks) | 556 | 0.0 | 1.0 | observational_clock_confounded |
| n232:mem75 | 0/5 | None | n/a | None | None | None |  |
- n232: memory100 / arithmetic100: 1.1418 [1.1255, 1.1597] (5 blocks); numerator faster: False; pooled clock ok: False
- n232: memory50 / arithmetic100: 1.0488 [1.0389, 1.0577] (5 blocks); numerator faster: False; pooled clock ok: False
- n152: memory100 / arithmetic100: 1.1610 [1.1446, 1.1777] (5 blocks); numerator faster: False; pooled clock ok: False
- n152: memory50 / arithmetic100: 1.0699 [1.0558, 1.0880] (5 blocks); numerator faster: False; pooled clock ok: False
- n152 absent / n232 mem50 (shared-service balance, not stock-GPU win): 1.0058 [0.9967, 1.0154] (5 blocks); numerator faster: False; pooled clock ok: False
- n152 absent / n232 mem100 (shared-service balance, not stock-GPU win): 0.9239 [0.9112, 0.9393] (5 blocks); numerator faster: True; pooled clock ok: False

## E4 service (mean over acquisitions; ratio to 304)


## E5 / timeline

- e5:qwen32b_graph_b1_c4096__n200__absent__ktrace:a0: completed=True slowdown=1.0868036858264336 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e5:qwen32b_graph_b1_c4096__n304__absent__ktrace:a0: completed=True slowdown=1.0756173680683543 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None

## E8 guardrails

- e8:qwen7b_graph_b8_c4096__n96__absent: 3/3 ms/step=[12.3523, 12.1409, 11.8527] prefill_ms=[2377.3, 2217.9, 2152.3] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n96__absent__prefill: 3/3 ms/step=[19.6638, 19.0188, 18.8249] prefill_ms=[1428.5, 1399.6, 1369.2] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n200__absent: 3/3 ms/step=[8.6519, 7.9618, 7.8546] prefill_ms=[1378.7, 1226.0, 1199.4] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n200__absent__prefill: 3/3 ms/step=[17.3214, 17.4116, 17.3219] prefill_ms=[693.1, 696.5, 692.1] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n304__absent: 3/3 ms/step=[6.3142, 6.2367, 6.3658] prefill_ms=[839.2, 809.4, 871.8] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n304__absent__prefill: 3/3 ms/step=[16.9656, 16.883, 16.8504] prefill_ms=[481.2, 480.9, 478.9] J/token=[] W=[]
- e8:qwen32b_graph_b1_c16384__n304__absent: 3/3 ms/step=[17.0633, 16.529, 16.5811] prefill_ms=[2484.7, 2473.8, 2450.2] J/token=[] W=[]
- e8:qwen32b_graph_b1_c16384__n200__absent: 3/3 ms/step=[17.1351, 17.0075, 17.1354] prefill_ms=[3627.9, 3656.4, 3622.5] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n304__absent: 3/3 ms/step=[4.7749, 4.7703, 5.3326] prefill_ms=[120.7, 122.8, 192.2] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n200__absent: 3/3 ms/step=[5.0359, 4.9631, 5.0081] prefill_ms=[178.5, 178.0, 178.6] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n200__absent__energy: 3/3 ms/step=[16.513, 16.2055, 16.3787] prefill_ms=[717.4, 705.9, 699.6] J/token=[15.8508, 16.0145, 15.8129] W=[960.9, 988.3, 966.3]
- e8:qwen32b_graph_b1_c4096__n304__absent__energy: 3/3 ms/step=[15.7801, 15.6563, 15.4831] prefill_ms=[487.0, 491.2, 486.3] J/token=[15.3657, 15.5127, 15.2894] W=[974.4, 991.2, 987.6]
