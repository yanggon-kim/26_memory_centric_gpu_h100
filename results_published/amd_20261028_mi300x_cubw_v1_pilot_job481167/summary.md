# cubw_v1 pilot — mi300x

## Gates

- placement_ok: **True**
- lifecycle_ok: **False**
- outputs_ok: **False**
- leak_ok: **False**
- clock_ok: **False**
- release_e2: **False**
- release_e3_minimal: **False**
- release_e3_full: **False**
- release_e5e8: **False**
- outputs: {'runs': 27, 'complete': 25, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {'qwen32b_graph_b1_c4096__n232__arith100': 0.6051194159183565, 'qwen32b_graph_b1_c4096__n152__arith100': 0.6144102654384086, 'qwen32b_graph_b1_c4096__n152__leakprobe': 0.5849124889107989, 'qwen32b_graph_b1_c4096__n96__leakprobe': 0.6144955976170119}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1319, 'gfx_max': 2021, 'gfx_ratio': 1.532221379833207, 'mem_min': 1059, 'mem_max': 1300, 'samples': 498, 'coverage_min': 0.3387130835317605}

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
| comma_list_0-7,16-23 | 16 | 16 | {'0': 2, '1': 2, '2': 2, '3': 2, '4': 2, '5': 2, '6': 2, '7': 2} | True | 0 |
| wrong_index_mask | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| empty_xcd_0-6 | 45 | 7 | {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 38} | False | 3 |

## E2 latency ratio R(n) = T(n)/T(304)

### e1_e2_pilot:bridge
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 1/1 | 2.2384 | n/a | None (gfx xNone) |   |
| 232 | 1/1 | 2.3406 | 1.0457 [1.0390, 1.0519] (1 blocks) | False (gfx x1.064) | pilot one pilot acquisition; never promoted to a production comparison |
| 200 | 1/1 | 2.346 | 1.0481 [1.0412, 1.0545] (1 blocks) | False (gfx x1.057) | pilot one pilot acquisition; never promoted to a production comparison |
| 152 | 1/1 | 2.5757 | 1.1507 [1.1429, 1.1580] (1 blocks) | False (gfx x1.071) | pilot one pilot acquisition; never promoted to a production comparison |
| 96 | 1/1 | 2.7268 | 1.2182 [1.2100, 1.2262] (1 blocks) | False (gfx x1.092) | pilot one pilot acquisition; never promoted to a production comparison |
Smallest qualified noninferior n: None

### e1_e2_pilot:qwen14b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 1/1 | 8.6978 | n/a | None (gfx xNone) |   |
| 232 | 1/1 | 9.0919 | 1.0453 [1.0418, 1.0491] (1 blocks) | False (gfx x1.114) | pilot one pilot acquisition; never promoted to a production comparison |
| 200 | 1/1 | 8.9078 | 1.0242 [1.0209, 1.0276] (1 blocks) | False (gfx x1.106) | pilot one pilot acquisition; never promoted to a production comparison |
| 152 | 1/1 | 9.6169 | 1.1057 [1.1012, 1.1107] (1 blocks) | False (gfx x1.165) | pilot one pilot acquisition; never promoted to a production comparison |
| 96 | 1/1 | 9.6469 | 1.1091 [1.1056, 1.1129] (1 blocks) | False (gfx x1.271) | pilot one pilot acquisition; never promoted to a production comparison |
Smallest qualified noninferior n: None

### e1_e2_pilot:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 1/1 | 17.7512 | n/a | None (gfx xNone) |   |
| 232 | 1/1 | 18.2524 | 1.0282 [1.0256, 1.0309] (1 blocks) | False (gfx x1.246) | pilot one pilot acquisition; never promoted to a production comparison |
| 200 | 1/1 | 18.015 | 1.0149 [1.0121, 1.0175] (1 blocks) | False (gfx x1.199) | pilot one pilot acquisition; never promoted to a production comparison |
| 152 | 1/1 | 19.233 | 1.0835 [1.0809, 1.0861] (1 blocks) | False (gfx x1.456) | pilot one pilot acquisition; never promoted to a production comparison |
| 96 | 1/1 | 19.4328 | 1.0947 [1.0921, 1.0974] (1 blocks) | False (gfx x1.381) | pilot one pilot acquisition; never promoted to a production comparison |
Smallest qualified noninferior n: None

### e1_e2_pilot:qwen32b:gpu7
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 1/1 | 17.4089 | n/a | None (gfx xNone) |   |
| 232 | 1/1 | 17.9997 | 1.0339 [1.0327, 1.0353] (1 blocks) | False (gfx x1.296) | pilot one pilot acquisition; never promoted to a production comparison |
| 200 | 1/1 | 17.805 | 1.0227 [1.0218, 1.0238] (1 blocks) | False (gfx x1.343) | pilot one pilot acquisition; never promoted to a production comparison |
| 152 | 1/1 | 18.9312 | 1.0874 [1.0863, 1.0887] (1 blocks) | False (gfx x1.46) | pilot one pilot acquisition; never promoted to a production comparison |
| 96 | 1/1 | 19.1906 | 1.1023 [1.1009, 1.1038] (1 blocks) | False (gfx x1.529) | pilot one pilot acquisition; never promoted to a production comparison |
Smallest qualified noninferior n: None

## E3 T(n, z)

### e1_e3_pilot:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n152:absent | 1/1 | 19.5469 | n/a | None | None | None |  |
| n152:arith100 | 1/1 | 20.7672 | 1.0624 [1.0596, 1.0655] (1 acq) | 0.0 | 10.7 | 1.0 | pilot |
| n152:mem100 | 1/1 | 25.5235 | 1.3058 [1.3038, 1.3078] (1 acq) | 1188 | 0.0 | 1.0 | pilot |
| n152:mem50 | 1/1 | 22.0196 | 1.1265 [1.1247, 1.1284] (1 acq) | 601 | 0.0 | 1.0 | pilot |
| n232:absent | 1/1 | 18.4946 | n/a | None | None | None |  |
| n232:arith100 | 1/1 | 19.6336 | 1.0616 [1.0585, 1.0646] (1 acq) | 0.0 | 10.5 | 1.0 | pilot |
| n232:inactive | 1/1 | 18.4543 | 0.9978 [0.9956, 1.0001] (1 acq) | 0.0 | 0.0 | 1.0 | pilot |
| n232:mem100 | 1/1 | 23.4677 | 1.2689 [1.2666, 1.2712] (1 acq) | 1050 | 0.0 | 1.0 | pilot |
| n232:mem50 | 1/1 | 20.7535 | 1.1221 [1.1201, 1.1242] (1 acq) | 515 | 0.0 | 1.0 | pilot |
- n232: memory100 / arithmetic100: 1.1953 [1.1925, 1.1981] (1 acq); numerator faster: False; pooled clock ok: False
- n232: memory50 / arithmetic100: 1.0570 [1.0546, 1.0595] (1 acq); numerator faster: False; pooled clock ok: False
- n152: memory100 / arithmetic100: 1.2290 [1.2255, 1.2323] (1 acq); numerator faster: False; pooled clock ok: False
- n152: memory50 / arithmetic100: 1.0603 [1.0572, 1.0632] (1 acq); numerator faster: False; pooled clock ok: False
- n152 absent / n232 mem50 (shared-service balance, not stock-GPU win): 0.9419 [0.9407, 0.9430] (1 acq); numerator faster: True; pooled clock ok: False
- n152 absent / n232 mem100 (shared-service balance, not stock-GPU win): 0.8329 [0.8319, 0.8339] (1 acq); numerator faster: True; pooled clock ok: False

### e1_leak:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n96:leakprobe | 1/1 | 25.8061 | n/a | 0.0 | 45.3 | 1.0 |  |
| n152:leakprobe | 1/1 | 23.4376 | n/a | 0.0 | 33.8 | 1.0 |  |

## E4 service (mean over acquisitions; ratio to 304)

### e4_pilot
- chase_c1 [ns_per_dependent_load]: 152: 147.4 (1.014), 200: 150.1 (1.033), 232: 149.7 (1.030), 304: 145.3, 96: 144.2 (0.993)
- chase_c19456 [ns_per_dependent_load]: 152: 740.9 (0.982), 200: 741.5 (0.983), 232: 746.2 (0.990), 304: 754.1, 96: 786.6 (1.043)
- chase_c304 [ns_per_dependent_load]: 152: 705.6 (0.964), 200: 712.3 (0.973), 232: 708.8 (0.968), 304: 732.3, 96: 692.4 (0.946)
- chase_c4864 [ns_per_dependent_load]: 152: 665.7 (0.955), 200: 677.4 (0.972), 232: 686.1 (0.984), 304: 697.0, 96: 657.6 (0.943)
- fma [useful_tflop_per_s]: 152: 61.6 (0.639), 200: 71.7 (0.743), 232: 77.8 (0.806), 304: 96.5, 96: 44.8 (0.464)
- gather1g [useful_gb_per_s]: 152: 452.5 (0.730), 200: 575.0 (0.927), 232: 607.8 (0.980), 304: 620.1, 96: 311.9 (0.503)
- grid1216 [useful_gb_per_s]: 152: 3480.2 (0.785), 200: 4421.1 (0.997), 232: 4416.5 (0.996), 304: 4433.3, 96: 4083.9 (0.921)
- grid304 [useful_gb_per_s]: 152: 2861.7 (0.998), 200: 2892.2 (1.008), 232: 2877.3 (1.003), 304: 2868.0, 96: 2812.4 (0.981)
- grid9728 [useful_gb_per_s]: 152: 4386.0 (1.014), 200: 4399.1 (1.017), 232: 4385.6 (1.014), 304: 4325.4, 96: 4257.0 (0.984)
- ilp1 [useful_gb_per_s]: 152: 3151.8 (0.715), 200: 3874.3 (0.879), 232: 3706.8 (0.841), 304: 4407.0, 96: 2701.4 (0.613)
- ilp16 [useful_gb_per_s]: 152: 3933.5 (0.925), 200: 3995.9 (0.940), 232: 4021.4 (0.946), 304: 4252.8, 96: 2994.8 (0.704)
- ilp2 [useful_gb_per_s]: 152: 3913.2 (0.909), 200: 3993.7 (0.928), 232: 3924.7 (0.912), 304: 4304.9, 96: 2933.5 (0.681)
- ilp4 [useful_gb_per_s]: 152: 3783.6 (0.891), 200: 3941.2 (0.928), 232: 3960.7 (0.932), 304: 4248.3, 96: 2911.7 (0.685)
- ilp8 [useful_gb_per_s]: 152: 3948.6 (0.931), 200: 3983.6 (0.939), 232: 3983.7 (0.939), 304: 4241.5, 96: 2991.9 (0.705)
- mfma [useful_tflop_per_s]: 152: 567.3 (0.525), 200: 711.1 (0.658), 232: 811.2 (0.751), 304: 1080.8, 96: 380.8 (0.352)
- reuse128m [useful_gb_per_s]: 152: 5666.9 (0.992), 200: 5820.2 (1.018), 232: 6339.6 (1.109), 304: 5714.9, 96: 6451.1 (1.129)
- reuse32m [useful_gb_per_s]: 152: 15368.3 (0.627), 200: 19943.3 (0.814), 232: 22467.5 (0.917), 304: 24505.2, 96: 11184.1 (0.456)
- reuse512m [useful_gb_per_s]: 152: 4965.8 (1.090), 200: 5007.7 (1.100), 232: 4648.8 (1.021), 304: 4553.9, 96: 4933.4 (1.083)
- reuse8m [useful_gb_per_s]: 152: 16640.1 (0.543), 200: 20339.8 (0.664), 232: 22284.5 (0.727), 304: 30653.9, 96: 11215.9 (0.366)
- seqread4g [useful_gb_per_s]: 152: 3995.1 (0.880), 200: 4375.1 (0.964), 232: 4315.3 (0.951), 304: 4538.0, 96: 3961.9 (0.873)
- stride16 [useful_gb_per_s]: 152: 462.7 (0.860), 200: 513.6 (0.955), 232: 497.4 (0.925), 304: 538.0, 96: 369.2 (0.686)
- stride4 [useful_gb_per_s]: 152: 1060.4 (0.983), 200: 1041.7 (0.966), 232: 1059.5 (0.982), 304: 1078.9, 96: 1047.3 (0.971)
- comp_fma [useful_tflop_per_s]: competitor: 17.6
- comp_read1g [useful_gb_per_s]: competitor: 2275.7
- comp_seqread4g [useful_gb_per_s]: competitor: 2290.3

## E5 / timeline

- e1_timeline:qwen32b_graph_b1_c4096__n304__absent__ktrace:a0: completed=True slowdown=1.075791893340378 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e1_timeline:qwen32b_graph_b1_c4096__n152__absent__ktrace:a0: completed=True slowdown=1.0989794487131685 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e5_pilot:qwen32b_graph_b1_c4096__n304__absent__pmc_FETCH_SIZE:a0: completed=False slowdown=3.871293960475883 outputs_match=True decode_ranges=0 dispatches/tick=None counter/tick=None counter coverage=None
- e5_pilot:qwen32b_graph_b1_c4096__n304__absent__pmc_FETCH_SIZE_excl:a0: completed=False slowdown=None outputs_match=None decode_ranges=0 dispatches/tick=None counter/tick=None counter coverage=None

## E8 guardrails

- e8_pilot:qwen7b_graph_b8_c4096__n304__absent: 1/1 ms/step=[6.7661] prefill_ms=[952.9] J/token=[] W=[]
- e8_pilot:qwen32b_graph_b1_c4096__n304__absent__prefill: 1/1 ms/step=[19.2095] prefill_ms=[559.1] J/token=[] W=[]
- e8_pilot:qwen7b_graph_b8_c4096__n152__absent: 1/1 ms/step=[9.7456] prefill_ms=[1563.0] J/token=[] W=[]
- e8_pilot:qwen32b_graph_b1_c4096__n152__absent__prefill: 1/1 ms/step=[20.6202] prefill_ms=[976.4] J/token=[] W=[]
- e8_pilot:qwen32b_graph_b1_c16384__n304__absent: 1/1 ms/step=[19.7393] prefill_ms=[3128.8] J/token=[] W=[]
- e8_pilot:qwen32b_graph_b1_c4096__n304__absent__energy: 1/1 ms/step=[18.7095] prefill_ms=[611.2] J/token=[14.075] W=[752.3]
