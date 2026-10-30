# cubw_v1 auto — mi300x

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **False**
- leak_ok: **False**
- leak_ok_v11: **False**
- clock_ok: **False**
- release_e2: **False**
- release_e3_minimal: **False**
- release_e3_full: **False**
- release_e5e8: **False**
- outputs: {'runs': 164, 'complete': 18, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {'qwen32b_graph_b1_c4096__n232__arith100_L0': 0.6134239190414151, 'qwen32b_graph_b1_c4096__n232__arith100_L1': 0.6074255676792297, 'qwen32b_graph_b1_c4096__n232__arith100': 0.6118072036686908}
- leak v1.1 (relative to placement-proven native co-runners): {'qwen32b_graph_b1_c4096__n232__arith100_L0': {'decoder_ratio': 0.614325027653671, 'native_ratios': {'arith100': 0.7487547781676919, 'mem100': 0.5916548094181086}, 'relative': 0.8204622468747521}, 'qwen32b_graph_b1_c4096__n152__leakprobe_L0': {'decoder_ratio': 0.6215657018073409, 'native_ratios': {'mem100': 0.6497325737372103, 'arith100': 0.8113221005180311}, 'relative': 0.7661145942042867}, 'qwen32b_graph_b1_c4096__n232__arith100_L1': {'decoder_ratio': 0.6085701142505496, 'native_ratios': {'arith100': 0.7615823728788071, 'mem100': 0.5900305321714111}, 'relative': 0.7990863968530862}, 'qwen32b_graph_b1_c4096__n152__leakprobe_L1': {'decoder_ratio': 0.6666166434654347, 'native_ratios': {'arith100': 0.8703847187066464, 'mem100': 0.7014318799017402}, 'relative': 0.7658873474433212}, 'qwen32b_graph_b1_c4096__n232__arith100': {'decoder_ratio': 0.6118072036686908, 'native_ratios': {'arith100': 0.7601500539508071, 'mem100': 0.5889208532588958}, 'relative': 0.804850569290736}}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1300, 'gfx_max': 1964, 'gfx_ratio': 1.5107692307692309, 'mem_min': 1049, 'mem_max': 1300, 'samples': 518, 'coverage_min': 0.6405621450384805}

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
| 304 | 0/3 | None | n/a | None (gfx xNone) |   |
| 232 | 0/3 | None | n/a | None (gfx xNone) |   |
| 200 | 0/3 | None | n/a | None (gfx xNone) |   |
| 152 | 0/3 | None | n/a | None (gfx xNone) |   |
| 96 | 0/3 | None | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

### e2:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 0/5 | None | n/a | None (gfx xNone) |   |
| 232 | 0/5 | None | n/a | None (gfx xNone) |   |
| 200 | 0/5 | None | n/a | None (gfx xNone) |   |
| 152 | 0/5 | None | n/a | None (gfx xNone) |   |
| 96 | 0/5 | None | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

### e1_e2_pilot:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 2/2 | 17.9258 | n/a | None (gfx xNone) |   |
| 232 | 2/2 | 18.3252 | 1.0223 [1.0204, 1.0242] (2 blocks) | False (gfx x1.29) | pilot one pilot acquisition; never promoted to a production comparison |
| 200 | 2/2 | 18.0941 | 1.0094 [1.0072, 1.0118] (2 blocks) | False (gfx x1.312) | pilot one pilot acquisition; never promoted to a production comparison |
| 152 | 2/2 | 19.0647 | 1.0635 [1.0604, 1.0668] (2 blocks) | False (gfx x1.457) | pilot one pilot acquisition; never promoted to a production comparison |
| 96 | 2/2 | 19.3282 | 1.0782 [1.0741, 1.0824] (2 blocks) | False (gfx x1.511) | pilot one pilot acquisition; never promoted to a production comparison |
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
| n152:absent | 0/5 | None | n/a | None | None | None |  |
| n152:arith100 | 0/5 | None | n/a | None | None | None |  |
| n152:mem100 | 0/5 | None | n/a | None | None | None |  |
| n152:mem25 | 0/5 | None | n/a | None | None | None |  |
| n152:mem50 | 0/5 | None | n/a | None | None | None |  |
| n152:mem75 | 0/5 | None | n/a | None | None | None |  |
| n200:absent | 0/5 | None | n/a | None | None | None |  |
| n200:arith100 | 0/5 | None | n/a | None | None | None |  |
| n200:mem100 | 0/5 | None | n/a | None | None | None |  |
| n200:mem25 | 0/5 | None | n/a | None | None | None |  |
| n200:mem50 | 0/5 | None | n/a | None | None | None |  |
| n200:mem75 | 0/5 | None | n/a | None | None | None |  |
| n232:absent | 0/5 | None | n/a | None | None | None |  |
| n232:arith100 | 0/5 | None | n/a | None | None | None |  |
| n232:inactive | 0/1 | None | n/a | None | None | None |  |
| n232:mem100 | 0/5 | None | n/a | None | None | None |  |
| n232:mem25 | 0/5 | None | n/a | None | None | None |  |
| n232:mem50 | 0/5 | None | n/a | None | None | None |  |
| n232:mem75 | 0/5 | None | n/a | None | None | None |  |

### e1_e3_pilot:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n232:absent | 1/1 | 18.271 | n/a | None | None | None |  |
| n232:arith100 | 1/1 | 19.4736 | 1.0658 [1.0634, 1.0681] (1 acq) | 0.0 | 10.7 | 1.0 | pilot |
| n232:inactive | 1/1 | 18.1568 | 0.9937 [0.9914, 0.9960] (1 acq) | 0.0 | 0.0 | 1.0 | pilot |
| n232:mem100 | 1/1 | 22.499 | 1.2314 [1.2287, 1.2339] (1 acq) | 1043 | 0.0 | 1.0 | pilot |
- n232: memory100 / arithmetic100: 1.1554 [1.1535, 1.1571] (1 acq); numerator faster: False; pooled clock ok: False

### e1_leak:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n152:leakprobe | 2/2 | 22.7775 | n/a | 0.0 | 34.7 | 1.0 |  |
| n232:arith100 | 2/2 | 19.482 | n/a | 0.0 | 10.7 | 1.0 |  |

## E4 service (mean over acquisitions; ratio to 304)

### e4
- chase_c1 [ns_per_dependent_load]: 152: 147.3 (0.947), 200: 152.2 (0.978), 232: 150.2 (0.966), 304: 155.6, 96: 147.7 (0.950)
- chase_c19456 [ns_per_dependent_load]: 152: 741.0 (0.977), 200: 743.1 (0.979), 232: 746.1 (0.983), 304: 758.7, 96: 786.2 (1.036)
- chase_c304 [ns_per_dependent_load]: 152: 657.6 (0.981), 200: 642.5 (0.958), 232: 636.6 (0.949), 304: 670.5, 96: 624.0 (0.931)
- chase_c4864 [ns_per_dependent_load]: 152: 653.2 (0.944), 200: 664.0 (0.960), 232: 682.0 (0.986), 304: 692.0, 96: 648.7 (0.937)
- fma [useful_tflop_per_s]: 152: 60.3 (0.647), 200: 71.1 (0.763), 232: 76.4 (0.820), 304: 93.2, 96: 43.8 (0.470)
- gather1g [useful_gb_per_s]: 152: 452.0 (0.745), 200: 560.8 (0.924), 232: 604.8 (0.997), 304: 606.6, 96: 311.7 (0.514)
- grid1216 [useful_gb_per_s]: 152: 3491.0 (0.789), 200: 4412.6 (0.997), 232: 4409.7 (0.996), 304: 4425.8, 96: 4050.8 (0.915)
- grid304 [useful_gb_per_s]: 152: 2875.0 (1.000), 200: 2896.3 (1.008), 232: 2887.5 (1.005), 304: 2873.8, 96: 2826.6 (0.984)
- grid9728 [useful_gb_per_s]: 152: 4352.6 (1.006), 200: 4386.4 (1.014), 232: 4373.5 (1.011), 304: 4325.5, 96: 4212.0 (0.974)
- ilp1 [useful_gb_per_s]: 152: 3150.5 (0.718), 200: 3859.9 (0.880), 232: 3685.5 (0.840), 304: 4385.7, 96: 2518.9 (0.574)
- ilp16 [useful_gb_per_s]: 152: 3920.5 (0.923), 200: 3991.3 (0.940), 232: 4013.9 (0.945), 304: 4247.0, 96: 2990.4 (0.704)
- ilp2 [useful_gb_per_s]: 152: 3909.1 (0.951), 200: 3982.2 (0.969), 232: 3929.6 (0.956), 304: 4110.1, 96: 2937.6 (0.715)
- ilp4 [useful_gb_per_s]: 152: 3786.2 (0.892), 200: 3937.7 (0.928), 232: 3960.3 (0.933), 304: 4242.9, 96: 2861.4 (0.674)
- ilp8 [useful_gb_per_s]: 152: 3928.3 (0.971), 200: 3970.2 (0.981), 232: 3982.2 (0.984), 304: 4046.5, 96: 2918.6 (0.721)
- mfma [useful_tflop_per_s]: 152: 564.8 (0.540), 200: 699.6 (0.669), 232: 795.0 (0.761), 304: 1045.3, 96: 377.1 (0.361)
- reuse128m [useful_gb_per_s]: 152: 5714.8 (1.004), 200: 5799.3 (1.019), 232: 6352.6 (1.116), 304: 5692.4, 96: 6459.4 (1.135)
- reuse32m [useful_gb_per_s]: 152: 15381.8 (0.647), 200: 19027.2 (0.801), 232: 21538.9 (0.906), 304: 23765.1, 96: 11274.9 (0.474)
- reuse512m [useful_gb_per_s]: 152: 4987.2 (1.096), 200: 5000.6 (1.099), 232: 4654.4 (1.023), 304: 4550.2, 96: 4988.0 (1.096)
- reuse8m [useful_gb_per_s]: 152: 16639.7 (0.567), 200: 20208.7 (0.689), 232: 22328.9 (0.761), 304: 29351.3, 96: 11220.5 (0.382)
- seqread4g [useful_gb_per_s]: 152: 3990.4 (0.874), 200: 4367.0 (0.957), 232: 4314.4 (0.945), 304: 4563.5, 96: 3941.0 (0.864)
- stride16 [useful_gb_per_s]: 152: 462.2 (0.863), 200: 507.7 (0.948), 232: 497.2 (0.928), 304: 535.6, 96: 368.9 (0.689)
- stride4 [useful_gb_per_s]: 152: 1063.8 (0.986), 200: 1037.4 (0.962), 232: 1058.3 (0.981), 304: 1078.9, 96: 1040.1 (0.964)
- comp_fma [useful_tflop_per_s]: competitor: 17.6
- comp_read1g [useful_gb_per_s]: competitor: 2308.9
- comp_seqread4g [useful_gb_per_s]: competitor: 2320.4

## E5 / timeline

- e5:qwen32b_graph_b1_c4096__n200__absent__ktrace:a0: completed=None slowdown=None outputs_match=None decode_ranges=None dispatches/tick=None counter/tick=None counter coverage=None
- e5:qwen32b_graph_b1_c4096__n304__absent__ktrace:a0: completed=None slowdown=None outputs_match=None decode_ranges=None dispatches/tick=None counter/tick=None counter coverage=None

## E8 guardrails

- e8:qwen7b_graph_b8_c4096__n96__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n96__absent__prefill: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n200__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n200__absent__prefill: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n304__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n304__absent__prefill: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c16384__n304__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c16384__n200__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n304__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n200__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n200__absent__energy: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n304__absent__energy: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
