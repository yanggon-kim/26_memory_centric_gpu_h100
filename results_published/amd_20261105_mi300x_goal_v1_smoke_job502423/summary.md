# cubw_v1 pilot — mi300x

## Gates

- placement_ok: **False**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok: **False**
- leak_ok_v11: **False**
- leak_ok_v2: **False**
- leak_ok_v3: **False**
- leak_ok_v4: **True**
- clock_ok: **False**
- release_e2: **False**
- release_e3_minimal: **False**
- release_e3_minimal_v2: **False**
- release_e3_minimal_v3: **False**
- release_e3_minimal_v4: **False**
- release_e3_full: **False**
- release_e5e8: **True**
- outputs: {'runs': 5, 'complete': 5, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {}
- leak v1.1 (relative to placement-proven native co-runners): {}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': None, 'gfx_max': None, 'gfx_ratio': None, 'mem_min': None, 'mem_max': None, 'samples': 0, 'coverage_min': None}

## Placement

| probe | distinct CUs | expected | per-XCD | balanced | rc |
|---|---:|---:|---|---|---:|
| L0_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L0_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L0_n96 | 96 | 96 | {'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12} | True | 0 |
| L0_n264 | 264 | 264 | {'0': 33, '1': 33, '2': 33, '3': 33, '4': 33, '5': 33, '6': 33, '7': 33} | True | 0 |
| L0_competitor_256-303 | 48 | 48 | {'0': 6, '1': 6, '2': 6, '3': 6, '4': 6, '5': 6, '6': 6, '7': 6} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

## E3 T(n, z)

## E4 service (mean over acquisitions; ratio to 304)


## E6 dispatch probe (wvSplitK-shaped launches; per-workgroup CU and timing)

| step | blocks | LDS | kernel ms | CUs used | max WG/CU | CUs with >=2 | serialized WGs | WG per SE (XCD0) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| n128_g304_L0 | 304 | 65536 | 0.403 | 128 | 3 | 128 | 176.0 | [9, 10, 10, 9] |
| n128_g128_L0 | 128 | 65536 | 0.374 | 128 | 1 | 0 | 0.0 | [4, 4, 4, 4] |
| n128_g128_lds0_L0 | 128 | 1 | 0.371 | 128 | 1 | 0 | 0.0 | [4, 4, 4, 4] |

## E5 / timeline

- g2_route:qwen32b_graph_b1_c4096__n304__absent__stock__ktrace:a0: completed=True slowdown=None outputs_match=None decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- g2_route:qwen32b_graph_b1_c4096__n128__absent__g128__ktrace:a0: completed=True slowdown=None outputs_match=None decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- g2_route:qwen7b_graph_b8_c4096__n304__absent__stock__ktrace:a0: completed=True slowdown=None outputs_match=None decode_ranges=13 dispatches/tick=363.0 counter/tick=None counter coverage=None
- g2_route:qwen7b_graph_b8_c4096__n128__absent__g128__ktrace:a0: completed=True slowdown=None outputs_match=None decode_ranges=13 dispatches/tick=363.0 counter/tick=None counter coverage=None
- g7_counters:qwen32b_graph_b1_c4096__n304__absent__stock__eager__FETCH_SIZE:a0: completed=True slowdown=None outputs_match=None decode_ranges=6 dispatches/tick=733.0 counter/tick={'FETCH_SIZE': 63538896.19593254} counter coverage=0.8938586803525412
- g7_counters:qwen32b_graph_b1_c4096__n128__absent__g128__eager__FETCH_SIZE:a0: completed=True slowdown=None outputs_match=None decode_ranges=6 dispatches/tick=733.0 counter/tick={'FETCH_SIZE': 63548483.542328045} counter coverage=0.8939489089557302

## Leak test v4 (per-CU start-delay probe)

- qwen32b_graph_b1_c4096__n256__gapprobe_L0: {'expected_cus': 48, 'alone_cus': 48, 'flag_excess': 0.1, 'alone_late_max': 0.25001373822634004, 'launches': [90987, 87930, 89018, 85455, 26669, 25353], 'native_flags': [0, 0], 'native_max_excess': [0.00012765610537071592, 0.00012357051382866387], 'positive_flags': [48, 48], 'positive_top8_excess': [[0.6814463752852379, 0.6841963606576561, 0.6853782692645252, 0.6886789631957696, 0.6933122696155871, 0.7015739277982564, 0.7083384330922063, 0.7309936317347667], [0.6443061232142188, 0.6507445683075253, 0.6541192730208628, 0.663991900708688, 0.6664666841651355, 0.669766395440399, 0.6864524354119015, 0.692890880505208]], 'decoder_flags': [0], 'decoder_max_excess': [0.0020115962607975386], 'info_positive_flags': {'arith100': 32, 'mem100': 48}, 'sensitive': True, 'ok': True}

## E8 guardrails

