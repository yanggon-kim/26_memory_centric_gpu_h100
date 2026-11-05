# cubw_v1 pilot — mi350x

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
| L0_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L0_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L0_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L0_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L0_competitor_224-255 | 32 | 32 | {'0': 4, '1': 4, '2': 4, '3': 4, '4': 4, '5': 4, '6': 4, '7': 4} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

## E3 T(n, z)

## E4 service (mean over acquisitions; ratio to 304)


## E6 dispatch probe (wvSplitK-shaped launches; per-workgroup CU and timing)

| step | blocks | LDS | kernel ms | CUs used | max WG/CU | CUs with >=2 | serialized WGs | WG per SE (XCD0) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| n192_g256_L0 | 256 | 163840 | 0.321 | 192 | 2 | 64 | 64.0 | [8, 8, 8, 8] |
| n192_g192_L0 | 192 | 163840 | 0.241 | 192 | 1 | 0 | 0.0 | [6, 6, 6, 6] |
| n192_g192_lds0_L0 | 192 | 1 | 0.244 | 192 | 1 | 0 | 0.0 | [6, 6, 6, 6] |

## E5 / timeline

- g2_route:qwen32b_graph_b1_c4096__n256__absent__stock__ktrace:a0: completed=True slowdown=None outputs_match=None decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- g2_route:qwen32b_graph_b1_c4096__n192__absent__g192__ktrace:a0: completed=True slowdown=None outputs_match=None decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- g2_route:qwen7b_graph_b8_c4096__n256__absent__stock__ktrace:a0: completed=True slowdown=None outputs_match=None decode_ranges=13 dispatches/tick=363.0 counter/tick=None counter coverage=None
- g2_route:qwen7b_graph_b8_c4096__n192__absent__g192__ktrace:a0: completed=True slowdown=None outputs_match=None decode_ranges=13 dispatches/tick=363.0 counter/tick=None counter coverage=None
- g7_counters:qwen32b_graph_b1_c4096__n256__absent__stock__eager__FETCH_SIZE:a0: completed=True slowdown=None outputs_match=None decode_ranges=6 dispatches/tick=733.0 counter/tick={'FETCH_SIZE': 63601722.484126985} counter coverage=0.8947393115196662
- g7_counters:qwen32b_graph_b1_c4096__n192__absent__g192__eager__FETCH_SIZE:a0: completed=True slowdown=None outputs_match=None decode_ranges=6 dispatches/tick=733.0 counter/tick={'FETCH_SIZE': 63617148.99289021} counter coverage=0.8949522510231923

## Leak test v4 (per-CU start-delay probe)

- qwen32b_graph_b1_c4096__n224__gapprobe_L0: {'expected_cus': 32, 'alone_cus': 32, 'flag_excess': 0.1, 'alone_late_max': 3.114326942042376e-05, 'launches': [96329, 97411, 98519, 86038, 23787, 16510], 'native_flags': [0, 0], 'native_max_excess': [0.00019470391416253913, 0.00036471945756624884], 'positive_flags': [8, 8], 'positive_top8_excess': [[0.8663534774795509, 0.8698519316509635, 0.8724205641156885, 0.8747646388804947, 0.880219443680532, 0.8818466316672354, 0.884194431476622, 0.8888667855527279], [0.1797309116345187, 0.18506996237652065, 0.1857846384600957, 0.1914491108588725, 0.2067624835015049, 0.20798163682054469, 0.22202291987431355, 0.22933783978855243]], 'decoder_flags': [0], 'decoder_max_excess': [0.001907075991633483], 'info_positive_flags': {'arith100': 8, 'mem100': 8}, 'sensitive': True, 'ok': True}

## E8 guardrails

