# cubw_v1 pilot — mi350x

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok: **False**
- leak_ok_v11: **False**
- leak_ok_v2: **False**
- clock_ok: **False**
- release_e2: **True**
- release_e3_minimal: **False**
- release_e3_minimal_v2: **False**
- release_e3_full: **False**
- release_e5e8: **True**
- outputs: {'runs': 5, 'complete': 5, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {}
- leak v1.1 (relative to placement-proven native co-runners): {}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1547, 'gfx_max': 2202, 'gfx_ratio': 1.4234001292824823, 'mem_min': 2000, 'mem_max': 2000, 'samples': 502, 'coverage_min': 0.9457354955681602}

## Placement

| probe | distinct CUs | expected | per-XCD | balanced | rc |
|---|---:|---:|---|---|---:|
| L0_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L0_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L0_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L0_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

### e1_e2_pilot:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 256 | 1/1 | 12.2297 | n/a | None (gfx xNone) |   |
| 192 | 1/1 | 14.7247 | 1.2040 [1.2030, 1.2051] (1 blocks) | False (gfx x1.423) | pilot one pilot acquisition; never promoted to a production comparison |
| 128 | 1/1 | 14.508 | 1.1863 [1.1851, 1.1875] (1 blocks) | False (gfx x1.423) | pilot one pilot acquisition; never promoted to a production comparison |
Smallest qualified noninferior n: None

### e6_decode:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 128 | 1/1 | 15.4029 | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

## E3 T(n, z)

## E4 service (mean over acquisitions; ratio to 304)

### e4_pilot
- chase_c256 [ns_per_dependent_load]: 128: 543.6 (0.991), 256: 548.5
- fma [useful_tflop_per_s]: 128: 64.7 (0.631), 256: 102.5
- ilp1 [useful_gb_per_s]: 128: 4419.4 (0.680), 256: 6499.1
- ilp8 [useful_gb_per_s]: 128: 4291.8 (0.757), 256: 5666.3
- seqread4g [useful_gb_per_s]: 128: 5728.0 (0.891), 256: 6429.6

## E5 / timeline

- e6_timeline:qwen32b_graph_b1_c4096__n256__absent__ktrace:a0: completed=True slowdown=1.0911197056297361 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None

## E6 kernel-shape controls (median us per call; ratio to 304)

- down:linear: 128: 78.3 (1.549), 256: 50.6 (1.000)
- down:wvsplitk_dev: 128: 67.0 (1.439), 256: 46.6 (1.000)
- down:wvsplitk_n: 128: 65.6 (1.412), 256: 46.5 (1.000)
- gate_up:linear: 128: 125.3 (1.484), 256: 84.5 (1.000)
- gate_up:llmm1: 128: 135.7 (1.311), 256: 103.5 (1.000)
- gate_up:wvsplitk_dev: 128: 121.7 (1.363), 256: 89.3 (1.000)
- gate_up:wvsplitk_n: 128: 103.3 (1.232), 256: 83.8 (1.000)
- lm_head:linear: 128: 324.1 (1.355), 256: 239.2 (1.000)
- lm_head:llmm1: 128: 370.1 (1.271), 256: 291.1 (1.000)
- lm_head:wvsplitk_dev: 128: 308.5 (1.280), 256: 241.0 (1.000)
- lm_head:wvsplitk_n: 128: 272.7 (1.178), 256: 231.5 (1.000)
- o_proj:linear: 128: 21.7 (1.379), 256: 15.7 (1.000)
- o_proj:llmm1: 128: 15.7 (1.346), 256: 11.7 (1.000)
- o_proj:wvsplitk_dev: 128: 12.7 (1.292), 256: 9.8 (1.000)
- o_proj:wvsplitk_n: 128: 12.0 (1.163), 256: 10.3 (1.000)
- qkv:linear: 128: 25.9 (1.529), 256: 16.9 (1.000)
- qkv:llmm1: 128: 21.0 (1.339), 256: 15.7 (1.000)
- qkv:wvsplitk_dev: 128: 17.0 (1.295), 256: 13.1 (1.000)
- qkv:wvsplitk_n: 128: 15.6 (1.148), 256: 13.6 (1.000)

## Leak test v2 (per-CU probe)

- qwen32b_graph_b1_c4096__n192__cuprobe_L0: {'expected_cus': 64, 'alone_cus': 64, 'covered_cus': [64, 64, 64, 64, 64], 'native_min_q': [1.0, 1.0], 'positive_cus_below': [0], 'decoder_min_q': [1.0], 'decoder_cus_below': [0], 'sensitive': False, 'threshold': 0.95, 'ok': False}
- qwen32b_graph_b1_c4096__n128__cuprobe_L0: {'expected_cus': 128, 'alone_cus': 128, 'covered_cus': [128, 128, 128, 128, 128], 'native_min_q': [1.0, 1.0], 'positive_cus_below': [0], 'decoder_min_q': [1.0], 'decoder_cus_below': [0], 'sensitive': False, 'threshold': 0.95, 'ok': False}

## E8 guardrails

