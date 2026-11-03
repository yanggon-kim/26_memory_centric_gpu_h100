# cubw_v1 pilot — mi350x

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok: **False**
- leak_ok_v11: **False**
- leak_ok_v2: **False**
- leak_ok_v3: **False**
- clock_ok: **False**
- release_e2: **True**
- release_e3_minimal: **False**
- release_e3_minimal_v2: **False**
- release_e3_minimal_v3: **False**
- release_e3_full: **False**
- release_e5e8: **True**
- outputs: {'runs': 4, 'complete': 4, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {}
- leak v1.1 (relative to placement-proven native co-runners): {}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1659, 'gfx_max': 2189, 'gfx_ratio': 1.319469559975889, 'mem_min': 2000, 'mem_max': 2000, 'samples': 328, 'coverage_min': 0.9419938891484908}

## Placement

| probe | distinct CUs | expected | per-XCD | balanced | rc |
|---|---:|---:|---|---|---:|
| L0_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L0_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L0_n208 | 208 | 208 | {'0': 26, '1': 26, '2': 26, '3': 26, '4': 26, '5': 26, '6': 26, '7': 26} | True | 0 |
| L0_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L0_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L0_n168 | 168 | 168 | {'0': 21, '1': 21, '2': 21, '3': 21, '4': 21, '5': 21, '6': 21, '7': 21} | True | 0 |
| L0_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L0_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L0_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

### e1_e2_pilot:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 256 | 1/1 | 12.48 | n/a | None (gfx xNone) |   |
| 232 | 0/0 | None | n/a | None (gfx xNone) |   |
| 208 | 1/1 | 15.1482 | 1.2138 [1.2118, 1.2157] (1 blocks) | False (gfx x1.319) | pilot one pilot acquisition; never promoted to a production comparison |
| 192 | 0/0 | None | n/a | None (gfx xNone) |   |
| 184 | 0/0 | None | n/a | None (gfx xNone) |   |
| 168 | 0/0 | None | n/a | None (gfx xNone) |   |
| 152 | 0/0 | None | n/a | None (gfx xNone) |   |
| 128 | 0/0 | None | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

### e6_decode:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 192 | 1/1 | 12.6555 | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

### e6_balanced:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 208 | 1/1 | 12.5953 | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

## E3 T(n, z)

## E4 service (mean over acquisitions; ratio to 304)


## E6 dispatch probe (wvSplitK-shaped launches; per-workgroup CU and timing)

| step | blocks | LDS | kernel ms | CUs used | max WG/CU | CUs with >=2 | serialized WGs | WG per SE (XCD0) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| n256_g256 | 256 | 163840 | 0.198 | 256 | 1 | 0 | 0.0 | [8, 8, 8, 8] |
| n256_g256_lds0 | 256 | 1 | 0.198 | 256 | 1 | 0 | 0.0 | [8, 8, 8, 8] |
| n232_g256 | 256 | 163840 | 0.322 | 232 | 2 | 24 | 24.0 | [8, 8, 8, 8] |
| n232_g232 | 232 | 163840 | 0.346 | 232 | 2 | 8 | 6.4 | [7, 7, 8, 7] |
| n232_g224 | 224 | 163840 | 0.205 | 224 | 1 | 0 | 0.0 | [7, 7, 7, 7] |
| n232_g232_lds0 | 232 | 1 | 0.287 | 232 | 2 | 8 | 0.0 | [7, 7, 8, 7] |
| n208_g256 | 256 | 163840 | 0.316 | 208 | 2 | 48 | 48.0 | [8, 8, 8, 8] |
| n208_g208 | 208 | 163840 | 0.372 | 200 | 2 | 8 | 8.0 | [6, 7, 6, 7] |
| n208_g192 | 192 | 163840 | 0.239 | 192 | 1 | 0 | 0.0 | [6, 6, 6, 6] |
| n208_g208_lds0 | 208 | 1 | 0.314 | 200 | 2 | 8 | 0.0 | [6, 7, 6, 7] |
| n192_g256 | 256 | 163840 | 0.314 | 192 | 2 | 64 | 64.0 | [8, 8, 8, 8] |
| n192_g192 | 192 | 163840 | 0.234 | 192 | 1 | 0 | 0.0 | [6, 6, 6, 6] |
| n192_g192_lds0 | 192 | 1 | 0.236 | 192 | 1 | 0 | 0.0 | [6, 6, 6, 6] |
| n184_g256 | 256 | 163840 | 0.319 | 184 | 2 | 72 | 72.0 | [8, 8, 8, 8] |
| n184_g184 | 184 | 163840 | 0.409 | 184 | 2 | 8 | 4.8 | [6, 6, 6, 5] |
| n184_g160 | 160 | 163840 | 0.264 | 160 | 1 | 0 | 0.0 | [5, 5, 5, 5] |
| n184_g184_lds0 | 184 | 1 | 0.349 | 184 | 2 | 8 | 0.0 | [6, 6, 6, 5] |
| n168_g256 | 256 | 163840 | 0.314 | 168 | 2 | 88 | 88.0 | [8, 8, 8, 8] |
| n168_g168 | 168 | 163840 | 0.449 | 168 | 2 | 8 | 6.4 | [5, 5, 6, 5] |
| n168_g160 | 160 | 163840 | 0.264 | 160 | 1 | 0 | 0.0 | [5, 5, 5, 5] |
| n168_g168_lds0 | 168 | 1 | 0.379 | 168 | 2 | 8 | 0.0 | [5, 5, 6, 5] |
| n152_g256 | 256 | 163840 | 0.314 | 152 | 2 | 104 | 104.0 | [8, 8, 8, 8] |
| n152_g152 | 152 | 163840 | 0.484 | 152 | 2 | 8 | 4.8 | [5, 5, 5, 4] |
| n152_g128 | 128 | 163840 | 0.313 | 128 | 1 | 0 | 0.0 | [4, 4, 4, 4] |
| n152_g152_lds0 | 152 | 1 | 0.419 | 152 | 2 | 8 | 0.0 | [5, 5, 5, 4] |
| n128_g256 | 256 | 163840 | 0.317 | 128 | 2 | 128 | 128.0 | [8, 8, 8, 8] |
| n128_g128 | 128 | 163840 | 0.317 | 128 | 1 | 0 | 0.0 | [4, 4, 4, 4] |
| n128_g128_lds0 | 128 | 1 | 0.312 | 128 | 1 | 0 | 0.0 | [4, 4, 4, 4] |

## E5 / timeline


## E6 kernel-shape controls (median us per call; ratio to 304)

- down:linear: 192: 66.3 (1.313), 208: 66.1 (1.311), 256: 50.5 (1.000)
- down:wvsplitk_c192: 208: 47.8
- down:wvsplitk_dev: 192: 60.4 (1.304), 208: 58.7 (1.269), 256: 46.3 (1.000)
- down:wvsplitk_n: 192: 47.6 (1.031), 208: 65.3 (1.414), 256: 46.2 (1.000)
- gate_up:linear: 192: 128.4 (1.528), 208: 129.4 (1.540), 256: 84.0 (1.000)
- gate_up:llmm1: 192: 111.4 (1.145), 208: 109.7 (1.128), 256: 97.3 (1.000)
- gate_up:wvsplitk_c192: 208: 83.9
- gate_up:wvsplitk_dev: 192: 113.2 (1.278), 208: 112.2 (1.266), 256: 88.6 (1.000)
- gate_up:wvsplitk_n: 192: 83.8 (0.997), 208: 121.1 (1.442), 256: 84.0 (1.000)
- lm_head:linear: 192: 250.8 (1.094), 208: 255.2 (1.113), 256: 229.2 (1.000)
- lm_head:llmm1: 192: 301.0 (1.092), 208: 301.2 (1.093), 256: 275.5 (1.000)
- lm_head:wvsplitk_c192: 208: 225.1
- lm_head:wvsplitk_dev: 192: 303.5 (1.297), 208: 302.6 (1.293), 256: 234.0 (1.000)
- lm_head:wvsplitk_n: 192: 224.8 (0.979), 208: 332.5 (1.448), 256: 229.6 (1.000)
- o_proj:linear: 192: 21.9 (1.391), 208: 22.0 (1.396), 256: 15.8 (1.000)
- o_proj:llmm1: 192: 12.3 (1.068), 208: 12.6 (1.098), 256: 11.5 (1.000)
- o_proj:wvsplitk_c192: 208: 10.1
- o_proj:wvsplitk_dev: 192: 12.9 (1.306), 208: 13.2 (1.339), 256: 9.8 (1.000)
- o_proj:wvsplitk_n: 192: 10.3 (1.005), 208: 12.5 (1.222), 256: 10.2 (1.000)
- qkv:linear: 192: 26.3 (1.534), 208: 26.4 (1.543), 256: 17.1 (1.000)
- qkv:llmm1: 192: 17.2 (1.117), 208: 17.1 (1.110), 256: 15.4 (1.000)
- qkv:wvsplitk_c192: 208: 13.2
- qkv:wvsplitk_dev: 192: 16.9 (1.304), 208: 17.0 (1.315), 256: 12.9 (1.000)
- qkv:wvsplitk_n: 192: 13.1 (0.990), 208: 16.3 (1.229), 256: 13.3 (1.000)

## Leak test v3 (per-CU duration probe)

- qwen32b_graph_b1_c4096__n192__cuprobe_t_L0: {'expected_cus': 64, 'alone_cus': 64, 'covered_cus': [64, 64, 64, 64, 64, 64, 64], 'native_min_q': [0.9683597837513023, 0.9807145054943321], 'positive_cus_below': [0, 0, 0], 'decoder_min_q': [0.9917476366450032], 'decoder_cus_below': [0], 'sensitive': False, 'threshold': 0.9183597837513022, 'ok': False, 'positive_treatments': ['arith100', 'cuprobe', 'mem100']}
- qwen32b_graph_b1_c4096__n128__cuprobe_t_L0: {'expected_cus': 128, 'alone_cus': 128, 'covered_cus': [128, 128, 128, 128, 128, 128, 128], 'native_min_q': [0.9869620419590767, 0.9656235476342716], 'positive_cus_below': [0, 0, 0], 'decoder_min_q': [0.9699096274547889], 'decoder_cus_below': [0], 'sensitive': False, 'threshold': 0.9156235476342716, 'ok': False, 'positive_treatments': ['arith100', 'cuprobe', 'mem100']}

## E8 guardrails

