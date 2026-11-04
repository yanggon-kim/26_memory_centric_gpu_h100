# cubw_v1 pilot — mi300x

## Gates

- placement_ok: **True**
- lifecycle_ok: **False**
- outputs_ok: **True**
- leak_ok: **False**
- leak_ok_v11: **False**
- leak_ok_v2: **False**
- leak_ok_v3: **False**
- leak_ok_v4: **False**
- clock_ok: **False**
- release_e2: **True**
- release_e3_minimal: **False**
- release_e3_minimal_v2: **False**
- release_e3_minimal_v3: **False**
- release_e3_minimal_v4: **False**
- release_e3_full: **False**
- release_e5e8: **True**
- outputs: {'runs': 2, 'complete': 2, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {}
- leak v1.1 (relative to placement-proven native co-runners): {}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1328, 'gfx_max': 1668, 'gfx_ratio': 1.2560240963855422, 'mem_min': 1062, 'mem_max': 1300, 'samples': 432, 'coverage_min': 0.9544076717113602}

## Placement

| probe | distinct CUs | expected | per-XCD | balanced | rc |
|---|---:|---:|---|---|---:|
| L0_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L0_n288 | 288 | 288 | {'0': 36, '1': 36, '2': 36, '3': 36, '4': 36, '5': 36, '6': 36, '7': 36} | True | 0 |
| L0_n272 | 272 | 272 | {'0': 34, '1': 34, '2': 34, '3': 34, '4': 34, '5': 34, '6': 34, '7': 34} | True | 0 |
| L0_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L0_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L0_n224 | 224 | 224 | {'0': 28, '1': 28, '2': 28, '3': 28, '4': 28, '5': 28, '6': 28, '7': 28} | True | 0 |
| L0_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L0_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L0_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L0_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L0_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

### e1_e2_pilot:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 1/1 | 17.961 | n/a | None (gfx xNone) |   |
| 288 | 0/0 | None | n/a | None (gfx xNone) |   |
| 272 | 1/1 | 19.0336 | 1.0597 [1.0573, 1.0622] (1 blocks) | False (gfx x1.256) | pilot one pilot acquisition; never promoted to a production comparison |
| 256 | 0/0 | None | n/a | None (gfx xNone) |   |
| 232 | 0/0 | None | n/a | None (gfx xNone) |   |
| 224 | 0/0 | None | n/a | None (gfx xNone) |   |
| 200 | 0/0 | None | n/a | None (gfx xNone) |   |
| 192 | 0/0 | None | n/a | None (gfx xNone) |   |
| 152 | 0/0 | None | n/a | None (gfx xNone) |   |
| 128 | 0/0 | None | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

### e6_decode:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 256 | 1/1 | 17.9957 | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

### e6_balanced:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 1/1 | 18.0051 | n/a | None (gfx xNone) |   |
| 272 | 1/1 | 18.0066 | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

## E3 T(n, z)

## E4 service (mean over acquisitions; ratio to 304)


## E6 dispatch probe (wvSplitK-shaped launches; per-workgroup CU and timing)

| step | blocks | LDS | kernel ms | CUs used | max WG/CU | CUs with >=2 | serialized WGs | WG per SE (XCD0) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| n304_g304 | 304 | 65536 | 0.259 | 304 | 1 | 0 | 0.0 | [9, 10, 10, 9] |
| n304_g288 | 288 | 65536 | 0.326 | 288 | 1 | 0 | 0.0 | [9, 9, 9, 9] |
| n304_g304_lds0 | 304 | 1 | 0.258 | 304 | 1 | 0 | 0.0 | [9, 10, 10, 9] |
| n288_g304 | 304 | 65536 | 0.427 | 288 | 2 | 16 | 16.0 | [9, 10, 10, 9] |
| n288_g288 | 288 | 65536 | 0.327 | 288 | 1 | 0 | 0.0 | [9, 9, 9, 9] |
| n288_g288_lds0 | 288 | 1 | 0.258 | 288 | 1 | 0 | 0.0 | [9, 9, 9, 9] |
| n272_g304 | 304 | 65536 | 0.412 | 272 | 2 | 32 | 32.0 | [9, 10, 10, 9] |
| n272_g272 | 272 | 65536 | 0.372 | 264 | 2 | 8 | 8.0 | [8, 9, 8, 9] |
| n272_g256 | 256 | 65536 | 0.341 | 256 | 1 | 0 | 0.0 | [8, 8, 8, 8] |
| n272_g272_lds0 | 272 | 1 | 0.319 | 264 | 2 | 8 | 0.0 | [8, 9, 8, 9] |
| n256_g304 | 304 | 65536 | 0.399 | 256 | 2 | 48 | 48.0 | [9, 10, 10, 9] |
| n256_g256 | 256 | 65536 | 0.346 | 256 | 1 | 0 | 0.0 | [8, 8, 8, 8] |
| n256_g256_lds0 | 256 | 1 | 0.346 | 256 | 1 | 0 | 0.0 | [8, 8, 8, 8] |
| n232_g304 | 304 | 65536 | 0.380 | 232 | 2 | 72 | 72.0 | [9, 10, 10, 9] |
| n232_g232 | 232 | 65536 | 0.395 | 232 | 2 | 8 | 6.4 | [7, 7, 8, 7] |
| n232_g224 | 224 | 65536 | 0.329 | 224 | 1 | 0 | 0.0 | [7, 7, 7, 7] |
| n232_g232_lds0 | 232 | 1 | 0.393 | 232 | 2 | 8 | 0.0 | [7, 7, 8, 7] |
| n224_g304 | 304 | 65536 | 0.372 | 224 | 2 | 80 | 80.0 | [9, 10, 10, 9] |
| n224_g224 | 224 | 65536 | 0.257 | 224 | 1 | 0 | 0.0 | [7, 7, 7, 7] |
| n224_g224_lds0 | 224 | 1 | 0.327 | 224 | 1 | 0 | 0.0 | [7, 7, 7, 7] |
| n200_g304 | 304 | 65536 | 0.287 | 200 | 2 | 104 | 104.0 | [9, 10, 10, 9] |
| n200_g200 | 200 | 65536 | 0.481 | 200 | 2 | 8 | 6.4 | [6, 6, 7, 6] |
| n200_g192 | 192 | 65536 | 0.336 | 192 | 1 | 0 | 0.0 | [6, 6, 6, 6] |
| n200_g200_lds0 | 200 | 1 | 0.350 | 200 | 2 | 8 | 0.0 | [6, 6, 7, 6] |
| n192_g304 | 304 | 65536 | 0.283 | 192 | 2 | 112 | 112.0 | [9, 10, 10, 9] |
| n192_g192 | 192 | 65536 | 0.267 | 192 | 1 | 0 | 0.0 | [6, 6, 6, 6] |
| n192_g192_lds0 | 192 | 1 | 0.262 | 192 | 1 | 0 | 0.0 | [6, 6, 6, 6] |
| n152_g304 | 304 | 65536 | 0.362 | 152 | 3 | 141 | 152.0 | [9, 10, 10, 9] |
| n152_g152 | 152 | 65536 | 0.490 | 152 | 2 | 8 | 4.8 | [5, 5, 5, 4] |
| n152_g128 | 128 | 65536 | 0.360 | 128 | 1 | 0 | 0.0 | [4, 4, 4, 4] |
| n152_g152_lds0 | 152 | 1 | 0.434 | 152 | 2 | 8 | 0.0 | [5, 5, 5, 4] |
| n128_g304 | 304 | 65536 | 0.360 | 128 | 3 | 128 | 176.0 | [9, 10, 10, 9] |
| n128_g128 | 128 | 65536 | 0.354 | 128 | 1 | 0 | 0.0 | [4, 4, 4, 4] |
| n128_g128_lds0 | 128 | 1 | 0.357 | 128 | 1 | 0 | 0.0 | [4, 4, 4, 4] |

## E5 / timeline


## E6 kernel-shape controls (median us per call; ratio to 304)

- down:linear: 272: 87.1 (0.992), 304: 87.8 (1.000)
- down:wvsplitk_c256: 272: 69.2
- down:wvsplitk_c288: 304: 69.4 (1.000)
- down:wvsplitk_dev: 272: 77.4 (1.117), 304: 69.3 (1.000)
- down:wvsplitk_n: 272: 71.2 (1.026), 304: 69.4 (1.000)
- gate_up:linear: 272: 166.6 (1.043), 304: 159.7 (1.000)
- gate_up:llmm1: 272: 178.6 (1.078), 304: 165.7 (1.000)
- gate_up:wvsplitk_c256: 272: 135.9
- gate_up:wvsplitk_c288: 304: 135.3 (1.000)
- gate_up:wvsplitk_dev: 272: 146.3 (1.038), 304: 141.0 (1.000)
- gate_up:wvsplitk_n: 272: 142.6 (1.053), 304: 135.4 (1.000)
- lm_head:linear: 272: 477.6 (1.109), 304: 430.7 (1.000)
- lm_head:llmm1: 272: 498.5 (1.112), 304: 448.4 (1.000)
- lm_head:wvsplitk_c256: 272: 371.0
- lm_head:wvsplitk_c288: 304: 369.6 (1.000)
- lm_head:wvsplitk_dev: 272: 404.6 (1.049), 304: 385.8 (1.000)
- lm_head:wvsplitk_n: 272: 401.5 (1.089), 304: 368.8 (1.000)
- o_proj:linear: 272: 20.8 (0.968), 304: 21.5 (1.000)
- o_proj:llmm1: 272: 19.6 (0.996), 304: 19.7 (1.000)
- o_proj:wvsplitk_c256: 272: 14.5
- o_proj:wvsplitk_c288: 304: 14.5 (1.000)
- o_proj:wvsplitk_dev: 272: 16.6 (1.117), 304: 14.9 (1.000)
- o_proj:wvsplitk_n: 272: 14.5 (0.971), 304: 14.9 (1.000)
- qkv:linear: 272: 35.2 (1.175), 304: 30.0 (1.000)
- qkv:llmm1: 272: 25.1 (1.023), 304: 24.6 (1.000)
- qkv:wvsplitk_c256: 272: 17.7
- qkv:wvsplitk_c288: 304: 19.1 (1.000)
- qkv:wvsplitk_dev: 272: 25.3 (1.105), 304: 22.9 (1.000)
- qkv:wvsplitk_n: 272: 18.0 (1.009), 304: 17.8 (1.000)

## E8 guardrails

