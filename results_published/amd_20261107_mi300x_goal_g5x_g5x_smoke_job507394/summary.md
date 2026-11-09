# cubw_v1 g5x_smoke — mi300x

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok: **False**
- leak_ok_v11: **False**
- leak_ok_v2: **False**
- leak_ok_v3: **False**
- leak_ok_v4: **True**
- clock_ok: **False**
- release_e2: **True**
- release_e3_minimal: **False**
- release_e3_minimal_v2: **False**
- release_e3_minimal_v3: **False**
- release_e3_minimal_v4: **True**
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
| L0_n264 | 264 | 264 | {'0': 33, '1': 33, '2': 33, '3': 33, '4': 33, '5': 33, '6': 33, '7': 33} | True | 0 |
| L0_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L0_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L0_n136 | 136 | 136 | {'0': 17, '1': 17, '2': 17, '3': 17, '4': 17, '5': 17, '6': 17, '7': 17} | True | 0 |
| L0_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L0_competitor_256-303 | 48 | 48 | {'0': 6, '1': 6, '2': 6, '3': 6, '4': 6, '5': 6, '6': 6, '7': 6} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

## E3 T(n, z)

## E4 service (mean over acquisitions; ratio to 304)


## E5 / timeline


## Leak test v4 (per-CU start-delay probe)

- qwen32b_graph_b1_c4096__n256__gapprobe_L0: {'expected_cus': 48, 'alone_cus': 48, 'flag_excess': 0.1, 'alone_late_max': 0.25020615949598124, 'launches': [90949, 735, 89566, 87679, 88369, 85069, 27090, 26093], 'native_flags': [0, 0, 0, 0], 'native_max_excess': [0.01479006362465381, 0.0007072968030473873, 7.650435411787893e-06, 2.44781925021198e-05], 'positive_flags': [43, 43], 'positive_top8_excess': [[0.5466403456749043, 0.6237293776332982, 0.6544694853779631, 0.656672665568099, 0.6598924156826533, 0.6614877226428543, 0.6640628634299558, 0.6892329483592935], [0.48583289030984356, 0.6545086206601591, 0.6553058048048469, 0.6707719765434197, 0.6711238448122677, 0.6758841690941383, 0.6989271462873202, 0.7100830283309245]], 'decoder_flags': [0], 'decoder_max_excess': [0.0015053316095058244], 'info_positive_flags': {'mem100': 44, 'arith100': 36}, 'sensitive': True, 'ok': True}

## E8 guardrails

