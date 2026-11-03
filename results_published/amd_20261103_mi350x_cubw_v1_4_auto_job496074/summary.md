# cubw_v1 auto — mi350x

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok: **True**
- leak_ok_v11: **True**
- leak_ok_v2: **False**
- clock_ok: **False**
- release_e2: **True**
- release_e3_minimal: **True**
- release_e3_minimal_v2: **False**
- release_e3_full: **False**
- release_e5e8: **True**
- outputs: {'runs': 94, 'complete': 94, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {'qwen32b_graph_b1_c4096__n192__arith100_L1': 0.8749665789480344, 'qwen32b_graph_b1_c4096__n192__arith100': 0.8504520478742643}
- leak v1.1 (relative to placement-proven native co-runners): {'qwen32b_graph_b1_c4096__n192__arith100_L1': {'decoder_ratio': 0.9125493851398452, 'native_ratios': {'mem100': 1.0180677728677976, 'arith100': 0.9135025970039967}, 'relative': 0.8963542599617731}, 'qwen32b_graph_b1_c4096__n128__leakprobe_L1': {'decoder_ratio': 0.8021266712219027, 'native_ratios': {'mem100': 0.8215764947817066, 'arith100': 0.8788929850209436}, 'relative': 0.9126556758247291}, 'qwen32b_graph_b1_c4096__n192__arith100': {'decoder_ratio': 0.8504520478742643, 'native_ratios': {'mem100': 0.9761392543449833, 'arith100': 0.8758805333458685}, 'relative': 0.8712404957476496}}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1432, 'gfx_max': 2204, 'gfx_ratio': 1.5391061452513966, 'mem_min': 2000, 'mem_max': 2000, 'samples': 5446, 'coverage_min': 0.6152590436979348}

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
| L0_n80 | 80 | 80 | {'0': 10, '1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10} | True | 0 |
| L0_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |
| L1_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L1_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L1_n208 | 208 | 208 | {'0': 26, '1': 26, '2': 26, '3': 26, '4': 26, '5': 26, '6': 26, '7': 26} | True | 0 |
| L1_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L1_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L1_n168 | 168 | 168 | {'0': 21, '1': 21, '2': 21, '3': 21, '4': 21, '5': 21, '6': 21, '7': 21} | True | 0 |
| L1_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L1_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L1_n80 | 80 | 80 | {'0': 10, '1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10} | True | 0 |
| L1_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |
| L2_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L2_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L2_n208 | 208 | 208 | {'0': 26, '1': 26, '2': 26, '3': 26, '4': 26, '5': 26, '6': 26, '7': 26} | True | 0 |
| L2_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L2_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L2_n168 | 168 | 168 | {'0': 21, '1': 21, '2': 21, '3': 21, '4': 21, '5': 21, '6': 21, '7': 21} | True | 0 |
| L2_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L2_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L2_n80 | 80 | 80 | {'0': 10, '1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10} | True | 0 |
| L2_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |
| L3_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L3_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L3_n208 | 208 | 208 | {'0': 26, '1': 26, '2': 26, '3': 26, '4': 26, '5': 26, '6': 26, '7': 26} | True | 0 |
| L3_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L3_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L3_n168 | 168 | 168 | {'0': 21, '1': 21, '2': 21, '3': 21, '4': 21, '5': 21, '6': 21, '7': 21} | True | 0 |
| L3_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L3_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L3_n80 | 80 | 80 | {'0': 10, '1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10} | True | 0 |
| L3_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |
| L4_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L4_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L4_n208 | 208 | 208 | {'0': 26, '1': 26, '2': 26, '3': 26, '4': 26, '5': 26, '6': 26, '7': 26} | True | 0 |
| L4_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L4_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L4_n168 | 168 | 168 | {'0': 21, '1': 21, '2': 21, '3': 21, '4': 21, '5': 21, '6': 21, '7': 21} | True | 0 |
| L4_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L4_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L4_n80 | 80 | 80 | {'0': 10, '1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10} | True | 0 |
| L4_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |
| L5_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L5_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L5_n208 | 208 | 208 | {'0': 26, '1': 26, '2': 26, '3': 26, '4': 26, '5': 26, '6': 26, '7': 26} | True | 0 |
| L5_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L5_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L5_n168 | 168 | 168 | {'0': 21, '1': 21, '2': 21, '3': 21, '4': 21, '5': 21, '6': 21, '7': 21} | True | 0 |
| L5_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L5_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L5_n80 | 80 | 80 | {'0': 10, '1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10} | True | 0 |
| L5_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |
| L6_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L6_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L6_n208 | 208 | 208 | {'0': 26, '1': 26, '2': 26, '3': 26, '4': 26, '5': 26, '6': 26, '7': 26} | True | 0 |
| L6_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L6_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L6_n168 | 168 | 168 | {'0': 21, '1': 21, '2': 21, '3': 21, '4': 21, '5': 21, '6': 21, '7': 21} | True | 0 |
| L6_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L6_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L6_n80 | 80 | 80 | {'0': 10, '1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10} | True | 0 |
| L6_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |
| L7_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L7_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L7_n208 | 208 | 208 | {'0': 26, '1': 26, '2': 26, '3': 26, '4': 26, '5': 26, '6': 26, '7': 26} | True | 0 |
| L7_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L7_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L7_n168 | 168 | 168 | {'0': 21, '1': 21, '2': 21, '3': 21, '4': 21, '5': 21, '6': 21, '7': 21} | True | 0 |
| L7_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L7_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L7_n80 | 80 | 80 | {'0': 10, '1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10} | True | 0 |
| L7_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |
| comma_list_0-7,16-23 | 16 | 16 | {'0': 2, '1': 2, '2': 2, '3': 2, '4': 2, '5': 2, '6': 2, '7': 2} | True | 0 |
| wrong_index_mask | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

### e2:qwen14b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 256 | 3/3 | 6.7685 | n/a | None (gfx xNone) |   |
| 232 | 0/0 | None | n/a | None (gfx xNone) |   |
| 208 | 0/0 | None | n/a | None (gfx xNone) |   |
| 192 | 3/3 | 7.8545 | 1.1604 [1.0854, 1.2102] (3 blocks) | False (gfx x1.322) | observational_clock_confounded pooled clock envelope failed |
| 184 | 0/0 | None | n/a | None (gfx xNone) |   |
| 168 | 3/3 | 8.2153 | 1.2138 [1.1427, 1.2792] (3 blocks) | False (gfx x1.312) | observational_clock_confounded pooled clock envelope failed |
| 152 | 0/0 | None | n/a | None (gfx xNone) |   |
| 128 | 3/3 | 7.7086 | 1.1389 [1.0661, 1.2093] (3 blocks) | False (gfx x1.325) | observational_clock_confounded pooled clock envelope failed |
| 80 | 3/3 | 12.3098 | 1.8187 [1.6907, 1.9670] (3 blocks) | False (gfx x1.335) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

### e2:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 256 | 8/8 | 12.6963 | n/a | None (gfx xNone) |   |
| 232 | 8/8 | 15.8805 | 1.2508 [1.1895, 1.3186] (8 blocks) | False (gfx x1.534) | observational_clock_confounded pooled clock envelope failed |
| 208 | 8/8 | 14.9394 | 1.1767 [1.1463, 1.2026] (8 blocks) | False (gfx x1.533) | observational_clock_confounded pooled clock envelope failed |
| 192 | 8/8 | 14.9756 | 1.1795 [1.1382, 1.2267] (8 blocks) | False (gfx x1.533) | observational_clock_confounded pooled clock envelope failed |
| 184 | 8/8 | 15.1922 | 1.1966 [1.1523, 1.2458] (8 blocks) | False (gfx x1.536) | observational_clock_confounded pooled clock envelope failed |
| 168 | 8/8 | 15.1207 | 1.1910 [1.1480, 1.2317] (8 blocks) | False (gfx x1.531) | observational_clock_confounded pooled clock envelope failed |
| 152 | 8/8 | 14.9756 | 1.1795 [1.1405, 1.2236] (8 blocks) | False (gfx x1.531) | observational_clock_confounded pooled clock envelope failed |
| 128 | 8/8 | 14.7147 | 1.1590 [1.1204, 1.1965] (8 blocks) | False (gfx x1.533) | observational_clock_confounded pooled clock envelope failed |
| 80 | 8/8 | 23.5886 | 1.8579 [1.8112, 1.9001] (8 blocks) | False (gfx x1.538) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

### e1_e2_pilot:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 256 | 2/2 | 12.4521 | n/a | None (gfx xNone) |   |
| 232 | 0/0 | None | n/a | None (gfx xNone) |   |
| 208 | 0/0 | None | n/a | None (gfx xNone) |   |
| 192 | 2/2 | 15.036 | 1.2075 [1.1852, 1.2268] (2 blocks) | False (gfx x1.331) | pilot one pilot acquisition; never promoted to a production comparison |
| 184 | 0/0 | None | n/a | None (gfx xNone) |   |
| 168 | 2/2 | 14.7632 | 1.1856 [1.1717, 1.1964] (2 blocks) | False (gfx x1.327) | pilot one pilot acquisition; never promoted to a production comparison |
| 152 | 0/0 | None | n/a | None (gfx xNone) |   |
| 128 | 2/2 | 14.5969 | 1.1722 [1.1479, 1.1939] (2 blocks) | False (gfx x1.327) | pilot one pilot acquisition; never promoted to a production comparison |
| 80 | 2/2 | 23.3875 | 1.8782 [1.8499, 1.9006] (2 blocks) | False (gfx x1.337) | pilot one pilot acquisition; never promoted to a production comparison |
Smallest qualified noninferior n: None

### e6_decode:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 232 | 5/5 | 14.675 | 1.1590 [1.1401, 1.1783] (5 blocks) | False (gfx x1.362) | observational_clock_confounded pooled clock envelope failed |
| 208 | 5/5 | 17.0838 | 1.3492 [1.3116, 1.3849] (5 blocks) | False (gfx x1.425) | observational_clock_confounded pooled clock envelope failed |
| 192 | 5/5 | 12.5194 | 0.9888 [0.9646, 1.0092] (5 blocks) | False (gfx x1.349) | observational_clock_confounded pooled clock envelope failed |
| 184 | 5/5 | 17.8804 | 1.4122 [1.3878, 1.4364] (5 blocks) | False (gfx x1.371) | observational_clock_confounded pooled clock envelope failed |
| 168 | 5/5 | 16.8452 | 1.3304 [1.2977, 1.3613] (5 blocks) | False (gfx x1.368) | observational_clock_confounded pooled clock envelope failed |
| 152 | 5/5 | 19.5901 | 1.5472 [1.5235, 1.5717] (5 blocks) | False (gfx x1.37) | observational_clock_confounded pooled clock envelope failed |
| 128 | 5/5 | 14.3701 | 1.1349 [1.1124, 1.1573] (5 blocks) | False (gfx x1.367) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

## E3 T(n, z)

### e3:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n128:absent | 0/5 | None | n/a | None | None | None |  |
| n128:arith100 | 0/5 | None | n/a | None | None | None |  |
| n128:mem100 | 0/5 | None | n/a | None | None | None |  |
| n128:mem50 | 0/5 | None | n/a | None | None | None |  |
| n192:absent | 0/5 | None | n/a | None | None | None |  |
| n192:arith100 | 0/5 | None | n/a | None | None | None |  |
| n192:inactive | 0/1 | None | n/a | None | None | None |  |
| n192:mem100 | 0/5 | None | n/a | None | None | None |  |
| n192:mem50 | 0/5 | None | n/a | None | None | None |  |

### e1_e3_pilot:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n192:absent | 1/1 | 15.6094 | n/a | None | None | None |  |
| n192:arith100 | 1/1 | 15.3935 | 0.9862 [0.9755, 0.9978] (1 acq) | 0.0 | 28.0 | 1.0 | pilot |
| n192:inactive | 1/1 | 15.9779 | 1.0236 [1.0084, 1.0390] (1 acq) | 0.0 | 0.0 | 1.0 | pilot |
| n192:mem100 | 1/1 | 20.6486 | 1.3228 [1.3036, 1.3426] (1 acq) | 2326 | 0.0 | 1.0 | pilot |
- n192: memory100 / arithmetic100: 1.3414 [1.3250, 1.3569] (1 acq); numerator faster: False; pooled clock ok: False

### e1_leak:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n128:leakprobe | 1/1 | 15.6254 | n/a | 0.0 | 47.2 | 1.0 |  |
| n192:arith100 | 1/1 | 15.321 | n/a | 0.0 | 28.8 | 1.0 |  |

## E4 service (mean over acquisitions; ratio to 304)

### e4
- chase_c1 [ns_per_dependent_load]: 128: 99.7 (0.718), 168: 99.8 (0.718), 192: 101.5 (0.731), 256: 138.9, 80: 98.7 (0.710)
- chase_c16384 [ns_per_dependent_load]: 128: 647.1 (1.017), 168: 664.8 (1.045), 192: 649.6 (1.021), 256: 636.1, 80: 690.6 (1.086)
- chase_c256 [ns_per_dependent_load]: 128: 524.1 (0.978), 168: 562.4 (1.050), 192: 554.1 (1.034), 256: 535.7, 80: 536.9 (1.002)
- chase_c4096 [ns_per_dependent_load]: 128: 578.0 (1.009), 168: 577.0 (1.007), 192: 584.8 (1.021), 256: 572.7, 80: 565.9 (0.988)
- fma [useful_tflop_per_s]: 128: 64.8 (0.638), 168: 77.7 (0.765), 192: 86.7 (0.854), 256: 101.6, 80: 33.9 (0.334)
- gather1g [useful_gb_per_s]: 128: 771.8 (0.704), 168: 897.6 (0.819), 192: 998.1 (0.911), 256: 1095.9, 80: 427.3 (0.390)
- grid1024 [useful_gb_per_s]: 128: 5930.7 (0.995), 168: 5724.7 (0.960), 192: 6174.8 (1.036), 256: 5962.0, 80: 3626.0 (0.608)
- grid256 [useful_gb_per_s]: 128: 2323.9 (0.973), 168: 2235.7 (0.936), 192: 2231.9 (0.935), 256: 2388.1, 80: 2160.5 (0.905)
- grid8192 [useful_gb_per_s]: 128: 5280.6 (0.897), 168: 5751.8 (0.977), 192: 5891.0 (1.000), 256: 5889.0, 80: 3382.7 (0.574)
- ilp1 [useful_gb_per_s]: 128: 4414.8 (0.678), 168: 4369.0 (0.671), 192: 4379.8 (0.673), 256: 6507.6, 80: 2222.0 (0.341)
- ilp16 [useful_gb_per_s]: 128: 3973.4 (0.753), 168: 5035.9 (0.955), 192: 5607.0 (1.063), 256: 5273.6, 80: 2212.7 (0.420)
- ilp2 [useful_gb_per_s]: 128: 3923.1 (0.653), 168: 5184.7 (0.863), 192: 5271.8 (0.877), 256: 6010.2, 80: 2168.5 (0.361)
- ilp4 [useful_gb_per_s]: 128: 4049.3 (0.698), 168: 4837.0 (0.834), 192: 5398.0 (0.931), 256: 5799.1, 80: 2148.6 (0.371)
- ilp8 [useful_gb_per_s]: 128: 4292.2 (0.756), 168: 5103.9 (0.899), 192: 5786.1 (1.019), 256: 5677.3, 80: 2213.4 (0.390)
- mfma [useful_tflop_per_s]: 128: 567.3 (0.502), 168: 682.0 (0.603), 192: 754.7 (0.668), 256: 1130.1, 80: 282.0 (0.250)
- reuse128m [useful_gb_per_s]: 128: 7796.7 (1.115), 168: 7791.0 (1.114), 192: 7110.4 (1.017), 256: 6992.5, 80: 6459.2 (0.924)
- reuse32m [useful_gb_per_s]: 128: 16989.3 (0.592), 168: 18615.4 (0.649), 192: 19237.7 (0.671), 256: 28690.4, 80: 8645.0 (0.301)
- reuse512m [useful_gb_per_s]: 128: 7240.7 (1.126), 168: 6460.3 (1.005), 192: 6238.5 (0.970), 256: 6428.3, 80: 5298.6 (0.824)
- reuse8m [useful_gb_per_s]: 128: 17313.6 (0.545), 168: 18330.2 (0.577), 192: 20767.2 (0.654), 256: 31741.3, 80: 8811.8 (0.278)
- seqread4g [useful_gb_per_s]: 128: 5743.4 (0.895), 168: 5621.0 (0.876), 192: 5360.5 (0.835), 256: 6416.2, 80: 3572.4 (0.557)
- stride16 [useful_gb_per_s]: 128: 760.4 (1.007), 168: 691.3 (0.916), 192: 740.9 (0.982), 256: 754.8, 80: 452.5 (0.599)
- stride4 [useful_gb_per_s]: 128: 1376.2 (0.874), 168: 1505.3 (0.956), 192: 1435.9 (0.912), 256: 1574.2, 80: 789.4 (0.501)
- comp_fma [useful_tflop_per_s]: competitor: 33.7
- comp_read1g [useful_gb_per_s]: competitor: 3463.8
- comp_seqread4g [useful_gb_per_s]: competitor: 3327.6

## E5 / timeline

- e1_timeline:qwen32b_graph_b1_c4096__n256__absent__ktrace:a0: completed=True slowdown=1.1015638101276768 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e1_timeline:qwen32b_graph_b1_c4096__n128__absent__ktrace:a0: completed=True slowdown=1.0911518057966014 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n256__absent__ktrace:a0: completed=True slowdown=1.0764514688358926 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n232__absent__ktrace:a0: completed=True slowdown=1.0670899521831687 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n208__absent__ktrace:a0: completed=True slowdown=1.0854484799854571 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n192__absent__ktrace:a0: completed=True slowdown=1.0511844394340546 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n184__absent__ktrace:a0: completed=True slowdown=1.0439370547421445 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n168__absent__ktrace:a0: completed=True slowdown=1.0396430024253418 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n152__absent__ktrace:a0: completed=True slowdown=1.045769808785683 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n128__absent__ktrace:a0: completed=True slowdown=1.0809391375810176 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n192__absent__cu192__ktrace:a0: completed=True slowdown=0.9823175793631178 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n168__absent__cu168__ktrace:a0: completed=True slowdown=1.2411442290530097 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_eager_counters:qwen32b_graph_b1_c4096__n256__absent__eager__FETCH_SIZE:a0: completed=True slowdown=5.179962402335993 outputs_match=True decode_ranges=6 dispatches/tick=733.0 counter/tick={'FETCH_SIZE': 63630668.99619709} counter coverage=1.0
- e6_eager_counters:qwen32b_graph_b1_c4096__n256__absent__eager__TCC_EA0_RDREQ_LEVEL:a0: completed=True slowdown=3.9048315912230938 outputs_match=True decode_ranges=6 dispatches/tick=733.0 counter/tick={'TCC_EA0_RDREQ_LEVEL': 1075437596925.2513} counter coverage=1.0
- e6_eager_counters:qwen32b_graph_b1_c4096__n168__absent__eager__FETCH_SIZE:a0: completed=True slowdown=4.433638950745454 outputs_match=True decode_ranges=6 dispatches/tick=733.0 counter/tick={'FETCH_SIZE': 63626468.93303572} counter coverage=1.0
- e6_eager_counters:qwen32b_graph_b1_c4096__n168__absent__eager__TCC_EA0_RDREQ_LEVEL:a0: completed=True slowdown=3.3105997099473923 outputs_match=True decode_ranges=6 dispatches/tick=733.0 counter/tick={'TCC_EA0_RDREQ_LEVEL': 742241806605.6217} counter coverage=0.9995524661281824

## E6 kernel-shape controls (median us per call; ratio to 304)

- down:linear: 128: 61.5 (1.247), 152: 66.7 (1.352), 168: 66.5 (1.350), 184: 67.4 (1.366), 192: 67.0 (1.359), 208: 66.5 (1.349), 232: 68.7 (1.394), 256: 49.3 (1.000)
- down:wvsplitk_dev: 128: 52.6 (1.054), 152: 55.7 (1.116), 168: 58.1 (1.165), 184: 59.9 (1.201), 192: 60.7 (1.217), 208: 59.3 (1.189), 232: 59.1 (1.184), 256: 49.9 (1.000)
- down:wvsplitk_n: 128: 51.7 (1.148), 152: 73.7 (1.638), 168: 49.7 (1.105), 184: 69.3 (1.540), 192: 48.5 (1.079), 208: 65.3 (1.452), 232: 44.8 (0.996), 256: 45.0 (1.000)
- gate_up:linear: 128: 123.8 (1.501), 152: 123.5 (1.497), 168: 133.8 (1.621), 184: 126.4 (1.532), 192: 128.9 (1.562), 208: 130.4 (1.581), 232: 83.4 (1.011), 256: 82.5 (1.000)
- gate_up:llmm1: 128: 135.0 (1.371), 152: 132.7 (1.348), 168: 121.5 (1.234), 184: 119.1 (1.210), 192: 111.8 (1.136), 208: 111.1 (1.129), 232: 104.4 (1.060), 256: 98.5 (1.000)
- gate_up:wvsplitk_dev: 128: 107.2 (1.239), 152: 107.2 (1.240), 168: 108.6 (1.256), 184: 109.2 (1.263), 192: 110.6 (1.279), 208: 109.3 (1.263), 232: 108.5 (1.254), 256: 86.5 (1.000)
- gate_up:wvsplitk_n: 128: 102.3 (1.231), 152: 145.2 (1.746), 168: 137.5 (1.654), 184: 129.5 (1.557), 192: 83.3 (1.002), 208: 121.2 (1.458), 232: 115.8 (1.393), 256: 83.1 (1.000)
- lm_head:linear: 128: 308.5 (1.347), 152: 316.1 (1.380), 168: 277.5 (1.211), 184: 303.0 (1.323), 192: 251.5 (1.098), 208: 251.2 (1.097), 232: 262.0 (1.144), 256: 229.1 (1.000)
- lm_head:llmm1: 128: 363.8 (1.342), 152: 359.9 (1.327), 168: 325.9 (1.202), 184: 321.7 (1.186), 192: 302.6 (1.116), 208: 300.5 (1.108), 232: 290.5 (1.071), 256: 271.1 (1.000)
- lm_head:wvsplitk_dev: 128: 284.3 (1.196), 152: 295.2 (1.242), 168: 326.5 (1.373), 184: 298.7 (1.257), 192: 304.4 (1.280), 208: 299.0 (1.258), 232: 303.9 (1.278), 256: 237.7 (1.000)
- lm_head:wvsplitk_n: 128: 269.3 (1.203), 152: 413.5 (1.847), 168: 435.2 (1.944), 184: 350.8 (1.567), 192: 225.4 (1.007), 208: 333.8 (1.491), 232: 320.0 (1.429), 256: 223.9 (1.000)
- o_proj:linear: 128: 21.6 (1.357), 152: 21.6 (1.357), 168: 21.6 (1.359), 184: 21.9 (1.377), 192: 22.3 (1.398), 208: 22.1 (1.386), 232: 15.8 (0.996), 256: 15.9 (1.000)
- o_proj:llmm1: 128: 15.6 (1.364), 152: 15.3 (1.339), 168: 13.5 (1.185), 184: 13.6 (1.191), 192: 12.4 (1.087), 208: 12.4 (1.084), 232: 11.8 (1.033), 256: 11.4 (1.000)
- o_proj:wvsplitk_dev: 128: 12.7 (1.284), 152: 12.7 (1.278), 168: 12.8 (1.295), 184: 12.9 (1.301), 192: 13.1 (1.318), 208: 13.3 (1.337), 232: 13.7 (1.386), 256: 9.9 (1.000)
- o_proj:wvsplitk_n: 128: 11.8 (1.155), 152: 16.6 (1.626), 168: 11.1 (1.083), 184: 15.8 (1.545), 192: 10.4 (1.020), 208: 14.6 (1.435), 232: 10.2 (1.001), 256: 10.2 (1.000)
- qkv:linear: 128: 25.8 (1.537), 152: 25.7 (1.532), 168: 26.0 (1.549), 184: 26.0 (1.553), 192: 26.2 (1.564), 208: 26.3 (1.569), 232: 16.8 (1.002), 256: 16.8 (1.000)
- qkv:llmm1: 128: 20.7 (1.313), 152: 20.8 (1.323), 168: 18.1 (1.150), 184: 18.3 (1.164), 192: 17.0 (1.078), 208: 16.2 (1.028), 232: 16.2 (1.030), 256: 15.7 (1.000)
- qkv:wvsplitk_dev: 128: 16.5 (1.273), 152: 16.5 (1.271), 168: 16.8 (1.296), 184: 17.0 (1.307), 192: 17.1 (1.312), 208: 17.3 (1.333), 232: 18.5 (1.420), 256: 13.0 (1.000)
- qkv:wvsplitk_n: 128: 15.3 (1.162), 152: 22.3 (1.687), 168: 18.0 (1.362), 184: 20.1 (1.523), 192: 13.3 (1.006), 208: 16.5 (1.250), 232: 13.3 (1.011), 256: 13.2 (1.000)

## Leak test v2 (per-CU probe)

- qwen32b_graph_b1_c4096__n192__cuprobe_L2: {'expected_cus': 64, 'alone_cus': 64, 'covered_cus': [64, 64, 64, 64, 64], 'native_min_q': [0.9999799580722872, 0.9999799580722871], 'positive_cus_below': [0], 'decoder_min_q': [0.999979958072287], 'decoder_cus_below': [0], 'sensitive': False, 'threshold': 0.9499799580722871, 'ok': False}
- qwen32b_graph_b1_c4096__n128__cuprobe_L2: {'expected_cus': 128, 'alone_cus': 128, 'covered_cus': [128, 128, 128, 128, 128], 'native_min_q': [0.9999872981870261, 0.9999922320267576], 'positive_cus_below': [0], 'decoder_min_q': [0.9999839422941473], 'decoder_cus_below': [0], 'sensitive': False, 'threshold': 0.9499872981870261, 'ok': False}
- qwen32b_graph_b1_c4096__n192__cuprobe_L3: {'expected_cus': 64, 'alone_cus': 64, 'covered_cus': [64, 64, 64, 64, 64], 'native_min_q': [0.9999802706083362, 0.9999822347430601], 'positive_cus_below': [0], 'decoder_min_q': [0.999980270608336], 'decoder_cus_below': [0], 'sensitive': False, 'threshold': 0.9499802706083361, 'ok': False}
- qwen32b_graph_b1_c4096__n128__cuprobe_L3: {'expected_cus': 128, 'alone_cus': 128, 'covered_cus': [128, 128, 128, 128, 128], 'native_min_q': [0.9999833566340457, 0.9999922474411305], 'positive_cus_below': [0], 'decoder_min_q': [0.9999714276556956], 'decoder_cus_below': [0], 'sensitive': False, 'threshold': 0.9499833566340457, 'ok': False}

## E8 guardrails

- e8:qwen7b_graph_b8_c4096__n80__absent: 3/3 ms/step=[8.692, 8.6465, 8.7277] prefill_ms=[1245.7, 1270.4, 1249.8] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n80__absent__prefill: 3/3 ms/step=[24.1574, 24.1953, 23.694] prefill_ms=[754.7, 743.9, 728.5] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n168__absent: 3/3 ms/step=[5.5835, 5.5623, 5.5434] prefill_ms=[676.7, 662.4, 661.8] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n168__absent__prefill: 3/3 ms/step=[15.1525, 17.2012, 15.1457] prefill_ms=[366.5, 453.0, 366.2] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n256__absent: 3/3 ms/step=[4.6788, 4.6387, 4.6819] prefill_ms=[475.2, 479.7, 513.7] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n256__absent__prefill: 3/3 ms/step=[14.7026, 14.0005, 15.0683] prefill_ms=[329.1, 298.3, 353.1] J/token=[] W=[]
- e8:qwen32b_graph_b1_c16384__n256__absent: 3/3 ms/step=[13.8238, 14.0522, 13.594] prefill_ms=[1474.9, 1500.5, 1372.9] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n256__absent: 3/3 ms/step=[4.2672, 4.1478, 4.0542] prefill_ms=[78.0, 68.1, 68.4] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n256__absent__energy: 3/3 ms/step=[12.8176, 12.5862, 12.4184] prefill_ms=[294.9, 282.4, 283.9] J/token=[11.5653, 11.3933, 11.4243] W=[902.5, 905.4, 920.0]
- e8:qwen32b_graph_b1_c16384__n168__absent: 3/3 ms/step=[15.6178, 16.1059, 15.7236] prefill_ms=[1793.3, 1897.2, 1787.6] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n168__absent: 3/3 ms/step=[4.7704, 4.7587, 4.67] prefill_ms=[104.7, 92.2, 94.3] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n168__absent__energy: 3/3 ms/step=[14.7755, 15.1418, 14.5986] prefill_ms=[373.5, 390.8, 369.9] J/token=[12.4176, 12.588, 12.4123] W=[840.5, 831.5, 850.2]
