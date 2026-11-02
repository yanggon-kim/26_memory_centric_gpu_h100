# cubw_v1 auto — mi325x

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok: **False**
- leak_ok_v11: **False**
- leak_ok_v2: **True**
- clock_ok: **False**
- release_e2: **True**
- release_e3_minimal: **False**
- release_e3_minimal_v2: **True**
- release_e3_full: **False**
- release_e5e8: **True**
- outputs: {'runs': 66, 'complete': 66, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {}
- leak v1.1 (relative to placement-proven native co-runners): {}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1450, 'gfx_max': 2087, 'gfx_ratio': 1.4393103448275861, 'mem_min': 1257, 'mem_max': 1500, 'samples': 4214, 'coverage_min': 0.3882771825875877}

## Placement

| probe | distinct CUs | expected | per-XCD | balanced | rc |
|---|---:|---:|---|---|---:|
| L0_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L0_n272 | 272 | 272 | {'0': 34, '1': 34, '2': 34, '3': 34, '4': 34, '5': 34, '6': 34, '7': 34} | True | 0 |
| L0_n248 | 248 | 248 | {'0': 31, '1': 31, '2': 31, '3': 31, '4': 31, '5': 31, '6': 31, '7': 31} | True | 0 |
| L0_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L0_n216 | 216 | 216 | {'0': 27, '1': 27, '2': 27, '3': 27, '4': 27, '5': 27, '6': 27, '7': 27} | True | 0 |
| L0_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L0_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L0_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L0_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L1_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L1_n272 | 272 | 272 | {'0': 34, '1': 34, '2': 34, '3': 34, '4': 34, '5': 34, '6': 34, '7': 34} | True | 0 |
| L1_n248 | 248 | 248 | {'0': 31, '1': 31, '2': 31, '3': 31, '4': 31, '5': 31, '6': 31, '7': 31} | True | 0 |
| L1_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L1_n216 | 216 | 216 | {'0': 27, '1': 27, '2': 27, '3': 27, '4': 27, '5': 27, '6': 27, '7': 27} | True | 0 |
| L1_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L1_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L1_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L1_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L2_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L2_n272 | 272 | 272 | {'0': 34, '1': 34, '2': 34, '3': 34, '4': 34, '5': 34, '6': 34, '7': 34} | True | 0 |
| L2_n248 | 248 | 248 | {'0': 31, '1': 31, '2': 31, '3': 31, '4': 31, '5': 31, '6': 31, '7': 31} | True | 0 |
| L2_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L2_n216 | 216 | 216 | {'0': 27, '1': 27, '2': 27, '3': 27, '4': 27, '5': 27, '6': 27, '7': 27} | True | 0 |
| L2_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L2_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L2_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L2_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L3_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L3_n272 | 272 | 272 | {'0': 34, '1': 34, '2': 34, '3': 34, '4': 34, '5': 34, '6': 34, '7': 34} | True | 0 |
| L3_n248 | 248 | 248 | {'0': 31, '1': 31, '2': 31, '3': 31, '4': 31, '5': 31, '6': 31, '7': 31} | True | 0 |
| L3_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L3_n216 | 216 | 216 | {'0': 27, '1': 27, '2': 27, '3': 27, '4': 27, '5': 27, '6': 27, '7': 27} | True | 0 |
| L3_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L3_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L3_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L3_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L4_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L4_n272 | 272 | 272 | {'0': 34, '1': 34, '2': 34, '3': 34, '4': 34, '5': 34, '6': 34, '7': 34} | True | 0 |
| L4_n248 | 248 | 248 | {'0': 31, '1': 31, '2': 31, '3': 31, '4': 31, '5': 31, '6': 31, '7': 31} | True | 0 |
| L4_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L4_n216 | 216 | 216 | {'0': 27, '1': 27, '2': 27, '3': 27, '4': 27, '5': 27, '6': 27, '7': 27} | True | 0 |
| L4_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L4_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L4_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L4_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L5_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L5_n272 | 272 | 272 | {'0': 34, '1': 34, '2': 34, '3': 34, '4': 34, '5': 34, '6': 34, '7': 34} | True | 0 |
| L5_n248 | 248 | 248 | {'0': 31, '1': 31, '2': 31, '3': 31, '4': 31, '5': 31, '6': 31, '7': 31} | True | 0 |
| L5_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L5_n216 | 216 | 216 | {'0': 27, '1': 27, '2': 27, '3': 27, '4': 27, '5': 27, '6': 27, '7': 27} | True | 0 |
| L5_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L5_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L5_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L5_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L6_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L6_n272 | 272 | 272 | {'0': 34, '1': 34, '2': 34, '3': 34, '4': 34, '5': 34, '6': 34, '7': 34} | True | 0 |
| L6_n248 | 248 | 248 | {'0': 31, '1': 31, '2': 31, '3': 31, '4': 31, '5': 31, '6': 31, '7': 31} | True | 0 |
| L6_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L6_n216 | 216 | 216 | {'0': 27, '1': 27, '2': 27, '3': 27, '4': 27, '5': 27, '6': 27, '7': 27} | True | 0 |
| L6_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L6_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L6_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L6_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |
| L7_n304 | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| L7_n272 | 272 | 272 | {'0': 34, '1': 34, '2': 34, '3': 34, '4': 34, '5': 34, '6': 34, '7': 34} | True | 0 |
| L7_n248 | 248 | 248 | {'0': 31, '1': 31, '2': 31, '3': 31, '4': 31, '5': 31, '6': 31, '7': 31} | True | 0 |
| L7_n232 | 232 | 232 | {'0': 29, '1': 29, '2': 29, '3': 29, '4': 29, '5': 29, '6': 29, '7': 29} | True | 0 |
| L7_n216 | 216 | 216 | {'0': 27, '1': 27, '2': 27, '3': 27, '4': 27, '5': 27, '6': 27, '7': 27} | True | 0 |
| L7_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L7_n184 | 184 | 184 | {'0': 23, '1': 23, '2': 23, '3': 23, '4': 23, '5': 23, '6': 23, '7': 23} | True | 0 |
| L7_n152 | 152 | 152 | {'0': 19, '1': 19, '2': 19, '3': 19, '4': 19, '5': 19, '6': 19, '7': 19} | True | 0 |
| L7_competitor_232-303 | 72 | 72 | {'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, '6': 9, '7': 9} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

### e2:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 8/8 | 15.5806 | n/a | None (gfx xNone) |   |
| 272 | 8/8 | 17.3943 | 1.1164 [1.0882, 1.1444] (8 blocks) | False (gfx x1.422) | observational_clock_confounded pooled clock envelope failed |
| 248 | 8/8 | 16.8309 | 1.0802 [1.0564, 1.1000] (8 blocks) | False (gfx x1.417) | observational_clock_confounded pooled clock envelope failed |
| 232 | 8/8 | 16.3879 | 1.0518 [1.0288, 1.0708] (8 blocks) | False (gfx x1.399) | observational_clock_confounded pooled clock envelope failed |
| 216 | 8/8 | 15.9847 | 1.0259 [1.0033, 1.0444] (8 blocks) | False (gfx x1.386) | observational_clock_confounded pooled clock envelope failed |
| 200 | 8/8 | 15.9007 | 1.0205 [0.9982, 1.0389] (8 blocks) | False (gfx x1.396) | observational_clock_confounded pooled clock envelope failed |
| 184 | 8/8 | 15.7107 | 1.0083 [0.9864, 1.0265] (8 blocks) | False (gfx x1.381) | observational_clock_confounded pooled clock envelope failed |
| 152 | 8/8 | 17.7713 | 1.1406 [1.1156, 1.1616] (8 blocks) | False (gfx x1.439) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

### e6_decode:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 272 | 5/5 | 16.6169 | 1.0561 [1.0259, 1.0858] (5 blocks) | False (gfx x1.399) | observational_clock_confounded pooled clock envelope failed |
| 248 | 5/5 | 18.673 | 1.1867 [1.1479, 1.2274] (5 blocks) | False (gfx x1.437) | observational_clock_confounded pooled clock envelope failed |
| 232 | 5/5 | 17.427 | 1.1075 [1.0760, 1.1387] (5 blocks) | False (gfx x1.431) | observational_clock_confounded pooled clock envelope failed |
| 216 | 5/5 | 19.0371 | 1.2099 [1.1769, 1.2422] (5 blocks) | False (gfx x1.439) | observational_clock_confounded pooled clock envelope failed |
| 200 | 5/5 | 19.7474 | 1.2550 [1.2208, 1.2889] (5 blocks) | False (gfx x1.44) | observational_clock_confounded pooled clock envelope failed |
| 184 | 5/5 | 19.4738 | 1.2376 [1.2029, 1.2725] (5 blocks) | False (gfx x1.441) | observational_clock_confounded pooled clock envelope failed |
| 152 | 5/5 | 20.9539 | 1.3317 [1.2938, 1.3699] (5 blocks) | False (gfx x1.443) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

## E3 T(n, z)

## E4 service (mean over acquisitions; ratio to 304)


## E5 / timeline

- e6_timeline:qwen32b_graph_b1_c4096__n304__absent__ktrace:a0: completed=True slowdown=1.062326080578847 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n272__absent__ktrace:a0: completed=True slowdown=1.063623280158015 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n248__absent__ktrace:a0: completed=True slowdown=1.0818203966323803 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n232__absent__ktrace:a0: completed=True slowdown=1.0706519693948169 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n216__absent__ktrace:a0: completed=True slowdown=1.070764060877471 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n200__absent__ktrace:a0: completed=True slowdown=1.0714925457397875 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n184__absent__ktrace:a0: completed=True slowdown=1.0800644706153737 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n152__absent__ktrace:a0: completed=True slowdown=1.0722629636123056 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n232__absent__cu232__ktrace:a0: completed=True slowdown=1.1623494957568572 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n200__absent__cu200__ktrace:a0: completed=True slowdown=1.3540126385469382 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_eager_counters:qwen32b_graph_b1_c4096__n304__absent__eager__FETCH_SIZE:a0: completed=False slowdown=None outputs_match=None decode_ranges=0 dispatches/tick=None counter/tick=None counter coverage=None
- e6_eager_counters:qwen32b_graph_b1_c4096__n304__absent__eager__TCC_EA0_RDREQ_LEVEL:a0: completed=False slowdown=None outputs_match=None decode_ranges=0 dispatches/tick=None counter/tick=None counter coverage=None
- e6_eager_counters:qwen32b_graph_b1_c4096__n200__absent__eager__FETCH_SIZE:a0: completed=False slowdown=None outputs_match=None decode_ranges=0 dispatches/tick=None counter/tick=None counter coverage=None
- e6_eager_counters:qwen32b_graph_b1_c4096__n200__absent__eager__TCC_EA0_RDREQ_LEVEL:a0: completed=False slowdown=None outputs_match=None decode_ranges=0 dispatches/tick=None counter/tick=None counter coverage=None

## E6 kernel-shape controls (median us per call; ratio to 304)

- down:linear: 152: 79.5 (1.064), 184: 86.6 (1.158), 200: 88.7 (1.186), 216: 91.5 (1.224), 232: 97.9 (1.310), 248: 95.7 (1.280), 272: 76.6 (1.024), 304: 74.8 (1.000)
- down:wvsplitk_dev: 152: 74.2 (1.212), 184: 63.4 (1.035), 200: 63.2 (1.032), 216: 67.0 (1.094), 232: 69.3 (1.133), 248: 69.6 (1.138), 272: 69.8 (1.140), 304: 61.2 (1.000)
- down:wvsplitk_n: 152: 79.9 (1.327), 184: 77.4 (1.287), 200: 77.4 (1.287), 216: 74.8 (1.243), 232: 60.2 (1.001), 248: 71.3 (1.186), 272: 58.8 (0.977), 304: 60.2 (1.000)
- gate_up:linear: 152: 152.3 (1.129), 184: 148.2 (1.098), 200: 146.4 (1.085), 216: 144.2 (1.068), 232: 145.7 (1.080), 248: 140.6 (1.042), 272: 142.1 (1.053), 304: 134.9 (1.000)
- gate_up:llmm1: 152: 185.4 (1.472), 184: 163.2 (1.295), 200: 158.0 (1.254), 216: 145.6 (1.156), 232: 141.4 (1.123), 248: 141.0 (1.119), 272: 136.1 (1.080), 304: 126.0 (1.000)
- gate_up:wvsplitk_dev: 152: 127.5 (1.042), 184: 122.6 (1.002), 200: 121.2 (0.990), 216: 119.4 (0.975), 232: 129.9 (1.061), 248: 127.9 (1.045), 272: 128.5 (1.050), 304: 122.4 (1.000)
- gate_up:wvsplitk_n: 152: 154.5 (1.337), 184: 140.6 (1.217), 200: 142.4 (1.232), 216: 140.5 (1.216), 232: 139.1 (1.204), 248: 136.0 (1.177), 272: 126.8 (1.098), 304: 115.5 (1.000)
- lm_head:linear: 152: 670.3 (1.699), 184: 558.1 (1.414), 200: 578.2 (1.465), 216: 436.2 (1.105), 232: 430.3 (1.090), 248: 433.9 (1.100), 272: 459.0 (1.163), 304: 394.6 (1.000)
- lm_head:llmm1: 152: 512.1 (1.425), 184: 457.6 (1.273), 200: 419.7 (1.168), 216: 408.6 (1.137), 232: 390.8 (1.087), 248: 391.6 (1.089), 272: 382.2 (1.063), 304: 359.4 (1.000)
- lm_head:wvsplitk_dev: 152: 350.4 (1.069), 184: 312.9 (0.955), 200: 310.9 (0.949), 216: 321.7 (0.981), 232: 337.8 (1.031), 248: 341.9 (1.043), 272: 350.4 (1.069), 304: 327.8 (1.000)
- lm_head:wvsplitk_n: 152: 427.3 (1.375), 184: 390.8 (1.258), 200: 398.1 (1.281), 216: 374.2 (1.204), 232: 413.5 (1.331), 248: 367.5 (1.182), 272: 362.0 (1.165), 304: 310.8 (1.000)
- o_proj:linear: 152: 20.1 (1.125), 184: 20.4 (1.147), 200: 20.6 (1.158), 216: 21.3 (1.194), 232: 20.9 (1.173), 248: 20.8 (1.167), 272: 18.5 (1.039), 304: 17.8 (1.000)
- o_proj:llmm1: 152: 20.5 (1.364), 184: 18.1 (1.209), 200: 17.5 (1.168), 216: 16.7 (1.115), 232: 16.0 (1.068), 248: 15.8 (1.052), 272: 15.5 (1.036), 304: 15.0 (1.000)
- o_proj:wvsplitk_dev: 152: 17.2 (1.311), 184: 14.9 (1.133), 200: 15.2 (1.161), 216: 14.9 (1.138), 232: 15.8 (1.201), 248: 15.6 (1.187), 272: 15.7 (1.200), 304: 13.1 (1.000)
- o_proj:wvsplitk_n: 152: 16.0 (1.243), 184: 16.1 (1.249), 200: 17.1 (1.326), 216: 15.7 (1.219), 232: 13.1 (1.012), 248: 16.1 (1.247), 272: 12.9 (0.999), 304: 12.9 (1.000)
- qkv:linear: 152: 37.9 (1.381), 184: 30.6 (1.112), 200: 31.1 (1.132), 216: 32.0 (1.164), 232: 32.7 (1.192), 248: 32.9 (1.198), 272: 34.3 (1.248), 304: 27.5 (1.000)
- qkv:llmm1: 152: 28.6 (1.366), 184: 25.5 (1.217), 200: 24.1 (1.152), 216: 22.9 (1.093), 232: 23.0 (1.100), 248: 23.1 (1.102), 272: 22.2 (1.060), 304: 20.9 (1.000)
- qkv:wvsplitk_dev: 152: 21.3 (1.290), 184: 18.4 (1.115), 200: 17.7 (1.072), 216: 17.9 (1.082), 232: 18.5 (1.119), 248: 19.2 (1.162), 272: 20.0 (1.210), 304: 16.5 (1.000)
- qkv:wvsplitk_n: 152: 22.2 (1.272), 184: 23.0 (1.320), 200: 22.9 (1.312), 216: 20.4 (1.168), 232: 18.0 (1.029), 248: 21.6 (1.240), 272: 17.9 (1.028), 304: 17.4 (1.000)

## Leak test v2 (per-CU probe)

- qwen32b_graph_b1_c4096__n232__cuprobe_L0: {'expected_cus': 72, 'alone_cus': 72, 'covered_cus': [72, 72, 72, 72, 72], 'native_min_q': [1.0, 0.999998102322757], 'positive_cus_below': [8], 'decoder_min_q': [0.99999252839211], 'decoder_cus_below': [0], 'sensitive': True, 'threshold': 0.9499981023227569, 'ok': True}
- qwen32b_graph_b1_c4096__n152__cuprobe_L0: {'expected_cus': 152, 'alone_cus': 152, 'covered_cus': [152, 152, 152, 152, 152], 'native_min_q': [0.8853965479680421, 0.8812783224755251], 'positive_cus_below': [8], 'decoder_min_q': [0.8898889105655085], 'decoder_cus_below': [0], 'sensitive': True, 'threshold': 0.831278322475525, 'ok': True}

## E8 guardrails

