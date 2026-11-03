# cubw_v1 auto — mi300x

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
- outputs: {'runs': 108, 'complete': 108, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {}
- leak v1.1 (relative to placement-proven native co-runners): {}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1294, 'gfx_max': 1959, 'gfx_ratio': 1.5139103554868625, 'mem_min': 1069, 'mem_max': 1300, 'samples': 2034, 'coverage_min': 0.5119193829404012}

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
| 304 | 8/8 | 17.5365 | n/a | None (gfx xNone) |   |
| 272 | 8/8 | 18.7275 | 1.0679 [1.0641, 1.0712] (8 blocks) | False (gfx x1.355) | observational_clock_confounded pooled clock envelope failed |
| 248 | 8/8 | 18.4578 | 1.0525 [1.0479, 1.0564] (8 blocks) | False (gfx x1.376) | observational_clock_confounded pooled clock envelope failed |
| 232 | 8/8 | 18.112 | 1.0328 [1.0298, 1.0358] (8 blocks) | False (gfx x1.374) | observational_clock_confounded pooled clock envelope failed |
| 216 | 8/8 | 17.8755 | 1.0193 [1.0156, 1.0230] (8 blocks) | False (gfx x1.406) | observational_clock_confounded pooled clock envelope failed |
| 200 | 8/8 | 17.8333 | 1.0169 [1.0147, 1.0188] (8 blocks) | False (gfx x1.412) | observational_clock_confounded pooled clock envelope failed |
| 184 | 8/8 | 17.7189 | 1.0104 [1.0073, 1.0132] (8 blocks) | False (gfx x1.43) | observational_clock_confounded pooled clock envelope failed |
| 152 | 8/8 | 19.0286 | 1.0851 [1.0806, 1.0897] (8 blocks) | False (gfx x1.514) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

### e6_decode:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 272 | 5/5 | 18.3407 | 1.0433 [1.0387, 1.0481] (5 blocks) | False (gfx x1.342) | observational_clock_confounded pooled clock envelope failed |
| 248 | 5/5 | 19.5884 | 1.1143 [1.1065, 1.1223] (5 blocks) | False (gfx x1.361) | observational_clock_confounded pooled clock envelope failed |
| 232 | 5/5 | 18.9049 | 1.0754 [1.0693, 1.0817] (5 blocks) | False (gfx x1.342) | observational_clock_confounded pooled clock envelope failed |
| 216 | 5/5 | 19.8107 | 1.1270 [1.1175, 1.1364] (5 blocks) | False (gfx x1.373) | observational_clock_confounded pooled clock envelope failed |
| 200 | 5/5 | 20.3597 | 1.1582 [1.1480, 1.1685] (5 blocks) | False (gfx x1.398) | observational_clock_confounded pooled clock envelope failed |
| 184 | 5/5 | 20.0514 | 1.1406 [1.1311, 1.1507] (5 blocks) | False (gfx x1.433) | observational_clock_confounded pooled clock envelope failed |
| 152 | 5/5 | 21.1306 | 1.2020 [1.1918, 1.2120] (5 blocks) | False (gfx x1.484) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

## E3 T(n, z)

### e3:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n152:absent | 5/5 | 19.0887 | n/a | None | None | None |  |
| n152:arith100 | 5/5 | 19.9253 | 1.0438 [1.0380, 1.0496] (5 blocks) | 0.0 | 11.8 | 1.0 | observational_clock_confounded |
| n152:mem100 | 5/5 | 24.178 | 1.2666 [1.2483, 1.2863] (5 blocks) | 1253 | 0.0 | 1.0 | observational_clock_confounded |
| n152:mem50 | 5/5 | 21.2375 | 1.1126 [1.1044, 1.1213] (5 blocks) | 632 | 0.0 | 1.0 | observational_clock_confounded |
| n232:absent | 5/5 | 18.1076 | n/a | None | None | None |  |
| n232:arith100 | 5/5 | 19.0876 | 1.0541 [1.0460, 1.0622] (5 blocks) | 0.0 | 10.9 | 1.0 | observational_clock_confounded |
| n232:mem100 | 5/5 | 22.5532 | 1.2455 [1.2261, 1.2647] (5 blocks) | 1082 | 0.0 | 1.0 | observational_clock_confounded |
| n232:mem50 | 5/5 | 20.024 | 1.1058 [1.0947, 1.1181] (5 blocks) | 540 | 0.0 | 1.0 | observational_clock_confounded |
- n232: memory100 / arithmetic100: 1.1816 [1.1615, 1.2017] (5 blocks); numerator faster: False; pooled clock ok: False
- n232: memory50 / arithmetic100: 1.0491 [1.0362, 1.0609] (5 blocks); numerator faster: False; pooled clock ok: False
- n152: memory100 / arithmetic100: 1.2134 [1.1988, 1.2277] (5 blocks); numerator faster: False; pooled clock ok: False
- n152: memory50 / arithmetic100: 1.0659 [1.0609, 1.0704] (5 blocks); numerator faster: False; pooled clock ok: False
- n152 absent / n232 mem50 (shared-service balance, not stock-GPU win): 0.9533 [0.9419, 0.9649] (5 blocks); numerator faster: True; pooled clock ok: False
- n152 absent / n232 mem100 (shared-service balance, not stock-GPU win): 0.8464 [0.8318, 0.8616] (5 blocks); numerator faster: True; pooled clock ok: False

## E4 service (mean over acquisitions; ratio to 304)


## E5 / timeline

- e6_timeline:qwen32b_graph_b1_c4096__n304__absent__ktrace:a0: completed=True slowdown=1.1091021508697156 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n272__absent__ktrace:a0: completed=True slowdown=1.100698496345204 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n248__absent__ktrace:a0: completed=True slowdown=1.0857474500599618 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n232__absent__ktrace:a0: completed=True slowdown=1.0934447788200234 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n216__absent__ktrace:a0: completed=True slowdown=1.10680793793388 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n200__absent__ktrace:a0: completed=True slowdown=1.091036670958293 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n184__absent__ktrace:a0: completed=True slowdown=1.0904628948389234 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n152__absent__ktrace:a0: completed=True slowdown=1.0901288630917438 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n232__absent__cu232__ktrace:a0: completed=True slowdown=1.1649881208014272 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_timeline:qwen32b_graph_b1_c4096__n200__absent__cu200__ktrace:a0: completed=True slowdown=1.2616192503478514 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e6_eager_counters:qwen32b_graph_b1_c4096__n304__absent__eager__FETCH_SIZE:a0: completed=False slowdown=None outputs_match=None decode_ranges=0 dispatches/tick=None counter/tick=None counter coverage=None
- e6_eager_counters:qwen32b_graph_b1_c4096__n304__absent__eager__TCC_EA0_RDREQ_LEVEL:a0: completed=True slowdown=3.0923628956150373 outputs_match=True decode_ranges=6 dispatches/tick=733.0 counter/tick={'TCC_EA0_RDREQ_LEVEL': 672002127845.6111} counter coverage=0.9995596844164375
- e6_eager_counters:qwen32b_graph_b1_c4096__n200__absent__eager__FETCH_SIZE:a0: completed=False slowdown=3.6619426398565307 outputs_match=True decode_ranges=0 dispatches/tick=None counter/tick=None counter coverage=None
- e6_eager_counters:qwen32b_graph_b1_c4096__n200__absent__eager__TCC_EA0_RDREQ_LEVEL:a0: completed=True slowdown=3.1639899218942062 outputs_match=True decode_ranges=6 dispatches/tick=733.0 counter/tick={'TCC_EA0_RDREQ_LEVEL': 667577875358.8359} counter coverage=0.9995524661281824

## E6 kernel-shape controls (median us per call; ratio to 304)

- down:linear: 152: 95.2 (1.098), 184: 96.5 (1.112), 200: 96.9 (1.117), 216: 100.3 (1.156), 232: 104.7 (1.207), 248: 103.2 (1.190), 272: 86.4 (0.996), 304: 86.7 (1.000)
- down:wvsplitk_dev: 152: 80.0 (1.176), 184: 73.8 (1.085), 200: 76.8 (1.129), 216: 78.2 (1.149), 232: 78.4 (1.152), 248: 78.0 (1.147), 272: 77.2 (1.135), 304: 68.0 (1.000)
- down:wvsplitk_n: 152: 84.8 (1.248), 184: 82.1 (1.209), 200: 82.8 (1.219), 216: 80.7 (1.188), 232: 69.4 (1.021), 248: 76.8 (1.130), 272: 67.5 (0.993), 304: 68.0 (1.000)
- gate_up:linear: 152: 173.9 (1.133), 184: 170.0 (1.107), 200: 164.4 (1.071), 216: 164.7 (1.073), 232: 166.7 (1.086), 248: 161.7 (1.053), 272: 161.2 (1.050), 304: 153.5 (1.000)
- gate_up:llmm1: 152: 222.2 (1.387), 184: 205.3 (1.282), 200: 197.0 (1.230), 216: 204.1 (1.274), 232: 180.5 (1.127), 248: 172.2 (1.075), 272: 169.7 (1.059), 304: 160.2 (1.000)
- gate_up:wvsplitk_dev: 152: 143.7 (1.068), 184: 141.4 (1.051), 200: 135.3 (1.005), 216: 140.4 (1.044), 232: 141.1 (1.049), 248: 142.6 (1.060), 272: 141.7 (1.053), 304: 134.5 (1.000)
- gate_up:wvsplitk_n: 152: 155.0 (1.187), 184: 147.2 (1.127), 200: 145.8 (1.116), 216: 148.7 (1.138), 232: 147.2 (1.127), 248: 143.1 (1.096), 272: 138.3 (1.059), 304: 130.6 (1.000)
- lm_head:linear: 152: 639.0 (1.543), 184: 543.2 (1.312), 200: 546.5 (1.320), 216: 550.5 (1.329), 232: 463.5 (1.119), 248: 467.9 (1.130), 272: 479.5 (1.158), 304: 414.1 (1.000)
- lm_head:llmm1: 152: 615.4 (1.388), 184: 572.5 (1.292), 200: 546.9 (1.234), 216: 563.5 (1.271), 232: 495.7 (1.118), 248: 479.5 (1.082), 272: 473.2 (1.068), 304: 443.2 (1.000)
- lm_head:wvsplitk_dev: 152: 396.5 (1.056), 184: 381.0 (1.015), 200: 372.0 (0.991), 216: 392.0 (1.044), 232: 389.6 (1.038), 248: 386.5 (1.030), 272: 394.1 (1.050), 304: 375.4 (1.000)
- lm_head:wvsplitk_n: 152: 424.4 (1.176), 184: 409.9 (1.135), 200: 408.6 (1.132), 216: 406.6 (1.126), 232: 404.7 (1.121), 248: 389.9 (1.080), 272: 391.5 (1.084), 304: 361.0 (1.000)
- o_proj:linear: 152: 22.2 (1.079), 184: 22.2 (1.080), 200: 22.7 (1.106), 216: 22.3 (1.086), 232: 23.5 (1.145), 248: 22.1 (1.077), 272: 20.7 (1.007), 304: 20.5 (1.000)
- o_proj:llmm1: 152: 22.0 (1.154), 184: 20.9 (1.100), 200: 20.3 (1.069), 216: 20.3 (1.066), 232: 20.4 (1.073), 248: 19.4 (1.021), 272: 19.6 (1.028), 304: 19.0 (1.000)
- o_proj:wvsplitk_dev: 152: 18.1 (1.246), 184: 16.1 (1.114), 200: 15.5 (1.073), 216: 16.2 (1.117), 232: 16.9 (1.167), 248: 16.6 (1.147), 272: 16.7 (1.154), 304: 14.5 (1.000)
- o_proj:wvsplitk_n: 152: 16.9 (1.161), 184: 16.8 (1.157), 200: 16.9 (1.160), 216: 16.8 (1.154), 232: 14.4 (0.992), 248: 15.8 (1.086), 272: 14.6 (1.005), 304: 14.5 (1.000)
- qkv:linear: 152: 39.3 (1.389), 184: 32.0 (1.132), 200: 32.7 (1.155), 216: 32.7 (1.156), 232: 33.7 (1.193), 248: 34.5 (1.219), 272: 35.9 (1.269), 304: 28.3 (1.000)
- qkv:llmm1: 152: 30.7 (1.264), 184: 27.6 (1.133), 200: 26.5 (1.090), 216: 26.9 (1.107), 232: 26.2 (1.078), 248: 25.4 (1.043), 272: 25.2 (1.037), 304: 24.3 (1.000)
- qkv:wvsplitk_dev: 152: 21.5 (1.226), 184: 19.2 (1.094), 200: 18.9 (1.076), 216: 18.8 (1.073), 232: 19.2 (1.092), 248: 19.2 (1.092), 272: 20.3 (1.156), 304: 17.6 (1.000)
- qkv:wvsplitk_n: 152: 23.7 (1.320), 184: 23.8 (1.328), 200: 24.2 (1.350), 216: 21.5 (1.195), 232: 18.3 (1.021), 248: 22.1 (1.231), 272: 18.1 (1.010), 304: 17.9 (1.000)

## Leak test v2 (per-CU probe)

- qwen32b_graph_b1_c4096__n232__cuprobe_L0: {'expected_cus': 72, 'alone_cus': 72, 'covered_cus': [72, 72, 72, 72, 72], 'native_min_q': [0.999978174536214, 0.9999888531700939], 'positive_cus_below': [8], 'decoder_min_q': [0.999974560831766], 'decoder_cus_below': [0], 'sensitive': True, 'threshold': 0.9499781745362139, 'ok': True}
- qwen32b_graph_b1_c4096__n152__cuprobe_L0: {'expected_cus': 152, 'alone_cus': 152, 'covered_cus': [152, 152, 152, 152, 152], 'native_min_q': [0.8866353641804233, 0.8567908518738369], 'positive_cus_below': [8], 'decoder_min_q': [0.8563857982187532], 'decoder_cus_below': [0], 'sensitive': True, 'threshold': 0.8067908518738368, 'ok': True}
- qwen32b_graph_b1_c4096__n232__cuprobe_L1: {'expected_cus': 72, 'alone_cus': 72, 'covered_cus': [72, 72, 72, 72, 72], 'native_min_q': [0.9999849542301, 0.9999800005323777], 'positive_cus_below': [8], 'decoder_min_q': [0.9999841102536532], 'decoder_cus_below': [0], 'sensitive': True, 'threshold': 0.9499800005323776, 'ok': True}
- qwen32b_graph_b1_c4096__n152__cuprobe_L1: {'expected_cus': 152, 'alone_cus': 152, 'covered_cus': [152, 152, 152, 152, 152], 'native_min_q': [0.8874464329064718, 0.8511869562553999], 'positive_cus_below': [8], 'decoder_min_q': [0.8759590608222139], 'decoder_cus_below': [0], 'sensitive': True, 'threshold': 0.8011869562553998, 'ok': True}

## E8 guardrails

