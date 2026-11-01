# cubw_v1 production — mi300x

## Gates

- placement_ok: **True**
- lifecycle_ok: **False**
- outputs_ok: **True**
- leak_ok: **False**
- leak_ok_v11: **False**
- clock_ok: **False**
- release_e2: **True**
- release_e3_minimal: **False**
- release_e3_full: **False**
- release_e5e8: **True**
- outputs: {'runs': 25, 'complete': 25, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {}
- leak v1.1 (relative to placement-proven native co-runners): {}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1294, 'gfx_max': 2015, 'gfx_ratio': 1.5571870170015456, 'mem_min': 1068, 'mem_max': 1300, 'samples': 850, 'coverage_min': 0.4949770262779862}

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
| 304 | 3/3 | 8.644 | n/a | None (gfx xNone) |   |
| 232 | 3/3 | 9.0852 | 1.0510 [1.0468, 1.0539] (3 blocks) | False (gfx x1.139) | observational_clock_confounded pooled clock envelope failed |
| 200 | 3/3 | 8.8728 | 1.0265 [1.0239, 1.0295] (3 blocks) | False (gfx x1.171) | observational_clock_confounded pooled clock envelope failed |
| 152 | 3/3 | 9.5647 | 1.1065 [1.1035, 1.1092] (3 blocks) | False (gfx x1.262) | observational_clock_confounded pooled clock envelope failed |
| 96 | 3/3 | 9.5222 | 1.1016 [1.0963, 1.1057] (3 blocks) | False (gfx x1.291) | observational_clock_confounded pooled clock envelope failed |
Smallest qualified noninferior n: None

### e2:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 5/5 | 17.7781 | n/a | None (gfx xNone) |   |
| 232 | 5/5 | 18.2035 | 1.0239 [1.0203, 1.0270] (5 blocks) | False (gfx x1.321) | observational_clock_confounded pooled clock envelope failed |
| 200 | 5/5 | 17.9887 | 1.0118 [1.0102, 1.0135] (5 blocks) | False (gfx x1.359) | observational_clock_confounded pooled clock envelope failed |
| 152 | 5/5 | 19.0283 | 1.0703 [1.0641, 1.0745] (5 blocks) | False (gfx x1.479) | observational_clock_confounded pooled clock envelope failed |
| 96 | 5/5 | 19.246 | 1.0826 [1.0753, 1.0880] (5 blocks) | False (gfx x1.557) | observational_clock_confounded pooled clock envelope failed |
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

## E4 service (mean over acquisitions; ratio to 304)


## E5 / timeline

- e5:qwen32b_graph_b1_c4096__n200__absent__ktrace:a0: completed=True slowdown=1.0633067812149999 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e5:qwen32b_graph_b1_c4096__n304__absent__ktrace:a0: completed=True slowdown=1.0549836308265985 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None

## E8 guardrails

- e8:qwen7b_graph_b8_c4096__n96__absent: 3/3 ms/step=[10.8161, 10.7729, 10.8321] prefill_ms=[2041.2, 2040.2, 2043.7] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n96__absent__prefill: 3/3 ms/step=[20.2673, 20.2125, 20.404] prefill_ms=[1246.2, 1232.2, 1247.5] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n200__absent: 3/3 ms/step=[7.8115, 7.8147, 7.8268] prefill_ms=[1263.7, 1264.6, 1262.1] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n200__absent__prefill: 3/3 ms/step=[19.7358, 18.8696, 19.6741] prefill_ms=[785.4, 768.2, 787.4] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n304__absent: 3/3 ms/step=[6.6386, 6.8335, 6.6331] prefill_ms=[942.2, 928.8, 940.6] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n304__absent__prefill: 3/3 ms/step=[18.7734, 18.8373, 18.7803] prefill_ms=[551.3, 552.9, 550.1] J/token=[] W=[]
- e8:qwen32b_graph_b1_c16384__n304__absent: 3/3 ms/step=[19.0282, 19.1234, 19.057] prefill_ms=[2811.6, 2803.2, 2815.8] J/token=[] W=[]
- e8:qwen32b_graph_b1_c16384__n200__absent: 3/3 ms/step=[19.2035, 19.7987, 19.282] prefill_ms=[3805.8, 3836.6, 3816.2] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n304__absent: 3/3 ms/step=[5.1195, 5.1122, 5.1335] prefill_ms=[140.4, 141.9, 140.5] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n200__absent: 3/3 ms/step=[5.3518, 5.2849, 5.3211] prefill_ms=[195.0, 194.9, 195.1] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n200__absent__energy: 3/3 ms/step=[18.3286, 18.3002, 18.3096] prefill_ms=[783.2, 783.8, 782.2] J/token=[13.7908, 13.768, 13.7739] W=[752.4, 752.3, 752.3]
- e8:qwen32b_graph_b1_c4096__n304__absent__energy: 3/3 ms/step=[18.554, 18.1715, 18.4243] prefill_ms=[566.9, 564.1, 563.5] J/token=[13.9584, 13.6734, 13.8585] W=[752.3, 752.5, 752.2]
