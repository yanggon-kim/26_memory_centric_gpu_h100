# cubw_v1 auto — mi325x

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **False**
- leak_ok: **False**
- leak_ok_v11: **True**
- clock_ok: **False**
- release_e2: **False**
- release_e3_minimal: **False**
- release_e3_full: **False**
- release_e5e8: **False**
- outputs: {'runs': 109, 'complete': 21, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {'qwen32b_graph_b1_c4096__n232__arith100': 0.7984380806613881, 'qwen32b_graph_b1_c4096__n152__arith100': 0.8389708299018613, 'qwen32b_graph_b1_c4096__n232__arith100_L3': 0.7883794730793332}
- leak v1.1 (relative to placement-proven native co-runners): {'qwen32b_graph_b1_c4096__n232__arith100': {'decoder_ratio': 0.7984380806613881, 'native_ratios': {'arith100': 0.8668176013739904, 'mem100': 0.8115665124084588}, 'relative': 0.9211142913985432}, 'qwen32b_graph_b1_c4096__n152__arith100': {'decoder_ratio': 0.8353894709379531, 'native_ratios': {'arith100': 0.8631173714302185, 'mem100': 0.8081021356978147}, 'relative': 0.9678747046345283}, 'qwen32b_graph_b1_c4096__n232__arith100_L3': {'decoder_ratio': 0.785014076104561, 'native_ratios': {'arith100': 0.8631173714302185, 'mem100': 0.8081021356978147}, 'relative': 0.9095102266378472}, 'qwen32b_graph_b1_c4096__n152__leakprobe_L3': {'decoder_ratio': 0.7482306590577632, 'native_ratios': {'mem100': 0.7894317654172215, 'arith100': 0.8605110272641269}, 'relative': 0.8695189664642147}}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': 1494, 'gfx_max': 2076, 'gfx_ratio': 1.3895582329317269, 'mem_min': 1285, 'mem_max': 1500, 'samples': 748, 'coverage_min': 0.6661113819667596}

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
| comma_list_0-7,16-23 | 16 | 16 | {'0': 2, '1': 2, '2': 2, '3': 2, '4': 2, '5': 2, '6': 2, '7': 2} | True | 0 |
| wrong_index_mask | 304 | 304 | {'0': 38, '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38} | True | 0 |
| empty_xcd_0-6 | 45 | 7 | {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 38} | False | 3 |

## E2 latency ratio R(n) = T(n)/T(304)

### e2:qwen14b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 0/3 | None | n/a | None (gfx xNone) |   |
| 232 | 0/3 | None | n/a | None (gfx xNone) |   |
| 200 | 0/3 | None | n/a | None (gfx xNone) |   |
| 152 | 0/3 | None | n/a | None (gfx xNone) |   |
| 96 | 0/3 | None | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

### e2:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 0/3 | None | n/a | None (gfx xNone) |   |
| 232 | 0/3 | None | n/a | None (gfx xNone) |   |
| 200 | 0/3 | None | n/a | None (gfx xNone) |   |
| 152 | 0/3 | None | n/a | None (gfx xNone) |   |
| 96 | 0/3 | None | n/a | None (gfx xNone) |   |
Smallest qualified noninferior n: None

### e1_e2_pilot:qwen14b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 1/1 | 7.7951 | n/a | None (gfx xNone) |   |
| 232 | 1/1 | 9.1839 | 1.1782 [1.1623, 1.1939] (1 blocks) | False (gfx x1.171) | pilot one pilot acquisition; never promoted to a production comparison |
| 200 | 1/1 | 8.2087 | 1.0531 [1.0472, 1.0595] (1 blocks) | False (gfx x1.18) | pilot one pilot acquisition; never promoted to a production comparison |
| 152 | 1/1 | 9.1354 | 1.1719 [1.1677, 1.1764] (1 blocks) | False (gfx x1.191) | pilot one pilot acquisition; never promoted to a production comparison |
| 96 | 1/1 | 9.435 | 1.2104 [1.1975, 1.2244] (1 blocks) | False (gfx x1.19) | pilot one pilot acquisition; never promoted to a production comparison |
Smallest qualified noninferior n: None

### e1_e2_pilot:qwen32b
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 1/1 | 16.2326 | n/a | None (gfx xNone) |   |
| 232 | 1/1 | 17.3547 | 1.0691 [1.0441, 1.0968] (1 blocks) | False (gfx x1.175) | pilot one pilot acquisition; never promoted to a production comparison |
| 200 | 1/1 | 16.4613 | 1.0141 [0.9930, 1.0371] (1 blocks) | False (gfx x1.29) | pilot one pilot acquisition; never promoted to a production comparison |
| 152 | 1/1 | 18.7231 | 1.1534 [1.1307, 1.1780] (1 blocks) | False (gfx x1.23) | pilot one pilot acquisition; never promoted to a production comparison |
| 96 | 1/1 | 17.9875 | 1.1081 [1.0874, 1.1312] (1 blocks) | False (gfx x1.251) | pilot one pilot acquisition; never promoted to a production comparison |
Smallest qualified noninferior n: None

### e1_e2_pilot:qwen32b:gpu7
| n | complete | ms/step | R(n) [95% CI] | pooled clock vs 304 | disposition |
|---:|---:|---:|---|---|---|
| 304 | 1/1 | 16.2407 | n/a | None (gfx xNone) |   |
| 232 | 1/1 | 16.4387 | 1.0122 [0.9996, 1.0259] (1 blocks) | False (gfx x1.151) | pilot one pilot acquisition; never promoted to a production comparison |
| 200 | 1/1 | 16.858 | 1.0380 [1.0050, 1.0732] (1 blocks) | False (gfx x1.217) | pilot one pilot acquisition; never promoted to a production comparison |
| 152 | 1/1 | 17.8709 | 1.1004 [1.0860, 1.1161] (1 blocks) | False (gfx x1.205) | pilot one pilot acquisition; never promoted to a production comparison |
| 96 | 1/1 | 18.4341 | 1.1351 [1.1194, 1.1520] (1 blocks) | False (gfx x1.216) | pilot one pilot acquisition; never promoted to a production comparison |
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
| n232:inactive | 0/1 | None | n/a | None | None | None |  |
| n232:mem100 | 0/3 | None | n/a | None | None | None |  |
| n232:mem25 | 0/3 | None | n/a | None | None | None |  |
| n232:mem50 | 0/3 | None | n/a | None | None | None |  |
| n232:mem75 | 0/3 | None | n/a | None | None | None |  |

### e1_e3_pilot:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n152:absent | 1/1 | 17.8321 | n/a | None | None | None |  |
| n152:arith100 | 1/1 | 18.6342 | 1.0450 [1.0392, 1.0498] (1 acq) | 0.0 | 14.6 | 1.0 | pilot |
| n152:mem100 | 1/1 | 21.2982 | 1.1944 [1.1748, 1.2168] (1 acq) | 1299 | 0.0 | 1.0 | pilot |
| n232:absent | 1/1 | 17.153 | n/a | None | None | None |  |
| n232:arith100 | 1/1 | 17.1836 | 1.0018 [0.9891, 1.0134] (1 acq) | 0.0 | 13.9 | 1.0 | pilot |
| n232:inactive | 1/1 | 17.3291 | 1.0103 [0.9848, 1.0362] (1 acq) | 0.0 | 0.0 | 1.0 | pilot |
| n232:mem100 | 1/1 | 20.0584 | 1.1694 [1.1545, 1.1835] (1 acq) | 1170 | 0.0 | 1.0 | pilot |
- n232: memory100 / arithmetic100: 1.1673 [1.1583, 1.1773] (1 acq); numerator faster: False; pooled clock ok: False
- n152: memory100 / arithmetic100: 1.1430 [1.1234, 1.1656] (1 acq); numerator faster: False; pooled clock ok: False
- n152 absent / n232 mem100 (shared-service balance, not stock-GPU win): 0.8890 [0.8833, 0.8941] (1 acq); numerator faster: True; pooled clock ok: False

### e1_leak:qwen32b
| cell | complete | T ms/step | ratio to absent (same n) | competitor GB/s | TFLOP/s | coverage | disposition |
|---|---:|---:|---|---:|---:|---:|---|
| n152:leakprobe | 1/1 | 19.1096 | n/a | 0.0 | 46.9 | 1.0 |  |
| n232:arith100 | 1/1 | 17.4865 | n/a | 0.0 | 13.7 | 1.0 |  |

## E4 service (mean over acquisitions; ratio to 304)

### e4
- chase_c1 [ns_per_dependent_load]: 152: 147.0 (0.955), 200: 154.8 (1.005), 232: 151.3 (0.983), 304: 153.9, 96: 145.2 (0.943)
- chase_c19456 [ns_per_dependent_load]: 152: 759.1 (0.990), 200: 759.0 (0.990), 232: 775.8 (1.012), 304: 766.5, 96: 799.4 (1.043)
- chase_c304 [ns_per_dependent_load]: 152: 854.9 (0.975), 200: 864.4 (0.986), 232: 866.3 (0.988), 304: 876.5, 96: 833.1 (0.951)
- chase_c4864 [ns_per_dependent_load]: 152: 779.4 (0.943), 200: 787.9 (0.953), 232: 805.2 (0.974), 304: 826.7, 96: 784.9 (0.949)
- fma [useful_tflop_per_s]: 152: 62.7 (0.558), 200: 79.5 (0.707), 232: 85.0 (0.756), 304: 112.4, 96: 44.9 (0.399)
- gather1g [useful_gb_per_s]: 152: 427.9 (0.661), 200: 581.0 (0.897), 232: 607.7 (0.939), 304: 647.5, 96: 311.9 (0.482)
- grid1216 [useful_gb_per_s]: 152: 3501.4 (0.723), 200: 4805.8 (0.992), 232: 4768.9 (0.985), 304: 4842.9, 96: 4149.4 (0.857)
- grid304 [useful_gb_per_s]: 152: 2503.6 (0.897), 200: 2590.1 (0.928), 232: 2586.1 (0.927), 304: 2790.8, 96: 2510.2 (0.899)
- grid9728 [useful_gb_per_s]: 152: 4701.1 (1.003), 200: 4772.0 (1.018), 232: 4754.2 (1.014), 304: 4688.7, 96: 4442.2 (0.947)
- ilp1 [useful_gb_per_s]: 152: 3321.1 (0.695), 200: 4244.3 (0.889), 232: 3980.9 (0.834), 304: 4775.2, 96: 2608.2 (0.546)
- ilp16 [useful_gb_per_s]: 152: 4160.9 (0.893), 200: 4387.6 (0.942), 232: 4382.8 (0.941), 304: 4657.0, 96: 2990.3 (0.642)
- ilp2 [useful_gb_per_s]: 152: 4068.9 (0.868), 200: 4378.3 (0.934), 232: 4238.6 (0.904), 304: 4686.6, 96: 2913.4 (0.622)
- ilp4 [useful_gb_per_s]: 152: 3956.9 (0.854), 200: 4304.3 (0.929), 232: 4327.6 (0.934), 304: 4632.6, 96: 2863.3 (0.618)
- ilp8 [useful_gb_per_s]: 152: 4147.1 (0.914), 200: 4354.8 (0.960), 232: 4354.2 (0.959), 304: 4538.4, 96: 2949.7 (0.650)
- mfma [useful_tflop_per_s]: 152: 515.7 (0.419), 200: 711.5 (0.578), 232: 817.1 (0.664), 304: 1229.9, 96: 370.3 (0.301)
- reuse128m [useful_gb_per_s]: 152: 5726.3 (1.004), 200: 5830.0 (1.022), 232: 6368.8 (1.116), 304: 5705.7, 96: 6461.6 (1.132)
- reuse32m [useful_gb_per_s]: 152: 15194.7 (0.619), 200: 19668.8 (0.801), 232: 22441.9 (0.914), 304: 24563.9, 96: 11292.4 (0.460)
- reuse512m [useful_gb_per_s]: 152: 5025.4 (1.016), 200: 5346.2 (1.081), 232: 4990.7 (1.009), 304: 4945.8, 96: 4941.5 (0.999)
- reuse8m [useful_gb_per_s]: 152: 15685.9 (0.498), 200: 20210.9 (0.641), 232: 22296.7 (0.708), 304: 31513.5, 96: 11221.1 (0.356)
- seqread4g [useful_gb_per_s]: 152: 4173.5 (0.843), 200: 4753.7 (0.960), 232: 4669.2 (0.943), 304: 4950.1, 96: 4047.4 (0.818)
- stride16 [useful_gb_per_s]: 152: 424.2 (0.712), 200: 542.9 (0.912), 232: 529.0 (0.888), 304: 595.4, 96: 314.5 (0.528)
- stride4 [useful_gb_per_s]: 152: 1112.4 (0.944), 200: 1111.6 (0.944), 232: 1146.2 (0.973), 304: 1178.0, 96: 1103.7 (0.937)
- comp_fma [useful_tflop_per_s]: competitor: 17.6
- comp_read1g [useful_gb_per_s]: competitor: 1949.6
- comp_seqread4g [useful_gb_per_s]: competitor: 1971.8
### e4_pilot
- chase_c1 [ns_per_dependent_load]: 152: 147.7 (0.902), 200: 151.9 (0.927), 232: 146.4 (0.894), 304: 163.8, 96: 143.9 (0.879)
- chase_c19456 [ns_per_dependent_load]: 152: 758.9 (0.989), 200: 756.6 (0.986), 232: 761.5 (0.992), 304: 767.6, 96: 797.8 (1.039)
- chase_c304 [ns_per_dependent_load]: 152: 839.4 (0.985), 200: 848.9 (0.996), 232: 843.9 (0.990), 304: 852.1, 96: 832.7 (0.977)
- chase_c4864 [ns_per_dependent_load]: 152: 778.2 (0.968), 200: 786.1 (0.978), 232: 827.0 (1.029), 304: 803.8, 96: 766.8 (0.954)
- fma [useful_tflop_per_s]: 152: 64.5 (0.603), 200: 79.0 (0.738), 232: 62.2 (0.581), 304: 107.1, 96: 42.7 (0.399)
- gather1g [useful_gb_per_s]: 152: 451.8 (0.831), 200: 581.4 (1.069), 232: 610.8 (1.123), 304: 543.9, 96: 312.1 (0.574)
- grid1216 [useful_gb_per_s]: 152: 3509.2 (0.725), 200: 4800.6 (0.992), 232: 4777.7 (0.988), 304: 4837.0, 96: 4161.9 (0.860)
- grid304 [useful_gb_per_s]: 152: 2606.7 (0.933), 200: 2695.9 (0.965), 232: 2687.6 (0.962), 304: 2794.2, 96: 2516.7 (0.901)
- grid9728 [useful_gb_per_s]: 152: 4742.9 (1.022), 200: 4788.4 (1.032), 232: 3145.3 (0.678), 304: 4640.4, 96: 4459.7 (0.961)
- ilp1 [useful_gb_per_s]: 152: 3328.8 (0.686), 200: 4249.3 (0.875), 232: 3987.6 (0.821), 304: 4854.9, 96: 2615.0 (0.539)
- ilp16 [useful_gb_per_s]: 152: 4304.7 (2.345), 200: 4388.8 (2.391), 232: 2527.4 (1.377), 304: 1835.9, 96: 2995.8 (1.632)
- ilp2 [useful_gb_per_s]: 152: 4260.6 (0.909), 200: 2600.7 (0.555), 232: 4259.4 (0.909), 304: 4686.7, 96: 2913.8 (0.622)
- ilp4 [useful_gb_per_s]: 152: 4084.1 (0.882), 200: 4205.3 (0.909), 232: 4337.5 (0.937), 304: 4628.8, 96: 2888.7 (0.624)
- ilp8 [useful_gb_per_s]: 152: 4292.6 (0.930), 200: 4359.7 (0.945), 232: 4375.2 (0.948), 304: 4615.7, 96: 2972.9 (0.644)
- mfma [useful_tflop_per_s]: 152: 568.1 (0.608), 200: 725.0 (0.775), 232: 779.3 (0.833), 304: 935.1, 96: 357.3 (0.382)
- reuse128m [useful_gb_per_s]: 152: 5674.6 (0.996), 200: 5845.8 (1.026), 232: 6349.2 (1.115), 304: 5695.2, 96: 6481.8 (1.138)
- reuse32m [useful_gb_per_s]: 152: 15516.7 (0.644), 200: 19004.5 (0.789), 232: 21817.1 (0.905), 304: 24094.6, 96: 11287.0 (0.468)
- reuse512m [useful_gb_per_s]: 152: 5032.3 (1.091), 200: 5351.7 (1.160), 232: 4976.6 (1.078), 304: 4614.6, 96: 4919.2 (1.066)
- reuse8m [useful_gb_per_s]: 152: 16669.0 (0.525), 200: 19915.3 (0.627), 232: 22550.2 (0.710), 304: 31771.6, 96: 11222.8 (0.353)
- seqread4g [useful_gb_per_s]: 152: 4185.6 (0.839), 200: 4753.7 (0.952), 232: 4658.2 (0.933), 304: 4991.5, 96: 4065.2 (0.814)
- stride16 [useful_gb_per_s]: 152: 433.6 (0.734), 200: 549.7 (0.930), 232: 523.9 (0.886), 304: 591.1, 96: 314.2 (0.532)
- stride4 [useful_gb_per_s]: 152: 1145.0 (0.972), 200: 1116.2 (0.948), 232: 1146.9 (0.974), 304: 1177.7, 96: 1108.0 (0.941)
- comp_fma [useful_tflop_per_s]: competitor: 17.6
- comp_read1g [useful_gb_per_s]: competitor: 1952.5
- comp_seqread4g [useful_gb_per_s]: competitor: 1973.7

## E5 / timeline

- e1_timeline:qwen32b_graph_b1_c4096__n304__absent__ktrace:a0: completed=True slowdown=1.1481518755531848 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e1_timeline:qwen32b_graph_b1_c4096__n152__absent__ktrace:a0: completed=True slowdown=1.072335309946249 outputs_match=True decode_ranges=13 dispatches/tick=731.0 counter/tick=None counter coverage=None
- e5:qwen32b_graph_b1_c4096__n200__absent__ktrace:a0: completed=None slowdown=None outputs_match=None decode_ranges=None dispatches/tick=None counter/tick=None counter coverage=None
- e5:qwen32b_graph_b1_c4096__n304__absent__ktrace:a0: completed=None slowdown=None outputs_match=None decode_ranges=None dispatches/tick=None counter/tick=None counter coverage=None

## E8 guardrails

- e8_pilot:qwen7b_graph_b8_c4096__n304__absent: 1/1 ms/step=[6.2683] prefill_ms=[826.7] J/token=[] W=[]
- e8_pilot:qwen32b_graph_b1_c4096__n304__absent__prefill: 1/1 ms/step=[21.1377] prefill_ms=[623.3] J/token=[] W=[]
- e8_pilot:qwen7b_graph_b8_c4096__n152__absent: 1/1 ms/step=[10.122] prefill_ms=[1571.1] J/token=[] W=[]
- e8_pilot:qwen32b_graph_b1_c4096__n152__absent__prefill: 1/1 ms/step=[18.7509] prefill_ms=[1006.5] J/token=[] W=[]
- e8_pilot:qwen32b_graph_b1_c16384__n304__absent: 1/1 ms/step=[16.9713] prefill_ms=[2617.3] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n96__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n96__absent__prefill: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n200__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n200__absent__prefill: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen7b_graph_b8_c4096__n304__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n304__absent__prefill: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c16384__n304__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c16384__n200__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n304__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:mistral7b_graph_b1_c4096__n200__absent: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
- e8:qwen32b_graph_b1_c4096__n304__absent__energy: 0/3 ms/step=[] prefill_ms=[] J/token=[] W=[]
