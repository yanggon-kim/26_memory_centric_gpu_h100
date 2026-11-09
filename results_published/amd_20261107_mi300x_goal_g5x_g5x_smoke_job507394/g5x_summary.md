# G5 extension — mi300x (g5x_smoke)

- placement_ok True, lifecycle_ok True, leak_ok_v4 True
- leak v5: available True, sensitive True, ok **True**
- G5 disposition: **isolated (leak v5)**

## Leak v5 per GPU and level

| cell | decoder T8 / M / flags | max native T8 / M | min positive T8 | thr T8 / M | storm kernels/s (matched) | ok |
|---|---|---|---|---|---|---|
| qwen32b_graph_b1_c4096__n256__gapprobe_L0 | 0.001 / 0.002 / 0 | 0.014 / 0.015 | 0.645 | 0.329 / 0.115 | storm_lds 0 (False), storm_small 137056 (True) | True |

## G5 (decoder time ratio vs competitor absent, same block)

- n=256: mem100 1.1972 [1.1942, 1.2002] (1 blocks) (868 GB/s); arith100 1.0698 [1.0668, 1.0729] (1 blocks); mem100/arith100 1.1190 [1.1172, 1.1206] (1 blocks)

Observational with respect to clock (controlled clocks not authorized).
