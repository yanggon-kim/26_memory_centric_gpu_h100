# G5 extension — mi325x (g5x)

- placement_ok True, lifecycle_ok True, leak_ok_v4 False
- leak v5: available True, sensitive True, ok **False**
- G5 disposition: **observational: isolation not established**

## Leak v5 per GPU and level

| cell | decoder T8 / M / flags | max native T8 / M | min positive T8 | thr T8 / M | storm kernels/s (matched) | ok |
|---|---|---|---|---|---|---|
| qwen32b_graph_b1_c4096__n128__gapprobe_L0 | 0.084 / 0.085 / 0 | 0.083 / 0.083 | 0.428 | 0.255 / 0.183 | storm_lds 171486 (True), storm_small 157107 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L1 | 0.084 / 0.084 / 0 | 0.005 / 0.005 | 0.434 | 0.220 / 0.105 | storm_small 154777 (True), storm_lds 175945 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L2 | 0.065 / 0.065 / 0 | 0.063 / 0.063 | 0.442 | 0.253 / 0.163 | storm_small 157185 (True), storm_lds 175903 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L3 | 0.085 / 0.086 / 0 | 0.084 / 0.084 | 0.411 | 0.247 / 0.184 | storm_small 155186 (True), storm_lds 170019 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L4 | 0.016 / 0.016 / 0 | -0.000 / -0.000 | 0.429 | 0.214 / 0.100 | storm_small 158002 (True), storm_lds 172755 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L5 | 0.085 / 0.085 / 0 | 0.070 / 0.070 | 0.437 | 0.253 / 0.170 | storm_small 155082 (True), storm_lds 176211 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L6 | 0.085 / 0.085 / 0 | 0.068 / 0.068 | 0.437 | 0.252 / 0.168 | storm_small 151964 (True), storm_lds 176060 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L7 | 0.084 / 0.084 / 0 | 0.078 / 0.078 | 0.427 | 0.252 / 0.178 | storm_small 157346 (True), storm_lds 176286 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L0 | 0.222 / 0.223 / 32 | 0.208 / 0.208 | 0.541 | 0.374 / 0.308 | storm_small 150375 (True), storm_lds 169955 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L1 | 0.250 / 0.251 / 16 | 0.186 / 0.186 | 0.503 | 0.344 / 0.286 | storm_lds 172234 (True), storm_small 152899 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L2 | 0.250 / 0.250 / 24 | 0.249 / 0.250 | 0.543 | 0.396 / 0.350 | storm_small 152647 (True), storm_lds 166499 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L3 | 0.223 / 0.224 / 32 | 0.222 / 0.222 | 0.441 | 0.331 / 0.322 | storm_small 153245 (True), storm_lds 170520 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L4 | 0.150 / 0.150 / 28 | 0.138 / 0.138 | 0.534 | 0.336 / 0.238 | storm_small 149557 (True), storm_lds 169882 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L5 | 0.160 / 0.160 / 32 | -0.001 / 0.000 | 0.435 | 0.217 / 0.100 | storm_lds 167129 (True), storm_small 152347 (True) | False |
| qwen32b_graph_b1_c4096__n256__gapprobe_L6 | 0.251 / 0.252 / 32 | 0.104 / 0.104 | 0.550 | 0.327 / 0.204 | storm_lds 171295 (True), storm_small 152586 (True) | False |
| qwen32b_graph_b1_c4096__n256__gapprobe_L7 | 0.251 / 0.251 / 32 | 0.250 / 0.250 | 0.481 | 0.365 / 0.350 | storm_small 149767 (True), storm_lds 171671 (True) | True |

## G5 (decoder time ratio vs competitor absent, same block)

- n=256: mem50 1.0651 [1.0592, 1.0737] (8 blocks) (463 GB/s); inactive 0.9993 [0.9963, 1.0019] (8 blocks); arith100 1.0546 [1.0500, 1.0589] (8 blocks); mem100 1.1571 [1.1456, 1.1680] (8 blocks) (954 GB/s); mem25 1.0286 [1.0262, 1.0306] (8 blocks) (231 GB/s); mem100/arith100 1.0973 [1.0903, 1.1039] (8 blocks)
- n=192: inactive 1.0000 [0.9973, 1.0030] (8 blocks); mem50 1.0751 [1.0721, 1.0787] (8 blocks) (525 GB/s); arith100 1.0449 [1.0359, 1.0530] (8 blocks); mem25 1.0347 [1.0330, 1.0364] (8 blocks) (271 GB/s); mem100 1.1787 [1.1689, 1.1883] (8 blocks) (1073 GB/s); mem100/arith100 1.1280 [1.1217, 1.1341] (8 blocks)
- n=128: mem100 1.2113 [1.2049, 1.2180] (8 blocks) (1232 GB/s); arith100 1.0337 [1.0259, 1.0410] (8 blocks); mem25 1.0423 [1.0395, 1.0448] (8 blocks) (307 GB/s); inactive 0.9999 [0.9979, 1.0016] (8 blocks); mem50 1.0882 [1.0850, 1.0909] (8 blocks) (598 GB/s); mem100/arith100 1.1718 [1.1644, 1.1786] (8 blocks)

Observational with respect to clock (controlled clocks not authorized).
