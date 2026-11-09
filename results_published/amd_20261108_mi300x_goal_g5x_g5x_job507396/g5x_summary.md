# G5 extension — mi300x (g5x)

- placement_ok True, lifecycle_ok True, leak_ok_v4 False
- leak v5: available True, sensitive True, ok **False**
- G5 disposition: **observational: isolation not established**

## Leak v5 per GPU and level

| cell | decoder T8 / M / flags | max native T8 / M | min positive T8 | thr T8 / M | storm kernels/s (matched) | ok |
|---|---|---|---|---|---|---|
| qwen32b_graph_b1_c4096__n128__gapprobe_L0 | 0.085 / 0.085 / 0 | 0.083 / 0.083 | 0.425 | 0.254 / 0.183 | storm_lds 173749 (True), storm_small 155970 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L1 | 0.069 / 0.069 / 0 | 0.042 / 0.042 | 0.425 | 0.233 / 0.142 | storm_lds 174795 (True), storm_small 156045 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L2 | 0.052 / 0.052 / 0 | 0.052 / 0.052 | 0.423 | 0.237 / 0.152 | storm_small 156472 (True), storm_lds 174265 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L3 | 0.085 / 0.085 / 0 | 0.084 / 0.085 | 0.431 | 0.258 / 0.185 | storm_lds 174206 (True), storm_small 155933 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L4 | 0.084 / 0.084 / 0 | 0.085 / 0.085 | 0.430 | 0.258 / 0.185 | storm_lds 172807 (True), storm_small 155062 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L5 | 0.024 / 0.024 / 0 | 0.022 / 0.022 | 0.437 | 0.230 / 0.122 | storm_small 156347 (True), storm_lds 174286 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L6 | 0.085 / 0.085 / 0 | 0.025 / 0.025 | 0.459 | 0.242 / 0.125 | storm_lds 172649 (True), storm_small 155314 (True) | True |
| qwen32b_graph_b1_c4096__n128__gapprobe_L7 | 0.063 / 0.063 / 0 | 0.044 / 0.044 | 0.464 | 0.254 / 0.144 | storm_lds 173967 (True), storm_small 156643 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L0 | 0.251 / 0.251 / 28 | 0.250 / 0.250 | 0.588 | 0.419 / 0.350 | storm_lds 156598 (True), storm_small 139777 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L1 | 0.041 / 0.042 / 0 | 0.001 / 0.001 | 0.592 | 0.296 / 0.101 | storm_lds 156173 (True), storm_small 142405 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L2 | 0.251 / 0.251 / 32 | 0.001 / 0.001 | 0.499 | 0.250 / 0.101 | storm_lds 157415 (True), storm_small 139875 (True) | False |
| qwen32b_graph_b1_c4096__n256__gapprobe_L3 | 0.251 / 0.251 / 28 | 0.250 / 0.250 | 0.604 | 0.427 / 0.350 | storm_small 142503 (True), storm_lds 157160 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L4 | 0.250 / 0.251 / 12 | 0.102 / 0.102 | 0.473 | 0.287 / 0.202 | storm_small 141063 (True), storm_lds 155042 (True) | False |
| qwen32b_graph_b1_c4096__n256__gapprobe_L5 | 0.251 / 0.251 / 32 | 0.250 / 0.250 | 0.586 | 0.418 / 0.350 | storm_lds 156673 (True), storm_small 142259 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L6 | 0.131 / 0.131 / 20 | 0.096 / 0.186 | 0.614 | 0.355 / 0.286 | storm_lds 155616 (True), storm_small 141428 (True) | True |
| qwen32b_graph_b1_c4096__n256__gapprobe_L7 | 0.251 / 0.251 / 28 | 0.250 / 0.250 | 0.618 | 0.434 / 0.350 | storm_small 144014 (True), storm_lds 157216 (True) | True |

## G5 (decoder time ratio vs competitor absent, same block)

- n=256: inactive 0.9996 [0.9969, 1.0024] (8 blocks); arith100 1.0583 [1.0543, 1.0625] (8 blocks); mem25 1.0442 [1.0407, 1.0474] (8 blocks) (221 GB/s); mem50 1.0875 [1.0848, 1.0904] (8 blocks) (432 GB/s); mem100 1.1885 [1.1844, 1.1921] (8 blocks) (864 GB/s); mem100/arith100 1.1230 [1.1192, 1.1267] (8 blocks)
- n=192: arith100 1.0561 [1.0537, 1.0583] (8 blocks); mem25 1.0462 [1.0425, 1.0508] (8 blocks) (247 GB/s); inactive 1.0018 [0.9998, 1.0041] (8 blocks); mem50 1.0985 [1.0939, 1.1032] (8 blocks) (483 GB/s); mem100 1.2185 [1.2134, 1.2236] (8 blocks) (950 GB/s); mem100/arith100 1.1538 [1.1498, 1.1578] (8 blocks)
- n=128: inactive 1.0019 [0.9999, 1.0042] (8 blocks); arith100 1.0557 [1.0521, 1.0593] (8 blocks); mem50 1.1152 [1.1112, 1.1189] (8 blocks) (540 GB/s); mem100 1.2616 [1.2546, 1.2684] (8 blocks) (1083 GB/s); mem25 1.0556 [1.0521, 1.0597] (8 blocks) (281 GB/s); mem100/arith100 1.1951 [1.1887, 1.2007] (8 blocks)

Observational with respect to clock (controlled clocks not authorized).
