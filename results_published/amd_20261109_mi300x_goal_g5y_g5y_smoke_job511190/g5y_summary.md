# G5 with leak test v6 — mi300x (g5y_smoke)

- placement_ok True, lifecycle_ok True
- leak v6: valid True, sensitive True, ok **True** (by level {256: True, 192: True, 128: True})
- G5 disposition: **isolated (leak v6)**

## Leak v6 per GPU, level and balanced list

| GPU / n / list | CUs | valid (missing / multi) | decoder T8 / M / flags | max native T8 / M | positive T8 / flags | thr T8 / M | ok |
|---|---|---|---|---|---|---|---|
| L0 / 256 / s0 | 32 | True (0.000 / 0.000) | 0.002 / 0.002 / 0 | 0.002 / 0.002 | 0.508 / 8 | 0.255 / 0.102 | True |
| L0 / 256 / s1 | 32 | True (0.000 / 0.000) | 0.003 / 0.003 / 0 | 0.001 / 0.001 | 0.543 / 8 | 0.272 / 0.101 | True |
| L0 / 192 / s0 | 96 | True (0.000 / 0.000) | 0.002 / 0.002 / 0 | 0.002 / 0.002 | 0.612 / 8 | 0.307 / 0.102 | True |
| L0 / 192 / s1 | 96 | True (0.000 / 0.000) | 0.002 / 0.002 / 0 | 0.002 / 0.002 | 0.614 / 8 | 0.308 / 0.102 | True |
| L0 / 128 / s0 | 160 | True (0.000 / 0.000) | 0.002 / 0.002 / 0 | 0.002 / 0.002 | 0.663 / 8 | 0.332 / 0.102 | True |
| L0 / 128 / s1 | 160 | True (0.000 / 0.000) | 0.002 / 0.002 / 0 | 0.002 / 0.002 | 0.630 / 8 | 0.316 / 0.102 | True |

## G5 (decoder time ratio vs competitor absent, same block)

- n=256: mem100 1.1887 [1.1863, 1.1911] (1 blocks) (835 GB/s); arith100 1.0606 [1.0581, 1.0631] (1 blocks); mem100/arith100 1.1208 [1.1190, 1.1225] (1 blocks)

Observational with respect to clock (controlled clocks not authorized).
