# H200 shoreline evidence progress

Ordinary, profiled, analytical and historical evidence retain separate scopes. Clock-rejected rows have raw values and no qualified matched confidence interval.

| Model | B | C | Engine | Acquisitions | Raw median step ms | Clock-qualified |
|---|---:|---:|---|---:|---:|---|
| bridge | 1 | 1024 | graph | 3 | 2.8640 | True |
| bridge | 1 | 1024 | eager | 3 | 9.6513 | True |
| bridge | 8 | 4096 | graph | 3 | 3.3725 | False |
| bridge | 8 | 4096 | eager | 3 | 9.8826 | False |
| mistral7b | 1 | 4096 | graph | 0 | NA | False |
| mistral7b | 1 | 4096 | eager | 0 | NA | False |
| mistral7b | 8 | 4096 | graph | 0 | NA | False |
| mistral7b | 8 | 4096 | eager | 0 | NA | False |
| qwen14b | 1 | 4096 | graph | 0 | NA | False |
| qwen14b | 1 | 4096 | eager | 0 | NA | False |

## Physical traffic versus logical demand

| Capture | Read bytes | Write bytes | D/U tick sum | D/U window unique |
|---|---:|---:|---:|---:|
| J2_bridge_eager_b1_c1024_full_acq00_ticks_only_v4 | 9376319488 | 28057856 | 1.0027387455343029 | 3.0081885646571367 |
| J2_qwen14b_graph_b1_c4096_full_acq00_ticks_only_v4 | 86414439424 | 73987072 | 1.0005480605143224 | 3.001621547369287 |
| J2_qwen14b_graph_b1_c4096_full_acq01_ticks_only_v4 | 86414073344 | 73917440 | 1.0005438218635088 | 3.0016088315127307 |
| J2_qwen14b_graph_b1_c4096_full_acq02_ticks_only_v4 | 86413971712 | 74879488 | 1.0005426451191919 | 3.0016053013064004 |

A near-three window ratio can reflect ordinary autoregressive recurrence. Aggregate ratios do not prove per-address exactly-once access or identify a KV hit rate.

## Remaining gates

Independent physical-service and clock joins, actual resource restriction/concurrency proofs, retention attribution, bounded overlap, energy qualification and trace semantics/transfer remain individually indexed in the case ledger. No physical compute/SRAM/PHY/package benefit is established.
