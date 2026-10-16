# H200 shoreline evidence progress

Ordinary, profiled, analytical and historical evidence retain separate scopes. Clock-rejected rows have raw values and no qualified matched confidence interval.

| Model | B | C | Engine | Acquisitions | Raw median step ms | Clock-qualified |
|---|---:|---:|---|---:|---:|---|
| bridge | 1 | 1024 | graph | 3 | 2.8367 | True |
| bridge | 1 | 1024 | eager | 3 | 9.6262 | True |
| bridge | 8 | 4096 | graph | 3 | 3.3870 | False |
| bridge | 8 | 4096 | eager | 3 | 9.9009 | False |
| mistral7b | 1 | 4096 | graph | 3 | 5.8015 | False |
| mistral7b | 1 | 4096 | eager | 3 | 10.4622 | False |
| mistral7b | 8 | 4096 | graph | 3 | 7.0002 | False |
| mistral7b | 8 | 4096 | eager | 3 | 10.7322 | False |
| qwen14b | 1 | 4096 | graph | 3 | 9.9820 | False |
| qwen14b | 1 | 4096 | eager | 3 | 15.2831 | False |

## Physical traffic versus logical demand

| Capture | Read bytes | Write bytes | D/U tick sum | D/U window unique |
|---|---:|---:|---:|---:|
| J2_bridge_eager_b1_c1024_full_acq00_range_api_v2 | 9376289024 | 27790336 | 1.0027354875999734 | 3.00823413516489 |
| J2_bridge_eager_b1_c1024_full_acq00_ticks_only_v4 | 9376605952 | 28651264 | 1.0027693810680396 | 3.008335816504442 |
| J2_bridge_eager_b1_c1024_full_acq01_ticks_only_v4 | 9376615680 | 28658176 | 1.0027704214168172 | 3.008338937579485 |
| J2_bridge_eager_b1_c1024_full_acq02_ticks_only_v4 | 9376347136 | 27531520 | 1.002741702315039 | 3.0082527794815928 |

A near-three window ratio can reflect ordinary autoregressive recurrence. Aggregate ratios do not prove per-address exactly-once access or identify a KV hit rate.

## Remaining gates

Independent physical-service and clock joins, actual resource restriction/concurrency proofs, retention attribution, bounded overlap, energy qualification and trace semantics/transfer remain individually indexed in the case ledger. No physical compute/SRAM/PHY/package benefit is established.
