# H200 shoreline evidence progress

Recovery update 2026-10-17: this generated table is an earlier 14-capture accounting snapshot. See [recovery audit](recovery_20261017.md) for 46/48 completed selected captures, interrupted allocation and all remaining gates.

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
| J2_bridge_eager_b1_c1024_full_acq00_range_api_v2 | 9376289024 | 27790336 | 1.0027354875999734 | 3.008178790944055 |
| J2_bridge_eager_b1_c1024_full_acq00_ticks_only_v4 | 9376605952 | 28651264 | 1.0027693810680396 | 3.008280470412917 |
| J2_bridge_eager_b1_c1024_full_acq01_ticks_only_v4 | 9376615680 | 28658176 | 1.0027704214168172 | 3.0082835914305397 |
| J2_bridge_eager_b1_c1024_full_acq02_ticks_only_v4 | 9376347136 | 27531520 | 1.002741702315039 | 3.008197434917748 |
| J2_bridge_eager_b1_c1024_layer0_acq00_ticks_only_v4 | 284777472 | 11245824 | 1.002899397401361 | 3.0084703591518824 |
| J2_bridge_eager_b1_c1024_layer0_acq01_ticks_only_v4 | 284857856 | 11393792 | 1.0031824853317177 | 3.0093195586326265 |
| J2_bridge_eager_b1_c1024_layer0_acq02_ticks_only_v4 | 284687104 | 11657728 | 1.0025811488681893 | 3.007515685850281 |
| J2_bridge_eager_b1_c1024_layer14_acq00_ticks_only_v4 | 284671488 | 9573120 | 1.0024936172054177 | 3.0070579454618223 |
| J2_bridge_eager_b1_c1024_layer14_acq01_ticks_only_v4 | 284707072 | 9091584 | 1.0026189291329497 | 3.0074338283810533 |
| J2_bridge_eager_b1_c1024_layer14_acq02_ticks_only_v4 | 284810240 | 9425408 | 1.0029822435703262 | 3.008523618427457 |
| J2_bridge_eager_b1_c1024_layer27_acq00_ticks_only_v4 | 284894976 | 9642752 | 1.0032806482322905 | 3.009418706530087 |
| J2_bridge_graph_b1_c1024_full_acq00_ticks_only_v4 | 9374737664 | 16038144 | 1.0025695793475655 | 3.0076810707652966 |
| J2_bridge_graph_b1_c1024_full_acq01_ticks_only_v4 | 9374433280 | 15654400 | 1.0025370273818703 | 3.007583415766527 |
| J2_bridge_graph_b1_c1024_full_acq02_ticks_only_v4 | 9374643200 | 16675584 | 1.0025594770133843 | 3.0076507640415406 |

A near-three window ratio can reflect ordinary autoregressive recurrence. Aggregate ratios do not prove per-address exactly-once access or identify a KV hit rate.

## Remaining gates

Independent physical-service and clock joins, actual resource restriction/concurrency proofs, retention attribution, bounded overlap, energy qualification and trace semantics/transfer remain individually indexed in the case ledger. No physical compute/SRAM/PHY/package benefit is established.
