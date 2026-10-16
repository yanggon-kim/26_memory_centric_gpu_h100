# H200 intermediate results

26 of 35 planned dense graph-baseline cases have passed the recorded timing and output gates.

All measurements use the allocated H200. Prefill and the first selected token are excluded from pure decode. Latency is milliseconds per batch decode step; throughput counts all requests.

| Run | Status | Step ms | Tokens/s | Board J/token |
|---|---|---:|---:|---:|
| bridge_eager_b1_c1024 | complete | 9.6334 | 103.8059 | NA |
| bridge_eager_b1_c128 | complete | 9.6500 | 103.6271 | NA |
| bridge_eager_b8_c4096 | complete | 9.9697 | 802.4304 | NA |
| bridge_graph_b1_c1024 | complete | 2.8768 | 347.6035 | NA |
| bridge_graph_b1_c1024_energy_n1024 | complete | 2.8380 | 352.3565 | NA |
| bridge_graph_b1_c1024_energy_n2048 | complete | 2.8521 | 350.6227 | 0.7509 |
| bridge_graph_b1_c128 | complete | 2.8434 | 351.6874 | NA |
| bridge_graph_b1_c128_smoke | complete | 2.8502 | 350.8519 | NA |
| bridge_graph_b1_c128_text_v1 | complete | 2.8717 | 348.2270 | NA |
| bridge_graph_b1_c4096 | complete | 2.9079 | 343.8965 | NA |
| bridge_graph_b4_c1024 | complete | 3.0020 | 1332.4319 | NA |
| bridge_graph_b4_c128 | complete | 2.9682 | 1347.6094 | NA |
| bridge_graph_b4_c128_text_v1 | complete | 2.9586 | 1351.9952 | NA |
| bridge_graph_b4_c4096 | complete | 3.1747 | 1259.9696 | NA |
| bridge_graph_b8_c1024 | complete | 3.1471 | 2542.0253 | NA |
| bridge_graph_b8_c128 | complete | 3.0543 | 2619.2701 | NA |
| bridge_graph_b8_c4096 | complete | 3.3921 | 2358.4066 | NA |
| bridge_graph_b8_c4096_energy_n1024 | failed | NA | NA | NA |
| bridge_graph_b8_c4096_energy_n1024_selected_pair | complete | 3.3182 | 2410.9389 | NA |
| bridge_graph_b8_c4096_energy_n2048 | complete | 3.3296 | 2402.6873 | 0.1340 |
| bridge_hf_compiled_b1_c1024 | complete | 2.4857 | 402.3020 | NA |
| bridge_hf_compiled_b1_c128 | complete | 2.3470 | 426.0843 | NA |
| bridge_hf_compiled_b8_c4096 | complete | 5.5387 | 1444.3741 | NA |
| bridge_hf_eager_b1_c128_smoke | complete | 14.1552 | 70.6453 | NA |
| mistral7b_eager_b1_c1024 | complete | 10.6768 | 93.6614 | NA |
| mistral7b_graph_b1_c1024 | complete | 5.7008 | 175.4134 | NA |
| mistral7b_graph_b1_c1024_energy_n1024 | complete | 5.6592 | 176.7037 | 2.5242 |
| mistral7b_graph_b1_c128 | complete | 5.6582 | 176.7334 | NA |
| mistral7b_graph_b1_c128_smoke | failed | NA | NA | NA |
| mistral7b_graph_b1_c128_smoke_tokenizer_resolved | complete | 5.5322 | 180.7615 | NA |
| mistral7b_graph_b1_c4096 | complete | 5.8442 | 171.1090 | NA |
| mistral7b_graph_b4_c1024 | complete | 5.9571 | 671.4722 | NA |
| mistral7b_graph_b4_c128 | complete | 5.8003 | 689.6137 | NA |
| mistral7b_graph_b4_c4096 | complete | 6.3551 | 629.4173 | NA |
| mistral7b_graph_b8_c1024 | complete | 6.2089 | 1288.4818 | NA |
| mistral7b_graph_b8_c128 | complete | 5.9450 | 1345.6658 | NA |
| mistral7b_graph_b8_c4096 | complete | 6.9722 | 1147.4208 | NA |
| mistral7b_graph_b8_c4096_energy_n1024 | complete | 6.9559 | 1150.1028 | 0.4603 |
| mistral7b_hf_eager_b1_c128_smoke | complete | 16.1448 | 61.9393 | NA |
| qwen14b_graph_b1_c1024 | complete | 9.8787 | 101.2284 | NA |
| qwen14b_graph_b1_c1024_energy_n1024 | complete | 9.8203 | 101.8303 | 4.9665 |
| qwen14b_graph_b1_c128 | complete | 9.8159 | 101.8756 | NA |
| qwen14b_graph_b1_c128_smoke | complete | 9.7605 | 102.4539 | NA |
| qwen14b_graph_b1_c4096 | complete | 10.0575 | 99.4281 | NA |
| qwen14b_graph_b4_c1024 | complete | 10.2083 | 391.8387 | NA |
| qwen14b_graph_b4_c128 | complete | 10.0041 | 399.8376 | NA |
| qwen14b_graph_b4_c4096 | complete | 10.8297 | 369.3541 | NA |
| qwen14b_graph_b4_c4096_energy_n1024 | complete | 10.5164 | 380.3575 | 1.4390 |
| qwen14b_hf_eager_b1_c128_smoke | complete | 24.3326 | 41.0971 | NA |
| qwen32b_graph_b1_c1024 | complete | 19.0336 | 52.5386 | NA |
| qwen32b_graph_b1_c1024_energy_n1024 | complete | 18.6753 | 53.5467 | 10.7110 |
| qwen32b_graph_b1_c128 | complete | 18.9853 | 52.6723 | NA |
| qwen32b_graph_b1_c128_smoke | complete | 18.9749 | 52.7013 | NA |
| qwen32b_hf_eager_b1_c128_smoke | complete | 31.8187 | 31.4280 | NA |
| qwen7b_eager_b1_c128_smoke | failed | NA | NA | NA |
| qwen7b_graph_b1_c128_smoke | failed | NA | NA | NA |
| qwen7b_hf_eager_b1_c128_smoke | complete | 14.4827 | 69.0478 | NA |

## Interpretation

Graph, eager, HF compiled, energy, and profiled executions have separate identities. Differences between these programs require software and correctness analysis before any architectural interpretation.

Qualified board energy uses raw cumulative-counter and sampled-power estimates with at least five seconds, five counter changes, and at most ten percent disagreement. Agreement provides an internal crosscheck; component energy and absolute accuracy remain uncalibrated.

## Remaining work

See ../STATUS.json and ../results/fixed_bandwidth_case_manifest.json for completed, failed, and pending cases. Hardware-counter attribution, representative kernel coverage, and native trace-tool validation retain their own evidence gates.

## Preserved failed attempts

- bridge_graph_b8_c4096_energy_n1024: no selected cases. Excluded from valid performance comparisons.
- mistral7b_graph_b1_c128_smoke: expected str, bytes or os.PathLike object, not NoneType. Excluded from valid performance comparisons.
- materialize_mistral7b_smoke: failed; preserved log logs/materialize_mistral7b_smoke.log. Later attempts retain separate identities.
- materialize_mistral7b_smoke_hf_layout: failed; preserved log logs/materialize_mistral7b_smoke_hf_layout.log. Later attempts retain separate identities.
- qwen7b_graph_hf_smoke_tokens: failed; preserved log logs/qwen7b_graph_hf_smoke_tokens.log. Later attempts retain separate identities.
- qwen7b_eager_hf_smoke_tokens: failed; preserved log logs/qwen7b_eager_hf_smoke_tokens.log. Later attempts retain separate identities.
- qwen7b_full_dynamic_reference: failed; preserved log logs/qwen7b_full_dynamic_reference.log. Later attempts retain separate identities.
- qwen7b_profile_b1_c1024_sequence: failed; preserved log logs/qwen7b_profile_b1_c1024_sequence.log. Later attempts retain separate identities.
- qwen7b_graph_profile_b1_c1024_sequence: failed; preserved log logs/qwen7b_graph_profile_b1_c1024_sequence.log. Later attempts retain separate identities.
- qwen7b_profile_b8_c4096_sequence: failed; preserved log logs/qwen7b_profile_b8_c4096_sequence.log. Later attempts retain separate identities.
- qwen7b_graph_profile_b8_c4096_sequence: failed; preserved log logs/qwen7b_graph_profile_b8_c4096_sequence.log. Later attempts retain separate identities.
- bridge_range_b1_c1024: failed; preserved log logs/bridge_range_b1_c1024.log. Later attempts retain separate identities.
- bridge_range_b8_c4096: failed; preserved log logs/bridge_range_b8_c4096.log. Later attempts retain separate identities.
