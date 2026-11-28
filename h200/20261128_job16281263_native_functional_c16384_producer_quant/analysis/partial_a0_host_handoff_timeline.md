# Partial acquisition0host handoff observations

Post-archive derived metadata; original captures/certificates unchanged. Send-completion observation is an upper observation bound, not isolated wire time. Consumer whole request includes CPU bookkeeping. Host intervals overlap and are not additive. File IPC/harness request span is not deployed TTFT, offered-load goodput, clean GPU/event latency, layer attribution or a qualified slowdown ratio. One paired acquisition is not the required3independent-acquisition comparison. Rejected clocks and stock-after timeout remain binding.

Producer clock gate failed; consumer individual clock gate passed. The full cohort is partial and has no qualified producer ratio. All ten completed paired timed requests are retained.

| Host observation (ms) | Raw minimum | Raw median | Raw maximum |
|---|---:|---:|---:|
| producer_first_step_host_ms | 2191.533546 | 2197.350204 | 2200.354229 |
| p0_return_to_all_sends_observation_host_ms | 6.743556 | 14.296121 | 18.958517 |
| p0_return_to_consumer_first_file_read_host_ms | 15.590565 | 20.026623 | 26.592464 |
| p0_return_to_consumer_submission_host_ms | 16.214061 | 20.751577 | 27.273710 |
| consumer_first_handoff_call_host_ms | 94.629049 | 94.973806 | 98.644391 |
| consumer_following62_calls_host_ms | 1240.307595 | 1250.295038 | 1297.590736 |
| consumer_whole63_output_request_host_ms | 1338.961587 | 1349.276863 | 1402.682039 |
| p0_return_to_first_consumer_output_host_ms | 113.795050 | 120.816882 | 126.242811 |
| send_observation_relative_to_first_consumer_output_host_ms | -113.187357 | -106.394086 | -99.086612 |
| two_process_harness_producer_submit_to_consumer_last_return_host_ms | 3558.991929 | 3566.433414 | 3618.089496 |

## Interpretation limits

The first token file is published after producer step return and before the explicit all-send observation. Consumer admission and native sending can overlap. The first consumer engine call can include loading and blocking. Following62call sum excludes CPU gaps between calls; whole-request time retains them. These observations do not isolate physical transfer or kernel time. Preserve signed send-observation offsets; a late poll is not proof that data was unavailable until that timestamp.

The original run members match the verified archive. This derived JSON and report postdate that archive; they add no GPU measurement.

## Reproduction and pinned-source scope

Command: `/storage/scratch1/6/jkim4112/envs/h200_priority_20261125/bin/python h200/20261127_resident_preparation/h200_partial_host_handoff.py --root /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261128_job16281263_native_functional_c16384_producer_quant --acquisition 0`. The versioned CPU reader reproduces all10request rows and10metric summaries. Installed native engine/connector source hashes match the captured pins. The sender removes the final item before completing its blocking send; queue-empty alone is insufficient, so the collector separately polls actual sent-ID markers. Its observed timestamp remains a host observation bound. Source inspection is not concurrent-load or physical-traffic validation. Evidence: `../results/partial_host_handoff_reproduction_CPU_review.json`.
