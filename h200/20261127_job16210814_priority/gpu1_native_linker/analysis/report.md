# Spare-device native-linker repair cohort

Same allocated UUID, new frozen source cohort. Original native counter pilot timed out before any trial while the host linker wrapper waited. Explicit same-GCC/native-linker selection is an experiment-local build choice; installed tools and the other GPU worker are unchanged. No new GPU allocation.

- Limitation graph_b1_c4096_stock_f100_a0_traffic_reference: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu1_native_linker/results/graph_b1_c4096_stock_f100_a0_traffic_reference.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation P3a_observer_clean_a0: Observer pilot clock envelope failed; all trials retained and no clean/observed latency ratio qualified. Evidence: /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16210814_priority/gpu1_native_linker/runs/P3a_observer_clean_a0/run.json. Tried: Passive clock evidence; matched clean and observed producers retained. Resume: Review pooled acquisition gates and observer intrusion separately; no favorable resampling.

- Limitation P3a_observer_observed_a0: Observer pilot clock envelope failed; all trials retained and no clean/observed latency ratio qualified. Evidence: /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16210814_priority/gpu1_native_linker/runs/P3a_observer_observed_a0/run.json. Tried: Passive clock evidence; matched clean and observed producers retained. Resume: Review pooled acquisition gates and observer intrusion separately; no favorable resampling.
