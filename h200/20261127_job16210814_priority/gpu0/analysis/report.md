# GPU 0: GPU-522f3fae-e05d-fc45-ba23-34c699fa4185

- Limitation optimized_restriction_prior_block: Earlier job16184819 measured graph context escape and a bounded capture repair failed during graph-memory profiling; this cohort prioritizes the explicitly separate eager sweep.. Evidence: h200/20261127_job16184819_priority/analysis/report.md. Tried: Preserve graph failures; no repeated graph repair on this allocation. Resume: Require separate eager stock/full-green/cut proofs on this UUID; optimized restriction stays blocked.

- Limitation eager_b1_c4096_stock_f100_a0_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_stock_f100_a0_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation eager_b1_c4096_green_f100_a0_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_green_f100_a0_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation eager_b1_c4096_green_f85_a0_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_green_f85_a0_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation eager_b1_c4096_green_f75_a0_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_green_f75_a0_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation eager_b1_c4096_green_f100_a0_interleaved_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_green_f100_a0_interleaved_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation eager_b1_c4096_green_f60_a0_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_green_f60_a0_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation eager_b1_c4096_green_f45_a0_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_green_f45_a0_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation eager_b1_c4096_green_f100_a0_end_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_green_f100_a0_end_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

## eager B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | 132 | False | NA | NA |
| 0.85 | 112 | False | NA | NA |
| 0.75 | 96 | False | NA | NA |
| 0.6 | 80 | False | NA | NA |
| 0.45 | 56 | False | NA | NA |
| 0.3 | 40 | False | NA | NA |

- Limitation eager_b1_c4096_budget_a1: Next comparable curve bundle does not fit before actual closeout cutoff. Evidence: h200/20261127_job16210814_priority/gpu0/results/budget_eager_b1_c4096.json. Tried: Proved requested placements; staged first broad acquisition and measured-cost refinement; every completed acquisition retained. Resume: Resume a fresh identified cohort with enough time; no cross-allocation timing pooling or favorable-clock retries.

## eager B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | 132 | False | NA | NA |
| 0.85 | 112 | False | NA | NA |
| 0.75 | 96 | False | NA | NA |
| 0.6 | 80 | False | NA | NA |
| 0.45 | 56 | False | NA | NA |
| 0.3 | 40 | False | NA | NA |

- Limitation P3a_observer_clean_a0: Observer pilot clock envelope failed; all trials retained and no clean/observed latency ratio qualified. Evidence: /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16210814_priority/gpu0/runs/P3a_observer_clean_a0/run.json. Tried: Passive clock evidence; matched clean and observed producers retained. Resume: Review pooled acquisition gates and observer intrusion separately; no favorable resampling.

- Limitation P3a_observer_observed_a0: Observer pilot clock envelope failed; all trials retained and no clean/observed latency ratio qualified. Evidence: /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16210814_priority/gpu0/runs/P3a_observer_observed_a0/run.json. Tried: Passive clock evidence; matched clean and observed producers retained. Resume: Review pooled acquisition gates and observer intrusion separately; no favorable resampling.
