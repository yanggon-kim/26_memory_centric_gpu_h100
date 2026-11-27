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

- Limitation P3a_observer_clean_a1: Observer pilot clock envelope failed; all trials retained and no clean/observed latency ratio qualified. Evidence: /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16210814_priority/gpu0/runs/P3a_observer_clean_a1/run.json. Tried: Passive clock evidence; matched clean and observed producers retained. Resume: Review pooled acquisition gates and observer intrusion separately; no favorable resampling.

- Scheduling-only continuation: original remaining-bundle estimate declined; use the maximum observed acquisition plus60s per remaining acquisition and a15minute CPU-closeout reserve before the actual Slurm end. Same UUID, original frozen numerical sources, build environment and all scientific gates retained. Critical raw files already persist on shared storage. See `../session_finish_curve.json`, `../results/eager_curve_resume_transition.json` and `../results/eager_curve_resume_budget.json`.

- Limitation eager_b1_c4096_stock_f100_a1_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_stock_f100_a1_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

## 2026-11-27: job16210814 preempted during the second eager acquisition

Slurm accounting records PREEMPTED, start 09:38:37 and end 10:39:07 Eastern, elapsed3630s (2.0166666667 allocated GPU-hours). Evidence: `h200/20261127_job16210814_priority/environment/final_slurm_accounting.psv` and `logs/eager_curve_finish_supervisor.log` under that root. The complete nine-producer eager acquisition0 is retained; acquisition1 completed only its stock control before full-Green was interrupted. Acquisition2 was not captured. Actual decoder placements were proved for132/112/96/80/56/40SM, but the three-acquisition curve remains partial and the recorded clock gate failures preclude NVIDIA latency thresholds. A same-source continuation with an explicit15minute closeout reserve was attempted; preemption ended it before the scheduled two-hour limit. Resume in a fresh identified allocation/cohort, retain all partial records, and do not pool timings across UUIDs. Repeated fresh model initialization makes the current protocol too slow for a reliably one-hour allocation; a versioned protocol that reduces initialization overhead needs bounded correctness and actual-placement pilots before use.
