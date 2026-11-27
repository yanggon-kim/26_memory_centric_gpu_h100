# GPU 0: GPU-522f3fae-e05d-fc45-ba23-34c699fa4185

- Limitation optimized_restriction_prior_block: Earlier job16184819 measured graph context escape and a bounded capture repair failed during graph-memory profiling; this cohort prioritizes the explicitly separate eager sweep.. Evidence: h200/20261127_job16184819_priority/analysis/report.md. Tried: Preserve graph failures; no repeated graph repair on this allocation. Resume: Require separate eager stock/full-green/cut proofs on this UUID; optimized restriction stays blocked.

- Limitation eager_b1_c4096_stock_f100_a0_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_stock_f100_a0_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation eager_b1_c4096_green_f100_a0_rebind_v1: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16210814_priority/gpu0/results/eager_b1_c4096_green_f100_a0_rebind_v1.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.
