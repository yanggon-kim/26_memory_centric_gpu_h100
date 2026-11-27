# GPU 1: GPU-28fa5fe0-0a71-78dc-ff99-a1fb75e9d459

## Bounded clock-conditioning diagnostic

One independent producer per arm, three warmups and three timed trials; no three-acquisition latency claim. memory_settle_v3 is a fixed one-second private-buffer workload after prefill, outside decode timing; it changes cache history and TTFT. Any future conditioned SM curve needs its own placement, HF and3x10/source/clock/isolation proof and must stay separate from unconditioned results.

Evidence: `../results/clock_conditioning_pilot.json`. All failed windows are retained; no clock or power settings were changed. This diagnostic does not qualify clean bandwidth, eager placement or thresholds.

- Limitation graph_b1_c4096_stock_f100_a0_resident_followup_full_shape: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16228671_priority/gpu1/results/graph_b1_c4096_stock_f100_a0_resident_followup_full_shape.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c4096_stock_f100_a1_resident_followup_full_shape: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16228671_priority/gpu1/results/graph_b1_c4096_stock_f100_a1_resident_followup_full_shape.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c4096_stock_f100_a2_resident_followup_full_shape: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16228671_priority/gpu1/results/graph_b1_c4096_stock_f100_a2_resident_followup_full_shape.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c16384_stock_f100_a0_resident_followup_full_shape: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16228671_priority/gpu1/results/graph_b1_c16384_stock_f100_a0_resident_followup_full_shape.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c16384_stock_f100_a1_resident_followup_full_shape: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16228671_priority/gpu1/results/graph_b1_c16384_stock_f100_a1_resident_followup_full_shape.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

## 2026-11-27T18:46:52.497840+00:00: allocation preempted

Slurm accounting records job16228671PREEMPTED at2026-11-27T13:45:47-05:00 after3627seconds, before its1:30requested limit. This is2.015allocatedGPU-hours; overlapping step16228671.1 used the same pair and is not an additional allocation. Evidence: `environment/slurm_accounting_terminal.txt`, the batch epilog and GPU1's final process/monitor/log.

The completed3acquisition resident eager SM curve, optimized full-device4Ktriplicate, clock diagnostic, and both functional serving/prefill pilots remain intact. At16K the optimized placement and strict dynamic/StaticCache references passed, and acquisitions0/1completed. Acquisition2was interrupted during initialization: its preserved `run.json` has statusrunning and zero cases, not a completed or unsupported measurement. No32Kcase launched. This limits repeated16Kstatistics and all cross-context latency conclusions; full4Ktimings also failed clocks.

Tried: separate immutable producers, continuous compact checkpoints, completed-unit archives before preemption, and a dependency-held replacement. Job16233972is already submitted, now eligible for a separate fixed-conditioned B1/C4096curve and conditioned full-device contexts. Resume each incomplete quantitative comparison as a fresh identified cohort with3acquisitions; do not pool these two16Kacquisitions across GPUs or promote clock-rejected results.

## graph B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | None | False | NA | NA |
| 0.85 | None | False | NA | NA |
| 0.75 | None | False | NA | NA |
| 0.6 | None | False | NA | NA |
| 0.45 | None | False | NA | NA |
| 0.3 | None | False | NA | NA |

## graph B1/C16384 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | None | False | NA | NA |
| 0.85 | None | False | NA | NA |
| 0.75 | None | False | NA | NA |
| 0.6 | None | False | NA | NA |
| 0.45 | None | False | NA | NA |
| 0.3 | None | False | NA | NA |
