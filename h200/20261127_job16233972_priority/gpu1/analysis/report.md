# GPU 1: GPU-77cd4a86-b663-9b1f-80cd-4a0af9be721c

## Bounded clock-conditioning diagnostic

One independent producer per arm, three warmups and three timed trials; no three-acquisition latency claim. memory_settle_v3 is a fixed one-second private-buffer workload after prefill, outside decode timing; it changes cache history and TTFT. Any future conditioned SM curve needs its own placement, HF and3x10/source/clock/isolation proof and must stay separate from unconditioned results.

Evidence: `../results/clock_conditioning_pilot.json`. All failed windows are retained; no clock or power settings were changed. This diagnostic does not qualify clean bandwidth, eager placement or thresholds.

- Limitation conditioned_full_context_scope: Full-device context comparisons use the prospectively fixed memory_settle_v3 scope; original unconditioned timings remain separate.. Evidence: configs/*memory_settle_v3.json; results/clock_conditioning_pilot.json. Tried: Own stock placement/HF proof and3x10acquisitions per shape, with all warmup/timed conditioning receipts verified. Resume: Interpret decode only; added untimed conditioning changes cache history and TTFT. No deployment optimization or original unconditioned latency claim.

- Limitation clock_conditioning_result: Both diagnostic arms failed the predeclared <=5% SM-clock envelope on this physical GPU: unconditioned 360–1350 MHz (3.75x), memory_settle_v3 390–1395 MHz (3.576923x); memory clock stayed3201 MHz. Exact outputs and conditioning receipts passed. Evidence: results/clock_conditioning_pilot.json. Tried: One prospectively declared unconditioned/one-second fixed-buffer pair; original dynamic and unchanged StaticCache references; no clock/power changes. Resume: Retain both arms; no latency, clean bandwidth or threshold qualification transfers. Continue independent full-GPU functional/traffic work; qualify any subsequent comparison with all original clock and source gates.

- Limitation graph_b1_c4096_stock_f100_a0_resident_followup_full_shape_original_memory_settle_v3: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16233972_priority/gpu1/results/graph_b1_c4096_stock_f100_a0_resident_followup_full_shape_original_memory_settle_v3.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c4096_stock_f100_a1_resident_followup_full_shape_original_memory_settle_v3: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16233972_priority/gpu1/results/graph_b1_c4096_stock_f100_a1_resident_followup_full_shape_original_memory_settle_v3.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c4096_stock_f100_a2_resident_followup_full_shape_original_memory_settle_v3: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16233972_priority/gpu1/results/graph_b1_c4096_stock_f100_a2_resident_followup_full_shape_original_memory_settle_v3.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.
