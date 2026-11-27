# GPU 1: GPU-77cd4a86-b663-9b1f-80cd-4a0af9be721c

## Bounded clock-conditioning diagnostic

One independent producer per arm, three warmups and three timed trials; no three-acquisition latency claim. memory_settle_v3 is a fixed one-second private-buffer workload after prefill, outside decode timing; it changes cache history and TTFT. Any future conditioned SM curve needs its own placement, HF and3x10/source/clock/isolation proof and must stay separate from unconditioned results.

Evidence: `../results/clock_conditioning_pilot.json`. All failed windows are retained; no clock or power settings were changed. This diagnostic does not qualify clean bandwidth, eager placement or thresholds.

- Limitation conditioned_full_context_scope: Full-device context comparisons use the prospectively fixed memory_settle_v3 scope; original unconditioned timings remain separate.. Evidence: configs/*memory_settle_v3.json; results/clock_conditioning_pilot.json. Tried: Own stock placement/HF proof and3x10acquisitions per shape, with all warmup/timed conditioning receipts verified. Resume: Interpret decode only; added untimed conditioning changes cache history and TTFT. No deployment optimization or original unconditioned latency claim.

- Limitation clock_conditioning_result: Both diagnostic arms failed the predeclared <=5% SM-clock envelope on this physical GPU: unconditioned 360–1350 MHz (3.75x), memory_settle_v3 390–1395 MHz (3.576923x); memory clock stayed3201 MHz. Exact outputs and conditioning receipts passed. Evidence: results/clock_conditioning_pilot.json. Tried: One prospectively declared unconditioned/one-second fixed-buffer pair; original dynamic and unchanged StaticCache references; no clock/power changes. Resume: Retain both arms; no latency, clean bandwidth or threshold qualification transfers. Continue independent full-GPU functional/traffic work; qualify any subsequent comparison with all original clock and source gates.

- Limitation graph_b1_c4096_stock_f100_a0_resident_followup_full_shape_original_memory_settle_v3: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16233972_priority/gpu1/results/graph_b1_c4096_stock_f100_a0_resident_followup_full_shape_original_memory_settle_v3.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c4096_stock_f100_a1_resident_followup_full_shape_original_memory_settle_v3: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16233972_priority/gpu1/results/graph_b1_c4096_stock_f100_a1_resident_followup_full_shape_original_memory_settle_v3.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c4096_stock_f100_a2_resident_followup_full_shape_original_memory_settle_v3: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16233972_priority/gpu1/results/graph_b1_c4096_stock_f100_a2_resident_followup_full_shape_original_memory_settle_v3.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation full_graph_c4096_pooled_clock: All three conditioned optimized full-device acquisitions were captured, but their pooled SM-clock max/min ratio3.565217 exceeds1.05; memory ratio1.0. Raw medians33.190529/33.180358/32.939082ms per timed tick are descriptive only.. Evidence: results/full_graph_b1_c4096.json. Tried: Own full-device decoder placement, strict dynamic/unchanged StaticCache references,3fresh acquisitions x10trials and fixed one-second conditioning; original clock rule retained. Resume: No clean optimized latency, confidence interval, bandwidth normalization or comparison to eager is qualified. Retain all observations and advance the16Kfunctional/context case; do not resample favorable trials..

- Limitation live_thermal_slowdown: Read-only owned-pair diagnostic at20:28UTC: GPU1 reports SwThermalSlowdown in25/25samples,87–88C,345–855MHz SM clocks,241–285W at unchanged560W limit. GPU0 reports no thermal flag,32–38C.. Evidence: ../environment/clock_events_20261127T2028.json. Tried: Five-second NVML event-mask/pstate/clock/temperature/power readback on allocated UUIDs only, with no CUDA kernels/contexts or settings changes; current incomplete16Kcomparison retained. Resume: This supports a current thermal limitation, not retrospective proof of every earlier clock failure. GPU1 clock-rejected timings stay unqualified. Finish/publish current cases within budget, avoid this node for the next owned job and re-inventory fresh UUIDs; cooling/system changes remain outside scope..

- Limitation graph_b1_c16384_stock_f100_a0_resident_followup_full_shape_original_memory_settle_v3: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16233972_priority/gpu1/results/graph_b1_c16384_stock_f100_a0_resident_followup_full_shape_original_memory_settle_v3.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c16384_stock_f100_a1_resident_followup_full_shape_original_memory_settle_v3: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16233972_priority/gpu1/results/graph_b1_c16384_stock_f100_a1_resident_followup_full_shape_original_memory_settle_v3.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation graph_b1_c16384_stock_f100_a2_resident_followup_full_shape_original_memory_settle_v3: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16233972_priority/gpu1/results/graph_b1_c16384_stock_f100_a2_resident_followup_full_shape_original_memory_settle_v3.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation full_graph_c16384_pooled_clock: All three conditioned optimized full-device C16384acquisitions and strict dualHF gates completed, but pooled SM-clock ratio3.913043 exceeds1.05; memory ratio1.0. Raw medians35.395486/35.499924/35.805847ms per tick are descriptive only.. Evidence: results/full_graph_b1_c16384.json. Tried: Own stock placement/HF proof,3fresh acquisitions x10timed trials and fixed memory conditioning; all clock-rejected observations retained; live thermal event mask recorded separately. Resume: No clean long-context latency/bandwidth or eager/optimized comparison is qualified. Re-inventory a healthy physical pair for the next eligible comparison; C32704remains unlaunched. Do not resample favorable trials or alter cooling/clocks/power..

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
