# GPU 1: GPU-7838118b-b9d4-2865-898f-f2dc79e741e7

- Limitation graph_b1_c4096_stock_f100_a0_traffic_reference: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16184819_priority/gpu1/results/graph_b1_c4096_stock_f100_a0_traffic_reference.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.

- Limitation P2a_triplicate_scope: Three independent physical DRAM captures passed exact replay/counter-reader gates, but clean reference clock envelope failed; L2 traffic, complete kernel attribution and restricted-arm traffic are absent.. Evidence: analysis/P2a_triplicate_counter_report.md; results/P2a_triplicate_counter_review.json. Tried: Retained all3 acquisitions/raw checksums and same-source audit; report physical bytes and instrumented duration separately; no clean bandwidth join. Resume: Collect eligible versioned L2/kernel evidence and qualify timing/clock joins; never replace physical bytes with logical demand or resample for favorable clocks.

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
