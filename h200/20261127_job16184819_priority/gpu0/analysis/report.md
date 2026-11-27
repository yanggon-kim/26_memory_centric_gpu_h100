# GPU 0: GPU-2779f29d-9f2a-0102-9d29-b982904324b9

- Limitation graph_b1_c4096_green_f100_a0_proof: Producer, exact output, granted-SM scope, or measured decoder placement gate failed. Evidence: h200/20261127_job16184819_priority/gpu0/logs/graph_b1_c4096_green_f100_a0_proof.log. Tried: Versioned green-context path and separate NSYS measured-decode proof. Resume: Inspect preserved failure; bounded repair before reusing this arm.

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

- Limitation graph_b1_c4096_green_f100_a0_proof: Optimized pilot failed a required gate; bounded graph-capture stream binding and per-tick rebinding follows. Evidence: h200/20261127_job16184819_priority/gpu0/results/graph_b1_c4096_green_f100_a0_proof.json. Tried: Initial context/stream binding and NSYS check; pinned vLLM source creates an ordinary capture stream and initializes its stream tracker on first use. Resume: Require exact outputs and a fresh proof for every count using the versioned capture_bind_v2 route; no success inferred from source inspection.

- Limitation graph_b1_c4096_green_f100_a0_proof_capture_bind_v2: Producer, exact output, granted-SM scope, or measured decoder placement gate failed. Evidence: h200/20261127_job16184819_priority/gpu0/logs/graph_b1_c4096_green_f100_a0_proof_capture_bind_v2.log. Tried: Versioned green-context path and separate NSYS measured-decode proof. Resume: Inspect preserved failure; bounded repair before reusing this arm.

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

- Limitation capture_bind_v2_import_order: Bounded adapter failed before model initialization: importing gpu_model_runner before baseline.main calls gpu_target.select leaves CUDA_VISIBLE_DEVICES in UUID form; pinned vLLM device_id_to_physical_device_id attempts int(UUID) and raises ValueError. No repaired decoder placement was measured.. Evidence: logs/graph_b1_c4096_green_f100_a0_proof_capture_bind_v2.log; runs/graph_b1_c4096_green_f100_a0_proof_capture_bind_v2/failure.json. Tried: Retained original mixed-context proof and failed adapter import; current frozen worker proceeds to separately labeled stock references/eager fallback. Resume: Version an explicit identity-checked numeric-mask preflight before early vLLM imports; CPU regression test, fresh frozen cohort and fresh NSYS proofs required; never patch running frozen sources.

- Limitation graph_b1_c4096_stock_f100_a0_unrestricted_fallback_capture_bind_v2: Clock envelope failed; all trials retained. Evidence: h200/20261127_job16184819_priority/gpu0/results/graph_b1_c4096_stock_f100_a0_unrestricted_fallback_capture_bind_v2.json. Tried: Passive clock sampling and per-trial endpoints; no clock settings changed. Resume: Independent source/clock-qualified acquisition; no favorable trial selection.
