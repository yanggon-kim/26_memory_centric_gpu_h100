# Optimized32B import-preflight repair cohort

Fresh source cohort on the same allocated UUID. No pooling with the original mixed-context or failed import attempts. Exact HF outputs, actual SM grants, real NSYS graph-kernel placement and3x10 timing gates remain binding.

- Limitation graph_b1_c4096_green_f100_a0_proof_integration_check_capture_bind_v2: Producer, exact output, granted-SM scope, or measured decoder placement gate failed. Evidence: h200/20261127_job16184819_priority/gpu1_import_preflight/logs/graph_b1_c4096_green_f100_a0_proof_integration_check_capture_bind_v2.log. Tried: Versioned green-context path and separate NSYS measured-decode proof. Resume: Inspect preserved failure; bounded repair before reusing this arm.

- Limitation import_preflight_integration: Corrected capture adapter still failed full-green measured proof. Evidence: results/integration_pilot.json. Tried: One fresh full-green NSYS pilot after identity-checked import preflight. Resume: Keep optimized restriction blocked; retain separately collected eager/full-GPU cases; no further repair cycle in this follow-up.

- Limitation capture_bind_context_stack: Corrected UUID preflight passed and capture hook ran once with actual132SM/context2, but vLLM profile_cudagraph_memory exits through PyTorch GreenContext.pop_context assertion: popped context differs from expected. No decoder run publication or NSYS measurement exists for this repaired arm.. Evidence: runs/graph_b1_c4096_green_f100_a0_proof_integration_check_capture_bind_v2/failure.json; runs/graph_b1_c4096_green_f100_a0_proof_integration_check_capture_bind_v2/capture_binding_audit.json. Tried: One source-versioned full-green integration pilot after numeric-mask CPU regression fix; retain audit/failure; no installed-package edits. Resume: Optimized path remains blocked; collect separate rebind_v1 eager proofs and full references. A future context-stack repair requires versioned implementation and fresh bounded proof, not source-based placement claims.

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

## eager B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | None | False | NA | NA |
| 0.85 | None | False | NA | NA |
| 0.75 | None | False | NA | NA |
| 0.6 | None | False | NA | NA |
| 0.45 | None | False | NA | NA |
| 0.3 | None | False | NA | NA |

## Terminal disposition

Parent allocation PREEMPTED at01:21:12 Eastern. Optimized corrected adapter failed during graph-memory profiling with a Green Context pop assertion. Eager stock NSYS proof and independent HF outputs passed; eager full-green proof was interrupted during model loading, so no eager restriction result exists. No complete decoder curve or NVIDIA threshold. CPU reconciliation: `../results/terminal_cohort_review.json`. GPU-hours are included once in the parent accounting.
