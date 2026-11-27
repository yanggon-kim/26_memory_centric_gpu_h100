# Post-core functional serving/prefill pilots

Same allocated GPU0, new frozen source/workspace after the original core worker returned. Original core observations are retained separately. The other allocated GPU may independently run context cases; shared host CPU concurrency remains part of this functional pilot environment. No new allocation, clock/power or system changes.

## P3a_functional_policies

functional_passed. bounded instrumented functional proof only; not3acquisitions, loaded serving, tuning, SLOs, energy or physical traffic. Evidence: `/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16228671_priority/gpu0_postcore/runs/P3a_functional_policies/run.json`; strict reference: True.

## P4a_functional_prefill

functional_passed. bounded instrumented functional proof only; not3acquisitions, loaded serving, tuning, SLOs, energy or physical traffic. Evidence: `/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16228671_priority/gpu0_postcore/runs/P4a_functional_prefill/run.json`; strict reference: True.

## 2026-11-27: first repeated prefill comparison deferred by remaining budget

The completed P4a functional pilot at1K passed whole/chunk256 semantics and both HF reference routes. A versioned controller for whole/256/1024/4096/whole-end,3independent acquisitions and10timed trials, is ready. At admission review the remaining time before job16228671's existing launch cutoff was below its1122second full-comparison estimate plus60second margin. Evidence: `h200/20261127_resident_preparation/results/prefill_deferred_budget_16228671.json`; no additional GPU producer was launched. This limits the result to functional pilot evidence, not a repeated prefill comparison or ordinary TTFT claim. Tried: measured-pilot budget estimation and fail-closed admission, with a CPU test confirming insufficient budget launches no producer. Resume at the beginning of a future owned allocation, with fresh source/UUID/reference qualification and separate CPU review.

CPU repair follow-up: separate_policy_v2now preserves temporary waiting/skipped entries using individual native prepends in queue order. Five policy CPU checks and three pilot contracts passed, including multiple decoder preemptions and an exception after preemption. Actual GPU-pressure validation remains pending. A further source-level limit is explicit: replay of a preempted request can combine original-prompt and generated-history queries; the current strict separate-policy mixed-step assertion may reject it. Recompute classification/phase semantics require a versioned definition and GPU proof before pressure robustness is claimed. No assertion was weakened and no failed request was promoted.
