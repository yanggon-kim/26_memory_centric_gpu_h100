# Research contract

Read `docs/primary_specification.md` thoroughly, beginning with Sections 0 and 14. It is the primary specification. This repository contains partial H100 Phase 0–2 evidence. Start with `docs/RENTAL_RESUME.md`, then `STATUS.json`, `results/case_manifest.json` and `docs/opportunity_report.md`. Resume unfinished gates; do not call the prior study complete or repeat expensive completed captures by default. Historical RTX evidence is not H100 evidence. A recorded running process belongs to the previous rental and is not evidence of a live process or successful build here.

Start with inventory and the H100 playbook. Use only the explicitly allocated physical GPU UUID. Multiple GPUs may exist; never expose them or introduce tensor parallelism, quantization, a different precision, or shared-GPU contention without explicit scope authorization. Do not alter drivers, power limits, clocks, firmware, or cluster settings. Scripts use startup snapshots and a cooperating lock, not a guarantee against external interference.

Run smoke before baseline and trace micro before a large capture. Keep immutable model revisions, BF16 weights/KV, seed 20261013, warmup ≥3, trials ≥3, fixed EOS-ignored greedy outputs. Preserve OOM, correctness failures, unsupported opcodes, empty traces, and partial runs. Reduce scope through a new recorded configuration; never quietly skip a case or weaken correctness to obtain a favorable result.

Always separate prefill, first selected token, timed decode steps, aggregate generated tokens, scheduler wall, GPU-event elapsed, and instrumented execution. Verify every request reaches the one-token prefill barrier before pure decode. Compilation/eager differences are evidence to investigate; the strict StaticCache gate remains intact. Trace kernels are instrumented eager executions and need their own measured eager calibration.

Energy must retain raw counter and sampled-power estimates. Primary J/token stays NA unless the declared window/cadence/agreement checks pass; even agreement is not external absolute calibration. Never rescale one sensor to match another. Classify measured, analytical, simulated and illustrative artifacts distinctly.

Logical tensor/chunk studies use synthetic full-block fills. Do not call them GPU instruction traces, real HBM savings, area-matched SRAM, or proof of architectural benefit. Dense adapters reject MoE. A single-layer repetition does not preserve the full model weight working set.

Before rental termination, verify nonempty NVBit instructions/kernel lists, postprocessing, a real sm_90 trace replay with positive cycles/instructions/kernel statistics, archive transfer checksums, pinned source/config/runtime manifests and free/storage budgets. Exit code zero alone is not replay success. Do not report H100 calibration from the V100 legacy fixture or relabel sm_120 traces as Hopper.

Keep an experiment log, completed/failed case manifest and machine-readable schema. Update STATUS only after actual evidence exists. Do not begin physical architecture modifications before Phase 2 opportunity and calibration gates are met.
