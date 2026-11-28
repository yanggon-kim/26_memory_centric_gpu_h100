# GPU 1: GPU-5fdf3697-20d4-8564-a920-317ecda4e1a9

## Acquisition 0

capture_passed_pending_CPU_review. instrumented complete prefill plus first selected token, followed by separate63decode ticks; source/output/clock/isolation/prefix coverage and cross-acquisition statistics require CPU review; no physical traffic or ordinary TTFT claim. Evidence: `/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16238357_priority/gpu1/runs/P4a_prefill_c1024_a0/prefill_resident.json`.

## Acquisition 1

capture_passed_pending_CPU_review. instrumented complete prefill plus first selected token, followed by separate63decode ticks; source/output/clock/isolation/prefix coverage and cross-acquisition statistics require CPU review; no physical traffic or ordinary TTFT claim. Evidence: `/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16238357_priority/gpu1/runs/P4a_prefill_c1024_a1/prefill_resident.json`.

## Acquisition 2

capture_passed_pending_CPU_review. instrumented complete prefill plus first selected token, followed by separate63decode ticks; source/output/clock/isolation/prefix coverage and cross-acquisition statistics require CPU review; no physical traffic or ordinary TTFT claim. Evidence: `/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16238357_priority/gpu1/runs/P4a_prefill_c1024_a2/prefill_resident.json`.

- Limitation prefill_CPU_review: Prefill capture/source/phase-clock qualification is incomplete or rejected. Evidence: results/prefill_CPU_review.json. Tried: Own smoke/HF, exact prefix and63decode checks, phase-specific clock sampling and original3x10/source/isolation gates. Resume: Inspect exact errors and raw windows; do not pool UUIDs or resample for favorable clocks. Preserve instrumented scope and resume a fresh complete acquisition set when incomplete.
