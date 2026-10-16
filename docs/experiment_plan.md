# Prioritized H100 experiment plan

**Priority superseded2026-10-16 for H200:** execute [jobs J0–J11](h200_shoreline_jobs.md), starting with actual local-status reconciliation and reuse of valid Phase0/1 work. The new [governing plan](shoreline_aware_research_plan_2.0.md) compares compute-to-SRAM, compute-to-bandwidth and joint designs. Preserve the original numbered implementation list below as context, not a demand to finish the old portfolio first.

For the next rental, execute the ordered continuation tasks in `RENTAL_RESUME.md`. This original plan is the overall contract, not a claim that every listed step is still unattempted.

1. Complete environment, version, UUID and disk inventory; validate CPU package contracts. Record failures and unavailable tools.
2. Establish a driver-compatible BF16 bridge with the immutable Qwen2.5-1.5B revision. Run B1/C128 smoke with three warmups and three measured trials. Check all token IDs and the one-token prefill barrier before expanding.
3. If a compatible optimized vLLM route is independently validated, run its controlled baseline. Otherwise keep any eager HF bridge as a distinct measured control and mark optimized-baseline gate incomplete. Add B1/C1024 then B4/B8 and longer contexts only after actual memory and disk fit.
4. Measure uninstrumented timing before energy/profiling. Keep GPU events, scheduler wall, first token, pure decode and aggregate token counts separate. Test energy with the raw-counter and sampled-power agreement gates; report primary J/token as NA on failure.
5. Identify actual sm_90 kernels and available Nsight metrics. Bound selected decode launches and retain a larger-batch compute-active counterexample. Mark unsupported or ambiguous metrics NA.
6. Query actual L2 bytes; run the dense logical reuse model over LRU, selective weight pins and shared-prefix policies, block granularities and recurrence lengths. Label all outputs analytical.
7. Only with sufficient free space, build pinned trace tools and capture a small dynamic-input native trace. Require positive instruction records and kernel lists before a bounded whole-model bridge capture. Then postprocess, replay real sm_90 with positive kernel/cycle/instruction statistics, and compare against measured eager execution. Preserve all failed and partial artifacts.
8. Consolidate the Section 9 result schema, case manifest, experiment log, calibration limits and opportunity report. Advance to physical architecture work only if realistic incremental opportunity and native calibration gates pass.

Space and compatibility stop rules: do not silently switch precision, quantize, add a GPU, weaken correctness or substitute CUDA 13 on the R570 driver. If a required artifact cannot fit alongside its raw/grouped/archive headroom, record the incomplete gate. Workspace-only storage cannot complete verified durable transfer.
