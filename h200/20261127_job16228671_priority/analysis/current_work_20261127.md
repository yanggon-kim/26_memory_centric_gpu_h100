# Current two-H200 work — job16228671

Updated 2026-11-27T18:04:28.857506+00:00. This is a live progress record, not a completed comparison.

Allocation began2026-11-27T12:45:20-05:00 on atl1-1-02-014-16-0. Scheduled end is2026-11-27T14:15:21-05:00; Slurm reports first preemption eligibility2026-11-27T13:45:20-05:00. Both workers reserve15minutes for closeout. Only the allocated UUIDs are used:

- GPU0: GPU-d6a286c1-f8de-4287-55f5-2ba29b153b9c.
- GPU1: GPU-28fa5fe0-0a71-78dc-ff99-a1fb75e9d459.

GPU0 passed the resident eager stock/full-Green/40SM actual-decoder pilot and independent same-shape64-output dynamic-cache and unchanged StaticCache HF checks. The full seven-level placement proof is running; three independent nine-condition acquisitions follow if its gates and remaining budget permit. This is the separately declared resident protocol; prior per-point model-init timings are not pooled. See gpu0/results/resident_pilot_hf.json and resident_b1_c4096_a0_pilot_*.json. No qualified latency curve is claimed yet.

GPU1 passed both HF reference paths for the unconditioned optimized full-device clock pilot. Its unconditioned clock envelope failed (1260–1980MHz SM; memory3201MHz), so those observations are diagnostic. The separate fixed memory_settle_v3 pilot is running. It changes cache history and adds untimed TTFT; it cannot promote unconditioned timings. Longer-context cases wait for the explicit core disposition.

Frozen runtime source identity: `3e08ca0ce0d8b265b436dfae285772922f91f5b44b3a3173c5e6eed2d7a4a07e`. Later repository edits do not change this running frozen cohort. Worker reports and limitations record new outcomes as they occur; limits and quantitative gates are unchanged. Native KV connector and new serving/prefill policy adapters remain prospective and are not enabled in this job.

## Resident decoder launch geometry: CPU review of saved captures

All seven placement captures contain2553decoder kernel launches over the selected three ticks. The six Green settings have exactly identical launch signatures (symbol, grid/block, registers, shared/local allocations and counts); the CPU review is saved in `results/eager_decoder_grid_catalog.json` and `analysis/eager_decoder_grid_catalog.md`. Missing per-operation NVTX ranges remain unattributed. This is launch metadata, not occupancy, issue/stalls, measured physical traffic or clean timing.

The prior fresh-model-per-point cohort showed an output-projection symbol/grid variant change between112and96SM. The new resident cohort does not show such a change across these selected ranges. Protocol and physical GPU differ, so the observation does not isolate a cause or establish a latency transition. These datasets must remain distinct; a retuning intervention with identical-settings controls is still pending.
