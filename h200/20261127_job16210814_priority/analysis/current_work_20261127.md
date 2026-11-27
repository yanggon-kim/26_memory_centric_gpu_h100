# Current two-H200 work — 2026-11-27

Job16210814 is PREEMPTED. It ran09:38:37–10:39:07 Eastern for3630s, consuming2.0166666667 GPU-hours. No user jobs were running or queued at the post-preemption status check. Final accounting is in `../environment/final_slurm_accounting.psv`.

GPU0: restricted eager Qwen2.5-32B BF16, B1/C4096, stock132 and Green132/112/96/80/56/40SM. Real decoder NSYS proofs and independent exact-output reference passed. Acquisition0 captured all nine cases including interleaved/end full references, each3warmups and10timed trials. Acquisition1 stock completed; its full-Green case was interrupted. Acquisition2 is absent. The three-acquisition deliverable is partial. Clock envelopes failed; qualified latency and NVIDIA5/10/25% thresholds remain unresolved. Optimized graph restriction remains blocked by the previously recorded context/graph integration failures, so this is explicitly an eager fallback.

GPU1: after a scoped host-linker repair, all three optimized full-device DRAM+L2 profiler acquisitions completed with replay checks. These are physical instrumented counters, not clean timing, L2 hit rates or architectural savings. See `../gpu1_native_linker/results/P2a_graph_b1_c4096_a{0,1,2}.json`. The three clean/observed scheduler pairs also completed with exact outputs; `../gpu1_native_linker/results/P3a_observer_pilot.json` records six completed producers. This validates the decode-only observer pilot; loaded mixed serving, tuning and SLO comparisons remain pending. All six pilot clock gates failed, so an observer latency ratio is unqualified.

Work uses two independent single-GPU workers, not tensor parallelism. Shared-storage outputs survive preemption; completed bundles are committed and pushed. Only owned allocated devices are used. No clocks, power, drivers or cluster settings were changed. There is no surviving GPU worker after this allocation ended.

Next: finish CPU closeout/archive and prepare a bounded, versioned reduction in repeated model initialization before the next authorized two-H200 submission. Fresh UUID cohorts must remain separate. A new job has not yet been submitted at this status check.
