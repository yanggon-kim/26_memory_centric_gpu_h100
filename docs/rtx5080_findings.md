> Historical RTX 5080 study. Embedded paths refer to the prior RTX artifact tree. These are not H100 measurements or simulator calibration.

# Phase 0–2 findings: memory-centric LLM decode

**Result:** low-batch decode provides evidence for improving weight delivery, but “less compute and more memory is better for LLMs” is not a validated architectural conclusion. Extra ordinary LRU capacity has negligible benefit in the tested logical streams. Selective weight pinning has a conditional opportunity. Physical compute removal, fixed-area feasibility and energy/token improvement remain unproven.

## Measured portfolio

RTX5080 16GB GDDR7,84SM,64MiB L2; vLLM0.22.1/torch2.11/CUDA13.0 isolated environment. Original ABI-incompatible vLLM environment is preserved. Qwen2.5-Instruct0.5B/1.5B/3B cached pinned snapshots; BF16 weights/KV; seeded uniform synthetic prompts, greedy decoding, EOS ignored. Three warmups/trials per reported case. This is a bounded offline decoder study, not production serving or a model-quality evaluation.

Controlled Qwen1.5B vLLM timings exclude prefill through an asserted all-request first-token barrier. Each trial times63 decode iterations; latency is median milliseconds per batch step, throughput includes B×63 tokens. CUDA events include launch gaps; synchronized wall includes scheduler/assertion work.

| Batch | C128 ms | C1024 ms | C4096 ms |
|---:|---:|---:|---:|
| 1 | 4.920 | 4.926 | 4.935 |
| 4 | 5.336 | 5.420 | 6.037 |
| 8 | 5.390 | 5.643 | 6.470 |

Additional 0.5B/3B six-point async vLLM portfolios, initial HF/vLLM references and nine valid compiled StaticCache cases are retained. Async request-core latency and batch-window throughput use different boundaries; later-request prefill can overlap the batch window. Do not rank them as identical controlled measurements. StaticCache12-case sweep stopped at B16 correctness failure; its nine passing cases are a partial reference, not a complete12-case sweep.

## What the counters establish

Optimized B1/C1024 spends88% of profiled kernel time in GEMV. Representative MLP GEMV kernels reach79–88% sustained DRAM throughput,1.7–6.1% L2 hit rates,48–56% long-scoreboard stalls and zero Tensor Pipe activity. Runtime sm_120 SASS contains weight loads/unpack/HFMA2 dependencies consistent with these counters. Smaller linears/long-context attention are less bandwidth-saturated; latency/concurrency causality is unresolved.

B8/C4096 changes to BF16 WMMA GEMM (72% of profiled busy time) and substantial split-KV attention (21%). This provides a counterexample regime that must be included before trimming arithmetic. Profiler timing and selected first-layer counters are not unperturbed full-model measurements. See kernel_inventory.md and bottleneck_classification.md for scopes, intervention and uncertainty.

## Traffic opportunity and energy limits

At B1/C1024,128MiB ideal selective pinning reduces modeled read-fill traffic13.41/4.30/2.16% for0.5/1.5/3B. At512MiB,1.5B gives17.22%; B8/C4096 with75% shared prefix gives15.74%. General LRU remains negligible. Existing L2/GQA/prefix reuse is charged before added savings. Diagnostic sizes are not feasible die-area claims. Full-block fill accounting, recurrence/granularity sensitivity and eight invariants are verified; actual instruction request traces are unavailable.

All final primary energy/token entries are **NA**. StaticCache long windows show84–90% counter-versus-integrated-power disagreement; controlled vLLM2048-token runs show82–87%, despite approximately10–12s windows. Raw counter and sampled-power estimates remain available. No correction factor or independently calibrated energy benefit is asserted. Full-system power, dynamic/static components, HBM-specific traffic and SRAM/NoC area/power remain unavailable.

## Research decision and reproducibility

Proceed only with a falsifiable local selective-residency plus compute-right-sizing strawman after obtaining calibrated architectural/physical costs. Reject naive cache expansion as the lead mechanism. Do not proceed directly to a heterogeneous remote memory-serving cluster based on these data.

Primary artifacts:

- results/phase1_results.csv:46 scoped measured rows with required handoff fields and NA reasons.
- results/phase2_reuse/traffic_bound_curves.csv:405 analytical policy/capacity rows; sensitivity directories preserve separate assumptions.
- plots/ and results/phase02_artifact_manifest.json: standalone figures and hashed provenance.
- scripts/reproduce_phase02.sh and configs/vllm_requirements.lock.txt: sequential GPU/CPU reproduction and205-package environment.
- docs/opportunity_report.md, final_phase02_spec_review.md, final_measurement_quality_review.md: qualified decision and independent reviews.

The requested GPU/software work through Phase2 is delivered. Hard architectural handoff gaps are disclosed: detailed SM/CTA request traces, actual DSM/TMA ablations, full-system calibrated energy, SRAM/NoC PPA and modern simulator calibration remain. Phase3 was not implemented. See environment_inventory.md for simulator feasibility and unsupported paths.
