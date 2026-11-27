# H200 allocation16184819: partial campaign results

The optimized Qwen2.5-32B BF16/TP1 batch1/context4096 SM sweep remains unfinished. No NVIDIA5/10/25percent threshold is established. The allocation was preempted on27November2026 at01:21:12 Eastern after3619seconds: **2.0106 allocatedGPU-hours**, including startup and idle. Two authorized allocations so far total **4.0289GPU-hours**.

| Evidence | Result | Scope or limit |
|---|---|---|
| Stock optimized32B |132 SM; NSYS2550 launches/2499 graph nodes; independent HF exact outputs pass | Stock proof only; no restricted curve |
| Original full-green optimized proof | Driver grants132 SM, but actual decoder contexts include 0 and 2 | Restriction rejected; graph kernels escape to stock context |
| Corrected capture adapter | UUID preflight succeeds; graph-memory profiling fails PyTorch GreenContext.pop_context assertion | No repaired decoder measurement; bounded attempt retained |
| Separate eager cohort | Stock proof and HF exact outputs pass; full-green proof preempted during load | Eager restriction unresolved, not demonstrated unsupported |
| GPU1 physical DRAM |3 independent exact-replay captures, about65.08 GB read/token | Clean-clock/L2/kernel/restricted-arm joins missing |
| GPU1 native controls |10/10 condition curves source/clock qualified across132/112/96/80/56/40 SM | Native useful payload only; not decoder or physical HBM service |
| GPU0 physical DRAM |Acquisition 0 passes; acquisition 1 interrupted | SeparateUUID; never pool withGPU1 |
| Fresh-pair transfer |128MiB BF16 peer-copy correctness passes | Synthetic functional pilot; fresh real-KV handoff deferred |

## Evidence paths

- [Three-acquisition physical counter comparison](../gpu1/analysis/P2a_triplicate_counter_report.md), including original report SHA256s.
- [Qualified native controls](native_controls_report.md), full uncertainty/source/clock data in `../gpu1/results/native_priority_analysis.json`.
- [Original cohorts CPU closeout](allocation_closeout.md), with process interruptions and case gate issues.
- [Corrected source cohort](../gpu1_import_preflight/analysis/report.md); CPU review `../gpu1_import_preflight/results/terminal_cohort_review.json`.
- [Allocation accounting](../environment/final_slurm_accounting.psv); [limitations register](../../../docs/h200_measurement_limitations_20261125.md).

## Interpretation and continuation

All exact outputs, raw profilers, failed attempts and clock-rejected trials are retained. The full-GPU clean reference clock envelope failed (GPU1 SM1260–1980MHz); no qualified clean bandwidth or latency ratio follows from the counter captures. No driver, power, clock or MIG settings changed. Startup filesystem I/O consumed several minutes; the180s runtime timeout was resolved by a single logged600s retry.

The next two-H200 allocation should prioritize the separately labeled eager SM curve, with graph-capture binding disabled and explicit per-step rebinding. Require fresh actual UUIDs and same-engine stock/full-green/cut NSYS proofs,3 independent acquisitionsx10trials, and all scientific gates. Keep optimized restriction blocked with its observed failure evidence. Use the spare device for eligible full-GPU long-context/scheduler work; do not repeat already completed native/physical comparisons merely for favorable clocks.

The chronological in-allocation notes are preserved in `report_during_allocation.md`. Repeated sequential two-H200 submissions are authorized by the owner; no four-GPU or optional TP campaign is authorized yet.

## Later CPU correctness-scope audit

This allocation's full-length32B C4096 HF reference uses dynamic-cache SDPA. The historical32B StaticCache smoke coversC128; matching C4096 StaticCache agreement is not demonstrated by it. No new StaticCache mismatch is claimed. See `h200/20261127_resident_preparation/results/reference_gate_audit.json` and the limitations register. Existing clock-rejected timings remain unqualified; physical captures keep their recorded dynamic-reference scope. The next versioned reference requires both full-shape cache routes and the exact source checksum. This follow-up note postdates the archived allocation snapshot and adds no GPU measurement.
