# H200 campaign handoff snapshot

Observed 2026-11-28T14:39:36+00:00. Branch: `codex/h200-phase01`. Source/document snapshot commit: `559d6d45b2fb5129ea955ee1c954a0cebf3574f7`. This is a continuing campaign, not a completion claim.

Progress estimate: **approximately 20%**, a rough qualified-evidence planning estimate. CPU preparation, raw partial captures and pending jobs are listed separately; it is not a runtime forecast or a measured percentage of all matrix cells.

## Published NVIDIA result

The qualified fallback is **conditioned eager B1/C4096**, with an declared 1 second read/write operation on a private 256 MiB buffer after prefill and outside timed decode. It changes cache history. It does not establish an optimized restricted or unconditioned curve. All seven actual decoder-placement proofs, strict default dynamic/unchanged StaticCache outputs and all 27 timing/source/isolation/clock conditions passed in that cohort.

| Allowed latency above full using upper 95% confidence bound | Smallest tested demonstrated SM count |
|---|---:|
| 5% |112|
| 10% |80|
| 25% |40|

Counts are tested NVIDIA points only, independently derived from this cohort. No AMD threshold or interpolation is adopted. The nonmonotonic 80/96 SM observations are retained. Canonical curve SHA256: `59a787d8ec1660a16ba7c83a0c08cf1359eb7793a2062569621f9bcdfe0d63ed`. [Original qualified review](../../20261127_job16233972_priority/gpu0/results/resident_curve_CPU_review.json).

## Latest paired evidence and limitations

The fresh 16K reference and all 12 paired real-model KV functional requests passed on job 16281263. Each functional request requested 4 GiB KV; native host stores and loads each recorded 3.125 GiB. This is API payload evidence, not physical HBM/link bytes or a pure device-to-device transfer.

The fixed producer cohort stopped after its 165 second stock-after process timeout. It was **not preempted**: Slurm recorded FAILED after 2009 seconds. Acquisition 0 stock-before and native roles completed, stock-after retained 3 complete timed requests plus one partial, and acquisitions 1/2 were not launched. Completed stock/producer timing failed the original clock gate; no qualified producer slowdown ratio is available.

All ten completed native request timelines are retained in [the raw host timeline report](../../20261128_job 16281263_native_functional_c16384_producer_quant/analysis/partial_a0_host_handoff_timeline.md). Raw medians: producer first-step 2.197 s, first consumer handoff call 94.974 ms, following 62 calls 1.250 s. Sender completion observations preceded the first consumer call in 10/10 retained requests by a median 10.616 ms. These overlapping host observations do not isolate physical communication, pure GPU time or receiver blocking. End-to-end serving goodput and primary J/token remain unqualified.

## Current reservation and exact next command

Job **16285315** was **PENDING (Priority)** at the last scheduler observation, `2026-11-28T14:39:36+00:00`. It requests 2 H200s for up to 2 hours. Pending status grants no GPUs, UUIDs or remaining allocation time. No overlapping replacement job is authorized by this snapshot.

The prepared pipeline runs fresh known controls, stock/default dynamic/unchanged StaticCache and 12 paired functional requests before an independently gated instrumented connector capture/export/review. It requires actual owned idle UUIDs, frozen source equality and a complete 1800 second capture budget. Profile success alone will not qualify clean timing or serving.

Exact next command at the next recorded 5 minute polling time:

```bash
timeout 25s squeue -h -j 16285315 -o '%i|%T|%l|%Q|%M|%R'
```

After its actual terminal state, inspect the captured outcome, preserve/archive partial evidence and select the highest eligible unfinished case. Do not submit a duplicate while this job is pending/running.

## Remaining scope by family

| Family | Manifest disposition | Evidence / unfinished work |
|---|---|---|
| P0 | partial | Fresh inventory, stock/strict dual-HF and native transport controls passed on recorded pairs. Optimized graph SM restriction failed its bounded repair; fresh allocation gates remain mandatory. |
| P1a | partial | Complete qualified B1/C4096 conditioned eager fallback, 3 acquisitions × 9 conditions × 10 timed trials. Optimized restricted route remains blocked under the pinned stack. |
| P1b | partial | B8/C4096 exact default/StaticCache reference failed, including one bounded deterministic control. B32/other resident shapes remain pending; do not quietly skip the failure. |
| P2a | partial | Optimized full-device DRAM/L2 captures and native service evidence retained in their original scopes. Matched traffic across both sides of the conditioned eager transition and qualified clean timing remain unfinished. |
| P2b | pending | 32B disjoint-SM idle/arithmetic/streaming pressure comparison, actual overlap/placement and dose validation remain pending. |
| P3a | partial | Three policy functional pilots passed. Complete load/tuning/held-out SLO and energy comparisons remain pending; the later load pilot was preempted without completed timed traces. |
| P3b | partial | Real 16K model-KV proof passed; one completed native timing acquisition is raw/clock-rejected and the 3 acquisition cohort is partial. Dedicated connector attribution job 16285315 is pending. Full/restricted loaded worker and goodput remain unfinished. |
| P4a | partial | C1024 whole/chunk 3 × 5 × 10 captured with strict correctness; original clock gate rejected timing. Longer prompt/prefix/kernel/traffic attribution remains pending. |
| P4b | pending | 512/2048/4096 output, injected-prefill and matched cache/history comparisons remain pending. No new long-output GPU result is claimed by CPU preparation. |
| P4c | pending | Portable generator, scoring, exact-input and reference-checking preparation exists. Serial-input integration patch is unapplied; full original output references and offered-load GPU controller/held-out measurements remain pending. |
| P5a | partial | Native functional prerequisites passed, but KV/seat sweeps and equal-GPU two-replica versus 1P1D deployment/SLO/goodput comparisons remain unfinished. |
| P5b | deferred | Deferred: requires 4 explicitly allocated GPUs; current authorization/scheduler request is 2. |
| P5c | partial | Retained J0–J11 reuse audit and original native/treatment evidence exist. New 32B multishape calibration, isolated traffic/compute and frozen held-out prediction gates remain unfinished. |
| P5d | deferred | Deferred optional TP; no TP/precision expansion in the current two-device campaign. |

All 12 families eligible at one/two devices remain in scope; two deferred optional/larger-device rows are separately retained. A blocked adapter or correctness/clock gate is an evidence-bearing outcome, never silently converted into a passed comparison.

## Accounting and archive handoff

Closed allocations in this campaign total **15.162222 allocated GPU-hours**. The queued job contributes no allocated GPU-hours until it starts. Accounting charges allocated boards and wall time, not just active kernel time. [Latest allocation closeout](../results/allocation_closeout_16281263.json).

| Closed evidence unit | Verified archive SHA256 |
|---|---|
| 20261128_job 16281263_native_functional_c16384 | `bb0170afd0046ab801b1d50dac1ead43fca5b88ffe62ce50ff2ccaff0e53e7eb` |
| 20261128_job 16281263_native_functional_c16384_producer_quant | `ae40d1f9bcea51258847d95b41d9ed73200d3055d79666e34ffc403daac547cb` |

Both original archives were transferred and every member verified. Later host-staging/timeline/figure/closeout metadata explicitly postdate them; original captures and canonical certificates remain unchanged. See linked receipts in the input snapshot.

CPU preparation for portable references and future cache export has no new GPU measurement credit. No cache bundle has actually been exported. A future importer must validate compatibility/bytes and retain all fresh numerical gates.

[Campaign manifest](../results/campaign_progress.json); [Limitations register](../../../docs/h200_measurement_limitations_20261125.md).
