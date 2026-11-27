# Two-H200 campaign: complete raw NVIDIA eager curve

Updated 2026-11-27T18:29:05.070410+00:00. Allocation 16228671 remains active; this report distinguishes finished evidence from work in progress.

Qwen2.5-32B-Instruct BF16, batch 1 / context 4096, seed 20261013, 64 fixed outputs after the one-token prefill barrier. The separately labeled `resident_eager_v1` protocol captured all 3 independent acquisitions, 9 conditions each, 3 warmups and 10 timed trials per condition. Seven actual-decoder placement proofs, matching dynamic/unchanged StaticCache HF checks, and producer/source/isolation provenance passed.

**The raw eager curve is complete; qualified NVIDIA thresholds remain NA.** Full-device and most intermediate points fail the predeclared 5% SM-clock envelope. All trials and nonmonotonic observations are retained. Optimized Green graph restriction remains separately blocked by earlier measured context escape and bounded repair failure. This eager fallback does not normalize optimized decoder latency.

| Path | Actual SMs | Acquisition0raw ms/tick | Acquisition1 | Acquisition2 |
|---|---:|---:|---:|---:|
| stock | 132 | 20.8135 | 20.6459 | 20.7675 |
| green132 | 132 | 20.7671 | 20.8158 | 20.7584 |
| green112 | 112 | 22.0124 | 21.7344 | 21.6189 |
| green96 | 96 | 22.2352 | 21.9382 | 22.0744 |
| green80 | 80 | 21.5108 | 21.6885 | 21.5096 |
| green56 | 56 | 23.1994 | 23.1194 | 22.9668 |
| green40 | 40 | 24.8450 | 24.7503 | 24.7872 |

These are medians of ten retained timed trials per acquisition, not qualified confidence intervals or thresholds. Interleaved/end full-Green controls are included in the complete [CPU review](../gpu0/analysis/resident_curve_CPU_review.md). [Raw plot](../gpu0/analysis/eager_raw_sm_curve.png), [curve gates](../gpu0/results/curve_eager_b1_c4096.json), [strict HF reference](../gpu0/results/resident_pilot_hf.json) and [chronological worker report](../gpu0/analysis/report.md).

All seven placement ranges contain 2553 decoder kernel launches. All six restricted settings share identical launch signatures in this resident cohort. [Saved launch-geometry review](../gpu0/analysis/eager_decoder_grid_catalog.md). The prior per-point-model cohort used a different GPU and showed a kernel variant change; these protocols cannot isolate retuning causality. Missing resident operation ranges remain unattributed.

## Current independent work and next allocation

- GPU 1: optimized full-device context 4096 proof and both HF reference routes passed; repeated full-device context cases continue. See its [worker report](../gpu1/analysis/report.md). Its separate fixed-conditioned diagnostic held sampled timed SM clocks at 1980 MHz; this single functional pair is not a 3-acquisition latency comparison.
- GPU 0 post-core: verified original worker completion and zero selected-GPU compute processes before creating a separate frozen cohort for bounded serving-policy and chunked-prefill functional pilots. See [post-core report](../gpu0_postcore/analysis/report.md). These are instrumented functional prerequisites, not completed P3a/P4a quantitative families.
- Job 16233972: two H200s, 1:30 walltime, fixed-conditioned B1/C4096resident curve, queued with `afterany:16228671`; no overlapping allocation. Current allocation had 2537.5 seconds estimated work versus 2275.6 seconds before its launch cutoff, so that curve was deferred to the replacement.

## Preserved core evidence

Before allocation termination, the completed GPU 0 evidence unit was transferred to project storage with every archive member and destination SHA verified: `/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives/gpu0_d71e1de9a02c.tar.gz`. SHA256: `d29b6a85befbdb6f3fcb6ca7e16e86f33b4b4bb5ccdf03bc264533fb10923368`. Receipt: [archive transfer](../gpu0/results/archive_transfer_d71e1de9a02c.json). Other workers remain active and are outside this completed-unit archive. Later parent-level reports are published separately.

Limitations and attempted remedies are recorded in the affected worker reports and [central register](../../../docs/h200_measurement_limitations_20261125.md). A prospective atomic JSON replacement repair protects future snapshots after two transient live-manifest reads; final acquisition 2 JSON and all condition files passed review. The running original frozen source was unchanged. No AMD threshold, clean bandwidth join, area benefit, optimized restriction success or loaded-serving result is inferred.

## Further completed evidence

The optimized full-device B1/C4096 comparison captured all three acquisitions on GPU 1. Raw acquisition medians were 19.2274, 19.2960 and 19.3118 ms/tick; the pooled SM-clock ratio was 1.5904, so latency intervals and qualified cross-context ratios remain NA. Its own placement and matching dynamic/StaticCache references passed. At 16K, both correctness routes and placement passed; acquisitions 0 and 1 are complete and acquisition 2 is running. These are distinct from the GPU 0 eager curve.

Both GPU 0 post-core functional pilots passed CPU provenance/composition review: all nine serving requests matched 64 outputs, whole/separate and chunked/separate emitted no mixed steps, and chunked/mixed emitted actual mixed steps. All six trials per prefill condition (three warmup/three measured) executed the complete 1K prefix: one whole step or four 256-token chunks. Evidence: [CPU pilot review](../gpu0_postcore/results/functional_pilots_CPU_review.json). No loaded/tuned serving, ordinary TTFT, energy or physical-traffic comparison is inferred.

The completed post-core evidence unit is also archived with every member verified: `/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives/gpu0_postcore_ebe29f6fed81.tar.gz`, SHA256 `dc2e85613815da8e41b93ba0d3618a5f20274b1fa3185b9a3f54f8c9329aae5f`. Its snapshot precedes the additional CPU-review JSON; raw pilot artifacts are unchanged, and the CPU review is published separately in Git. The full five-budget, 3-by-10 prefill comparison was deferred before launch when 1048 seconds remained against 1182 seconds required admission budget. Its controller is ready for a later allocation; no partial GPU capture was started.

## 2026-11-27T18:46:52.497840+00:00: allocation preempted

Slurm accounting records job16228671PREEMPTED at2026-11-27T13:45:47-05:00 after3627seconds, before its1:30requested limit. This is2.015allocatedGPU-hours; overlapping step16228671.1 used the same pair and is not an additional allocation. Evidence: `environment/slurm_accounting_terminal.txt`, the batch epilog and GPU1's final process/monitor/log.

The completed3acquisition resident eager SM curve, optimized full-device4Ktriplicate, clock diagnostic, and both functional serving/prefill pilots remain intact. At16K the optimized placement and strict dynamic/StaticCache references passed, and acquisitions0/1completed. Acquisition2was interrupted during initialization: its preserved `run.json` has statusrunning and zero cases, not a completed or unsupported measurement. No32Kcase launched. This limits repeated16Kstatistics and all cross-context latency conclusions; full4Ktimings also failed clocks.

Tried: separate immutable producers, continuous compact checkpoints, completed-unit archives before preemption, and a dependency-held replacement. Job16233972is already submitted, now eligible for a separate fixed-conditioned B1/C4096curve and conditioned full-device contexts. Resume each incomplete quantitative comparison as a fresh identified cohort with3acquisitions; do not pool these two16Kacquisitions across GPUs or promote clock-rejected results.
