# H200 allocation 16228671: completed raw eager curve and partial follow-ups

**PREEMPTED** on 27 November 2026 at 13:45:47 Eastern, after 1 hour 27 seconds of the requested 90 minutes. Allocated GPU-hours: **2.015**, including staging and idle. The additional completed Slurm step used the same pair and is not charged again. [Terminal accounting](../environment/slurm_accounting_terminal.txt).

## NVIDIA SM-count curve

Qwen2.5-32B-Instruct BF16, batch 1 / context 4096, seed 20261013, 64 fixed outputs after the verified one-token prefill barrier. The separately labeled `resident_eager_v1` protocol completed 3 independent process acquisitions, 9 conditions each, 3 warmups and 10 timed trials per condition. All seven actual-decoder placement proofs, same-shape dynamic and unchanged StaticCache HF references, and producer/source/isolation checks passed.

**The measured raw eager curve is complete. Qualified NVIDIA thresholds remain NA:** unrestricted/full-Green controls and most intermediate settings failed the predeclared 5% clock envelope. All trials and nonmonotonic results remain. Optimized graph restriction is separately blocked by earlier measured context escape and bounded capture-repair failure; eager measurements are not optimized decoder normalization.

| Path | Actual SMs | Acquisition0raw median ms/tick | Acquisition1 | Acquisition2 |
|---|---:|---:|---:|---:|
| stock | 132 | 20.8135 | 20.6459 | 20.7675 |
| green132 | 132 | 20.7671 | 20.8158 | 20.7584 |
| green112 | 112 | 22.0124 | 21.7344 | 21.6189 |
| green96 | 96 | 22.2352 | 21.9382 | 22.0744 |
| green80 | 80 | 21.5108 | 21.6885 | 21.5096 |
| green56 | 56 | 23.1994 | 23.1194 | 22.9668 |
| green40 | 40 | 24.8450 | 24.7503 | 24.7872 |

Interleaved/end full-Green controls and all qualification records: [CPU curve review](../gpu0/analysis/resident_curve_CPU_review.md), [raw plot](../gpu0/analysis/eager_raw_sm_curve.png), [curve gates](../gpu0/results/curve_eager_b1_c4096.json), [strict HF reference](../gpu0/results/resident_pilot_hf.json). These are retained per-acquisition medians, not qualified confidence intervals or thresholds.

Each placement range contains 2553 decoder kernel launches. All six restricted settings share identical symbols/grid/block/register/shared/local launch signatures in this resident cohort. [Launch review](../gpu0/analysis/eager_decoder_grid_catalog.md). The earlier per-point-model cohort used a different GPU and showed a symbol/grid switch; these datasets do not isolate retuning causality. Missing resident operation ranges remain unattributed.

## Other completed and interrupted work

- On GPU 1, optimized full-device 4K captured 3 acquisitions of 10 timed trials; placement and both HF references passed. Raw acquisition medians were 19.2274, 19.2960 and 19.3118 ms/tick. Pooled SM-clock ratio 1.5904 rejects clean timing inference. [Context review](../gpu1/results/full_context_CPU_review.json).
- At 16K, placement and both HF references passed; acquisitions 0/1 captured 10 trials each. Preemption stopped acquisition 2 during model initialization, with its preserved run reporting zero cases. It is partial; no 32K case launched and no cross-context ratio qualified.
- The three-policy serving pilot passed actual composition and all nine 64-token outputs: separate policies emitted no mixed steps, while chunked/mixed emitted mixed steps. Independent dynamic/StaticCache references passed. The 1K prefill pilot also passed: whole executed one step and chunk256 executed four, each covering the complete prefix, followed by 63 separate decode ticks. [CPU functional review](../gpu0_postcore/results/functional_pilots_CPU_review.json). These are functional instrumented prerequisites, not loaded serving, tuning, held-out SLOs, energy, physical traffic or 3-acquisition prefill results.
- The fixed one-second private-buffer conditioning diagnostic held sampled timed SM clocks at 1980 MHz on the spare device, versus 1260–1980 MHz unconditioned. This is a single diagnostic pair. It changes cache history and adds untimed TTFT; its future conditioned curves remain separate from unconditioned results.
- The repeated five-budget 1K prefill comparison was deferred before GPU launch when 1048 seconds remained against 1182 seconds required admission budget. Its controller is ready for an early slot in a future allocation.

## Preservation and next allocation

Before preemption the completed core was archived with all 328 members and destination SHA verified: `/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives/gpu0_d71e1de9a02c.tar.gz`, SHA256 `d29b6a85befbdb6f3fcb6ca7e16e86f33b4b4bb5ccdf03bc264533fb10923368`. Both functional pilots were likewise archived with 208 members: `gpu0_postcore_ebe29f6fed81.tar.gz`, SHA256 `dc2e85613815da8e41b93ba0d3618a5f20274b1fa3185b9a3f54f8c9329aae5f`. These are pretermination snapshots; final CPU analysis metadata was published afterward. Raw measurements remain unchanged. The full closed-allocation archive follows this review.

Job 16233972 is already submitted and eligible after the dependency released. Its limit was reduced while pending to 75 minutes, based on the measured core duration and fixed-conditioning overhead, retaining a 15-minute closeout reserve. It requests two H200s for the separately declared conditioned B1/C4096resident curve and conditioned full-device contexts, with fresh UUID/source/placement/HF/source/clock/isolation gates. No original clock failure is promoted. Submission and limit-update receipts are under `h200/20261127_resident_preparation/results/`.

Branch: `codex/h200-phase01`. [Final case manifest](../results/case_manifest.json), [allocation reconciliation](../results/allocation_closeout.json), [limitations register](../../../docs/h200_measurement_limitations_20261125.md). Four closed allocations total 8.0606 allocated GPU-hours. No AMD threshold, clean bandwidth join, physical area benefit, optimized restriction success or loaded-serving result is inferred.

Exact monitoring command for the pending replacement:

```bash
timeout 25s squeue -h -j 16233972 -o '%i|%T|%l|%Q|%M|%R'
```

The full closed-allocation archive is verified: `/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives/20261127_job16228671_priority_420321f2d9fa.tar.gz`, SHA256 `34f54163e5c9b9869a24867700c6f6b39993cda1f43225a49b593e9ea91183a4`. All 834 members (82,245,854 source bytes) and destination readback passed. This final transfer receipt and subsequent planning metadata are published after the archive snapshot; original raw captures remain intact.
