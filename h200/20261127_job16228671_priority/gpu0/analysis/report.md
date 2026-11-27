# GPU 0: GPU-d6a286c1-f8de-4287-55f5-2ba29b153b9c

- Limitation resident_protocol_scope: Prior optimized graph restriction remains blocked; repeated model initialization prevented a complete fresh-model-per-point curve before preemption.. Evidence: h200/20261127_job16210814_priority/analysis/current_work_20261127.md. Tried: Versioned one-resident-model-per-acquisition eager protocol; bounded stock/full-green/40SM pilot before full proof and3fresh acquisitions. Resume: Never pool prior timings; retain exact outputs, actual decoder placement, all references and clock/isolation gates; optimized restriction remains independently blocked.

- Limitation resident_b1_c4096_a0: Resident curve clocks fail the predeclared envelope; all trials retained. Evidence: /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16228671_priority/gpu0/runs/resident_b1_c4096_a0/resident_sweep.json. Tried: New separate protocol; bounded actual-kernel proof, independent HF exact reference, original3x10 clock/source/isolation gates. Resume: Inspect the saved failure; do not pool protocols or resample for favorable clocks. Continue eligible full-GPU cases when the captured curve is quality-limited..

## eager B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | 132 | False | NA | NA |
| 0.85 | 112 | False | NA | NA |
| 0.75 | 96 | False | NA | NA |
| 0.6 | 80 | False | NA | NA |
| 0.45 | 56 | False | NA | NA |
| 0.3 | 40 | False | NA | NA |

## Resident decoder launch geometry: CPU review of saved captures

All seven placement captures contain2553decoder kernel launches over the selected three ticks. The six Green settings have exactly identical launch signatures (symbol, grid/block, registers, shared/local allocations and counts); the CPU review is saved in `results/eager_decoder_grid_catalog.json` and `analysis/eager_decoder_grid_catalog.md`. Missing per-operation NVTX ranges remain unattributed. This is launch metadata, not occupancy, issue/stalls, measured physical traffic or clean timing.

The prior fresh-model-per-point cohort showed an output-projection symbol/grid variant change between112and96SM. The new resident cohort does not show such a change across these selected ranges. Protocol and physical GPU differ, so the observation does not isolate a cause or establish a latency transition. These datasets must remain distinct; a retuning intervention with identical-settings controls is still pending.

- Limitation resident_b1_c4096_a1: Resident curve clocks fail the predeclared envelope; all trials retained. Evidence: /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16228671_priority/gpu0/runs/resident_b1_c4096_a1/resident_sweep.json. Tried: New separate protocol; bounded actual-kernel proof, independent HF exact reference, original3x10 clock/source/isolation gates. Resume: Inspect the saved failure; do not pool protocols or resample for favorable clocks. Continue eligible full-GPU cases when the captured curve is quality-limited..

## eager B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | 132 | False | NA | NA |
| 0.85 | 112 | False | NA | NA |
| 0.75 | 96 | False | NA | NA |
| 0.6 | 80 | False | NA | NA |
| 0.45 | 56 | False | NA | NA |
| 0.3 | 40 | False | NA | NA |

## 2026-11-27: transient incremental-manifest read during rewrite

Two CPU progress reads of job16228671's active acquisition2 `gpu0/runs/resident_b1_c4096_a2/resident_sweep.json` encountered an empty JSON payload while the producer was rewriting it; a following filesystem listing showed the manifest present at about1MiB and the producer still running. The shared `software_decode_experiments.dump` writes directly to the destination. This is a live-reader/preemption-preservation limitation, not an observed GPU/output failure. Conclusions must use finalized parseable artifacts; an interrupted in-place rewrite could lose the last manifest snapshot.

Bounded remedy: version the shared dump helper to write a unique sibling temporary file and atomically replace the destination, preserving identical JSON bytes and the previous complete snapshot if a write/rename fails. This applies only to newly frozen cohorts; the current core source stays unchanged. Resume validation after producer completion, preserving partial/invalid artifacts if interrupted, and verify final per-condition outputs/checksums before declaring the acquisition complete.

- Limitation resident_b1_c4096_a2: Resident curve clocks fail the predeclared envelope; all trials retained. Evidence: /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261127_job16228671_priority/gpu0/runs/resident_b1_c4096_a2/resident_sweep.json. Tried: New separate protocol; bounded actual-kernel proof, independent HF exact reference, original3x10 clock/source/isolation gates. Resume: Inspect the saved failure; do not pool protocols or resample for favorable clocks. Continue eligible full-GPU cases when the captured curve is quality-limited..

## eager B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | 132 | False | NA | NA |
| 0.85 | 112 | False | NA | NA |
| 0.75 | 96 | False | NA | NA |
| 0.6 | 80 | False | NA | NA |
| 0.45 | 56 | False | NA | NA |
| 0.3 | 40 | False | NA | NA |

## eager B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | 132 | False | NA | NA |
| 0.85 | 112 | False | NA | NA |
| 0.75 | 96 | False | NA | NA |
| 0.6 | 80 | False | NA | NA |
| 0.45 | 56 | False | NA | NA |
| 0.3 | 40 | False | NA | NA |

Final CPU review: [complete resident curve and raw acquisition table](resident_curve_CPU_review.md). All27condition outputs and producer/source/isolation records passed; quantitative clocks remain unresolved. The transient live-manifest reads were followed by a complete, parseable final manifest. The prospective atomic dump fix passed an interrupted-publication test and preserves exact JSON bytes; current frozen source is unchanged.

## eager B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | 132 | False | NA | NA |
| 0.85 | 112 | False | NA | NA |
| 0.75 | 96 | False | NA | NA |
| 0.6 | 80 | False | NA | NA |
| 0.45 | 56 | False | NA | NA |
| 0.3 | 40 | False | NA | NA |

## eager B1/C4096 curve

| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |
|---|---:|---|---:|---|
| stock | NA | False | NA | 1 (normalization) |
| 1.0 | 132 | False | NA | NA |
| 0.85 | 112 | False | NA | NA |
| 0.75 | 96 | False | NA | NA |
| 0.6 | 80 | False | NA | NA |
| 0.45 | 56 | False | NA | NA |
| 0.3 | 40 | False | NA | NA |
