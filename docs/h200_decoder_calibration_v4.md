# Decoder calibration and validation, version4

This executes the missing calibration/validation step for one narrow H200 family. No new measurements or successful validation are claimed by this implementation. The rejected J4 native fit remains rejected and supplies no coefficients.

## Scope and acceptance

One pinned Qwen2.5-1.5B (bridge) graph-engine family, BF16 weights/KV, TP1, seed20261013, all-request one-token prefill barrier,64 output tokens/63 timed decode steps. The separately labelled memory_settle_v3 conditioning changes cache history and adds untimed first-token latency. It does not repair the original unconditioned timing. GPU UUID/allocation, sources, clocks, isolation and raw failures are retained. No system clock/power/driver changes.

| Condition | Split |
|---|---|
| B1/C128 | training |
| B1/C1024 | training |
| B2/C512 | training |
| B8/C4096 | training |
| B2/C2048 | held out |
| B4/C1536 | held out |

This is a predeclared condition-level4/2 split (rounded70/30). No saved captures of the two held-out conditions were found when declaring it. Every quantitative condition requires three independent process acquisitions, ten timed trials and at least three warmups. Exact HF StaticCache/graph outputs and pooled <=5% SM/memory clocks are mandatory. No failed trial or condition is discarded and no favorable-clock retry is scheduled.

## Independent features

1. Physical memory service: new synthetic BF16 read/reduction controls use a power-of-two footprint at least as large as the checkpoint. Streaming and128-byte-chunk permutation are separate references. Ordinary CUDA-graph repetitions provide event time; independent NCU app-range captures provide actual read+write bytes. Beta is derived only from these controls, never decoder D/T. A producer/reader pilot must pass before expansion. The paged-like reference is synthetic, not a verified vLLM page-layout replay.
2. Compute reference: BF16 hot tiled torch.mm controls for QKV, output projection, gate/up, down, full-vocabulary head, QK and AV. Batch/contraction dimensions derive from pinned geometry; tiles512/1024 are declared reference choices, not asserted vLLM-issued tiles.128 repeated operations in a CUDA graph reduce CPU-launch gaps. Full-scope useful linear FLOPs must agree exactly with immutable checkpoint accounting. Attention work is summed over all63 lengths; its reference uses the mean length. Other instructions/softmax/scheduling and unknown padding/issued work remain outside useful F.
3. Decoder demand/latency: measure ordinary full63-step decode separately from NCU full63-step HBM bytes. Only measured trial0 emits calibration_decode_selected. All app-range replay processes publish full64-token outputs, prompt digest, barrier and clock endpoints. Three-tick captures cannot satisfy this scope. Profiled time is never substituted for ordinary latency.

Compute controls are synthetic hot-reference service measurements. They do not preserve the full model weight working set, isolate physical arithmetic-only time, or prove identical issued routes. All full decoder captures execute the complete model. Cache/route/issued-work assumptions remain explicit even if latency prediction passes.

## Fit, freeze and validate

C_reference is useful work divided by independently measured useful-work reference rates. M_reference is total physical decoder bytes divided by independent physical memory service. All times refer to the same63-step full-model scope.

Fit only training conditions to T=C+M-eta*min(C,M)+O with one common eta in[0,1] and one common O>=0. No per-condition residual or overlap parameter is allowed. Freeze parameter JSON,500 acquisition/trial bootstrap draws and design hash; commit/push them BEFORE starting held-out decoder acquisitions. Bootstrap intervals describe parameter/reference uncertainty; tile512 and paged-like alternatives are separate assumption scenarios. Unknown issued work is not filled with useful work.

Acceptance requires all predeclared conditions and gates, feasible coupled bounds, training maximum relative error <=10%, and both held-out median-time errors <=10%. A failed training-error gate stops before exposing held-out conditions. No refitting after held-out exposure. Incomplete inputs, source/clock failures, infeasible rates, or prediction errors yield rejected/NA status with raw evidence retained. Bootstrap intervals exclude model misspecification and are not used to excuse an accuracy failure.

Even a passed fit validates only conditional useful-work service-reference latency predictions for this declared decoder program. Physical compute-only/memory-only/overlap percentages remain unidentified. Report coupled residual/overlap bounds and reference sensitivity; do not stack unrelated counter percentages or interval endpoints. Physical architecture/energy extrapolation remains unvalidated.

## Execution and artifacts

`batch/h200_decoder_calibration_v4.sbatch` uses `configs/h200_batch_decoder_calibration_v4.json`. Submit afterany:14215309 so the existing failure-recovery allocation and repository lock finish first. Fresh inventory/smoke precede memory references, four training conditions, model freeze, then two held-out conditions. The required graph clock pilot from14215309 must have passed; if it was rejected or never completed, calibration records the unmet prerequisite and stops rather than retrying that failed clock protocol. One-acquisition pilots stop failing routes before expansion; each acquisition and the model freeze are checkpointed/pushed.

Main artifacts: results/decoder_calibration_v4.json; results/decoder_model_frozen_v4.json; results/decoder_calibration_disposition.json; analysis/decoder_calibration_v4.md; per-process native outputs, raw NCU reports/CSV and source/clock monitors. Large raw captures retain shared-storage hashes; this does not solve the private archive access blocker. A complete Slurm command sequence is separate from successful scientific validation.
