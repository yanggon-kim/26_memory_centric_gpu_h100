# Sequential two-H200 continuation: resident eager protocol

Prepared only; no new GPU measurement. The owner renewed authorization on2026-11-27 to keep submitting sequential Slurm jobs and finish the campaign families one by one. The current hardware scope remains twoH200s;4GPU worker ratios and optionalTP remain deferred.

Earlier fresh-model-per-point measurements survived preemption but did not capture3whole acquisitions on one UUID. This separate versioned protocol keeps one full32B BF16 model resident per fresh independent acquisition and runs9SM/control conditions sequentially. Its model/cache initialization history differs from earlier data, which will never be pooled. Every condition retains3warmups,10timed trials,64fixed outputs and the verified one-token prefill barrier. All original source, clock, isolation and exact-output gates remain.

A bounded NSYS pilot verifies stock/full-Green/40SM real decoder launches and independent full HF exact output before a7setting proof and the broad curve. The native-linker repair is scoped to recorded frozen wrappers and does not change system software. Critical raw files persist on shared storage; the closeout reserve is15minutes. Request90minutes to increase scheduling flexibility while budgeting useful comparisons for the first hour; preemption may still interrupt any case.

Spare work reuses prior3DRAM/L2 captures and3scheduler-observer pairs within their original source/device scopes. Missing full-GPU context acquisitions follow the core disposition; no loaded serving or calibrated bandwidth is inferred from those earlier pilots.

CPUcontracts passed:3resident binding/window/order checks, priority contracts,3eager-first tests and5allocation-review tests. GPUvalidation is pending. The exact next command is `sbatch --parsable batch/h200_priority_resident_20261127.sbatch`.

## Spare clock diagnostic

Saved job16210814 trial windows show SM clocks rising from roughly1,305–1,440MHz near decode entry to1,980MHz during the same interval. The rejection is not just an idle endpoint outside decode. On the spare GPU, one prospectively fixed unconditioned/memory_settle_v3 pair will investigate this while the core GPU runs. Each arm has one independent producer, three warmups and three timed trials with the original exact HF and clock checks. This is a functional diagnostic, not a qualified latency comparison.

The existing memory_settle_v3 adapter executes a fixed one-second256MiB private-buffer read/write workload after the prefill barrier, outside decode timing. It changes cache history and TTFT. No clock or power settings change. All failed windows remain; no favorable trials are selected. A conditioned restricted curve would need a separate declared protocol, actual-kernel proof, HF reference and three-acquisition/ten-trial/source/clock/isolation gates before any threshold.

## Future serving adapter audit — no GPU validation yet

The installed vLLM scheduler and native P2pNcclConnector sources were audited by path and checksum in `../results/serving_capability_audit.json`. A versioned three-policy scheduler adapter and three-request functional pilot are prepared. Six CPU checks passed for preserving held seats/queues through scheduling failures, request progress, exact prefixes and TTFT measured from offered arrival. These scripts are not enabled in job16228671 and are not serving results. Actual GPU input-batch/whole/mixed/output proof, observer intrusion, open-loop traces, tuning and3x10 repeated measurements remain pending.

The native1P1D connector is present in the pinned runtime. Its consumer loads all but the last prompt query; its default32GiB pool is pinned host memory. A future bounded connector pilot must record the actual transfer semantics, request-address metadata, payload/completion, KV seats and both devices before any loaded comparison. The existing HF handoff remains a separate software-functional result on its original pair.

## API reference and empirical requirement

[CUDA13.0.1 Green Context API documentation](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html) describes primary-context retention, explicit activation and the restriction that a Green context is current to one thread at a time. Those API statements do not establish resident32B decoder compatibility. Actual driver stream-resource queries, real decoder NSYS placement and full HF exact outputs remain mandatory.

## Prepared long-prompt adapter — not enabled or GPU validated

`scripts/h200_prefill_resident.py` prepares a separate whole/chunked-prefill acquisition protocol. Whole,256/1024/4096-token budgets and a whole-end reference share one resident full model per independent acquisition. The unchanged controlled decoder sees the same experimentally checked one-token barrier after all prompt chunks; exactly63later decode steps remain separate. Three CPU checks passed for reaching that barrier, rejecting extra tokens/premature completion and bounding a missing barrier.

This is an instrumented policy/observer/event scope, not a qualified clean TTFT or traffic result. Cold prompt means no prefix/KV reuse, not forced cold weights or a cache flush. It is not enabled in job16228671. A bounded whole/chunk256 GPU pilot, independent HF exact outputs and a validated physical-counter reader are required before the context matrix or quantitative conclusions.

## Saved physical/logical demand comparison

[The CPU accounting report](qwen32b_physical_logical_accounting.md) joins all three audited32B DRAM/L2 captures to exact checkpoint headers and replay-selected input tokens. Reads are approximately1.00055times the summed per-tick logical demand, consistent with aggregate repeated weight streaming. This does not prove address-level exactly-once reads, absence of reuse, a clean bandwidth join or architectural benefit. Detailed model/header, replay and raw-counter checksums remain in the accompanying JSON. No new GPU measurement was made.

The resident-mode continuation holder uses the same explicitly declared15minute closeout reserve as its workers. Earlier modes retain30minutes. This scheduling correction avoids ending an unpreempted90minute allocation after only60minutes; all scientific gates and actual Slurm EndTime checks remain unchanged.

## Optional allocated runtime I/O diagnostic

Earlier cold runtime imports consumed roughly408seconds before experiments. An optional read-only page-cache diagnostic is prepared for the next resident-mode allocation: read selected installed runtime libraries for at most60seconds and8GiB, with an80second subprocess limit. It runs only inside the allocation, imports no Torch/CUDA code and changes no installed bytes. Two CPU tests passed for read-budget bounds and exclusion of external symlinks/unrelated files.

This is a hypothesis about startup I/O, not an established cause or speedup. Failure is recorded and the original mandatory600second runtime/device/version check still runs unchanged. It does not change numerical programs, scientific gates or reported GPU-hours.

## 2026-11-27: eager SM proofs include a kernel/grid change

CPU reprocessing of all seven saved job16210814 eager placement traces found2,553launches per three-tick capture. Six Green settings share27launch signatures, but the attention output projection differs:132/112SM use `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` at grid4×20×1;96/80/56/40SM use the2x1variant at grid2×40×1. Each changing group has192launches, uniquely joined through CUDA correlation IDs and same-thread runtime API intervals to actual `self_attn_o_proj` NVTX ranges for64layers×3ticks. Block384×1×1,168registers/thread and164,308dynamic shared-memory bytes are unchanged. Evidence: `h200/20261127_resident_preparation/analysis/eager_decoder_grid_catalog.md` and source-bound catalog JSON.

This limits interpretation of the raw latency curve as an unchanged-kernel hardware-only SM intervention. The varying kernel path is observed once per setting; causality, repeatability, actual cluster dimensions and a latency transition are unresolved. No qualified threshold is inferred from instrumented or clock-rejected timings. The prepared resident protocol initializes library/model state differently, so it must remain separate and prove every setting's real decoder placement and kernel path. Resume with an actual API/operation-bound kernel-selection diagnostic and explicit retuning/identical-settings controls; do not infer cluster configuration from symbol substrings or assume fixed grids.
