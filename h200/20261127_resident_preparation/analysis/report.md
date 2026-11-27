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
