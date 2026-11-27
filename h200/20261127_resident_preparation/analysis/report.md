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

## 2026-11-27: matching-shape32B StaticCache correctness gate pending

The preserved Qwen32B StaticCache smoke at `h200/20261016_job14073758/runs/qwen32b_hf_eager_b1_c128_smoke/run.json` passed its existing C128 checks. November B1/C4096 full64output references used HF dynamic-cache SDPA; these do not demonstrate matching C4096 StaticCache agreement. Evidence and exact checksums: `h200/20261127_resident_preparation/results/reference_gate_audit.json`. No StaticCache mismatch is claimed for this target; the matching-shape gate is unmeasured. Prior clock-rejected timings remain unqualified, and old physical captures retain only their recorded dynamic-reference scope.

The future reference is versioned to reuse one HFmodel for3warm+3timed correctness sequences per dynamic and unchanged controlled StaticCache path, with the same prompt/count, BF16 weights/KV and explicit source checksum. Both must pass; no static failure may be promoted by dynamic agreement. Three CPU positive/negative gate checks and the existing priority harness passed. Resume with the allocated matching-shape gate before qualified curves or serving comparisons. A mismatch requires preserved first differences and bounded diagnosis; never weaken the existing mask/cache path or exact-token rule.

## Prospectively fixed conditioned follow-up on the same allocated device

If all three unconditioned resident acquisitions are captured but clock qualification fails, the primary worker may use remaining time for one separate memory_settle_v3 curve. Admission uses measured producer costs and the actual Slurm cutoff, including the15minute closeout reserve. The protocol is fixed in advance: one second of256MiB private-buffer read/write work after each prefill barrier, outside decode timing. Selection does not use favorable latency or discard any original trials.

The follow-up uses a new `gpu0_conditioned` root, workspace and frozen source cohort on the same allocatedGPU0. Its own stock/full-Green/40SM pilot, full7setting NSYS proof, dynamic plus unchanged StaticCache reference and three independent acquisitions are required. All warmup and timed conditioning receipts are checked, including the tenth trial. Clock/source/isolation and exact output gates remain unchanged.

Any qualified thresholds describe conditioned decode only. The workload changes cache history and adds untimed TTFT; it does not repair or replace original unconditioned latency. A blocked or clock-rejected conditioned curve remains explicitly blocked/unqualified and is not rerun merely to obtain favorable clocks. GPUvalidation is pending.

## 2026-11-27: retained aggregate checksum mismatch resolved by exact CPU reconstruction

The October consolidated report references treatment aggregate SHA256 `04906c57be9b0b96f9cb46b219d29d659e272806e717f603f6bed9ff127906d7`, while the current historical file has `b128aa71f0179f89d0dc272b5cd1cf062908038a6aa3af185cc820b5c87fcab1`. The expected exact version was not found in the tracked Git revisions for that path. The current file was preserved.

Recomputation from retained raw inputs using the pinned runtime and existing mathematical reader produced a new versioned copy whose complete bytes exactly match the expected04906c57SHA. All four qualified original contrasts exactly reproduce their reported ratios/intervals;24profile/run/process/source-manifest/monitor provenance checks passed, with no model-output or clock rule relaxed. Evidence: `h200/20261127_resident_preparation/results/retained_treatment_requalification.json` and `results/requalified_historical_treatment_statistics.json`.

This resolves reuse of those original bridge scopes through the reconstructed copy, not by silently modifying the old file. The original canonical-path checksum difference remains documented. No GPU recapture or32B transfer is inferred. The12independent native-service cells have separately verified direct source hashes and are unaffected. Resume through the source-hashed reuse audit and reconstructed copy; never overwrite the historical artifact to hide the discrepancy.

## 2026-11-27: prospective native connector request-ID gate

CPU inspection of pinned vLLM found that `InputProcessor.assign_request_id` appends independent random suffixes on each engine, while P2pNcclConnector transport keys contain the full internal request ID and layer name. Identical external IDs alone therefore do not establish identical producer/consumer tensor keys. Source paths/checksums and evidence are in `h200/20261127_resident_preparation/results/serving_capability_audit.json`. No GPU connector failure or transfer is claimed.

Before the P3b/P5a functional pilot, use the installed compatibility flag `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` prospectively on both isolated workers, assign globally unique paired IDs, and verify actual internal IDs plus per-layer send/receive keys. The flag is deprecated: fail closed if unavailable or keys disagree, then implement a versioned key adapter before loaded comparisons. The current allocation does not run this connector pilot; paired serving results remain pending.

## Prospective resident shape guard

Resident SM producers now reject configurations containing multiple workload shapes; resident prefill producers additionally require batch1. Their cached controlled-run responses are valid for exactly one declared shape per producer. Existing six CPU binding/barrier checks passed. These guards apply to future frozen cohorts; job16228671 keeps its already frozen source and single-shape configuration.

## Prepared sequential follow-up selection

The resident campaign now carries explicit batch/context/conditioning options through allocated bootstrap, coordinator, worker and bounded staging retry. Defaults preserve the current B1/C4096 unconditioned behavior. Supported prospective widths are8/32at4096, or batch1at16384/32704; each producer still admits exactly one shape. Analysis/plot metadata and paths retain the actual shape; width results cannot overwrite the B1alias. Existing scientific gates remain unchanged, and each new shape requires its own placement and dual HF pilot.

`batch/h200_priority_resident_followup_20261127.sbatch` is prepared but not submitted. It requires an explicit scope argument. After the current curve disposition, a separate fixed-conditioned B1/C4096 job can be submitted with `sbatch --parsable batch/h200_priority_resident_followup_20261127.sbatch --resident-conditioned`. Width sweeps follow the published broad curve; conditioned results retain their distinct cache/TTFT interpretation. Selection depends on actual current outcomes; no next job has been queued prematurely and no new GPU result is implied by these adapters.

## Prepared bounded post-core functional pilots

After the original GPU0worker returns with all3resident acquisitions captured, `scripts/h200_postcore_pilots.py` can use that same allocation's idle GPU0for P3a three-policy and P4a whole/chunk256 functional pilots. It requires the completed core disposition, a returned worker outcome, an idle current-process record, zero selected-GPU compute PIDs and the original producer lock. It checks owned RUNNING job/host/UUID, creates a separate workspace/frozen cohort, and uses the unchanged allocation/deadline/isolation supervisor. Two positive/negative CPU transition contracts passed. No old worker is stopped or source replaced.

Each pilot has a600second producer bound and a strict same-shape dynamic/StaticCache reference; a900second remaining-budget admission gate retains closeout reserve. They produce instrumented functional evidence only, not loaded serving, tuned policies, SLOs, physical traffic,3independent acquisitions or family completion. The other allocated GPU may concurrently run its own context cases; shared host CPU concurrency is recorded. These pilots are not yet launched.

## Dependent replacement submitted:16233972

After the completed unconditioned curve and CPU review, job16233972was submitted for the separate fixed-conditioned B1/C4096curve with1:30walltime and an `afterany:16228671` dependency. It cannot receive GPUs while the current allocation is running. Receipt and exact submission/source state: `results/submission_16233972.json`. Branch fast-forward pull was already up to date and ancestor7843c845was verified. This changes the earlier prospective-only submission status; no new allocation or conditioned-curve result exists yet.

## 2026-11-27: prospective conditioned full-device context scope

The pending replacement job16233972will apply its explicitly requested fixed memory_settle_v3 protocol to both the separate eager SM sweep and spare full-device decode-context comparisons. Every3warmup and3/10timed conditioning receipt must match the fixed declaration. New case labels/config files carry the protocol; mixed routes cannot form a qualified shape comparison. The original unconditioned results remain separate. This is conditioned decode only: cache history changes and the extra second lies outside decode timing, so it is not a TTFT/deployment improvement. Two new CPU positive/negative checks passed, along with the priority, resident, reference, eager and clock contracts. Actual GPU qualification remains pending on its own allocation.

A further owned idle-GPU0cohort is prepared for the first instrumented prefill comparison at context1024: whole,256,1024,4096andwhole-end,3independent process acquisitions with3warmups/10timed trials, plus a strict dynamic/unchanged StaticCache reference. The actual same-GPU functional pilot completed in151.13seconds. The prospective full-comparison scheduling estimate is1122seconds plus60seconds admission margin before the existing launch cutoff; preemption can happen sooner. The controller preserves each acquisition and requires separate CPU source/clock/isolation/prefix-coverage review. No ordinary TTFT, physical traffic or qualified comparison is inferred from capture alone.

## 2026-11-27: first repeated prefill comparison deferred by remaining budget

The completed P4a functional pilot at1K passed whole/chunk256 semantics and both HF reference routes. A versioned controller for whole/256/1024/4096/whole-end,3independent acquisitions and10timed trials, is ready. At admission review the remaining time before job16228671's existing launch cutoff was below its1122second full-comparison estimate plus60second margin. Evidence: `h200/20261127_resident_preparation/results/prefill_deferred_budget_16228671.json`; no additional GPU producer was launched. This limits the result to functional pilot evidence, not a repeated prefill comparison or ordinary TTFT claim. Tried: measured-pilot budget estimation and fail-closed admission, with a CPU test confirming insufficient budget launches no producer. Resume at the beginning of a future owned allocation, with fresh source/UUID/reference qualification and separate CPU review.

## Published broad-core gate for efficient spare-device continuation

The broad resident eager curve from job16228671is already captured and published. The next resident coordinator will verify its27raw condition checksums, seven placement proofs, process/source/isolation records, dualHF reference and review/session checksums, then pin an eligibility receipt in the new sessions. The spare worker rechecks the pinned review before proceeding to its own eligible full-device contexts after its clock diagnostic. This avoids waiting for the new conditioned core to repeat an already completed ordering prerequisite.

This transfers only proof that the broad curve was measured before expansion. Its original clock rejection remains: no latency, confidence interval, frequency qualification, optimized restriction success or physical-UUID result transfers. Every new context still requires its own placement and strict same-shape HF gates and3x10source/clock/isolation checks. Missing or changed prior evidence falls back to waiting for the current allocation's core disposition. Actual-data positive verification and a tampered-run checksum rejection passed; evidence: `results/published_core_gate_validation.json`.

## 2026-11-27: prospective serving-policy queue restoration under KV preemption

CPU inspection found that the v1 separate-policy decode mask installs temporary waiting/skipped queues and restores the original queue objects afterward. If the native scheduler preempts an eligible decoder into a temporary queue, restoring without merging could discard that request. Source evidence: `scripts/h200_serving_policy.py::schedule_phase` and pinned vLLM `Scheduler._preempt_request` (which prepends into waiting). The completed three-policy GPU pilot recorded zero preemptions and exact outputs for all requests, so its observed functional scope remains valid; it does not prove this pressure case.

This limits loaded/KV-pressure and seat-limit comparisons until a versioned repair preserves newly queued requests, queue order/identity, held seats and exception cleanup. Bounded remedy: merge temporary queue entries into their original queues in native order before restoration; add decode-preemption and failure-path CPU checks. A new GPU pilot with actual preemption is still required before claiming pressure robustness. Do not treat the old no-preemption pilot as that validation.

CPU repair follow-up: separate_policy_v2now preserves temporary waiting/skipped entries using individual native prepends in queue order. Five policy CPU checks and three pilot contracts passed, including multiple decoder preemptions and an exception after preemption. Actual GPU-pressure validation remains pending. A further source-level limit is explicit: replay of a preempted request can combine original-prompt and generated-history queries; the current strict separate-policy mixed-step assertion may reject it. Recompute classification/phase semantics require a versioned definition and GPU proof before pressure robustness is claimed. No assertion was weakened and no failed request was promoted.

## Prospective prefill controller and profiler-range gates

The first repeated prefill controller now requires a fresh owned-GPU whole/chunk256smoke and strict dynamic/unchanged StaticCache reference before any3x10baseline acquisition. The previous functional pilot is reused only as adapter/budget evidence, never as the new GPU's qualification. The prospective admission estimate is now1380seconds plus60seconds margin; the earlier1122second deferred plan remains historical. Current CPU budget, barrier and prefix-coverage checks passed. No new GPU capture occurred.

Source review also found that the prepared external-profiler flag previously selected measured0for every budget, potentially overwriting the single `prefill_interval_output.json` while a collector stopped after its first range. This would limit attribution of physical traffic to a declared budget. The versioned adapter now selects exactly one explicit `--profile-condition` (defaultwhole), records it in the manifest and leaves other conditions outside capture. A real bounded profiler/replay/reader pilot remains required before physical-traffic claims. The completed functional pilots did not use external profiling and are unaffected.
