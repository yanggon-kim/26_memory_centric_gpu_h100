# H200 agent assignment: shoreline-aware decode research

Instruction update: 2026-10-16. **New measurements below are pending; this document does not certify any H200 Phase 0/1 completion.** This is the active assignment for `codex/h200-phase01`. Preserve the H200 agent's existing local work, including results not yet pushed.

## 1. What changed and what governs this branch

Previously the assignment emphasized completing a broad Phase 0–2 model portfolio and testing whether compute area could become more on-chip memory at fixed external bandwidth. The new question is:

> Can some GPU compute resources be reallocated to SRAM, external-memory PHY/controller capacity and delivered HBM bandwidth, or both, to improve LLM decode under realistic area, placeable shoreline, package, power/thermal and cost constraints?

The user now wants experiments for this question, rather than automatic continuation of the old exhaustive portfolio. At the next safe case boundary, save the currently running valid measurement and prioritize the jobs below. Do not kill an unrelated user's job or discard a useful result because the direction changed. A negative result or justified pivot is an allowed research outcome.

Read [plan 2.0](shoreline_aware_research_plan_2.0.md) in full, especially Sections 2–7, 9–12 and 15. Its architecture scope supersedes [the preserved original brief](primary_specification.md). The old brief's Sections 0/14 remain methodological context; those are not the same sections in plan 2.0. Read [the common measurement contract](shoreline_hopper_measurements.md) for the detailed M1–M8/A–F methods, mathematical bounds, provenance and acceptance requirements. Read [recent RTX motivation](rtx5080_shoreline_motivation_20261016.md) as historical evidence only.

Precedence: this H200 assignment governs current priority, allocation, seeds and trace scope; the common contract supplies new measurement methodology; plan 2.0 governs research scope. Preserve stronger existing correctness, isolation, energy and trace-semantic gates. Original fixed-bandwidth and trace documents remain supporting sub-study procedures. Old instructions to complete every dense baseline before beginning characterization no longer apply.

Compare stock GPU, compute-to-SRAM, compute-to-bandwidth, joint designs and targeted request-delivery changes. More free die area does not guarantee more contiguous placeable PHY shoreline, memory-controller capacity, HBM stacks, interposer connectivity or cooling. Real H200 work characterizes motivation/service limits; physical architecture modifications, geometry/PPA and cycle-simulation validation are later stages.

## 2. First action: publish actual progress and audit the runnable paths

The branch snapshot inspected at `ad7d386` contains fresh H200 inventory/tool feasibility and new runners. It has no committed H200 `run.json`, session `STATUS.json` or intermediate-results table. This is **absence from the pushed snapshot**, not proof that your local Phase 1 work is unfinished. Inspect local files and live processes first. Publish compact status, selected/completed/failed/pending cases, sanitized logs and raw-artifact locations/hashes. Reuse valid finished measurements; do not restart Phase 0/1 wholesale or treat historical H100 rows as H200 results.

Recorded H200 facts: physical UUID GPU-a613c806-89b1-05d1-f71c-3d52ab46d246, 132 SMs, queried L2 62,914,560 bytes (60 MiB), driver615.71.09, explicit CUDA13.0.1, Python3.10.13, torch2.11.0/cu130, vLLM0.22.1. Actual NCU DRAM counters and NSYS micro timeline work. A native sm_90 micro produced640warp records and replayed4,121cycles/19,456active-thread instructions using a stock H100 parser-smoke config. None establishes optimized LLM correctness, H200 timing calibration, Hopper async semantics, or power accuracy. Remeasure identity if the allocation/runtime changed.

The recorded Slurm job14073758 has a launch cutoff15:03EDT and end15:33:58EDT on2026-10-16. Inspect current scheduler state and remaining budget; do not launch into an expired/preempted allocation or infer permission for a new job. Existing `h200_session.py` hardcodes paths/UUID/deadline. A different valid allocation needs an explicitly versioned configuration, not silent reuse. Inspect current quota, storage and transfer budgets rather than inheriting earlier H100's23GiB limit. No extra Slurm job, multi-GPU run or MIG/repartitioning is requested.

Audit `run_h200_phase01.py`, `profile_h200.py`, `h200_worker_metadata.py` and their actual dependencies before reuse. Three source-manifest entries in the pushed snapshot differ from committed scripts (`controlled_vllm_experiments.py`, `summarize_h200.py`, `h200_session.py`); bind subsequent captures to actual source hashes and preserve prior identities. Do not infer invalid data solely from a historical manifest mismatch. Existing `profile_h200.py` uses bounded eager kernel-replay samples; it cannot supply full-step HBM totals, ordinary cache state or graph traffic. Existing three-trial baselines are valid only for their declared scope, not automatically the new statistical contract.

Create a missing-capability/case ledger first: already qualified / needs raw requalification / requires adapter / planned / failed / unavailable. The new SM, independent-service, overlap and joint-model routes are not assumed implemented. Write meaningful positive/negative contracts before new producer/readers, run a bounded pilot, independently check provenance/correctness, then expand. Use actual script --help; this document does not invent executable flags.

## 3. Focused workloads and measurement rules

Keep BF16 weights/KV, TP1, immutable revisions, greedy EOS-ignored generation, actual prompt IDs and tokenizer identity. Reuse the H200 portfolio seed20261013 for its new matched controls. The older RTX-matched seed20261014 replication is an optional separate namespace, not a prerequisite for this H200 assignment; never pool seeds as one case.

Prioritize these contrasts using existing measurements wherever qualified:

| Priority case | Purpose |
|---|---|
| Qwen2.5-1.5B B1/C1024 | Small-model/low-batch cache, launch and latency contrast |
| Qwen2.5-1.5B B8/C4096 | Larger KV and execution-resource demand |
| First already smoke-qualified fitting Qwen7B, otherwise Mistral7B, B1/C4096 | Larger weight stream; if neither qualified, smoke Qwen7B first |
| Same chosen larger model, B8/C4096 if it fits | Compute-active counterexample; preserve OOM and use B4/C4096 if B8 fails |
| Already available14B, otherwise32B if no14B is available, B1/C4096 if fit | Model-scale extension after minimum contrasts; begin C128smoke and reduce toC1024 via a new case if necessary |

Use actual model geometry/GQA/sliding windows and loaded attention backend. Additional completed models are useful controls, but do not finish every old batch/context combination or start MoE before these new questions. Representative saved text follows synthetic smoke and has its own provenance. Active parameters alone do not determine MoE memory fit; routed access still needs a separate adapter.

New quantitative comparisons require three independent process acquisitions, ten accepted timed trials each and at least three complete warmups. Normally generate64tokens and time63pure-decode ticks after the verified one-token barrier. Preserve GPU-event and synchronized-wall scopes. Reuse existing three-trial captures as pilots/historical controls; collect fresh three-acquisition/ten-trial data only for selected comparisons that need it. Do not retroactively relabel replication counts.

Use source-frozen manifests, fresh output paths, a cooperating lock plus ongoing sampled process monitoring, actual clocks/thermal state and strict exact-token comparisons. Existing startup-only snapshots must be strengthened for new long runs or remain explicitly weaker evidence. No clocks/power/driver changes. New clock-matched joins require the common contract's <=5% pooled SM envelope and matching memory-clock regime across all acquisitions/replays; failed trials/replays cannot be dropped. Bound retries to two new attempts per rejected acquisition, selected by qualification rather than latency. Keep every attempt.

## 4. Ordered job list

### J0 — Reconcile, freeze and budget

Publish current session status and the capability/case ledger. Verify existing raw results/hashes and exact runtime/backend. At the next safe boundary stop automatic expansion of the old matrix. Do early bounded counter/trace feasibility checks; no broad campaign before its producer and reader pass. Preserve the existing allocation closeout buffer.

### J1 — Strong optimized baseline and actual decode timeline (M1)

Reuse or establish graph-optimized and eager baseline contrasts above, with exact output identity and scope. Capture separate bounded NSYS timelines with request/tick/operation attribution and actual launch counts. Report kernel busy-union shares separately from whole-step time and launch gaps. Graph/eager divergences remain rejected comparisons pending diagnosis. Do not attribute every symbol containing `cutlass` to a linear operation.

### J2 — Logical demand versus physical HBM traffic (M2)

For selected first/middle/last layers and complete decode, collect actual HBM read/write bytes and logical weight/historical-KV/input demand over selected ticks4/5/6. Save actual sequence/computed-token lengths, output head/tied storage and GQA/active-page semantics. Obtain graph and eager bytes independently when joining their timing. Preserve all intervening layers. Full-range counter methodology must be validated; limited kernel samples are not a whole-model total. Query H200-supported metrics; retain every replay publication/PID and original cache-state/perturbation limitations. TMA payload, L2 sectors and requested addresses are not physical HBM bytes.

Report D/U_tick_sum and D/U_window_unique separately. A near3window ratio may be ordinary autoregressive weight recurrence. Near-one aggregate reads can conceal rereads offset by cache hits and do not prove per-address exactly-once access. A signed D_read-N*W residual is not measured KV traffic. No logical-byte substitution for denied/unavailable counters.

### J3 — Actual arithmetic/issue/local-resource evidence (M3)

Bound real GEMV/GEMM, attention-forward/combine/cache-update launches; verify their operation via NVTX/runtime identity. Record normalized scalar/tensor activity, issue/eligible warps, scoreboard/barrier stalls, occupancy, register allocation, shared payload/allocated bytes/carveout, local sectors and grid/split/tile structure. Extract loaded sm_90/sm_90a SASS and dynamic work where supported. FA3 persistent grids, TMA/WGMMA and split metadata need actual validation; retain unknown productive/empty workers rather than infer them from grid size. Useful FLOPs and issued tensor work remain separate. Occupancy/stalls/raw cycle ratios are not decoder-wide compute utilization or exact DRAM-wait wall time. Do not transfer RTX's attention padding factor to H200.

### J4 — Independent service, latency and concurrency calibration (M4, A–D)

Measure stream/strided/gather/paged-like access, randomized dependent pointer chains, varying ILP/concurrency, scalar arithmetic, tensor GEMM and representative reductions. Use footprints below/near/above actual60MiBL2 plus a substantially larger stream. Run matched standalone and concurrent arithmetic/memory controls with GPU-event timing, physical read/write demand and actual instruction routes. Derive pattern-specific service curves and latency/MLP, not one universal copy-bandwidth ceiling. Use independent controls for P and beta; substituting D_decode/T_decode back into D/beta is a circular prediction. Test clock conditioning through a bounded versioned pilot if required, without modifying clocks or relaxing gates.

### J5 — SM allocation and delivered-service sweeps (M6/M7)

Query supported CUDA green-context allocation/rounding. Target fractions1.0/.75/.5/.3 of132SMs, rounded up to actual supported granularity; record returned counts and use them in case IDs. Preserve actual physical memory/L2/controller configuration. Pilot two contrasting shapes first, require all relevant library/default-stream/graph kernels to execute in the intended context, and prove graph activity during measured decode rather than only initialization. If graph restriction is unsupported after one implementation/review cycle and at most two pilot attempts per shape, report that limitation; retain best unrestricted graph baselines and qualified eager restriction separately. No MIG or uncontrolled background workload substitute.

Repeat ordinary latency, physical traffic and J4's independent stream/gather/pointer controls under the same supported allocations. Predetermine a5% latency-regression noninferiority envelope; the upper95% bound must pass, not a rounded point estimate. Fewer SMs remove arithmetic, LSU, issue and MLP together. They do not create SRAM or establish an iso-area compute-only change. A changed decode D/T is consumption, not proof of changed memory-controller capacity.

### J6 — Controlled memory-service sensitivity (extension)

After supported disjoint allocation and ordinary baselines, fix decoder resources near75% and reserve the allocator's returned disjoint remainder. Verify distinct contexts/resources and actual concurrent execution. Use the same partition and private-buffer allocation in all treatments: idle, owned streaming-memory competitor at50/100% requested duty, and register-arithmetic control at50/100%. Record actual achieved duty, clocks/power/physical traffic and HBM/NoC/L2 interference; never run against unrelated users. Stop the family as unsupported if placement/concurrency cannot be proven. No physical bandwidth increase is enacted. The result is sensitivity to shared-service competition, not a reversible experiment adding memory interfaces. Counter replay may change concurrency and needs separate validation. Do this after core J1–J5 and only while preserving closeout time; otherwise leave explicitly pending.

### J7 — Incremental retention and local SRAM opportunity (M5)

Use acquisition-matched normal/pressure comparisons, cold/steady-state controls and cross-token address reuse. For Qwen1.5B B1, use contexts1024/2048/4096 as initial probes and compute actual KV/cache ratios; add a longer fitting context only if all remain belowH200L2, retaining a versioned case. Keep B8/C4096 as the larger-KV contrast. A pressure buffer exceeding actualL2 is cache-history competition, not a guaranteed flush or larger SRAM. Record exact placement and direct cost: outside-layer pressure can still be inside full-decode ranges; do not subtract logical buffer bytes as measured physical corrections. Preserve writes/writeback attribution and timing including intervention cost.

If a supported retention policy or tiling/shared/RF software control exists, test it separately with unchanged computation, correctness and whole-model scope. Otherwise keep it analytical/unavailable. Per-layer payload fitting SRAM does not guarantee retention across full-model recurrence. Zero measured spills does not eliminate a concurrency opportunity. Keep SRAM capacity/policy, local concurrency and bandwidth effects separate.

### J8 — Complete-layer timing and bounded overlap (E/F)

Collect disjoint first/middle/last-layer CUDA events and whole-model ordinary/event-instrumented controls. Measure local perturbation where needed; a small whole-model change does not bound local event intrusion. Do not divide a first-to-last noncontiguous span by ticks. Inspect actual PM/replay pass layout; separate passes are marginal evidence, not simultaneous compute/memory overlap.

Use the common framework T=C+M-X+O with C=F/P, M=D/beta,0<=X<=min(C,M). O includes omitted work/address/on-chip service/launch/dependency/model error, not just host idle. Publish coupled feasible bounds and assumptions, not an exact physical wall-time partition. Empty intervals reject a model/rate choice. Fit common-family overlap coefficients only when sufficient qualified conditions exist, using reproducible70/30condition-level held-out validation and at least five distinct conditions/family; do not split replications between train/test or fit a free residual per row.

### J9 — Joint compute/SRAM/bandwidth sensitivity

Evaluate c={1,.75,.5,.25}, r={.7,1,1.2,1.4}, f={0,.05,.1,.2}, plus measurement-supported values:

```text
c = effective_compute_new / effective_compute_base
r = delivered_memory_service_new / delivered_memory_service_base
f = eliminated fraction of total physical memory service
C_new = C/c
M_new = M*(1-f)/r
memory-service improvement requires f > 1-r
```

Include compute-only, SRAM-only, bandwidth-only and joint cases; retain slowdowns. c is not automatically SM/die-area fraction, r is not nominal stack count, and f is not cache-fit fraction/hit rate. Same-residual/same-overlap and relaxed-overlap bounds remain conditional; propagate statistical uncertainty separately. State whether the real measurements support bandwidth, retention, request delivery, a different bottleneck, or no convincing opportunity.

### J10 — Qualified energy and negative regimes (M8)

Reuse valid board-energy captures or collect selected long windows with raw cumulative counter and sampled-power estimators, sensor cadence/updates and original agreement gates. Never rescale a sensor to match another. Sensor agreement is internal consistency, not external calibration or HBM/component energy. If invalid, report J/token NA and use bounded functional diagnostics before another sweep. Measure selected prefill and larger-batch cases as counterexamples to a decode-oriented resource trade. Do not let an unresolved energy sensor invalidate otherwise qualified timing/traffic evidence or become an energy-saving claim.

### J11 — Targeted trace/calibration and final evidence package

This revised assignment permits bounded native trace preparation beyond the old Phase0/1-only scope, **after** correct optimized baselines, a small semantic pilot and measured storage/runtime budgets. It does not require an immediate full-model instruction-trace campaign. Collect a bounded2–3complete eager decode-step bridge only if it can finish with postprocessing/verification/transfer before cutoff; otherwise preserve partial-layer scope and leave the full-model recurrence gate pending. Capture actual Hopper async/TMA/WGMMA/mbarrier semantics/descriptors/active masks and tensor/KV/request-page aliases. Never drop waits or multiply every warp record by32. Keep same-program ordinary eager calibration and strongest graph baseline separately. Stock H100 parser success is not H200 timing/power calibration. Resource/tiling changes may require new traces. No simulator architecture modification is requested during this rental.

## 5. Deliverables, implementation limits and closeout

Create a new versioned result namespace within the actual session; preserve original result rows. Emit:

- `shoreline_case_manifest.json`: J0–J11/M1–M8/A–F conditions, acquisition IDs, actual SM counts, implemented/missing adapters, planned/complete/failed/partial/unavailable disposition and raw hash references.
- `shoreline_measurements.jsonl` or CSV: ordinary timing, physical read/write bytes, logical/issued work, geometry/backend, metric numerator/denominator, all replay/source/clock/isolation coverage, CIs and explicit null reasons.
- `shoreline_service_calibration.json`, `shoreline_latency_bounds.json`, `shoreline_joint_sensitivity.json`: independent controls, held-out errors, coupled feasible/invalid rows and conditional c/r/f scenarios.
- `shoreline_opportunity_report.md` plus figures: traffic versus logical demand, latency/service versus actualSMallocation, kernel arithmetic/issue/resources, retention/pressure effects, bounded overlap, joint tradeoff and losing regimes. Separate statistical CIs from assumption envelopes.
- Current session STATUS/case manifest/log, source/runtime/model locks, trace/export manifest and sanitized raw-artifact locations with durable checksums/readback where available. Historical H100 root STATUS remains separate from H200 status.

The root readme/session summary must not advertise nonexistent files as completed evidence. If files exist only locally, publish their compact status or mark them unpublished. Preserve all failures, OOMs, permission denials, semantic gaps and incomplete resource proofs. Keep large raw databases/weights/traces outside Git; sanitize profiler/environment artifacts before publishing. Do not rely on node-local /tmp storage after allocation exit.

Priority under limited time: reconcile/publishJ0; finish the minimum contrasts and coreJ1–J5 plusJ8/J9analysis; collect selectedJ7/J10controls; runJ6/extendedmodels/largeJ11only after bounded pilots and while closeout remains possible. All declared unexecuted jobs remain pending in the ledger, not silently removed. Finish a clear continuation handoff at the cutoff. No new H200 work has been executed merely by preparing these instructions, and terminal disposition is not proof that every scientific gate passed.
