# Additive AMD assignment: activity, independent service and overlap calibration

Prepared 5 November 2026. Assignment ID OC-v0; goals OC0–OC7. Read [measurement framework V0](../compute_memory_overlap_calibration_framework_v0_20261105.md) and [area-allocation model V0](../area_allocation_mathematical_model_v0_20261105.md).

## 0. Authorization and coordination with the running campaign

The owner reports that the G1–G8 assignment has been read and is running, and now asks this agent to read and execute these additional measurements. This is an additive execution assignment, not another read-only onboarding checkpoint. Work autonomously within the owner's existing AMD allocation/submission permissions and restrictions. No new clock, power, firmware, driver or cluster-setting change is authorized.

**Preserve G1–G8. Do not cancel, restart, rewrite or duplicate its active jobs or frozen sources.** At the fetched snapshot `75ccd617`, the experiment log records smokes 502423/502424 and held productions 502425–502427. These are recorded submissions, not a verified live scheduler state. Reconcile actual jobs, allocation usage, newer commits and evidence on the AMD machine before acting. Retain the current campaign's remaining budget/time and output-delivery obligations; an older no-budget-cap instruction does not imply unlimited actual cluster quota.

Do CPU planning/capability review while jobs run. Run GPU pilots and follow-up acquisitions only on a verified free owner-allocated device or after the current campaign reaches a safe completion boundary. If another allocation is needed, use only the existing owner-authorized scheduler workflow and record its resource cost. Do not modify a queued/running release archive or plan to insert new cells. Publish a separate versioned plan, release and run roots for OC.

Do not change the frozen G8 model, predictions, held-out definitions or acceptance criteria. New calibration is a distinct model version with its own development/holdout ledger. Coordinate data exposure: do not inspect original withheld cases early or reuse exposed cases as fresh OC holdouts. Complete/judge the original G8 test on its original rules.

## 1. Goal and autonomy

We need a model that predicts how arithmetic, storage, request/control, communication and external interfaces affect decoder latency and J/token. The immediate gap is independent compute/memory service calibration and honest overlap/latency attribution, not one more utilization plot.

The shared framework defines the questions and evidence boundaries. Choose feasible tools, probes, shapes and masks for the actual AMD hardware. Adapt methods and build minimal missing adapters; do not assume a named counter, sampling facility or joint activity trace exists. Keep MI300X/MI325X/MI350X separate and prioritize the cohorts with usable instrumentation; inability to collect a metric is a legitimate outcome rather than a reason to change firmware or relabel another interface.

Reuse compatible G1–G8 measurements and readers before acquiring anything. Explain which inputs answer the OC goal, their source hashes and any scope mismatch. Publish only incremental work where prior evidence suffices. Existing candidates include `scripts/amd/goal/counter_audit.py`, `isa_meta.py`, `analyze_goal.py`, `serve_mix.py`, `cubw/e6_shapes.py`, `e6_counters.py` and native service controls. Static ISA census is not dynamic executed work, and the counter audit's candidate classification still needs architecture-specific interpretation.

## 2. OC0 — Reconcile, audit observability and publish an adapted plan

Deliver a job/source/data map of active and completed G1–G8 work, a per-cohort tool/metric capability table and a bounded OC execution plan. Separate ordinary timelines, counter/replay data, instruction samples, compatible time-domain activity and synthetic reference observables.

Record actual GPU UUIDs, software/container versions, counter definitions/units, firmware restrictions, coverage, estimated collection cost and potential serialization. Reuse the ongoing G4 audit. Do not probe a GPU being used by another job. Mark unsupported or disallowed routes unavailable early.

Before production, commit/push the adapted plan with hypotheses, resource budget, counts, cache/clock preparation, trial scopes, pilot gates, distortion criteria, model parameterization and planned fresh holdouts. Reuse existing source/clock/correctness gates; predeclare new estimator-specific criteria before data. Do not broaden a known broken profiler route into a large capture.

## 3. OC1 — Establish ordinary timeline and actual kernel demand

Select a small primary model/condition from completed G1/G2 and representative projection, attention and reduction/elementwise families. Include prefill and decode, then a budgeted second batch/context or model for transfer. Use actual full-model execution for baseline timing/history; isolated kernels are separate controls.

Collect ordinary event/wall timing, layer/whole-model scope, kernel/stream dependency timelines, launch/dispatch gaps and actual kernel route/grid/resources. Derive useful FLOPs and logical bytes from immutable geometry and include dynamic issued instruction/work estimates where available. Keep vector/matrix, address/control, reduction and padding categories distinct. If only static disassembly/resource metadata exist, state that dynamic work is unknown or bounded.

Return a reconciled kernel-demand table and measured gap/coverage evidence. Do not label every unaccounted interval compute-only or memory-only. Do not silently join graph timing with eager traffic/instructions without a validated bridge and explicit qualification.

## 4. OC2 — Measure independent arithmetic service

Build/reuse shape- and route-matched controls for the dominant arithmetic families. Use mostly on-chip operands where practical, with declared cache preparation, data size, instruction route, register/LDS use, occupancy and numerical validation. Where a reference changes the kernel shape or algorithm, report that mismatch and a reference-sensitivity range. A hot large GEMM is not a universal reference for GEMV or softmax.

Measure achievable operation-class rates across a few existing full/promising/boundary masks and actual grid choices. Reference code still performs local-memory/address work; do not call its time a physical compute-only duration. Never compute the reference rate from the decoder's own work divided by decoder time.

Return P_ref by operation/kernel family, metadata, trial-level rates, masks/clocks, applicability and limitations. If profiling issued work is unavailable, publish useful-rate references as such; do not invent exact issued counts.

## 5. OC3 — Measure independent memory/local-delivery service

Reuse G4 validation and G6 native controls to obtain pattern-matched streaming, gather/paged-like and cache-resident references at relevant footprints, grids, masks and independent-load levels. Add only missing cases. A synthetic paged pattern is not automatically vLLM's actual page layout. Record address/control work and resource usage.

Measure physical HBM bytes only if the audit establishes the observation point/units and adequate partition coverage. Retain pre-MALL, L2 and local-interface demands separately. Where accessible, derive L2/LDS/RF or port-related service constraints, with their own scope; unavailable RF bandwidth is not inferred from HBM bandwidth. Establish how issue/concurrency and residency change attainable service. Derive B_ref from independent controls, not decoder D/T.

Return reference curves and the Q/latency/issue observables or bounds that are actually supported. Missing HBM bytes or local-interface metrics remain NA; continue the independent arithmetic/timeline goals. Known profiler timeouts/firmware restrictions retain the existing bounded-attempt rules.

## 6. OC4 — Measure or bound activity overlap

Audit whether the available hardware/tool path supplies joint arithmetic/memory activity with a compatible time base and scope. Define memory activity explicitly: HBM transfer, cache service and outstanding requests are different.

If a candidate estimator exists, validate it with known sequential and deliberately pipelined/overlapping synthetic schedules, and a dependent-consumer control. Keep their work/output/traffic differences explicit. Show activity threshold, sample resolution, missing coverage, clock domain, overhead and the error of the estimator on controls. A PC sample or staging event is not automatically an interface busy interval.

Report the framework's four disjoint durations only if measurement semantics justify them. Otherwise report compatible marginal-duration bounds, or activity partition unavailable. Percent-of-peak metrics, per-CU averages and replayed separate observations cannot be forced into a complete device timeline. Profiled serialization cannot establish ordinary concurrency.

The goal is a defensible measured/bounded/unavailable verdict, not a mandatory point estimate. Whatever its outcome, keep activity intersection separate from the service-reference overlap calibrated in OC6 and from critical-path sensitivity.

## 7. OC5 — Constrain exposed dependencies and resource sensitivity

Reuse tuned-grid/CU contrasts and G5 memory competition with placement, overlap, arithmetic controls and sensitive isolation checks. Add only contrasts needed to distinguish arithmetic shortage, dispatch serialization, occupancy/local-resource limits and insufficient requests in flight. Inspect dominant-kernel instruction dependencies and any validated sampling/stall evidence before asserting a cause.

A CU cut is not an arithmetic-only intervention. Report response versus the actual mask/grid/resource change. Estimate sensitivity to independently measured compute or decoder memory service only where that service is identified; otherwise retain the observed treatment response. Do not subtract competitor throughput from peak to fabricate decoder bandwidth. Clock control remains unauthorized, and clock changes would affect more than arithmetic anyway.

Return a mechanism table connecting each causal claim to its control, observables, confounds and a falsifying outcome. Do not turn a stall fraction directly into reclaimable area or combine unrelated counters into a physical time partition.

## 8. OC6 — Fit, freeze and validate a service/overlap model

Start with the shared framework's family model T=O+C+M-eta*min(C,M), with C/M based on OC2/OC3 independent references. Another dependency/wave model is allowed if the initial structure is incompatible with observed scheduling; version and compare it rather than concealing the failure.

Use independently measured overhead where possible and a small shared parameterization. Test identifiability: unknown O and eta can trade off, so a tiny fit error alone does not prove a unique physical interpretation. Respect feasible coupled bounds and propagate reference/work uncertainty. Do not force an eta into [0,1] by inventing demand or arbitrary per-case overhead; record infeasibility and investigate.

Choose unexposed development/held-out cases, or freeze predictions before acquiring genuinely new confirmations. Existing G8 exposures remain recorded and cannot be reused as fresh holdouts. Commit/push frozen reference definitions, source hashes, parameters and predictions before the held-out job. Predeclare errors/criteria; <=10% ordinary median-time error per held-out case is a useful initial target if feasible. Any different criterion must be justified before data, not relaxed afterward. Report per-kernel and aggregate errors, bias and parameter ambiguity; kernel cancellation can hide a bad fit.

If physical HBM input is missing, do not claim an HBM-calibrated overlap model. Deliver a separately named interface/reference model or explicit unavailability, and preserve useful OC1/OC2 results. Rejected models remain results. No refitting on the same held-out data while calling it validation.

## 9. OC7 — Energy, serving transfer and model-input delivery

Reuse qualified G2/G3 matched D/PD scopes and add only justified missing transfer cases for the calibrated model. Preserve sensor, output, clock and isolation gates; primary J/token stays NA when required checks fail. Do not combine profiled latency with an unrelated energy window. A board measurement does not identify per-component leakage or physical removal savings.

Return component/operation/interface-specific service, demand, request, overhead and overlap constraints for the area optimizer. Include ranges/uncertainty and source applicability rather than forcing unknowns into constants. Arithmetic/SRAM/NoC/MC/PHY areas remain independent physical-model inputs; GPU experiments cannot derive them from utilization.

## 10. Progression, verification and delivery

Suggested order: OC0; reuse/align OC1; pilot OC2/OC3; capability-gated OC4; OC5 using current campaign evidence; freeze OC6; fresh held-out acquisition; OC7 synthesis. Reorder independent tasks to available safe allocations. Do not wait indefinitely for unavailable joint-activity instrumentation, and do not rerun the entire G1–G8 matrix.

New quantitative comparisons retain at least three independent process acquisitions, ten timed trials, and warmup >=3 unless a stronger existing contract applies. Preserve blocked multi-GPU pairing, fixed BF16/model/prompt/seed policy, exact outputs, clock gates and failures. Clearly label smaller pilots. Test changed readers against meaningful known inputs and preserve raw exports. Verify actual commands, expected nonempty artifacts and source identity before a completion claim.

Publish new artifacts on `amd/mi300x-phase0`:

| Artifact | Required contents |
|---|---|
| `docs/amd/overlap_calibration_execution_plan_<date>.md` | Adaptation, dependencies/current-job map, observables, methods, limits, predeclared criteria and allocation cost |
| `docs/amd/overlap_calibration_results_<date>.md` | OC0–OC7 findings, activity versus reference-overlap distinction, negative results and model implications |
| `results_published/<cohort>_<date>_overlap_calibration_job<id>/` | Immutable source/config/launch manifests, trial-level ordinary and reference data, outputs, clocks, raw-counter exports and coverage |
| `.../observability.json` | Every metric's definition, units, scope/time base, tool version, replay/concurrency behavior and measured/validated/unavailable status |
| `.../overlap_bounds.json` | Joint-activity values or coupled bounds, estimator validation/uncertainty, or explicit NA reasons |
| `.../service_references.json` | Independent arithmetic and memory/local-service rates, shapes, masks/resources, history, units and uncertainty |
| `.../calibration_frozen.json`, `.../validation.json` | Frozen model/holdout provenance and independent error/identifiability verdict |
| `.../model_inputs.json`, `.../oc_status.json` | V0 model constraints and per-goal terminal status with evidence/missing inputs/next action |
| Versioned scripts/readers and experiment-log links | Enough reproducible code and compact evidence for the coordinating agent to recompute main results |

Use unique goal/run names; preserve G1–G8 artifacts and its model_inputs. Keep new model IDs and provenance explicit. Raw paths/hashes without files do not demonstrate independent readback; use an existing authorized archive destination if available, otherwise record the delivery gap and preserve cluster data. Never embed credentials or blindly commit large profiler binaries/LFS data.

Commit/push at safe boundaries, fetch/reconcile concurrent commits, never force-push, and verify remote delivery. The final message should give branch/head, result files, actual cohorts/jobs, OC0–OC7 dispositions, which parameters are identifiable, which are only conditional, and what improved the fixed-area model. This assignment is complete only to its documented terminal scope; queued jobs or a plan are not completed measurements.
