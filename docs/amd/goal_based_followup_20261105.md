# Goal-based AMD follow-up: compute headroom, memory delivery, serving and energy

Prepared 5 November 2026. Intended reader: the agent operating the AMD allocation and returning evidence through GitHub on `amd/mi300x-phase0`.

**Later execution/update notice:** the owner has ordered G1–G8 and reports it running; see `goal_followup_execution_plan_20261105.md` and the experiment log for actual releases/submissions. The preparation-only checkpoint below describes the original handoff and is not a new stop instruction. The owner now adds [OC0–OC7 overlap/service calibration](overlap_calibration_followup_20261105.md), to be planned/executed incrementally without cancelling, rewriting or duplicating G1–G8 or contaminating its frozen G8 test. Read the [shared framework](../compute_memory_overlap_calibration_framework_v0_20261105.md); preserve current campaign evidence and use separate new artifacts.

## 0. Readiness and authorization

**Pulling or reading this document is preparation only. Wait for the owner's subsequent instruction to execute this follow-up before starting GPU experiments or submitting new allocations.** This document describes what to achieve once that instruction arrives; it does not start a job itself. Read it now, reconcile current work, and be ready to propose or execute an adapted plan when ordered.

Once execution is ordered, proceed autonomously within the authorized allocation and this research scope. Audit available hardware, tools, time and storage; choose sensible cases; implement missing adapters; run bounded pilots; revise methods when evidence exposes a problem; and complete the feasible work. Do not require the owner to choose every kernel parameter or approve routine reversible implementation decisions. Preserve stronger existing requirements and any later explicit owner instructions.

This assignment adds G1–G8 below to the completed E0–E8 and v1.3–v1.6 campaigns. Do not rerun those campaigns wholesale or overwrite their plans, raw data, summaries or judgments. Reconcile newer commits and ongoing jobs first. Do not interrupt work owned by another process or agent. Work on `amd/mi300x-phase0`; do not edit, rebase or push `codex/h200-phase01`.

Clock/power/driver/firmware/cluster changes remain outside the existing authorization. A later explicit authorization may permit a controlled-clock sub-study; this document does not grant it. If comparable clocks cannot be obtained within scope, preserve observational conclusions and finish the remaining useful work rather than quietly weakening the clock gate.

## 1. Project goal and why this follow-up exists

**Additional owner goal, 5 November 2026:** support a mathematical optimizer that returns component counts and fixed-area allocations among arithmetic, storage/request/communication and MC/PHY, with HBM stacks/package resources budgeted separately, for both decode-only and prefill-plus-decode. Read [mathematical model V0](../area_allocation_mathematical_model_v0_20261105.md), especially Sections 10–11. The coordinating agent owns area-model/optimizer development; the AMD agent supplies measured demand, service curves, kernel/dispatch behavior, traffic, energy and uncertainty from G1–G8. Include parameter constraints and unavailable quantities in the results. Do not infer mm² from CU masks, invent process/PHY/SRAM areas, or treat stock-GPU suboptimality as already proven. This adds a modeling deliverable to the research, not a ninth automatic GPU campaign or a change to the execution checkpoint.

The governing architecture specification is [shoreline-aware research plan 2.0](../shoreline_aware_research_plan_2.0.md). We investigate whether reallocating excess GPU arithmetic area toward SRAM, memory controllers/PHYs, HBM interfaces and memory delivery can improve system-relevant J/token under die area, placeable shoreline, capacity, package, performance and power constraints. Lower arithmetic utilization alone does not prove architectural opportunity. No benefit is assumed.

Recent AMD work gives unusually useful motivation:

- MI300X job 499334: tuned Qwen32B graph B1/C4096 at 128/304 active CUs has about 0.993x full-device baseline latency. MI350X job 497082: tuned execution at 192/256 CUs has about 1.009x latency; at 128 CUs, about 1.168x.
- Much of the earlier restriction penalty came from `wvSplitK` workgroup placement and serialization. Changing the grid to a shader-engine-balanced count can recover performance. On full MI300X, the prediction of serialization was refuted; retain that negative result.
- MI350X job 499332: a memory competitor raises stock-decoder latency about 25–27%, while the arithmetic control raises it about 2%, with the registered isolation probe passing.
- Earlier guardrails show prefill and larger-batch regressions. One MI350X energy comparison rises from 11.46 to 12.47 J/token despite lower reported power.
- All the above AMD timing comparisons fail the pooled clock gate. Some traffic counters observe requests before MALL, and absolute FETCH_SIZE remains unvalidated on gfx950. The old smooth E7 prediction model failed blind checks.

These are observations for specific kernels/workloads, not fixed-clock noninferiority, physical CU removal, universal HBM saturation or an energy-efficient new chip. The purpose of G1–G8 is to resolve the most important uncertainties and feed a defensible architectural model.

Read these before planning:

1. [AMD handbook](handbook/README.md), especially the current conclusions and revisions table.
2. [MI300X shader-engine review](mi300x_se_review_20261105.md), [MI350X balanced-sizing review](mi350x_followup_review_20261103.md), and [MI350X isolation/E3 review](mi350x_leak4_review_20261104.md).
3. [Three-cohort synthesis](compute_bandwidth_followup_results_20261103.md) and [published cohort index](../../results_published/COHORTS.md).
4. [Original E0–E8 contract](compute_bandwidth_followup_20261026.md) for measurement gates, historical evidence and source rules.

These baseline statements are pinned to branch head `5b988063e8391005643351594dd8db0394c4d735`; newer evidence must be reconciled explicitly.

## 2. Autonomy and the first deliverable

The goals and evidence boundaries below are binding. Suggested CU levels, tools and workload menus are starting points, not commands guaranteed to work on the current allocation. You may choose different feasible levels, counter routes, scheduling strategies or kernel implementations when they answer the same question. Separate changes to the workload/algorithm from changes to its implementation.

Before new production acquisitions, publish a versioned local execution plan and commit/push it when connectivity permits. Include hardware/cohort inventory, existing-job reconciliation, mapping to G1–G8, candidate workload/kernel/mask matrix, pilot and production counts, randomization, acceptance rules, budget/reserve, failure limits and source pins. Record explicitly what requires new code and what can reuse a validated adapter.

New exploratory findings may motivate a revised plan. Preserve its predecessor, distinguish pre-registered from exploratory analyses, and freeze the revised confirmatory rule before collecting its fresh validation data. Do not select favorable cells after looking at outcomes or keep repeating a failed gate until it happens to pass. Failure, no effect and unavailable instrumentation are valid outcomes.

Default production comparisons retain at least three independent process acquisitions and ten timed trials per cell, with compilation and at least three full warmups outside timing. Use a predeclared stronger blocked multi-GPU design where affordable. Reuse historical acquisition counts only for explicitly historical evidence. Pilots may be smaller but cannot be promoted to production. Preserve pinned BF16 weights/KV, TP1, prompts, revisions, seed policies and exact-output comparisons from the applicable campaign; version any intentional change. Keep cohorts separate.

Existing entry points worth auditing rather than assuming complete:

- `scripts/amd/cubw/plan.py`, `cubw.py`, `analyze.py`, `manifest.py`, `test_analyze.py`;
- `scripts/amd/cubw/e6_shapes.py`, `e6_kernels.py`, `e6_counters.py`, `e7_model.py`;
- `scripts/amd/cubw/svc_bench_v4.hip` through `svc_bench_v6.hip` and the corresponding recorded binary hashes;
- `scripts/sitecustomize.py` and the `CUBW_CU_COUNT_OVERRIDE` route;
- existing AMD energy/prefill runners and source/counter/clock/isolation records.

The v1.6.1 dispatch-cell tag fix is not yet GPU-exercised in the historical record. Historical MI350X judgment/in-flight generator scripts were not all preserved. Reproduce or replace missing analyses with committed, tested CPU readers; do not silently claim complete raw reproducibility.

## 3. G1 — Separate CU headroom from clock changes

**Question:** Does the tuned reduced-CU decoder preserve performance under comparable operating conditions, and how much of the observed result could come from changing clocks?

**Goal:** Produce an honest, repeatable restriction curve that distinguishes command success, correctness, placement, clock qualification and statistical uncertainty.

**Adaptable approach:** Begin with the primary Qwen32B graph B1/C4096 family, full device and a few promising/boundary masks. Randomize/interleave paired conditions on the same physical GPU, keep warmup and thermal preparation consistent, and retain in-window graphics/memory clocks, power and temperature with cadence/coverage. Observe placement for the tuned grid as well as the mask. Controlled clocks are a stronger optional method only if separately authorized and available; do not assume they are.

**Evidence required:** ordinary event and wall scopes, output digests, mask/kernel identities, clock distributions, existing pooled-clock gate and block-level uncertainty. Report whether the original predeclared 5% noninferiority criterion is met, including every gate; a narrow latency CI alone does not qualify it. Predeclared clock-stratified/regression analyses may quantify sensitivity but must not replace the original acceptance rule or establish causality without support.

**Useful terminal outcome:** a qualified curve, or an observational curve with a quantified clock limitation and a clear reason controlled inference remains unavailable. Do not claim this goal failed merely because clocks cannot be controlled; complete the feasible diagnostics and preserve the limitation.

## 4. G2 — Test serving guardrails with the tuned kernel

**Question:** Does the apparent headroom survive the workload mix relevant to a serving GPU?

**Goal:** Separate software launch tuning from hardware restriction and identify which phases/workloads actually tolerate fewer CUs.

**Adaptable approach:** For representative cases compare (A) full mask/stock kernel, (B) full mask/reduced grid, and (C) reduced mask/the same reduced grid. Where practical add reduced mask/stock kernel and tune the full-device baseline fairly. Keep a development/held-out split if searching grid parameters. Verify the kernel route: an override that does not affect the selected kernel is not an intervention.

Start with the successful decode family, then choose a budgeted subset spanning batch 1 versus larger batch, short versus long KV context, a second model size, another dense model family, and prefill. The earlier Qwen7B B8, Qwen32B C16K and Mistral7B controls are useful anchors, not mandatory exhaustive grids. If the intended claim includes mixed serving, add an explicit request mix and report TTFT/inter-token latency/throughput; otherwise restrict the conclusion to isolated phases. A phase-only benchmark is not production p99 serving validation.

**Evidence required:** exact outputs, actual kernel/grid/resource metadata, per-phase event and wall latency, throughput, and source/clock qualification. Report regressions alongside successes. Define an overall workload-weighted decision only with weights and guardrails frozen before the final comparison. Reusing only the favorable batch-1 cell is insufficient.

## 5. G3 — Measure matched energy after tuning

**Question:** Can a tuned reduced-CU configuration reduce measured energy per useful output, or does longer execution cancel lower power?

**Goal:** Determine energy for the same configurations and work that support the latency conclusion.

**Adaptable approach:** Choose the promising G2 point, one boundary/regressing point and an appropriate full-device baseline. Run separate sufficiently long energy windows with matched prompts, outputs, starting KV and context progression. Include tuning at full device where it helps. Record raw energy accumulators and sampled power using the available validated routes, timestamps, cadence, wrap/units, board/socket scope and integration method. Keep prefill, decode and total-request scopes explicit.

**Evidence required:** J/token, total joules, useful token count, elapsed time and average power for a common scope, acquisition-level uncertainty and all sensor/output/source/clock/isolation gates. State whether a value is qualified or merely an observational estimate. Primary J/token remains NA if its required measurement checks fail. Raw board energy is primary; any idle-subtracted estimate is a separately labelled sensitivity, never a substitute silently chosen for a gain.

**Interpretation:** `E'/E = (P'/P)*(T'/T)` for equal useful output. A CU mask retains the physical silicon; neither lower power nor an energy gain measures leakage savings from physically removing arithmetic units.

## 6. G4 — Validate the traffic observation point

**Question:** Which accessible measurements represent actual HBM reads/writes, and which represent requests served before memory-side cache?

**Goal:** Establish a per-cohort metric capability/validation table before making physical-byte or saturation claims.

**Adaptable approach:** Inventory available counters/tools without assuming gfx942 and gfx950 semantics or permissions match. Consult the installed definitions and primary vendor documentation; pin their versions/URLs. Test known-size reads, writes and mixed access with working sets spanning cache-resident reuse and substantially larger streaming traffic. Declare cache preparation and allocation/initialization scope. Derive expected transactions for the actual instruction/line behavior rather than assuming useful bytes must exactly equal every physical counter.

**Evidence required:** counter definitions, units, collection/aggregation scope, channel/XCD coverage, raw profiler exports, positive controls, overhead and repeatability. Show whether cache hits change the candidate HBM counter in the expected direction and whether aggregation omits any device partitions. Record trace/counter failures and instrumentation effects. Avoid broad expensive all-counter captures.

**Useful terminal outcome:** validated physical read/write metrics, or an explicit unavailable status with tested alternatives. Pre-MALL FETCH/TCC metrics and analytical `D/T` must not be relabelled as measured HBM bandwidth. An unavailable physical counter does not block the separate timing goals.

## 7. G5 — Test memory-service sensitivity with the tuned decoder

**Question:** Once dispatch penalties are reduced, does memory service still dominate the decoder's sensitivity, and how does that sensitivity change with CU allocation?

**Goal:** Produce a tuned-decoder CU-by-shared-service response surface with arithmetic and inactive controls.

**Adaptable approach:** Reuse/adapt the validated owned-competitor machinery. Choose several decoder masks and graded memory duty/intensity, with a fixed disjoint competitor pool for comparisons intended to vary only decoder CUs. Document unused CUs and actual physical topology. Every paired pressure comparison keeps the decoder mask/grid fixed. Revalidate lifecycle, actual overlap and a sensitive isolation positive control for changed kernels/cohorts; old probe success is not automatic validation of a new setup.

**Evidence required:** ordinary decoder latency and outputs, achieved competitor service with overlap coverage, clocks/power, placement/isolation sensitivity and kernel/grid identities. Preserve per-cohort results. If G4 supplies decoder-delivered bandwidth, estimate a predeclared local response such as `-delta(log T)/delta(log B_decoder)`; otherwise report latency against treatment/achieved competitor activity without inventing decoder bandwidth. The competitor's achieved throughput is an outcome too and must not be subtracted from advertised peak.

**Interpretation boundary:** restoring uncontested service is a same-chip intervention, not additional HBM interfaces or an above-stock bandwidth demonstration. A full-device uncontended baseline remains useful context but is not the same fixed-partition comparison. Arithmetic competitors can alter shared power/fabric behavior; measure those effects rather than assuming perfect isolation.

## 8. G6 — Identify the dispatch/request-generation knee

**Question:** What resources must a smaller execution structure retain to consume the available bandwidth?

**Goal:** Distinguish arithmetic throughput, workgroup serialization, occupancy and outstanding-memory-request limits.

**Adaptable approach:** Sweep CU counts and actual workgroup counts independently around the observed knee. Use placement probes and isolated representative weight GEMM/attention controls. Native independent-load/grid/resource sweeps can characterize request capacity, but do not equate their rate with decoder service automatically. Where justified, inspect compiled ISA and register/LDS metadata for the dominant kernel before interpreting stall/occupancy counters. Retain fixed-kernel and tuned-kernel curves separately. Changing independent loads or LDS usage can change arithmetic and access behavior; record and validate those changes.

**Evidence required:** per-kernel latency, workgroup placement/serialization, resident resources, request counts and concurrency proxies with their observation points, service curves, actual clocks and traffic volume where measurable. Compare a few balanced and unbalanced layouts rather than assuming that a multiple-of-32 heuristic transfers universally. Missing hardware observables remain missing; do not manufacture physical stall-time partitions from unrelated percentages.

**Working model:** `B_eff <= min(eta*B_pins, B_MC, B_NoC, N_active*q/tau)`, where q is outstanding data per active CU and tau its service latency. Test whether scheduling/request effects explain the observations better than a FLOP-only or smooth bandwidth-only model. Peak HBM divided by CUs is a hypothesis-generating ratio, not measured decoder bandwidth or a universal 42 GB/s/CU limit.

## 9. G7 — Quantify avoidable traffic and the SRAM opportunity

**Question:** How much physical traffic could additional on-chip storage plausibly eliminate, and what reuse would be necessary?

**Goal:** Reconcile logical demand, interface requests and physical traffic for identical execution scopes, without counting ordinary autoregressive rereads as within-token redundancy.

**Adaptable approach:** Derive weights, KV, activation/output and auxiliary demand from pinned shapes/dtypes and the actual kernel routes. Include whole-model pure decode as the main scope and selected layers/kernels for attribution. Use G4-validated physical counters if available. Budgeted cache-pressure/reuse-distance or address studies may help attribution; label synthetic and instrumented tests explicitly and retain correctness/overhead checks. Actual addresses requested are not an HBM-miss history by themselves.

**Evidence required:** `D_physical = D_compulsory + D_reread + D_spill + D_other` as an accounting framework, with unknown terms left unknown and potential writebacks/padding/transactions accounted for. Report read and write domains separately before a combined ratio. Compare consistent prefill, tick/window and whole-model boundaries; account for any initially resident data and history. Aggregate excess is a lead, not proof that cache eviction caused it. Across-token weight rereads may require near-model-scale retention; single-layer repetition changes that working set.

If physical traffic is unavailable, publish the logical/request analysis and explicitly leave the HBM amplification ratio unavailable. If physical traffic is near compulsory demand, report the limited opportunity rather than searching indefinitely for a cache win. Additional SRAM may help tiles/spills/local reuse, but larger register/shared-memory capacity alone cannot change an unchanged kernel's algorithm.

## 10. G8 — Replicate and validate a predictive model

**Question:** Can the mechanism predict a new configuration instead of merely explaining measured points?

**Goal:** Deliver a model with an honest held-out test and a separately identified architectural interpretation.

**Adaptable approach:** If allocated, MI325X balanced-sizing replication is a high-value missing experiment. Otherwise use a genuinely unmeasured CU/grid combination or workload on the available cohort and record MI325X as unavailable. Choose development and held-out sets before collection; respect already exposed data. Fit timing/traffic/request parameters on development evidence only, commit/push the frozen model and predictions before held-out acquisition, and judge against predeclared error/coverage/ranking criteria. Do not pool hardware cohorts into one universal fit without explicit architecture terms and independent validation.

**Evidence required:** a committed generator, exact inputs and hashes, frozen parameters/predictions, fresh held-out data, all gates and residuals, and a pass/reject/unavailable verdict. Select a scientifically useful small holdout matrix and tolerances before results, with justification. Preserve the rejected E7 history and do not refit after exposure while calling the same points held out. Misspecification is a result; another iteration needs new validation data.

Hardware cannot establish an iso-area PHY/NoC implementation or manufacturable irregular shoreline. Produce measured constraints for that later model, not a claim that the AMD mask emulates reclaimed area or a faster new chip.

## 11. Execution order and completion

Suggested order after the owner's execution instruction:

1. Reconcile jobs, inventory, source pins and budget; publish the adapted execution plan.
2. Run G1 pilots plus a small G4 capability audit. Stop broken instrumentation routes early.
3. Run G2, then G3 on the selected configurations. This is the highest-value serving/energy test.
4. Run G5 and G6, reusing validated controls and bounded representative cases.
5. Complete G7 to the level G4 permits. Lack of HBM counters does not invalidate unrelated goals.
6. Freeze G8 predictions, collect its holdouts, and synthesize.

You may reorder independent tasks to match allocation availability. Do not consume the allocation on an exhaustive Cartesian product; prioritize decisive contrasts, retain a failure/analysis/archive reserve, and document scope changes before new confirmatory data. A goal may finish with a qualified result, observational limitation, refutation, or evidenced unavailability. Distinguish such terminal dispositions from work not attempted because time ran out. Do not mark a partially attempted scope scientifically complete.

## 12. Return results through GitHub

Use versioned new artifacts on `amd/mi300x-phase0`. Suggested names are conventions, not a required date or assumed implementation:

| Artifact | Required content |
|---|---|
| `docs/amd/goal_followup_execution_plan_<date>.md` | Adaptation, case matrix, hypotheses/criteria, counts, order, dependencies, budget and authorization scope |
| `docs/amd/goal_followup_results_<date>.md` | Per-goal findings, negative results, sources, qualifications, architecture implications and remaining uncertainty |
| `results_published/<cohort>_<date>_goal_followup_job<id>/` | Cohort/source manifest, actual launch plan, gates, compact trial-level timing/energy tables, output digests, counter exports/definitions, reader versions and hashes |
| `results_published/<...>/goal_status.json` | G1–G8 status, actual evidence paths, qualification, missing observables, failed attempts and next action |
| `results_published/<...>/model_inputs.json` | V0-model inputs/constraints from G1–G8: component/operation/interface identity, units, cohort/workload/kernel/clock scope, estimate/range, measured versus fitted status, source hashes and applicability; explicit unknowns for unmeasured area and physical HBM quantities |
| `scripts/amd/<versioned readers/adapters>` | Reproducible generators/readers, including model fitting/freezing/judgment; meaningful verification of changed readers |
| `docs/amd/experiment_log.md` and cohort index | Dated progress and navigable links without rewriting historical outcomes |

Record raw paths, file sizes and SHA256 manifests; preserve actual journals, telemetry, outputs and profiler data. Git should contain enough compact acquisition/trial evidence and code for the receiving agent to recompute key comparisons. A summary alone, an inaccessible cluster path or a hash without its file is insufficient for independent readback. Do not commit credentials or blindly add large raw binaries/LFS payloads. Use an existing owner-authorized archive destination and independently verify transfer; if none is available, report that delivery gap and preserve raw data on cluster storage while continuing the other goals.

Commit/push progress at sensible acquisition or analysis boundaries. Fetch before pushing; preserve concurrent commits, use a safe fast-forward/reconciliation workflow, and never force-push another agent's work. Verify the final remote head contains the delivery artifacts. Do not claim upload/readback or job completion from a queued process or exit status alone.

The final message to the owner should give branch, commit, files to read, jobs/cohorts used, which G1–G8 goals actually finished, which comparisons are qualified versus observational, and precisely what is still unavailable. State explicitly whether evidence supports less arithmetic, memory-service sensitivity, SRAM opportunity, and J/token improvement separately. Include the existing AMD University Program acknowledgement when presenting or publishing these measurements.

## 13. What the agent should do immediately after reading

Before an execution order, acknowledge this handoff, summarize the eight goals and important constraints, identify obvious allocation/tool/budget questions from available context, and wait. Do not treat that acknowledgement as authorization to start a GPU job. After the owner orders execution, carry out the feasible adapted plan autonomously under the contract above.
