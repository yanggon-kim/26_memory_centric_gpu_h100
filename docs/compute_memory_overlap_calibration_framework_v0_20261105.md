# Compute–memory activity and service calibration — framework V0

Date: 5 November 2026. This is an additive measurement contract for the fixed-area allocation model. It distinguishes directly observed activity, effective service-reference overlap, resource sensitivity, and unidentifiable quantities. It adds no measured GPU result.

## 1. Goal and scopes

The model needs attainable arithmetic service, memory-interface demand/service, request-generation and scheduling limits, energy, and uncertainty for both decode and prefill. Activity percentages alone do not supply these inputs. Measure representative projection/GEMV/GEMM, attention, reduction and elementwise families in actual whole-model execution; do not assume one percentage describes every layer, context or batch.

Keep three observational scopes separate: ordinary full-model/layer execution; instrumented kernels or ranges; and synthetic/isolated reference controls. A repeated isolated layer has a different working set/history from the actual decoder. Align token/tick, prefill barrier, model/revision, dtype, kernel route, grid/resources, masks, clocks, cache preparation and observation windows. Log first output versus subsequent decode outputs and distinguish event time, wall time, kernel union and sums of durations. Concurrent kernel sums are not the execution interval.

## 2. Activity partition: definition, not automatically an observable

Choose an interval [0,T] and a compatible observation domain. Define binary c(t) for selected arithmetic activity and m(t) for selected memory activity. State whether memory means physical HBM transfer, cache service, or an outstanding request; these are different observables. Then

\[
A_C=\int c(t)dt,\quad A_M=\int m(t)dt,\quad X=\int c(t)m(t)dt,
\]
\[
T_C=A_C-X,\quad T_M=A_M-X,\quad T_{CM}=X,\quad
T_O=T-A_C-A_M+X.
\]

The four disjoint durations sum to T. T_O means neither selected activity is observed, not necessarily pure software overhead. A memory-dependent warp may wait while the selected memory interface has no transfer, and other instructions may execute outside the chosen arithmetic category.

Any-arithmetic-active at device scope, arithmetic busy averaged across CUs, and per-wave stall samples are not interchangeable. Neither are device HBM service and per-CU LSU activity. Do not combine them in these equations unless a compatible domain and denominator have been demonstrated. Percent-of-peak throughput is not binary busy time. Low tensor activity does not directly imply equally large reclaimable compute area.

If actual joint activity with a common time base is available, report its measurement definition, resolution, coverage, activity threshold, attribution, overhead and uncertainty. Timestamped instruction or warp samples are not automatically a complete device activity trace. On-chip staging/transfer events need not mark physical HBM transfer start/end. Validate any proposed estimator with controls whose sequential versus overlapping schedules are known.

## 3. Bounds when joint activity cannot be measured

If A_C and A_M are actual compatible active durations in the same interval but X is unknown:

\[
\boxed{\max(0,A_C+A_M-T)\le X\le\min(A_C,A_M).}
\]

For an illustrative T=1 ms, A_C=0.4 ms and A_M=0.8 ms, X lies in [0.2,0.4] ms. Consequently T_C lies in [0,0.2], T_M in [0.4,0.6], and T_O in [0,0.2] ms; the endpoints are coupled through the same X and must not be combined independently. These are not measured decoder values.

If compatible A_C/A_M are unavailable, even these numeric bounds are unavailable. Return that status rather than converting peak-utilization percentages or replayed counters to durations. Distinguish measured joint activity, bounded activity, model-derived service overlap and unavailable activity partition in every result table.

## 4. Independent service references

For kernel family k, measure actual issued work by operation class where possible, physical memory bytes at a validated observation point, and independent reference rates under matched operating/resource conditions:

\[
C_k=\sum_i F_{issued,i,k}/P_{ref,i,k},\qquad
M_k=D_{HBM,k}/B_{ref,k}.
\]

Declare the arithmetic-class composition: independent engines may overlap, so the sum can be a conservative reference rather than a literal service duration. If issued counts are unavailable, retain useful work separately and bound padding/control/other issued work; do not silently substitute useful work as exact issued demand.

Compute references should approximate the actual instruction route, shapes, register/LDS footprint and occupancy, with mostly on-chip operands where practical. Such controls still access registers/SRAM. A hot square GEMM is not automatically a valid service rate for skinny GEMV or softmax. Keep compute-shape references distinct from reference execution that happens to avoid HBM.

Memory references should span the relevant contiguous, paged/gather and reuse patterns, CU/grid counts, independent loads and footprint. They include address/control work and are not literally compute-free. Derive B_ref from those controls, not from decoder D/T; the latter makes a prediction circular. Derive P_ref independently rather than using the decoder's own F/T.

Maintain D_L1, D_L2, D_NoC and D_HBM separately, with read/write/request units and aggregation coverage. Calibrate local-interface service where observable. Do not sum the same transfer's pipelined HBM/MC/NoC costs as independent serial costs without a justified dependency model. Missing physical HBM bytes imply missing M_HBM, not permission to substitute pre-MALL bytes.

## 5. Effective overlap and latency calibration

A starting family-level model is

\[
T_k=O_k+C_k+M_k-\eta_f\min(C_k,M_k),\quad 0\le\eta_f\le1.
\]

O includes independently established non-scalable costs and a narrowly justified residual. Eta describes service-reference overlap for family f; it is not the directly observed activity intersection X. Do not label C-M or fitted residuals physical compute-only or memory-only durations. An improved dependency/wave model may replace this equation; preserve versions and compare predictive performance.

For positive C/M, define Z=T-O. This formulation is feasible only if

\[
\max(C,M)\le Z\le C+M,\qquad
\eta=(C+M+O-T)/\min(C,M).
\]

Unknown O and eta can compensate each other. Do not fit both freely per observation. Measure launch/dispatch gaps where possible, share a small justified parameter set across related cases, and report feasible parameter regions when no unique estimate is identifiable. A longer kernel time caused by low issue/occupancy must not be hidden by an unconstrained overhead. Propagate coupled input/reference uncertainty and report sensitivity to reference shape/route separately from sampling uncertainty.

Fit on development cases only; freeze parameters, reference definitions, source hashes, held-out cases and acceptance rules before new validation data. A useful initial target is <=10% ordinary median-time error on every declared held-out case, with separate tail/ranking/traffic checks when claimed. The agent may predeclare a different justified criterion before data if the design requires it, but must not relax an existing frozen test or excuse systematic bias with broad intervals. Parameter identifiability and held-out accuracy are different tests. Preserve rejection and NA outcomes.

The H200 decoder-calibration-v4 contract previously proposed related independent references; it was blocked before acquisition and does not supply validated coefficients. Its exact scope/clock/correctness requirements remain historical evidence, not a completed calibration to transfer to AMD.

## 6. Resource sensitivities and request capacity

To allocate area, estimate what changes execution time, not merely what is active:

\[
S_C=-\partial\log T/\partial\log P_{compute},\qquad
S_B=-\partial\log T/\partial\log B_{delivered}.
\]

Estimate these derivatives only when the perturbed service is independently characterized. A CU mask changes arithmetic, issuers, queues and resident state together; report its response as such. Memory competition changes shared service, cache/fabric and power, not necessarily only HBM. Its achieved throughput cannot be subtracted from peak to fabricate decoder bandwidth. Clock changes affect multiple paths and remain unauthorized unless separately granted.

Use matched fixed/tuned grids, placement, registers/LDS occupancy, independent-load/native controls and graded memory treatments to constrain

\[
B_{eff}\le\min(B_{HBM},B_{MC/PHY},B_{L2},B_{NoC},I\bar b,Q/\ell).
\]

More arithmetic may have almost zero benefit while more issuers or queues help. An instruction stall category or a busy percentage alone does not establish this causal distinction; combine instruction dependencies, controls and latency response.

## 7. Ordinary execution, replay and profiler distortion

Keep ordinary timing, timeline tracing, counter captures and synthetic controls in separate identified executions. Record clocks, outputs and scope for each. Check replay output, kernel coverage, cache preparation, route changes, process isolation and timing perturbation. Do not require profiler time to equal ordinary time or substitute it silently.

For competitors/concurrent kernels, establish ordinary concurrency first. Counter profilers can replay/serialize dispatches or change cache history. Counter results from such execution cannot establish ordinary overlap. Prefer compatible single-pass measurements when supported; otherwise return bounded/conditional/unavailable overlap and continue useful calibration. Do not run broad full-counter sweeps or apply NVIDIA event/counter assumptions to AMD.

Primary documentation starting points, to be pinned to installed versions during the capability audit:

- NVIDIA Nsight Compute Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/ — metric meaning, replay, cache and collection overhead.
- AMD ROCm Compute Profiler overview: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/quick-reference/rocprofiler-compute-at-a-glance.html — instruction/occupancy/memory views, replay and dispatch-serialization limitations.

Their feature lists do not prove availability or valid semantics on the allocated GPU. Retain architecture-specific probe results and definitions.

## 8. Energy and architecture boundary

Join matched long-window energy with the ordinary workload and active kernel policy, not profiled latency from a different execution. Preserve raw sensor estimates and gates. Report D and PD separately, including prompt/output ratios and workload mix. Component static/dynamic energy remains uncertain even with accepted board energy.

No timing or activity experiment alone gives mm² per engine, RF/LDS port, MC or PHY. Physical area and package parameters require independent implementation/literature/synthesis evidence with uncertainty. The allocation model must not turn stall percentages into physical area fractions. Deliver calibrated service/traffic/request constraints and unknowns, then validate the optimizer's predictions separately.
