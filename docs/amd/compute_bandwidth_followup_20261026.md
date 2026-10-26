# MI300 follow-up: less arithmetic, preserved request capacity, more delivered memory service

**Owner-requested assignment, 26 October 2026.** Read this file FIRST for the new MI300 experiments. Then read [AMD port plan](../AMD_PORT_PLAN.md), [measurement contract](../shoreline_hopper_measurements.md), and [research plan 2.0](../shoreline_aware_research_plan_2.0.md). This is an execution handoff, not a claim that the experiments below have run.

## 0. Objective, scope and precedence

Test whether dense, small-batch LLM decode can retain its performance with fewer active compute units, and whether increased available memory service can offset the restriction penalty. Separate arithmetic capacity from request generation, L2/MALL/fabric delivery, launch overhead, clocks and power.

The research goal remains joint compute/SRAM/controller/PHY/delivered-HBM-bandwidth design under logic area, placeable shoreline, reticle, package, capacity, energy and cost constraints. No architecture is assumed beneficial. CU masks are software interventions; they do not remove physical area or add HBM interfaces.

This assignment supersedes the old automatic exhaustive-portfolio ordering for **new follow-up work**. It does not cancel active jobs, overwrite historical results, loosen correctness/clock/energy gates, or authorize changing clocks, power caps, firmware, drivers, MIG/partition modes or cluster settings. Same-GPU owned competition described in E3 is within this assignment; it must use proven disjoint CU masks and explicit owned-PID isolation exceptions. TP/MI325X results remain separately labelled extensions, not substitutes for the primary single-MI300 study.

Start useful work autonomously within the existing allocation and budget. Do not start a new Slurm allocation, spend an unapproved allocation budget, interrupt another agent, or stop a running campaign merely because this document exists. If no permitted GPU allocation is available, complete the CPU audit/adapter/dry-run work and publish a concrete blocked launch disposition.

### 0.1 Reconcile current progress before launching

The base inspected for this handoff is `d01317691c6e8d90266e4894fbab2478a5741552` on `amd/mi300x-phase0`. Its October 26 [experiment log](experiment_log.md) records:

- MI325X job 470176 attempt 3 completed phase-1 lane commands, then timed out while a phase-level wait also waited for the node monitor.
- `scripts/amd/mi325x_full.sbatch` now excludes the monitor from lane waits and accepts `PHASES` selection.
- Resume job 481072 was submitted for phases 2–4. Its current scheduler state, saved outputs and scientific qualification are **not established by that historical log**.

Inspect the actual scheduler/allocation and local persistent results. Record completed, interrupted, failed and pending work. Reuse genuinely matching qualified evidence; do not rerun phase 1 or submit a duplicate job by default. Do not describe `rc=0`, an old PID, a dry-run or a published summary as proof of a currently running or qualified experiment.

MI300X and MI325X remain separate cohorts. If only MI325X is allocated, the same protocol may be implemented there with an explicitly named MI325X manifest, fresh baselines and no pooling with MI300X. Mark the requested MI300-specific cells unavailable until measured on MI300.

## 1. Motivation from our existing measurements

The audited numerical snapshot is commit `5fc029aae2ffefa00893648ee9891cbccd447416`. Newer data must be reconciled against actual source/counter/clock/correctness evidence.

| MI300X quantity | 304 CUs | 232 CUs | 152 CUs | 96 CUs |
|---|---:|---:|---:|---:|
| Native BF16 MFMA, TFLOP/s | 1071.12 | 791.62 | 566.43 | 380.97 |
| 4 GiB useful-read stream, TB/s | 4.5485 | 4.3164 | 4.0049 | 3.9744 |
| 32 MiB repeated useful read, TB/s | 24.242 | 21.074 | 15.281 | 11.127 |
| Qwen32B B1/C4096 graph, ms/step | 17.4868 | 18.0823 | 18.9548 | 19.0918 |
| Qwen7B B8/C4096 graph, ms/step | 6.6539 | 7.9305 | 9.4793 | 10.8926 |

At 96 CUs, streaming retains 87.38% while MFMA retains 35.57%; 32B B1 slows 9.18%, but 7B B8 slows 63.70%. These are power-managed observations, not isolated fixed-clock effects. Native service records are directly published; most LLM raw runs remain on cluster storage.

Other findings and corrections that must be carried forward:

1. Memory competition with a fixed 232-CU decoder raises latency by approximately 14.75%/23.57%/25.94%/24.70% for bridge B1, Qwen7B B1, Qwen7B B8 and Qwen32B B1. Arithmetic competitors also affect latency. Existing competitor summaries do not supply qualified ratio CIs and average overlapping windows without duration weighting.
2. Only 2/16 production M1 conditions pass the declared clock envelope. The published 2/17 includes a one-acquisition smoke; actual M1 coverage is 16 production triplets plus that smoke, not 17 triplets. M6 has 19/56 individually clock-passing cells, without a demonstrated pooled cross-condition gate. `qualified_acquisitions` means completed acquisitions in the old reader, not clock-qualified ones.
3. `FETCH_SIZE`/`WRITE_SIZE` observe traffic before Infinity Cache/MALL. Runtime Triton kernels were excluded. They do not establish complete physical HBM traffic or per-address exactly-once reads.
4. Old latency bounds combine graph T with eager traffic and useful FLOPs/reference rates. Their residual O is not a measured nonmemory time partition. Mistral graph classification silently retains only one of three requested ticks after two alignment mismatches.
5. CU masks retain silicon and clocks can rise. Prefill slows about 2.08–2.25x at 96 CUs. Energy savings from physically removing arithmetic remain untested.
6. The MI325X compute-matched attempt at 264 CUs did not isolate bandwidth: MFMA was approximately 997 versus MI300X's 1071 TFLOP/s and streaming fell below full MI300X. TP changes communication, aggregate cache/capacity and timing scope.

Primary sources are listed in Section 12. Preserve the old report and all failed acquisitions; put corrections in a new versioned report, not by rewriting historical measurements.

## 2. Hypotheses and falsification criteria

| Hypothesis | Supporting outcome | Falsifying or limiting outcome |
|---|---|---|
| H1: Arithmetic provision exceeds small-batch need. | Selected reduced-CU graph cases satisfy a predeclared noninferiority gate. | Latency regresses materially at moderate restriction; padding, kernels, clocks or request limits explain apparent headroom. |
| H2: Reduced-CU decode remains memory-service-sensitive. | At fixed decoder masks, ordinary latency improves as independently characterized shared service is restored, with arithmetic controls and valid gates. | No reproducible response, or response explained by clocks/cache/host effects rather than the claimed service variable. |
| H3: Request capacity is an enabling constraint. | Service-vs-CU curves and independent-load controls locate a request/MLP knee and predict held-out decoder sensitivity. | Removing CUs collapses delivery that extra nominal HBM cannot use, or the controls do not transfer. |
| H4: More available service offsets a measured CU penalty. | A correctly labelled lower-CU/higher-service cell outperforms a higher-CU/lower-service cell with matched work and qualified comparison. | Added service cannot overcome restriction cost in the tested range. This is a result, not a reason to change gates. |

Separate H4's **same-chip shared-service demonstration** from a future design exceeding the unmodified stock GPU. Removing a competitor restores access within the stock chip; it neither adds interfaces nor proves a win over the uncontended 304-CU baseline.

## 3. Run contract and bounded progression

### 3.1 Primary workload and statistics

- TP1, dense BF16 weights/KV, immutable model/tokenizer revisions, seed 20261013, fixed prompts, EOS-ignored greedy generation and the verified all-request one-token prefill barrier.
- Primary: Qwen2.5-32B B1/C4096, then Qwen2.5-14B B1/C4096. Graph serving is primary. Preserve exact graph/eager/backend/kernel identities.
- At least three full warmups after compilation, ten timed trials per independent process acquisition, and at least three independent acquisitions per quantitative cell. Use five acquisitions for selected decisive comparisons if the declared budget permits; freeze this before inspecting their outcome.
- Default 64 generated tokens, hence 63 pure-decode ticks. Record step versus aggregate-token denominators. Energy uses separately declared longer windows, with the same work/context/window matched across treatments.
- Randomize/interleave treatment order in acquisition blocks using a recorded seed/order. Pair by block where justified, bootstrap independent blocks, and keep ten serial trials nested inside each acquisition. Never treat all trials as independent processes.
- Bounded pilots may use one acquisition/three timed trials, clearly labelled pilots and never promoted to production comparisons.
- Exact outputs must be unchanged across the relevant baseline/restricted/competition replay comparison. Log all mismatches; do not silently switch precision, quantize, substitute another attention route or relax the reference gate.

### 3.2 Gates and failures

Maintain separate fields for command completion, numerical correctness, mask/placement, clock comparability, isolation, source identity, sample counts, counter coverage and final scientific disposition. Use `qualified`, `observational_clock_confounded`, `unavailable`, `failed` or `not_run` with explicit reasons; exit code alone is insufficient.

Retain the declared pooled <=5% clock envelope on actual comparison windows, including memory clocks, under the common contract. Record sample coverage/cadence. Endpoint checks and post-hoc interior-only filtering do not repair a failed primary gate. Individual cell passes do not establish a pooled comparison pass.

Do not change clocks/power settings in this assignment. If the gate fails, record the fixed-clock conclusion as unavailable. A separately predeclared small observational cohort may characterize automatic power management; it must retain failed clock flags, not claim controlled fixed-clock causality, and not be expanded into repeated favorable resampling. Complete independent CPU/service capability work rather than pretending the whole assignment succeeded.

Default operational retries: at most two attempts per fresh pilot/cell for a diagnosed execution/tool failure, all attempts retained. No automatic recapture merely to obtain favorable clocks or performance. A versioned repair must state the diagnosed cause and changed source/protocol. Preserve journals and partial outputs.

### 3.3 Isolation, storage and time

Use exactly the explicitly allocated GPU UUID. Confirm physical index versus HIP-visible index, GPU SKU, CU count, topology, clocks, memory, runtime/container digest, tools/counter permissions and free budgets. Current allocation state takes precedence over old IDs/paths. Reuse the existing cooperating lock and isolation checks; they do not prove absence of every external interferer.

No other experiment may use the target GPU during a case. E3 allows only the recorded owned competitor process and its proven disjoint mask. Capture PID/argv/context/stream identities and actual overlap. Stop/join owned workers with bounded cleanup and verify they are gone; never broadly kill matching user processes. Monitors are not lane completion prerequisites. Parent-shell command completion and child scientific qualification remain separate.

Use a fresh campaign root such as `results/amd_<date>_<sku>_cu_bw_v1/`, backed by persistent storage with a recorded location. Fresh acquisition/attempt paths are mandatory; do not overwrite or move historical runs to make a reader accept them. Reserve at least 30 minutes before allocation end for consolidation, hashes, transfer checks and closeout. Estimate each bulk cost from pilots and publish a budget table before expansion.

## 4. E0 — Reconciliation, capability audit and adapter preparation (first)

1. Read actual current job state/results, including the MI325X continuation above. Reconcile old `STATUS_AMD.json`, historical summaries and fresh local outputs; record mismatches and missing raw files.
2. Inventory the current allocation and produce environment/source/config/model/tool hashes and a read-only counter capability list. No root/system changes.
3. Audit existing scripts in Section 11. They are reusable components, not implementations of this entire follow-up.
4. Implement only the missing versioned adapters/readers: arbitrary supported CU levels, fixed reserved competitor masks, measured overlap, duration-weighted windows, independent trial/clock gates and dry-run manifest generation.
5. Before editing a parser/adapter, create meaningful CPU validation fixtures for failure/status separation, sample nesting, missing clocks, coverage mismatch and overlapping windows; check failure then minimal repair. Validate source-specific GPU paths only with bounded pilots.
6. Publish an immutable prelaunch manifest covering the cases, treatments, order, source pins, timeouts, retry policy, repetitions, gate definitions, output paths and budget. Dry run must enumerate commands without touching the GPU or starting a scheduler job.

**Output:** `environment.json`, `source_manifest.json`, `case_manifest.json`, `launch_plan.json`, adapter validation results and a reconciliation note. Mark any inaccessible raw artifact as missing rather than claiming its SHA was rechecked.

## 5. E1 — Correctness, mask, clock and instrumentation pilots

Use the bridge graph smoke, then the Qwen32B graph primary case, before bulk.

- Prove graph and library kernels obey each intended mask. The previous study found physical-index-sensitive `HSA_CU_MASK`, unsupported comma lists and unsafe empty-XCD masks. Revalidate these observations on this runtime. CU count multiples of eight/contiguous ranges are only historically tested conventions, not a substitute for placement proof.
- Use the available supported mask/placement evidence and independent arithmetic/load controls. A set environment string or reduced TFLOP/s alone is not proof that every graph/library role is restricted correctly. If actual placement cannot be verified, make that claim unavailable.
- Record compile/warmup/measurement windows. Do not introduce treatment-specific warmup loads or silently exclude clock ramp samples. Any common conditioning protocol creates a versioned program with its own baseline.
- Verify unprofiled outputs versus profiled replay outputs. Compare ordinary timing with a lightweight timeline pilot to assess instrumentation distortion before using that timeline for phase durations.
- Pilot the E3 process lifecycle, no-overlap idle control and live overlap of the disjoint active competitor. Do not release the grid until cleanup works.

If fixed-clock qualification is unavailable, publish that limitation and run only the predeclared minimal observational subset plus independent capability controls within budget. Do not automatically release the complete expensive matrix. E2/E3 can be scientifically unavailable even when commands complete.

## 6. E2 — Fine CU noninferiority boundary (primary latency experiment)

Start with Qwen32B B1/C4096. Requested nominal CU levels on the historical 304-CU SKU are **304, 232, 200, 152, 96**, subject to actual supported/balanced masks and E1. Use a fresh own-cohort 304-CU baseline.

If a requested level such as 200 cannot be realized with the required mask granularity/placement, record it as unsupported and preregister a demonstrated supported replacement (e.g. 192 or 208 only after validation). Never silently round a requested allocation or report the requested count as the observed count.

For each cell collect ordinary HIP-event elapsed and synchronized wall latency, exact outputs, context/output lengths, launch/backend identities and telemetry. Report acquisition-level estimates plus confidence intervals. Define

\[
R(n)=T(n)/T(304),\qquad
\operatorname{upperCI}_{95\%}[R(n)]\leq1.05.
\]

The selected smallest n must satisfy every source/output/isolation/mask/clock/statistical gate, not just have a favorable point estimate. State that the criterion is <=5% **latency** regression, not exact equality or <=5% throughput loss.

If a boundary lies between tested levels and budget permits, preregister a supported refinement before collecting it; retain original levels. Then repeat the five-level experiment for Qwen14B B1/C4096. Minimum full E2 is 2 models x 5 levels x 3 acquisitions = **30 production acquisitions**, not including pilots/refinement.

Do not select a favorable clock subset or pool acquisitions across allocations to fill a cell. If qualification fails, report the raw curve as observational and do not publish a controlled noninferiority conclusion.

## 7. E3 — Decoder CU allocation x shared-service treatment

### 7.1 Fixed partition design

For the historically tested 304-CU mapping, reserve **CUs 232–303** for the owned competitor for the entire matrix. The decoder uses supported balanced masks within **0–231**, at nominal levels **232, 200, 152, 96**. Unused decoder-side CUs remain unused; do not increase competitor size as the decoder shrinks. Verify actual runtime/XCD placement before accepting these masks.

The uncontended 304-CU baseline from E2 is a **separate reference** and must not be described as one of these disjoint-competitor cells. At n=232, compare a genuinely absent competitor with an allocated-but-inactive process control where useful to identify process overhead.

### 7.2 Treatment schedule and bounded release

At each decoder mask, the six primary treatments are **solo (competitor absent)**, nominal memory25/50/75/100 and arithmetic100. An allocated-but-inactive process is a separately named validation control, not a seventh treatment silently included in the grid counts. Below, “idle” denotes the primary solo reference unless that extra control is explicitly named. Nominal duty labels are settings, not delivered bandwidth fractions. Record achieved activity.

Pilot Qwen32B at n=232 and n=152 with idle, memory50, memory100 and arithmetic100, one pilot acquisition each. After E1/output/lifecycle/clock assessment, freeze the production grid and cost. The full primary grid is 2 models x 4 decoder masks x 6 treatments x 3 acquisitions = **144 production acquisitions**. It is conditional on budget and the pilot gates, not an instruction to launch blindly.

If clocks remain confounded, the default minimal observational subset is Qwen32B only, n=232/152, idle/memory50/memory100/arithmetic100, three fresh acquisitions per cell (24 production acquisitions). Predeclare it before timing inspection; retain original gates as failed and do not promote it to the full grid. A shorter-budget manifest can instead defer cells with explicit reasons.

### 7.3 Required records and analysis

- Ordinary decode timing and exact outputs; owned competitor per-window start/end, completed useful bytes/operations and native events; concurrent trace proofs in separate acquisitions.
- Use monotonic time or another demonstrated common time basis. Compute intersections of actual trial/competitor windows, account for covered duration, and deduplicate overlapping records. If uniform throughput within a coarse record is assumed, label the overlap attribution estimate. Do not invent subwindow precision.
- Treat competitor achieved throughput as a response to the decoder as well as an intervention. It is not an exogenous measured subtraction from decoder HBM bandwidth. Nominal treatment and measured shared-service response remain distinct.
- Compare latency within a fixed n, and directly contrast memory versus arithmetic controls using matched acquisition blocks and qualified CIs. Arithmetic controls do not eliminate every cache/fabric/power confound.

Primary product is **T(n, z)**, where z is the declared treatment. Call it T(n, B_available) only if B_available is independently identified with a valid byte domain/control and the limitations are stated. Do not infer bandwidth-scalable critical time from a slowdown alone.

## 8. E4 — Request-generation/service curves at the same CU allocations

At E2 levels (or the preregistered subset), measure independent controls using actual supported HIP code, not a guessed mapping from CUDA:

1. Large-footprint sequential reads (4 GiB or larger verified above caches): weight-streaming service.
2. Strided/gather/paged accesses: access efficiency and address-generation limits.
3. Multiple independent loads/chains and a grid-size sweep: outstanding concurrency/MLP.
4. Dependent chains: latency-sensitive service.
5. Small repeated reads straddling per-XCD L2 and MALL working-set regions: cache-side delivery.
6. Register BF16 MFMA and FP32 arithmetic: retained arithmetic service.

Reuse the existing 35-condition native suite where it actually matches; extend ILP/grid sweeps through versioned adapters. Do not repeat all historical native conditions solely for completeness. Target controls to the E2/E3 resource boundary, with matched own-cohort full-resource references, three acquisitions/ten trials and explicit counters/checks.

Report useful logical rates separately from physical counters. Record footprint, allocations, launch grid, load count, validation, occupancy/resources, clocks and byte conventions. A useful concurrency bound is

\[
B_{request}\lesssim N_{outstanding}q_{request}/L_{memory}.
\]

If more independent loads recover service under restriction, the data motivate preserving request capacity while reducing arithmetic. They do not establish the exact hardware mechanism or its area. Use matched-shape controls to test transfer to decoder kernels; do not promote one native peak into a universal LLM service rate.

## 9. E5 — Matched graph traffic, phase timeline and coverage

For primary E2 baseline/selected boundary cells and a small E3 subset:

- Collect graph-mode traffic for graph-mode timing, with identical model/output/backend/mask/treatment definitions. Do not substitute eager traffic to fill a failed graph capture.
- Preserve selected decode-position identities (initial pilot ticks 4/5/6), full-step/phase markers, counter pass layout, instrumented outputs and acquired kernel identity. Separate full 63-tick ordinary averages from short physical windows; a join requires an explicit stationarity assumption or matched ordinary subwindows.
- Query actual available L2/MALL/fabric/physical-memory counters and validate units/scope on known traffic. If MALL/true HBM observability is unavailable, report that fact and keep pre-MALL D separate from physical D_H. Do not infer missing HBM bytes by assuming a cache hit rate.
- Include runtime Triton kernels if supported. Otherwise enumerate excluded dispatches/time, collected coverage and missing byte attribution. Failed/misaligned replay groups are unavailable, not silently dropped while status remains `ok`.
- Use bounded counter sets and process-group timeouts. Do not profile the entire workload with an unrestricted full metric suite. Keep all attempts and output comparisons.

For compatible matched physical data:

\[
B_{consumed}=D_H/T,\qquad u_{BW}=D_H/(\beta_{ref}T).
\]

D/T is consumption, not an independent capacity measurement. Report active weight-kernel and whole-step intervals separately. Physical read+write, useful-read and pre-MALL denominators must remain distinguishable. Aggregate logical/physical ratios do not prove per-address exactly-once reads.

## 10. E6–E8 — Kernel efficiency, decision contrasts and guardrails

### E6: Does the apparent compute headroom survive efficient kernels?

From a lightweight graph timeline, identify dominant weight and attention kernels. Use runtime code-object/symbol/source evidence and resource fields to establish actual arithmetic/padding/access organization before a root-cause claim. Keep counter availability and disassembly scope explicit.

Reproduce representative shapes with supported kernels at selected CU allocations. Compare retained service and resource sensitivity with full graph decode. A kernel-shape benchmark is a control, not a full-model latency substitute; cache footprint/state and launch overhead differ. Any improved kernel receives its own exact-output gate and graph baseline. Do not infer physical overhead by subtracting counter-profiled busy union from unprofiled T without validating duration transfer.

### E7: Predeclared decisive comparisons and model validation

Compare within fixed n across service treatments, and lower-n/higher-available-service versus higher-n/lower-available-service. Include uncontended 304-CU and reduced-CU references explicitly. Name every denominator and competitor state.

A reduced-CU/uncontended cell beating a deliberately contended full-resource or larger-CU cell is a valid **shared-service balance demonstration**, not proof of beating the stock uncontended GPU or physically adding HBM interfaces. If a genuinely independently controlled greater-than-stock delivered-bandwidth route is unavailable, record that architecture-specific cell as unavailable. Do not change memory clocks or silently substitute TP/another SKU to manufacture it.

Model phase bottlenecks with

\[
B_{eff}\leq\min(B_{HBM},B_{MC/PHY},B_{L2},B_{NoC},B_{request}),
\]

\[
T_{phase}\approx O+\max(F_i/P_{eff},D_H/B_{eff},L_{issue/dependency}).
\]

Use compatible issued work or explicitly model useful-to-issued efficiency. Sequential phases require an appropriate sum/dependency graph. For restriction penalty κ=T_R/T_0, independently justified bandwidth-scalable critical fraction s, and bandwidth multiplier g relative to the restricted configuration, a conditional fixed-other-cost model is

\[
T'/T_0=\kappa[(1-s)+s/g],\qquad
s(1-1/g)>1-1/\kappa\quad\text{for improvement}.
\]

Neither s nor g is established by nominal competitor duty, D/T, pre-MALL byte/reference-rate ratios or stall percentages. If fitting service-response models, fit on a preregistered training subset, validate on held-out CU/treatment/model points, report residuals/uncertainty and reject failed predictions. Do not force a constant s to explain a response surface with changing bottlenecks.

Illustration only: historical κ=1.09179, assumed s=0.8 and g=1.2 give T'/T_0=0.946218 (~1.057x). None is a measured architecture speedup. The B8 restriction penalty κ≈1.637 cannot be offset by g=1.2 even with s=1 in that fixed-other-cost model.

### E8: Negative regimes and matched energy

For selected baseline/boundary/decisive cells, add Qwen7B B8/C4096 and prefill as mandatory guardrails, then a longer-context B1 point (e.g. C16384 if exact correctness/memory/budget permit), with a small-model graph control if needed to expose launch overhead. Do not automatically expand into the old exhaustive portfolio.

After timing selection, collect separate long-window board/socket energy with raw accumulator and sampled-power records, counts/cadence/agreement gates, exact outputs and equal output/context work across compared cells. Use three independent acquisitions and ten trials for a new quantitative comparison unless a stronger existing contract applies. If the required long-window cost does not fit, retain the energy cell as not run/unavailable rather than reducing counts silently.

\[
E'/E=p/S,
\]

where p is matched average-power ratio and S is speedup. Do not mix long-output energy with 64-token latency to calculate power. Component or physically removed-silicon energy savings remain unmeasured by masks.

## 11. Implementation map and deliverables

| Existing component | Reuse and required caution |
|---|---|
| `scripts/amd/preflight_node.py`, `env.sh`, `vllm_in_container.sh` | Fresh environment/UUID/runtime inventory; historical paths/allocation assumptions need review. |
| `scripts/controlled_vllm_experiments.py` | Existing graph timing/correctness/one-token barrier; audit source/output/telemetry gates before extension. |
| `scripts/amd/lane_cumask.sh` | Old fixed 304/232/152/96 sweep; needs explicit supported new levels and comparison qualification. FMA rate alone is insufficient graph mask proof. |
| `scripts/amd/lane_competitor.sh` | Old fixed 232/72 partition; needs fixed reserved partition across n, lifecycle proofs and a manifest rather than blind whole-grid launch. |
| `scripts/amd/summarize_competitor.py` | Replace overlap double counting and missing qualified ratio CI/clock gates in a new versioned reader; do not rewrite old summaries. |
| `scripts/amd/service_sweep.py`, `service_sweep_lane.sh`, `svc_bench.hip` | Existing service controls; inspect actual options before implementing targeted ILP/grid adapters. |
| `scripts/amd/decode_profile_bench.py`, `analyze_trace.py`, `analyze_counters.py`, `rocprof_watchdog.sh` | Bounded graph/traffic coverage and replay correctness; keep unavailable groups explicit. |
| `scripts/amd/summarize_timing.py`, `summarize_m6.py` | Keep command completion and clock qualification distinct; preserve flags in derived comparisons. |
| `scripts/amd/classify_kernels.py`, `latency_bounds.py` | Existing heuristic/conditional tools; not calibrated physical-time partitions; repair coverage handling only through new versions. |
| `scripts/amd/lane_energy.sh`, `lane_prefill.sh`, `summarize_energy.py` | Requalify long-window scope and counts; include full metadata in new derived comparisons. |
| `scripts/amd/publish_results.py`, `refresh_and_publish.sh` | Compact publishing is useful, but raw run JSONs/monitors/counter inputs need a verified durable archive too. |
| `scripts/amd/mi325x_full.sbatch` | Existing campaign/resume mechanism only. It does not implement this new matrix, and should not be launched to approximate the assignment. |

Implementers must inspect actual CLI help/source and make a new versioned launch plan; this document intentionally does not invent commands for adapters that do not exist yet. Complete CPU adapter/reader checks and a no-GPU dry run before bounded GPU pilots. This handoff commit itself contains documentation only.

Required campaign artifacts:

1. Environment/model/config/tool/source manifest, launch-plan manifest, budget/deadline and randomized acquisition order.
2. A case/attempt journal with exact argv/timeouts/PIDs/outputs/treatment identities, actual status and all failed/partial attempts.
3. Raw run JSONs, outputs/digests, complete telemetry/clock windows, competitor windows/lifecycle records, trace/counter inputs and coverage matrices.
4. Machine-readable tables for E2 noninferiority, T(n,z), E4 service curves, E5 byte domains/coverage, held-out model checks and E8 guards/energy, with explicit unavailable cells.
5. A new report, e.g. `docs/amd/compute_bandwidth_followup_results_<date>.md`, stating what passed, failed, was observational, was blocked or was not run; update the status/log only after evidence exists. Do not retroactively fix old counts or claims by changing raw history.
6. A durable raw archive and complete SHA256 manifest, transfer location/revision and checksum readback. A Git summary or remote listing is not full raw reproducibility. Avoid credentials, model weights and large raw profiler payloads in ordinary Git blobs.

Before rental end, stop only owned workers/monitors, verify cleanup, preserve failures, consolidate compact tables and commit/push progress at safe boundaries under the existing agent's Git authorization. Record outstanding gates so a new agent can resume. Do not label the whole study complete if fixed-clock inference, true HBM attribution, raw transfer or physical architecture questions remain unavailable.

## 12. Exact references for historical motivation

All following AMD snapshot paths use commit `5fc029aae2ffefa00893648ee9891cbccd447416`; moving local results must not silently replace their provenance.

- [Opportunity report](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/docs/amd/shoreline_opportunity_report.md): motivation, with overstatements corrected in Section 1 above.
- [Native service acquisition records](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/results_published/amd_20261018_mi300x/m4_service/service.jsonl) and [summary](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/results_published/amd_20261018_mi300x/m4_service/summary.json): directly auditable 560 records/140 cells, useful-rate conventions and native curves.
- [CU timing](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/results_published/amd_20261018_mi300x/m6_cumask/summary.json), [M1 timing](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/results_published/amd_20261018_mi300x/m1/summary.json), [prefill](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/results_published/amd_20261018_mi300x/analysis/m6_prefill_summary.json): timing and separate clock/sample-count flags.
- [Competition summary](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/results_published/amd_20261018_mi300x/j6_competitor/summary.json): recorded shared-service sensitivity; no independent physical bandwidth subtraction.
- [Latency-model inputs](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/results_published/amd_20261018_mi300x/analysis/shoreline_latency_bounds.json): conditional reference quantities and mixed graph/eager scope, not a physical partition.
- [Kernel classification](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/results_published/amd_20261018_mi300x/analysis/kernel_classification.json): reduced alignment coverage must remain visible.
- [Raw artifact inventory](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/5fc029aae2ffefa00893648ee9891cbccd447416/docs/amd/artifacts.md): publication versus external raw data.
- [Base commit with MI325X continuation log](https://github.com/yanggon-kim/26_memory_centric_gpu_h100/blob/d01317691c6e8d90266e4894fbab2478a5741552/docs/amd/experiment_log.md): current-as-recorded October 26 job history, not live scheduler proof.

Read this file together with actual newer local progress. The requested outcome is a stronger, falsifiable conclusion about compute/service balance, including a negative result if the data require it.
