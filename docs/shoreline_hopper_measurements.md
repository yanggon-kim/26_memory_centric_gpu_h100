# Common Hopper measurement contract for shoreline-aware decode research

Updated 2026-10-16. For the active H200 branch, read [the H200 jobs and overrides](h200_shoreline_jobs.md) first. H100-specific model ordering, seed conventions, historical state and rental assumptions below do not override the H200 allocation or its existing measurements. **All new Hopper M1–M8 and A–F measurements below are pending qualification.** This document requests future execution; it is not a measurement, completed implementation, or architecture validation. Historical H100 evidence remains indexed in STATUS.json and results/case_manifest.json.

## 1. Governing question and reading order

Can we reallocate some GPU compute resources to on-chip SRAM, external-memory PHY/controller capacity and delivered HBM bandwidth, or both, to improve decode latency/throughput and energy under explicit silicon, shoreline, package, thermal and cost constraints?

Read [plan 2.0](shoreline_aware_research_plan_2.0.md) in full, especially Sections 2–7, 9–12 and 15. It supersedes the architecture scope of [the original brief](primary_specification.md). The original brief's Sections 0 and 14 remain historical methodological context; the new plan has a different section structure. Read AGENTS.md, RENTAL_RESUME.md, current evidence manifests and this protocol before planning GPU work. Then use phase02_playbook.md, fixed_bandwidth_collection.md, traffic_reuse_replication.md and trace_collection.md for implementation detail. This protocol governs new experiment order, replication and acceptance if older instructions conflict. Stronger existing correctness, energy, isolation and trace-semantic gates remain binding.

Candidates are stock GPU; compute-to-SRAM; compute-to-bandwidth; joint SRAM/bandwidth; and targeted request-generation/NoC/local-resource designs. Do not preselect a winner. Fixed external bandwidth is one sub-study. Free die area is not interchangeable with placeable contiguous PHY shoreline, memory-controller capacity, interposer connections, HBM stacks, cooling or cost. The rental collects motivation and calibration evidence; physical geometry and architectural changes belong to later modeled/simulated stages, not hardware modifications on the rental.

## 2. First deliverable: inventory, implementation audit and bounded case manifest

Start with a fresh physical-GPU UUID/SKU inventory, profiler permissions, runtime correctness, CPU/RAM/disk/transfer budgets and rental deadline. Read historical failures before retrying. A previous process marked running is not a live process on this rental. Verify import paths, driver/toolkit compatibility, actual L2 bytes and SM count; do not inherit RTX or stock-SXM defaults. Use one authorized physical GPU, no tensor parallelism, no quantization and BF16 weights/KV. No clock/power/driver changes. No MIG/repartitioning or cross-user contention experiments are authorized by this protocol.

The shipped runners implement selected baseline/trace/logical paths; **the new SM controls, native service sweep, event/PM readers and mathematical analysis are not assumed implemented**. Audit each family as implemented/tested/native-validated/missing before running it. Existing vllm_identity.py can bind the prior rental's patched source; inspect/rematerialize a new explicitly versioned identity rather than treating historical paths/build manifests as proof. Add reversible adapters with correctness/negative-control checks and saved source/config hashes. Do not invent executable command-line flags. Use each existing script's --help and prove a small pilot before expansion. CPU checks do not certify GPU captures.

Create a fresh rental result root and shoreline_case_manifest.json before expensive captures. List planned IDs, exact condition, three acquisition IDs, stage, expected artifacts, selection rule, fit budget and pending status. Include failed/unavailable cases and reasons; do not reduce the declared matrix silently when a case fails.

### Model and shape selection

1. Validate the pinned Qwen2.5-1.5B bridge B1/C128, then B1/C1024 and B8/C4096. Keep optimized graph and eager routes separate.
2. For direct RTX traffic comparisons, add the immutable Qwen2.5-3B revision from traffic_reuse_replication.md and Mistral 7B. Read actual layer/head/GQA/sliding-window fields; no universal head geometry is assumed.
3. For H100-specific scale, add Qwen 7B, then 14B and conditional 32B within measured BF16 weights/KV/workspace/graph fit and time/storage budgets. Start B1/C128, then C1024/4096 and B4/8 where feasible. A larger batched case is mandatory as a compute-active counterexample. Use the old portfolio for additional declared shapes, not as proof that they fit.
4. For deep profiling, prioritize bridge B1/C1024 and B8/C4096, Qwen 3B B1/C4096 and B8/C4096, Mistral 7B B1/C4096, plus at least one fitting 14B/32B B1 case. If a model is unavailable/OOM, preserve its failure and select a new explicitly labeled alternative. Cover GQA/head-layout differences. MoE is conditional after dense cases and needs real prompt/routing provenance and a separately validated logical adapter.
5. Execute representative saved text after synthetic smoke. Preserve immutable model/tokenizer revisions, exact prompt IDs and output IDs. Seed 20261013 belongs to the original portfolio; use the separately versioned 20261014 traffic-replication namespace for RTX-matched comparisons and related new captures. Never pool seeds/prompts as the same case. Record native data/randomization seeds separately.

### Repeated-measurement and scope contract

New quantitative comparisons require **three independent process acquisitions per condition, at least three full warmup sequences and ten accepted timed trials per acquisition**, normally 64 generated tokens giving 63 pure-decode ticks. Historical three-trial data remains historical. A different length/window requires a new recorded protocol, not a rewrite of old runs. Report batch-ms/tick and aggregate B*timed_ticks tokens/s separately. Exclude prefill; prove all requests reach the one-token barrier. Keep first token, prefill, CUDA-event, synchronized wall and instrumented durations separate.

Freeze source/config/model/workload controls before collection. Serialize experiments with a lock and continuous sampled GPU-process monitoring; verify owned process ancestry, record observed clocks/temperatures, reject detected interference and preserve attempts. A lock alone does not exclude unrelated users; sampled isolation cannot exclude arbitrarily short overlap. Declare clock rules before inspecting performance: for new matched joins, require pooled valid SM and memory samples each within a 1.05 max/min envelope, no zero/missing readings, all trial endpoints and all profiler replay endpoints included. Use interior samples for ordinary runs where available. Endpoint-only profile coverage remains explicitly weaker. Conditioning changes need a new version; do not drop startup trials or a slow replay to rescue qualification. Never change clocks to force matching.

Ordinary timing is separate from NCU/NSYS/events/traces. Profile exact early decode ticks 4/5/6 after verifying the runtime's tick semantics; save actual sequence lengths and computed-token counts. Their bytes are not automatically representative of a 63-tick mean: inspect early/middle/late uninstrumented-window behavior or fit context growth, and label any stationarity assumption. Three ticks/replay passes are not independent acquisitions. Use hierarchical acquisition/trial bootstrap (10,000 draws, saved seed) and paired acquisition ratios for matched controls. With fewer than three qualified acquisitions report values and missingness, not a three-acquisition CI. Predeclare bounded retry counts; retain failures and independent IDs. Do not select retries by favorable latency.

## 3. Eight required motivation experiments

The following M1–M8 registry is the new study's portable execution contract. It expands the older traffic replication; it is not a claim that all older scripts already support it.

| ID | Experiment and concrete measurements | Question / acceptance artifact |
|---|---|---|
| M1 | Uninstrumented eager and strongest valid CUDA-graph vLLM baselines over the declared model/shape matrix; separate NSYS CUDA/NVTX timelines, three acquisitions, exact op attribution and launch counts. | Where is latency spent, and what is the strongest correct baseline? Emit timing table, throughput, kernel busy-union shares, launch/gap distributions and matched mode comparisons. |
| M2 | Algorithmic logical-byte/FLOP accounting plus NCU physical HBM reads/writes for first/middle/last decoder layers and full decode. Preserve all intervening layers. Repeat three acquisitions; obtain eager and graph bytes independently for their timing joins. | Is traffic close to a single useful stream, or is there evidence for extra physical demand? Emit per-tick and multi-tick logical/physical tables and rooflines with exact scope/denominators. |
| M3 | Bounded actual gate/up GEMV/GEMM, attention forward, combine and KV-update kernels; occupancy, arithmetic/issue activity, eligible warps, stalls, local sectors, launch geometry/register/shared resources. Extract loaded sm_90/sm_90a SASS and actual dynamic instruction evidence. | Is a kernel limited by arithmetic, dependencies, local resources, issue or delivery? Emit operation-verified kernel inventory and SASS/stall explanation before causal claims. |
| M4 | Independent native streaming, strided/gather, pointer-chase, arithmetic, matrix and mixed compute/memory controls; working sets spanning below/near/above actual L2, independent chains/ILP, CTA concurrency and SM availability. | What service/latency can this exact Hopper deliver for different access patterns? Emit raw event timing, physical bytes, clocks, correctness checks and service curves. A copy ceiling is not universal decode bandwidth. |
| M5 | Three-acquisition normal versus controlled competing-read cache-history experiment; cold/steady state and distinct cross-token windows; reuse/address evidence. Test supported retention/tiling policies separately if feasible. | Which baseline reads are already avoided, and is incremental retention plausible? Emit read/write deltas separately, full-model scope, unchanged outputs/instruction checks and no invented cache-hit attribution. |
| M6 | Supported same-GPU software SM-allocation sweep, nominal fractions 1.0/.75/.5/.3 rounded to actual supported SM groups; repeat ordinary eager timing, valid graph timing if supported, full physical bytes and selected resource/counter proofs. | How far can execution resources decrease before latency degrades? Emit actual allocated SM counts, per-kernel context/grid proof, three-acquisition ratios and CIs. If unsupported, mark unavailable; no MIG substitution or background busy-kernel emulation. |
| M7 | Repeat independent streaming/gather/pointer/MLP service controls under exactly the M6 allocations; compare physical traffic, event latency, occupancy, issue/stalls and active grid. | Does trimming SMs reduce independently attainable delivery? Emit beta_service(SM,pattern,footprint,ILP), latency curves and compute controls. Separate this from workload consumption D/T. |
| M8 | Long-window board energy on accepted full-model/selected SM cases, raw counter plus sampled power and gates; separately measure prefill as a higher-intensity counterexample. | Are energy claims valid, and where does a decode-oriented design lose elsewhere? Emit qualified or NA J/token, raw sensors, prefill/decode comparisons and larger-batch negative regimes. |

### M1/M2: algorithm, bytes and attribution

A useful reference for a dense layer is W=sum(actual distinct required parameter bytes, including applicable norms/biases), historical KV=2*B*L_historical*G*d*q, attention useful F=4*B*H*d*L_attention, linear useful F=2*B*sum(projection parameter elements). Current K/V is generated inside the layer; do not charge it as a historical external read, but do include it in attention operands. With shared prefixes, count actual storage aliases and request roles. Use actual valid lengths/sliding windows, not allocated cache length. Include head/logits once at whole-model scope and preserve embedding aliases/lookup semantics. Non-main operations and metadata omitted from F/U must be named. MiB/GiB and decimal MB/GB must be explicit.

For N selected ticks distinguish U_tick_sum (weights counted at each tick) from U_window_unique (unchanged weights counted once). D/U_window_unique near N can simply be expected autoregressive streaming. Logical demand is not a compulsory-DRAM oracle. Cached inputs can offset rereads elsewhere, so D/U_tick_sum near one neither proves exactly-once addresses nor establishes zero reuse. A signed D_read-N*W residual is not measured KV traffic and must not be clamped or called a KV hit rate.

Query available H100 counters/units first. Collect actual dram read/write bytes, L2 interface sectors, global/local LSU sectors or requests, and relevant launch/occupancy/issue/arithmetic/stall metrics with names, raw values, units and aggregation. Keep L1/L2 requested sectors distinct from HBM bytes. Logical TMA tile payload, NVBit addresses and allocated ranges do not replace physical HBM counters. Denied counters remain NA; repair permissions through provider-supported allocation rather than interpreting analytic bytes as hardware evidence.

Every selected kernel must match actual NVTX operation/layer/tick and runtime symbol, not just a name regex/launch skip. RTX's broad cutlass regex accidentally matched FlashAttention template names; retain a negative test for attention-combine misselection. Native Hopper may use FA3/TMA/WGMMA, different packing/persistence and split heuristics. Discover those independently, preserve launch/split/tile mapping, dynamic work, register/shared allocation granularities, driver carveout and occupancy limiters. Do not import RTX's approximately11x issued/useful attention arithmetic ratio. Derive issued tensor work from actual instruction shape, grouping, masks and execution counts; if unsupported report it unavailable.

An occupancy percentage is resident warps relative to a documented limit, not percent of compute throughput. Warp-active stall percentages are not wall-time fractions. Raw tensor/FMA cycle counts cannot be divided by an unrelated elapsed counter to manufacture utilization; validate documented numerator, denominator, aggregation and replay scope. Prefer supported normalized per-pipeline metrics and report scalar/tensor/address/LSU activity separately. PM counters collected on separate passes cannot demonstrate simultaneous compute/memory overlap.

### M5: controlled pressure and software-resource experiments

Begin with a competing read larger than actual L2 (the RTX reference used a persistent256MiB buffer). Record exact bytes and CUDA work, seed, cache policy, synchronization and when it runs. For the RTX-matched layer experiment, perform the read before each selected layer0 tick4/5/6, finish outside its NVTX counter range, and still execute intervening full-model layers. It is a cache-history intervention, not a guaranteed flush or an emulation of a larger SRAM. Randomize/interleave matched normal/pressure acquisitions when possible and preserve treatment order. Investigate changed writes as well as reads: moving dirty writeback outside a range can change charged traffic without reducing logical writes. Measure ordinary treatment latency separately, include treatment cost if reporting usable end-to-end benefit, and never infer speedup from counter-only captures.

For M6/M7 prefer a supported CUDA green-context route only after correctness/device support is proven. Preserve the real memory interface and verify actual resources used by every recorded kernel, including library/default-stream behavior, graph capture/replay and cleanup. Requested fractions are not actual allocation proofs. Existing compiled grids may not retune to a smaller context. New kernel retuning is a separately versioned control. Declare a5% latency-regression noninferiority threshold before running; acceptance needs the upper95% interval <=5%, not a rounded point estimate. Reduced SMs remove arithmetic, LSU, issue and MLP together; these software proxies do not isolate physical compute removal, create SRAM or establish equal area.

### M8: energy is a gated result

Keep the existing long-window/cadence/counter-versus-power rules in measurement_schema.md and phase02_playbook.md. Preserve raw endpoints/samples, actual API-call timing, distinct readings, sensor errors, clocks/thermal behavior and full board scope. Sensor agreement is internal consistency, not external absolute calibration or HBM component energy. If disagreement persists, report energy NA. A separately versioned polling-cadence/conditioning diagnostic may investigate it without changing clocks, discarding native trial failures or rescaling one sensor to match the other. Do not repeat a full diagnostic sweep before a bounded functional pilot passes. Timing and traffic conclusions can remain useful while energy is unavailable.

## 4. Six additional measurements for a bounded latency decomposition

These A–F are required extensions to M1–M8. They provide service estimates and controls for a mathematical decomposition; they cannot uniquely measure wall-time overlap from aggregate counters.

| ID | Additional collection | Required output / failure handling |
|---|---|---|
| A | Dependency-chain unloaded latency: serialized pointer chase at L1/L2/HBM-sized footprints, randomized linked data and varied independent chains; empty-loop/timer overhead controls. | Cycles/ns per dependent access with actual footprint, cache-state evidence, clock samples and uncertainty; occupancy changes are separate. Mark cache-level inference conditional. |
| B | Independent compute service: matched scalar FMA, integer/address operations, tensor GEMM shapes/dtypes and softmax/reduction-like controls; sweep resource allocation and intensity. | Useful and issued work, event time, actual instruction route, attained rates and counter checks. Published peak is a reference floor, not fitted effective service. |
| C | Independent memory service and demand: stream/gather/paged-like access, physical reads+writes, footprint/ILP/concurrency/SM sweeps. | Access-pattern-specific beta and latency/MLP curves; no circular beta=D_decode/T_decode substitution. Reuse M4/M7 captures where exact controls qualify; do not duplicate them merely to fill a label. |
| D | Concurrent arithmetic+memory native microbenchmarks spanning balanced/imbalanced demands and dependent/independent work, with identical standalone controls. | Actual concurrent T, standalone C/M references, physical traffic and same-pass overlap evidence if supported. Test interference; concurrent time need not equal max(C,M). Separate kernel concurrency from intra-kernel warp specialization. |
| E | Layer timing/launch/instrumentation controls: CUDA events around disjoint first/middle/last complete layers at selected ticks; full decode and empty-marker/control measurements, plus NSYS busy union/gaps. | Matched ordinary vs event-instrumented timing, exact layer boundaries and perturbation. Whole-model low overhead does not bound local layer intrusion. No first-to-last noncontiguous span divided by ticks as a layer duration. |
| F | PM/replay controls and predictive model validation: inspect actual pass layout, compare instrumented vs ordinary timing and exercise held-out native and decoder conditions. | Only simultaneously sampled aligned domains can support time-bin association; separate passes stay marginal. Raw bins/stalls never become an exact busy partition. Fit common-family coefficients, keep failure/NA records and report held-out errors. |

### Mathematical framework and constraints

For a declared common layer or full-model scope:

```text
T = C + M - X + O
C = F/P_effective, M = D/beta_effective
0 <= X <= min(C,M), O >= 0

O_feasible = [max(O_low, T-C-M), min(O_high, T-max(C,M))]
X(O) = C+M+O-T
compute_only(O) = C-X(O)
memory_only(O) = M-X(O)
overlap(O) = X(O)
```

O is residual cost: omitted arithmetic, address/on-chip service, launches, dependencies and other model mismatch can all contribute. It is not simply CPU idle time. Bounds on O need an independent basis; if absent use a broad declared nonnegative range and show non-identifiability. C/M are service equivalents, not directly measured physical busy times. Useful F omits padding and non-main work; derive issued work where possible and retain both scenarios. Empty feasible intervals reject the chosen rates/model. A nonempty interval is consistency, not validation. C-only/M-only/X interval extrema are coupled through the same O and cannot be combined arbitrarily. Also propagate acquisition uncertainty and effective-rate sensitivity; label assumption envelopes separately from statistical CIs.

Use D's native concurrent cases to test eta=X/min(C,M), not to assume perfect overlap. Aggregate three acquisitions per condition before a reproducible70/30 condition-level train/held-out split, with at least five distinct conditions in a fitted family and a nonempty held-out set. Avoid per-row free residual/eta fits that force exact agreement; report ordinary held-out time predictions, errors, rate extrapolation and rejected conditions. Decoder families/backends require their own validation; a successful native fit is not automatically a decoder fit. If training is too sparse or rates/trace semantics unsupported, publish feasible bounds and NA predictions rather than fitted truth. Do not fit on all points and call them held out.

For the architecture sensitivity study define:

```text
c = effective_compute_new/effective_compute_base
r = delivered_memory_service_new/delivered_memory_service_base
f = eliminated_fraction_of_total_physical_memory_service
C_new = C/c
M_new = M*(1-f)/r
T_new = C_new+M_new-X_new+O_new
memory_service_improves iff f > 1-r
```

Evaluate c={1,.75,.5,.25}, r={.7,1,1.2,1.4}, f={0,.05,.1,.2}, plus workload-supported values. c is not automatically an SM or die-area fraction; r is not nominal HBM stack count; f is not a cache-capacity fraction or aggregate sector-hit rate. Show SRAM-only, bandwidth-only, joint and compute-trim-only cases including slowdowns. Same-O/same-eta projections and a free-new-overlap envelope must be labeled conditional; architecture changes may alter both. Do not infer eliminated physical traffic from a synthetic capacity curve without address/counter/model support. `1/(1-p+p/r)` bandwidth sensitivity needs a justified bandwidth-sensitive wall fraction p, never a warp-stall percentage.

## 5. Trace and physical-model handoff

Native H100 collection does not require starting architectural simulation during the rental. First obtain correct optimized ordinary baselines and bounded sm_90 eager capture/replay pilots; then collect2–3 complete decode steps, including all intervening layers, LM head/sampling/cache updates. Larger-model selected layers need separate partial labels and retained inter-layer address chronology; a partial capture cannot establish whole-model cross-token retention.

Follow trace_collection.md and retain source/model/runtime/cubin/config hashes, kernel order and NVTX identity, actual-SKU clocks/cache/HBM/SM configuration, correctness tokens, uninstrumented same-program eager calibration, active masks, global/local/shared addresses and access sizes, parameter/KV aliases and request/page/slot mappings, unmapped scratch/temporaries and initial cache-state assumptions. Preserve observed TMA/WGMMA/mbarrier/async operations and required descriptors/register values. Dropping unsupported waits/barriers leaves semantics unvalidated. Positive parser replay alone is insufficient. Different resource allocations or changed tiling/schedules may need newly generated traces; a stock trace cannot automatically predict a redesigned kernel.

Provide later simulator work with workload demand and service limits for arithmetic, LSU/address issue, outstanding requests, cache banks/ports, NoC and controllers, rather than only SM count. The later architecture matrix must vary delivered bandwidth/controllers, SRAM capacity/policy/service, compute and delivery resources independently, and then apply sourced area/PHY/shoreline/package constraints. Distinguish effective bandwidth multipliers from a real extra stack/PHY implementation. H100 power/area and package feasibility remain uncalibrated unless independently established. The original source_change_map.md is historical fixed-bandwidth simulator guidance, not an instruction to hold HBM fixed in all new candidates.

## 6. Order, deliverables and completion gates

Execute in stages: inventory/correctness and one-family pilots; M1 timing/timeline and M2/M3 selected counters; M4 plus A/B/C service pilots; M5/M6/M7 matched interventions; D/E/F and mathematical validation; M8 energy/prefill; then bounded semantic trace collection/transfer. Use pilot costs to budget the full declared matrix. Counter-access and trace pilots happen early enough to discover blocked routes before spending the rental. Historical correctly scoped captures may satisfy a cell only after current hashes, runtime identity, raw provenance and scope gates are revalidated. Never silently pool different rentals/backends/clock envelopes. Spend remaining time on fitting larger models after the minimum contrast cases pass.

Under a fresh result root produce:

- shoreline_case_manifest.json: M1–M8/A–F planned conditions and acquisitions, required/missing fields, attempts, statuses and reason; per-case raw artifact hashes and measured/analytical/simulated class.
- shoreline_measurements.jsonl or CSV: layer/tick/kernel/range/engine/SM allocation, useful and issued F, logical U_tick_sum/U_window_unique, D_read/D_write, timing and metric numerator/denominator, source/replay IDs, clock/isolation coverage, uncertainty and NA reason.
- shoreline_service_calibration.json: ordinary and physical-counter service curves, validated native families, train/held-out split/errors and unsupported extrapolation.
- shoreline_latency_bounds.json: C/M/O/X inputs and provenance, coupled feasible intervals, rejected/infeasible rows and assumptions; no fabricated exact physical partition.
- shoreline_joint_sensitivity.json: c/r/f grid, same-O/eta and relaxed-overlap envelopes, negative regimes and explicit physical-feasibility/energy unknowns.
- shoreline_opportunity_report.md: strongest graph/eager baseline, evidence for compulsory streaming versus retention, SM/request-delivery tradeoff, H100-specific attention structure, larger-model/batch differences, prefill/energy disposition, and recommendations for later physical/simulator screening.
- Figures: ordinary timing versus model/batch/context; useful F/D roofline with independently measured reference curves; logical vs HBM reads/writes; latency and service vs actual SM allocation; kernel occupancy/issue/stalls with denominators; coupled latency bounds; c/r/f sensitivity and losing cases. Do not stack overlapping counters as a wall-time decomposition.
- Native trace/export manifests, calibration table and durable archive with full independent remote readback/checksum. Update experiment_log.md, opportunity_report.md and STATUS.json only with actual evidence; preserve original result rows and prior failures.

Continue reporting the older fixed_bandwidth_* and traffic_reuse_* artifacts required by their supplements, linking shared qualified captures rather than rerunning identical work. The old18baseline+2pressure+2address list defines conditions/minimum pilots, not final statistical sufficiency: the new quantitative normal and pressure comparisons need three independent acquisitions (18 baseline conditions become54 acquisitions, two pressure conditions become6). Both address cases start bounded and cannot support CIs without sufficient independently qualified repeats. Do not automatically triple expensive traces before a budgeted pilot.

Completion means every declared family has a terminal disposition, qualified comparisons have raw evidence and independent checks, unavailable fields remain NA, and final artifacts pass source/hash/link and archive readback checks. An unavailable family keeps its scientific gate incomplete; terminal disposition is not successful validation. Report which hypothesis survives or fails. This rental does not authorize a claim of better physical architecture merely because all scripts returned zero.
