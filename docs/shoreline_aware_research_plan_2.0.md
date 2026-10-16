# Shoreline-Aware Compute–Memory–Bandwidth Co-Design for Energy-Efficient LLM Decoding

**Research plan and agentic-experiment handoff — 15 October 2026**  
**Status:** Proposal and falsifiable research hypotheses, **not** a demonstrated architecture.  
**Target:** A GPU-like, HBM-attached accelerator for memory-intensive autoregressive LLM decoding.  
**Primary goal:** Minimize **system-relevant energy per generated token**, subject to specified decode performance, manufacturability, die, package, memory-capacity, and cost constraints.

> **One-sentence thesis.** Rather than assuming that unused GPU compute area should become on-chip SRAM, treat compute, cache capacity and bandwidth, HBM-controller/PHY capacity, off-chip HBM bandwidth, on-die interconnect, and *usable I/O shoreline* as coupled design resources. Then develop and validate a physically constrained optimizer and the targeted microarchitecture required to make its selected design practical.

> **For a Linux AI research agent:** Treat every claimed performance gain as a hypothesis until reproduced. Do not fabricate H100 trace availability, physical die layouts, PHY sizes, foundry rules, or energy/area measurements. Record sources, versions, assumptions, uncertainty ranges, and negative results.

---

## 1. Executive Summary

### 1.1 How our research question evolved

**Original hypothesis:** During small-batch LLM decode, arithmetic units can be underutilized because the GPU waits for model weights and/or KV cache data. Reclaim some compute area and enlarge register files, shared SRAM, L1, or L2.

**New observation motivating this proposal:** For a **dense, small-batch GEMV-like weight path**, weights often have little useful reuse *within one decode step*, and the reuse distance across steps can greatly exceed practical on-chip cache capacity. Extra SRAM capacity alone may therefore deliver disappointing reductions in HBM bytes/token. The input activation vector can still be reused, and this observation must not be generalized to all decode operations: larger batches, GEMM-like kernels, attention, GQA/MQA, shared prefixes, and cache residency policies may behave differently.

**Revised hypothesis:** If off-chip bandwidth is the dominant bottleneck, a better use of reclaimed SM/Tensor-Core area may be to support **more memory channels and HBM stacks**, including the requisite **memory controllers, PHYs, cache/partition bandwidth, NoC capacity, and request-generation capability**. Whether more HBM can be integrated depends partly on the shape of the compute die, the usable placement of I/O macros on its boundary (*shoreline/beachfront*), and interposer/package constraints.

**Research leap:** Use a **shoreline-aware, package-aware area/resource allocation model** to choose between compute, SRAM, off-chip bandwidth, and interconnect—and propose a **bottleneck-driven microarchitecture** (possibly a lighter memory-serving SM/engine, memory partition, or on-chip fabric) to realize the selected balance.

### 1.2 Proposed research contributions (to be earned experimentally)

1. **Physical-resource-aware design-space exploration (DSE):** Couple a compute/memory performance model to discrete HBM-stack counts, memory controller/PHY macro footprint, *placeable* shoreline, reticle bounding box, interposer placement/routing, package footprint, power, and thermal limits.
2. **Compute-to-bandwidth architectural reallocation:** Quantify whether and when shrinking compute structures makes it physically feasible to integrate one or more additional HBM interfaces and gain useful *delivered* HBM bandwidth at a fixed design budget. Compare this to spending the same area on SRAM or NoC alone.
3. **New enabling microarchitecture:** From the remaining bottleneck—L2 bank bandwidth, NoC bisection, LSU/AGU request rate, memory-level parallelism, load balancing, or compute—develop the *smallest* necessary hardware mechanism and validate it against strong simpler alternatives.
4. **Generalizable design principles:** Identify when SRAM-focused, HBM-bandwidth-focused, and balanced designs win across decode workloads, and when shoreline-aware modeling changes the optimizer's answer.

**Novelty warning:** Compute/buffer optimization, LLM-oriented accelerator modeling, irregular die shapes, shoreline-limited bandwidth, and external-memory fabrics each have prior work. Novelty must come from a **new validated coupling, physical-feasibility-aware optimizer, and/or specific microarchitectural mechanism**, not from claiming to have invented these concepts individually.

---

## 2. Verified Starting Facts vs. Working Hypotheses

| Statement | Classification | Basis / caveat |
|---|---|---|
| Full GH100 design supports **6 HBM stacks, 12 × 512-bit memory controllers, and 60 MB L2**. | Published specification | NVIDIA Hopper architecture overview [R1]. |
| H100 SXM configuration has **5 active/installed HBM3 stacks, 10 × 512-bit controllers, and 50 MB L2**, with **132 SMs**. | Published specification | NVIDIA [R1]. **Do not call an 8-stack configuration an H100 baseline.** |
| LLMCompass models physical vs. active memory channels, compute resources, buffer capacity, and LLM performance, with an area-oriented hardware cost model. | Published implementation/research | Its documentation explicitly distinguishes physical channels (area) from active channels (performance) [R2, R3]. It is **not** a package floorplanner. |
| Irregular, tabbed, stepped, wavy, or interlocking die perimeters have been proposed to enlarge I/O beachfront. | Published patent-application disclosure | US 2026/0182042 A1 [R4]. A patent disclosure is **not** measured evidence of H100 implementation feasibility or speedup. |
| GPU memory bandwidth scaling can be limited by GPU die shoreline. | Published research motivation | Memory Fabric [R5], among others. Its measured/estimated improvements are for its own HBM-to-HBM/multi-chiplet design, **not** proof that reshaping a GPU die raises its local HBM bandwidth. |
| Optical I/O on the compute-die shoreline can change the tradeoff among memory access, network capacity, and compute placement. | Published alternative approach | SiPAM [R6], which uses photonic memory pooling rather than directly adding conventional HBM PHYs through irregular die edges. |
| Memory-bound LLM GEMV automatically benefits from more on-chip SRAM. | **Not established** | Must test actual reuse, residency, and the active bottleneck. |
| Removing SM area permits extra HBM stacks within the **same** reticle/package footprint. | **Not established** | Depends on usable shoreline, PHY depth, physical HBM package clearance, routing/keep-outs, power, and thermal constraints. |
| A notched or stepped monolithic GPU die is viable and energy-optimal. | **Not established** | Must test a physically conservative abstract feasibility model and compare standard rectangles and bridge/chiplet alternatives. |

**Why start with 5 → 6 before exploring 5 → 7 or 5 → 8?** GH100 already defines a six-stack full configuration [R1]. Modeling 5 → 6 is a convenient, potentially lower-risk **first-order architecture comparison**; it is **not** an assertion that any production H100 SXM board can simply install another stack. Beyond the six-stack full GH100 configuration requires hypothetical new I/O/package designs and much more extensive feasibility analysis.

---

## 3. First-Principles Performance and Energy Model

### 3.1 Workload families must be separated

- **Dense weight GEMV, small batch:** weights often dominate off-chip bytes/token; the vector has reuse, but the weight working set may dwarf SRAM. HBM throughput or insufficient memory-level parallelism can dominate.
- **Batched weight GEMM:** weights are reused across requests; Tensor Core throughput and tile-level SRAM reuse can become important.
- **Attention decode:** streaming of K/V often grows with context length, but GQA/MQA, paged KV, prefix sharing, and kernels may introduce valuable SRAM or cache reuse.
- **Additional operators:** normalization, sampling, projections, dequantization, MoE routing. Quantization changes both bytes and arithmetic costs.
- **Prefill:** explicitly measure as a *guardrail*, even if optimizing primarily for decode, because sacrificing compute may severely regress prefill.

Always report the precise workload (model, precision, batch/concurrency, prompt length, sequence length, attention type, kernels, frameworks, clock/power state) rather than saying only “LLM decode.”

### 3.2 Minimal limiting-resource model

For a workload point `w`, approximate its compute/memory-limited execution time by:

\[
T_w \gtrsim \max\left(\frac{F_w}{P_{\mathrm{compute,eff}}},\;\frac{D_{\mathrm{HBM},w}}{B_{\mathrm{HBM,delivered}}},\;\frac{D_{\mathrm{NoC},w}}{B_{\mathrm{NoC,eff}}},\; T_{\mathrm{MLP/issue},w}\right).
\]

This is an *initial screening bound*, not a cycle-accurate timing model. Address contention, dependencies, pipeline overlap, cache policy, launch/synchronization, and thermal behavior must be captured in the later simulator.

Define delivered bandwidth as bounded by:

\[
B_{\mathrm{HBM,delivered}} \le \min\{N_{\mathrm{HBM}}\,B_{\mathrm{stack}},\ B_{\mathrm{MC/PHY}},\ B_{\mathrm{L2/partition}},\ B_{\mathrm{NoC}},\ B_{\mathrm{SM\ issue}}\}.
\]

A simple 5-to-6-stack thought experiment predicts **at most 20% higher nominal aggregate HBM bandwidth if stack technology/rate are identical**. It does **not** guarantee a 20% end-to-end speedup, or that interposer/package integration is possible.

### 3.3 Energy per token

\[
E_{\mathrm{token}} = E_{\mathrm{compute}}+E_{\mathrm{L1/L2/SRAM}}+E_{\mathrm{NoC}}+E_{\mathrm{MC/PHY}}+E_{\mathrm{HBM}}+E_{\mathrm{other}}+E_{\mathrm{idle/leakage\ during\ runtime}}.
\]

Adding HBM may increase standby/controller/PHY/termination energy while reducing execution time; increasing SRAM adds leakage and access energy; reducing compute can reduce some leakage, but previously idle arithmetic pipelines may already be clock-gated. Do not infer power reduction directly from low utilization.

**Primary output:** J/token, plus throughput, time/token, p50/p99 inter-token latency, prefill/TTFT regression, package power, cost per useful token where estimated, and explicit uncertainty bounds.

---

## 4. The Correct Physical Budget: Several Coupled Constraints, Not One Interchangeable Area

A GPU logic die and an HBM stack are separate pieces of silicon with different processes, yields, and costs. Logic mm² are **not fungible** with HBM stack mm². Fix or bound each independently:

### 4.1 Logic-die area

\[
A_{\mathrm{logic}} = A_{\mathrm{SMs}}+A_{\mathrm{L2}}+A_{\mathrm{MC/PHY}}+A_{\mathrm{NoC}}+A_{\mathrm{IO,other}}+A_{\mathrm{control/clock/power}}+A_{\mathrm{physical\ overhead}}.
\]

`A_MC/PHY` should be estimated independently: a memory controller is digital scheduling/control logic, whereas a PHY has strict signaling, bump/placement, and analog/physical constraints. Reduction of Tensor Cores or SMs does not instantly free equally placeable I/O real estate.

### 4.2 Geometry and reticle

Represent the GPU logic die as a 2-D polygon `D` with bounding box `(W_bbox, H_bbox)` and active silicon area `Area(D)`. Require:

\[
W_{bbox}\le W_{\mathrm{reticle}},\qquad H_{bbox}\le H_{\mathrm{reticle}},\qquad Area(D)\le A_{\mathrm{logic,budget}}.
\]

A roughly **26 × 33 mm** maximum single exposure field is a useful *screening example* (verify the actual foundry/process/design-rule envelope before using it as a manufacturing guarantee). Include scribe/seal margins and an explicit bounding-box orientation. Geometry that looks permissible by area may fail by reticle dimensions, and an irregular shape does not evade the bounding-box field constraint.

### 4.3 Package / interposer / thermal / economic constraints

Separately track:

- HBM stack footprint and keep-out/placement constraints;
- package/interposer footprint, substrate size and routing layers/capacity;
- PHY microbump footprint, escape routing, signal integrity and maximum allowed reach;
- stack and GPU power delivery, thermals and mechanical assembly/yield risks;
- optional total package cost and heterogeneous silicon/yield cost, with sensitivity ranges;
- required **HBM capacity**, not just bandwidth: all target model weights, KV data, runtime allocations must fit.

**Fair comparisons:** (A) fixed logic-die area + fixed package/interposer footprint + matched memory capacity floor; (B) fixed *cost* envelope allowing HBM/package expansion. Show both when they give different answers. Do not claim “same-area” while quietly enlarging the interposer or adding costly HBM stacks without reporting the change.

---

## 5. I/O Shoreline Must Mean *Placeable* PHY Capacity, Not Raw Perimeter

### 5.1 Definitions

- **Geometric shoreline** `L_geom`: length of the polygon boundary.
- **Eligible shoreline** `L_eligible`: segments facing an achievable HBM connection, after reserved I/O regions and keep-outs.
- **Placeable PHY capacity** `N_PHY_feasible`: maximum whole PHY macros that can be placed along eligible segments **with their required interior depth, bump footprint, spacing, corner clearance and routing**.
- **Attachable HBM stacks** `N_HBM_feasible`: the number that can fit in the interposer/package and connect electrically with valid signal lengths/routing/power.

The real feasibility condition is a **discrete placement-and-routing problem**:

\[
N_{\mathrm{HBM}} \le \min\{N_{\mathrm{PHY\ groups,feasible}}(D),\ N_{\mathrm{package\ fit}}(D),\ N_{\mathrm{routing/thermal\ feasible}}(D)\}.
\]

**Do not estimate stack count by raw perimeter divided by nominal PHY width alone.** One HBM stack may require multiple PHY/controller partitions; the grouping and correspondence must be parameterized from the target memory standard and candidate design.

### 5.2 Explore die shape families

1. **Rectangular, conventional (mandatory baseline):** optimize PHY placement on all feasible sides first.
2. **Rectangular with some compute replaced by MC/PHY/L2/NoC (mandatory control):** same shape; test whether irregularity is actually required.
3. **Elongated rectangle:** increases long-side shoreline but quickly hits reticle width/height limits on large monolithic dies.
4. **Coarse rectilinear notches/steps/tabs:** creates new straight boundary segments within the same reticle bounding box, at the possible expense of removed silicon and deeper PHY routing.
5. **Fine sawtooth / wavy boundaries (stretch experiment):** maximize geometric length but may be unplaceable for real macros and mechanically risky; use only if coarse shapes underperform.
6. **Alternative form factors:** chiplets, bridges, remote-HBM fabrics, optical I/O as *comparison points*, not necessarily to be implemented in the first paper.

### 5.3 Useful geometry identity—and its limitation

Carving a simple rectangular inward notch of width `w` and depth `d` from a straight edge increases the nominal perimeter by **`2d`** while *removing* a silicon area of `w d` (ignoring corner and manufacturing details). Thus, **shoreline can be created by sacrificing silicon**. But whether either added side wall accommodates a full PHY with its interior depth—and whether an HBM stack can actually be placed and routed in/near the notch—is an independent placement problem. In many configurations the notch could create *zero* additional useful HBM interfaces.

A coarse **stepped/notched** option is a reasonable first hypothesis, **not** an established best package layout. Some designs may prefer outward tabs, multi-sided rectangles, or chiplets once assembly and routing constraints are included.

### 5.4 Minimal geometry feasibility prototype

Implement in Python with polygons and mixed-integer/constraint programming (e.g., Shapely + OR-Tools CP-SAT, after environment check). Each candidate records:

`die_polygon`, `bbox`, `actual_silicon_area`, `reticle_pass`, `phy_site_ids`, `phy_macro_area`, `phy_depth`, `available_straight_segment`, `placement_keepouts`, `hbm_stack_locations`, `interposer_routes`, `max_reachable_hbm_stacks`, `cost/power_proxies`, `reason_for_rejection`.

At each phase explicitly state whether the test is (i) polygon-level **screening**, (ii) macro-level **floorplanning estimate**, or (iii) actual foundry-certified **place-and-route** (unlikely without industrial tools/PDK).

---

## 6. Architecture Design Variables and Joint Optimization

The primary architectural design vector is:

\[
\begin{aligned}
\mathbf{x}=\{&N_{\mathrm{SM}}, N_{\mathrm{Tensor/SM}}, N_{\mathrm{INT/SM}}, N_{\mathrm{LSU/AGU}},\\
&C_{\mathrm{SM\ SRAM}}, C_{\mathrm{L2}}, N_{\mathrm{L2banks}}, B_{\mathrm{L2}},\\
&N_{\mathrm{MC/PHY}}, N_{\mathrm{HBM}}, B_{\mathrm{NoC}}, Q_{\mathrm{memory\ requests}},\\
&S_{\mathrm{die}},\ \mathrm{PHY\ placement},\ \mathrm{HBM\ placement},\ m_{\mathrm{mapping/scheduling}}\}.
\end{aligned}
\]

Use **discrete** counts for SM groups/TCs, banks, MCs/PHYs, HBM stacks, and ports, rather than continuous unrealistic knobs. Include die-shape class and parameterized steps/notches. The optimization is:

\[
\min_{\mathbf{x},m} \sum_{w\in\mathcal{W}}p_w E_{\mathrm{token},w}(\mathbf{x},m)
\]

subject to (for each workload class as appropriate):

- `logic_area(x) <= logic_area_budget`;
- `bbox(x) within reticle`; physically realizable usable PHY shoreline;
- `hbm_phy_mapping(x)`, `package_fit(x)`, `routing(x)`, `thermal(x)` all feasible;
- `package_footprint(x) <= package_budget` and optionally `cost(x) <= cost_budget`;
- `T_decode,w(x) <= SLA_w` and explicit prefill/TTFT guardrail;
- `memory_capacity(x) >= workload_required_memory`;
- `power(x) <= cooling/power_budget`.

**Pareto views** should include J/token vs. decode latency; performance vs. cost; shoreline consumed vs. HBM attached; and compute area vs. **delivered** bandwidth. Do not collapse all objectives into a single unexplained scalar. Optimize for several workload mixtures and report sensitivity to the mixture weights `p_w`.

---

## 7. Candidate GPU Microarchitectures (Do Not Preselect a Winner)

| Candidate | What changes | First-order benefit sought | Likely new risk |
|---|---|---|---|
| **B0: Stock H100-like** | Published 5-stack SXM-like baseline | Reference | Experimental model and public configuration mismatch |
| **B1: Reduce compute only** | Fewer SMs / Tensor pipes, unchanged memory | Quantify removable compute and baseline energy | Reduced issue/MLP; prefill regression |
| **B2: Compute → SRAM** | Reduce compute and enlarge local/L2 caches | Test original hypothesis fairly | No weight reuse; banking/NoC cost |
| **B3: Compute → HBM/MC/PHY, rectangle** | Retain regular edge, add MC/PHY and stack if physically feasible | More external bandwidth without irregular geometry | Not enough eligible shoreline or package room |
| **B4: Compute → HBM/MC/PHY, stepped/notched** | Sacrifice/rearrange SM regions, reshape die, extend *usable* shoreline | Enable an otherwise infeasible stack/interface | Reticle bounding-box, placement, yield, routing |
| **B5: Joint HBM + L2 banks + NoC** | Rebalance memory partitions, routing, queues and bank throughput | Deliver added BW to SMs | NoC area/energy and hotspots |
| **B6: Minimal memory-serving engine / SM** | Small AGU/LSU/descriptor/request subsystem in place of some full compute | Preserve MLP and request rate after compute removal | Reinventing existing TMA; scheduler complexity |
| **B7: External alternatives** | Chiplet partitioning, active bridges, optical/remote fabric | Bound opportunity cost of irregular monolithic die | Different technology/cost assumptions |

**Important architectural nuance:** Removing an entire SM eliminates that SM's LSU, address-generation support, warp slots, outstanding miss capacity and access injection points. Reducing only Tensor Core count inside all SMs could preserve more memory-level parallelism—but save less area and complicate physical macro reorganization. Study both along a consistent budget.

**Memory-centric SM is a candidate, not a required conclusion.** If L2 banking or the NoC is the bottleneck, a revised memory partition/fabric may be simpler and better. If the remaining regular SMs can saturate the extra HBM, no new memory-serving engine is needed.

---

## 8. Research Challenges and Testable Optimization Targets

### Challenge A — Does the workload actually need bandwidth rather than SRAM capacity?

**Measurements:** FLOPs/token, HBM bytes/token, weights/KV/activation bytes, temporal reuse distance, cache miss behavior, achieved vs. theoretical DRAM throughput, page/cache interference, request-level parallelism.

**Optimization:** Select between extra stacks, bigger banks, weight pinning, or data reuse aware scheduling. **Falsify:** if bandwidth does not dominate the target cases, do not force an HBM-first story.

### Challenge B — Does compute trimming destroy memory throughput?

**Measurements:** issue/warp stall categories, LSU requests/cycle, active warps, MSHRs/outstanding requests, HBM channel occupancy, Tensor/INT utilization before/after removing SMs or pipes.

**Optimization:** keep load/store and address generation in a light SM, decouple a descriptor-driven request engine, or retain more ordinary SMs. Compare with existing Hopper TMA and ordinary asynchronous memory instructions [R1, R14].

### Challenge C — Can a notched die actually host the PHY/stack?

**Measurements:** count of placeable PHY blocks **not just perimeter**, edge/sidewall usable depth, corner/keep-out loss, HBM package placement, interposer track density/reach, reticle bounding-box occupancy, removed silicon vs. added IO silicon.

**Optimization:** adjust notch pitch/depth, coarse straight segments, PHY distribution among sides, L2/MC placement, or use chiplets/bridges if shape is counterproductive. The irregular-perimeter patent [R4] motivates possible shapes but **does not certify** these constraints for a commercial GPU.

### Challenge D — Where does extra HBM bandwidth get blocked inside the GPU?

**Measurements:** L2 request throughput by partition, DRAM-channel balance, memory queue stalls, NoC hop-bytes/flit contention, source injection bandwidth, MC scheduling, spatial locality.

**Optimization:** more L2 **banks/ports** rather than merely capacity; repartition by HBM channel; direct memory-partition crosslinks; memory request steering; widened/hierarchical NoC; locality-aware CTA mapping. All count against the die area and energy budget.

### Challenge E — Is the added HBM good for **energy**, not only latency?

**Measurements:** additional stack idle/active energy, MC/PHY power, leakage of deleted compute and added logic, NoC energy, runtime reduction, higher local thermal load and throttle effects.

**Optimization:** frequency/voltage/channel activation policies, right-size controller counts, select a different Pareto point if high stack counts raise J/token.

### Challenge F — What is the strongest simpler alternative?

**Controls:** Rectangular PHY rearrangement, high-clock same-5-stack configuration where physically/power feasible, more L2 banks without extra capacity, optimized existing TMA/DSM software; package changes via alternative chiplet/bridge schemes [R7–R9].

**Criterion:** New irregular geometry and memory-serving hardware must outperform the best **feasible, equal-budget** simpler solution. If they do not, a negative research result is still useful but the proposed novelty changes.

---

## 9. Research Methodology: Two-Layer Modeling + Targeted Cycle Simulation + Physical Screening

### Layer 1 — Fast workload/architecture/geometry DSE

- **LLMCompass [R2, R3]:** starting point for LLM operator performance, mapping and memory-channel parameter sweeps. Inspect original definitions of physical/active channels before extending the code.
- **ConfuciuX / DiGamma / ZigZag [R10–R12]:** methodological precedents for constrained compute/buffer co-optimization, mapping co-search, and modeling memory hierarchy.
- **New module A: PHY/shoreline feasibility:** polygon shapes, discrete macro placement, reticle, and package/HBM stacking checks.
- **New module B: bandwidth-delivery model:** HBM → PHY/MC → memory partition/L2 → NoC → requesting SM, accounting for finite queues, load balance, and request issue throughput.
- **New module C: cost/energy:** per-block area models, stack/module cost and power, package footprint, uncertainty envelopes.
- **Search:** first simple sweeps and branch-and-bound/CP-SAT for feasibility; then constrained Bayesian search/genetic search only if needed. Use area/cost-matched candidates, not wishful infinitely scalable bandwidth.

### Layer 2 — Detailed validation of a small Pareto subset

- **Accel-Sim 2.0 H100/H200 simulator [R13]:** cycle-level model to validate compute downsizing, L2 banks, memory partitions, NoC, throughput, and stall transfer. Check currently supported knobs and actual trace availability before any claims.
- **CACTI [R15] and power estimation tools such as AccelWattch [R16]:** rough SRAM/router/cache area and power; validate against known GPU configurations. No publicly accessible framework yields signoff-quality H100 4N PHY macro or irregular-die implementation from a configuration switch.
- **Custom C++/SystemC/RTL blocks:** minimal request engines, queueing/credit control, inter-partition routing, or L2 bank selection, implemented **only after trace analysis shows their need**.
- **Physical proxy:** macro floorplanning and polygon/interposer screening with clearly stated DRC assumptions; report uncertainty and, if no PDK, explicitly call this a **proxy** rather than a verified manufacturable layout.

### Existing lab hardware

The available local GPU is an **RTX 5080**, which can profile real LLM decode behavior, collect performance counters and energy estimates (subject to permissions), and validate low-reuse/high-bandwidth hypotheses on a Blackwell consumer GPU. It **cannot generate native H100 SASS traces**, and its GDDR7 memory system is not H100 HBM. H100-targeted cycle simulation must use verified compatible H100/H200 traces or obtain such traces from accessible Hopper hardware (e.g., short cloud rental). Do not silently replay RTX 5080 SASS as H100 SASS.

---

## 10. Executable Research Plan and Phase Gates

### Phase 0 — Research baseline, evidence inventory, and simulator bring-up

**Tasks**
1. Inventory local RTX 5080, driver/CUDA, Nsight, CPU/RAM/storage, permission and power-measurement capabilities.
2. Clone/pin LLMCompass and Accel-Sim versions. Read H100 memory-channel settings and H100 trace requirements directly from source/docs.
3. Create a reference table with URLs, DOI/date, what each source *actually* establishes, and what remains hypothesis.
4. Decide a first H100-like **5-stack SXM** config and a separately labeled **full 6-stack GH100** config, without conflating them.

**Deliverables:** `environment_inventory.md`, `reference_matrix.md`, `simulator_capabilities.md`, reproducible version/commit lockfile.  
**Gate:** Reproduce at least one LLMCompass inference point and demonstrate a simulator smoke test (or document the exact blocker).

### Phase 1 — Reproduce low-reuse decode behavior on the real RTX 5080

**Tasks**
1. Benchmark a small set of dense model shapes/precisions; split GEMV-like linear layers vs. attention/KV accesses vs. prefill.
2. Sweep batch 1, 2, 4, 8, 16 as memory allows; vary context length and quantization.
3. Measure HBM-like external-memory (GDDR7) throughput, cache hit/miss, SM/Tensor utilization, memory-level parallelism proxies, time/token, J/token; clearly label GPU differences.
4. Extract or derive bytes/token by data class. Validate on one hand-written GEMV microbenchmark and one real model decode.

**Deliverables:** `baseline_5080.csv`, plots, scripts, profiling configuration.  
**Gate:** Identify and reproduce at least one bandwidth-sensitive decode operating point; if not, modify workload or scope honestly.

### Phase 2 — Fast model: SRAM-vs-HBM sensitivity and roofline

**Tasks**
1. In LLMCompass or an independently checked spreadsheet/model, sweep SM count, compute throughput, cache size, memory bandwidth and memory channel count.
2. Compare **extra SRAM capacity** vs. **extra L2 banks/bandwidth** vs. **extra HBM stack bandwidth** with the same approximate reclaimed compute *logic* area.
3. Vary activation/weight reuse assumptions and measure an *optimistic bound* vs. policy-realistic performance.
4. Mark the thresholds at which L2, NoC or remaining compute takes over as the bottleneck.

**Deliverables:** `roofline_and_reuse_model.py`, `resource_sensitivity.csv`, hypothesis decision chart.  
**Gate:** Show whether off-chip bandwidth, rather than extra SRAM, has higher marginal value for the chosen workloads. A non-HBM conclusion is allowed.

### Phase 3 — Geometry-and-shoreline feasibility model

**Tasks**
1. Make regular rectangle, longer rectangle, same-rectangle/PHY-relocation, stepped, notched, tabbed candidate polygons.
2. Input actual or explicitly **parametric** PHY widths/depths, keep-outs, reticle field, HBM footprint, and routing reach; sweep unavailable physical data over ranges.
3. Solve discrete PHY macro placement; then HBM stack package placement and routing viability (screening model).
4. Record `raw_perimeter` vs. `usable_phy_count` vs. `attachable_hbm_stack_count`, including rejected designs.
5. Compare a straight-sided compute-reduced design before treating irregularity as essential.

**Deliverables:** `geometry_model/`, shape plots, `feasibility.csv`, `unknown_physical_parameters.md`.  
**Gate:** Find at least one **conservative, parameter-robust** stepped/notched layout that increases *attachable* HBM or identify why it does not; do not claim a win from perimeter alone.

### Phase 4 — Naive compute-to-bandwidth GPU strawman

**Tasks**
1. Start with **H100-like 5-stack** baseline and **hypothetical 6-stack** design (with appropriate MC/PHY/L2-bank allocation).
2. Trim only compute lanes vs. whole SMs, measuring retained warp/LSU/MLP capacity.
3. Under die and package budgets, add the memory infrastructure without assuming free PHY, router or SRAM area.
4. Calibrate fast model against cycle-level candidates and measure where nominal bandwidth fails to become delivered bandwidth.
5. If examining 7+ stacks, label as **new hypothetical package and memory-interface architecture** rather than GH100 activation.

**Deliverables:** `strawman_results.csv`, performance/energy/area decomposition, stall waterfall.  
**Gate:** Quantify real benefit and identify the dominant new bottleneck; if there is no bandwidth gain, return to the geometry/packaging model.

### Phase 5 — Targeted microarchitecture (choose from evidence)

**Possible choices**
- **Memory-partition/L2-bank rebalancing** for more controller channels;
- **NoC/hierarchical interconnect** or bank-local paths when extra channels cause congestion;
- **Minimal AGU/LSU request engine / memory-serving SM** only if trimmed compute has lost critical issue/MLP capacity;
- **Channel-aware request distribution**, scheduler, and queueing changes if HBM stacks are unbalanced;
- **Adaptive channel/stack power states** if added HBM capacity increases idle energy.

**Deliverables:** specification, implementation, unit tests, bounded analytical PPA, ablation.  
**Gate:** Improvement over the best feasible equal-budget **naive** candidate, not only over an untuned stock GPU.

### Phase 6 — Physical/cost uncertainty and competing strategies

**Tasks**
1. Sweep PHY macro dimensions, required channel shoreline, HBM stack footprint, reticle margin, NoC area/power, package cost and interface energy.
2. Evaluate risks of irregular-die fabrication/corners and note what requires industrial verification.
3. Compare a simplified active-bridge or chiplet alternative conceptually or via a limited model using related work [R7–R9].
4. Conduct uncertainty analysis: **Which conclusions survive 20–50% model uncertainty?** Do not assert universal numeric tolerances; use appropriate empirical ranges and justify them.

**Deliverables:** `sensitivity_and_feasibility.md`, uncertainty bands, rejected-design registry.  
**Gate:** Distinguish robust wins from parameter-sensitive hypotheticals.

### Phase 7 — End-to-end DSE, ablations, and architecture principles

**Required comparisons**
- Stock H100-like 5-stack;
- Compute-only reduction;
- Compute → SRAM (strong baseline);
- Compute → rectangular PHY/MC/HBM (strong baseline);
- Compute → stepped/notched shoreline + PHY/MC/HBM;
- The above + L2/NoC rebalancing;
- The above + optional memory-serving SM/engine if justified;
- Selected chiplet/bridge/remote-I/O alternatives when comparable.

**Required outputs:** Pareto frontiers for J/token, latency, package cost/footprint, power; area distribution; HBM stack count vs. *delivered* bandwidth; active PHY sites; sensitivity to model/batch/context; decode win vs. prefill regression.  
**Gate:** Publishable evidence for **why** the chosen shape/architecture wins, not merely a best configuration.

### Phase 8 — Reproducible research artifacts and paper

Produce scripts, environment lockfiles, simulator patches, result manifests, architecture spec, figures with provenance, performance/area/energy model assumptions, negative results, and a manuscript with precise claims/limitations.

---

## 11. The Experiments That Will Make or Break the Paper

| ID | Question | Key plot / result | Potential falsifier |
|---|---|---|---|
| E1 | Is SRAM capacity low-value for **this** decode workload? | Extra MB SRAM vs. HBM bytes/token and J/token | Sustained cache hits make SRAM clearly superior |
| E2 | Does compute trim preserve HBM-demand generation? | SM/TC removal vs. outstanding requests and HBM GB/s | LSU/MLP falls faster than useful compute is removed |
| E3 | Is shoreline actually limiting HBM stack count? | Geometry → **feasible PHY macros** → **routed stacks** | Package/thermal/routing blocks all added stacks |
| E4 | Is a notch materially better than a regular rectangle? | Iso-reticle/package comparison | Straight-sided PHY layout provides the same stack count |
| E5 | Does nominal HBM BW reach the compute arrays? | Nominal vs. delivered BW, L2/NoC stalls | Internal paths saturate and are too expensive to widen |
| E6 | Does the new microarchitecture earn its cost? | Same-area naive vs. optimized architecture ablation | Additional control/fabric consumes more energy/area than saved |
| E7 | Does energy/token improve at useful latency? | E/token–latency Pareto, prefill guardrail | More HBM idle power dominates run-time savings |
| E8 | Are conclusions robust to uncertain PHY/package costs? | Confidence/sensitivity maps | Gains vanish for credible parameter alternatives |

---

## 12. Deliverables, Repository Layout, and Agent Rules

Suggested reproducible directory layout:

```text
shoreline-gpu-research/
  README.md
  docs/
    research_plan.md
    reference_matrix.md
    assumptions_and_unknowns.md
    experiment_log.md
    simulator_capabilities.md
  configs/
    workloads.yaml
    gpu_baselines.yaml
    phy_package_constraints.yaml
  workloads/
    gemv/
    llm_decode/
  profiling/
    5080/
    h100_if_available/
  models/
    llmcompass_extension/
    geometry_floorplan/
    performance_bandwidth/
    energy_cost/
  simulation/
    accel_sim_patches/
    noc_models/
  results/
    raw/
    normalized/
    figures/
  tests/
    geometry_unit_tests/
    cost_model_tests/
    simulator_validation/
  scripts/
    run_baselines.sh
    sweep_candidates.py
    generate_pareto.py
```

Rules:

1. **Label provenance:** `REAL_GPU_MEASURED`, `PUBLISHED_SPEC`, `SIMULATED`, `ANALYTICAL_ESTIMATE`, `PARAMETRIC_ASSUMPTION`, or `ILLUSTRATIVE`.
2. **Never invent manufacturing numbers.** Missing PHY macro dimension, package die-shape rules, I/O density, power, routing limits, HBM placement, or 4N standard cell cost must be tracked as a *range or unknown*, not a precise manufactured fact.
3. **Preserve baseline correctness and fair comparison.** Compare at equal technology assumptions, memory capacity requirements, performance targets, logic budget, and separately reported package budget/cost.
4. **Do not fabricate cache hit rates or add memory channels without requests.** The model must capture actual demand, bank interleaving and interconnect bandwidth.
5. **Validate the H100-like model.** RTX 5080 measurements can validate qualitative decode behavior but not automatically an H100 HBM PHY or package geometry.
6. **Separate patent prior art from empirical demonstration.** Patent publications establish proposed structures; they do not prove energy improvements or commercial viability.
7. **Stop at phase gates if hypotheses fail;** propose evidence-driven alternatives and record negative results.
8. **Avoid irreversible/system-wide changes without authorization** (driver/firmware/power limit, privileged installs, large paid cloud allocations).

**First agent assignment:** Inventory machine and tools, reproduce one decode GEMV and one real decode case on the RTX 5080, reproduce one LLMCompass example, and create a *parametric* rectangular-vs-stepped PHY placement toy model. Do **not** install a complex physical design tool or modify Accel-Sim before generating those artifacts and reviewing their limitations.

---

## 13. How the Prior Art Fits—and Where Novelty Must Be Demonstrated

| Source | Already established by the source | Gap relevant to this proposal |
|---|---|---|
| NVIDIA Hopper [R1] | The actual SM, HBM-stack/controller/L2 configurations | Not a published optimization of irregular die shoreline for decode |
| LLMCompass [R2, R3] | LLM-centric performance/area DSE with channel parameters | Does not itself verify PHY placement, nonrectangular die, package routing |
| Morpheus [R17] | Idle cores' storage can extend effective GPU LLC for memory-bound apps | Not physical compute-to-HBM-stack and shoreline co-design; results do not generalize automatically to GEMV |
| ConfuciuX, DiGamma, ZigZag [R10–R12] | Compute/buffer/mapping DSE and search techniques | Not the proposed GPU-specific PHY/HBM shoreline feasibility coupling |
| Irregular die patent application [R4] | Tabbed/wavy/irregular I/O beachfront shapes as prior disclosed concepts | No verified GPU LLM decode energy-optimal physical architecture |
| Memory Fabric [R5] | Shoreline can constrain GPU memory scaling; alternative uses HBM-side D2D | Different data path; not a notched GPU die with extra conventional HBM PHYs |
| SiPAM [R6] | Shoreline as a resource for photonic memory-pooling I/O | Different physical I/O technology and optimization target |
| Memory-centric MCM GPU [R7] | Interposer bridge/fabric can rebalance off-chip bandwidth | Different multi-chiplet structure, no irregular monolithic shoreline requirement |
| Active bridge patent [R8] | A bridge can connect non-adjacent HBM to compute chiplets | Distinct alternative to adding conventional beachfront; requires fair cost/latency comparison |
| Chiplet-Gym [R9] | Package-aware PPAC optimization method | Does not by itself resolve monolithic irregular GPU die + HBM PHY placement |

**Novelty test:** If all candidate designs can be represented as “smaller GPU + faster DRAM” without a meaningful geometry feasibility model or new validated bottleneck-resolution mechanism, the result is likely an application-specific DSE rather than a strong architecture contribution. Conversely, show (a) that physical shoreline *changes which design is feasible or Pareto-optimal*, and (b) a hardware mechanism necessary for the selected design to realize its delivered-bandwidth gain.

---

## 14. References (verified public starting points)

**Architecture, baseline, DSE**

- **[R1]** NVIDIA, [*NVIDIA Hopper Architecture In-Depth*](https://developer.nvidia.com/blog/?p=45555) (2022). Official reference for full GH100 vs. H100 SXM: HBM stack, MC, L2, SM counts, die size, and TMA architecture.
- **[R2]** H. Zhang, A. Ning, R. B. Prabhakar, D. Wentzlaff, [*LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference*](https://doi.org/10.1109/ISCA59077.2024.00082), ISCA (2024). LLM performance, mapping and area-cost evaluation.
- **[R3]** Princeton University, [LLMCompass public repository — `docs/run.md`](https://github.com/PrincetonUniversity/LLMCompass/blob/main/docs/run.md), showing `memory_channel_physical_count` and `memory_channel_active_count`.
- **[R10]** S.-C. Kao, G. Jeong, T. Krishna, [*ConfuciuX: Autonomous Hardware Resource Assignment for DNN Accelerators using Reinforcement Learning*](https://www.microarch.org/micro53/papers/738300a622.pdf), MICRO (2020).
- **[R11]** S.-C. Kao et al., [*DiGamma: Domain-aware Genetic Algorithm for HW-Mapping Co-optimization for DNN Accelerators*](https://arxiv.org/abs/2201.11220), DATE (2022).
- **[R12]** [*ZigZag: Enlarging Joint Architecture-Mapping Design Space Exploration for DNN Accelerators*](https://doi.org/10.1109/TC.2021.3059962), IEEE Transactions on Computers (2021).
- **[R17]** S. Darabi et al., [*Morpheus: Extending the Last Level Cache Capacity in GPU Systems Using Idle GPU Core Resources*](https://arxiv.org/abs/2209.10914), MICRO (2022). Hardware/software cache extension with idle cores. Also examined from the user's supplied Morpheus PDF.

**I/O shoreline, physical die/package structure, memory fabrics**

- **[R4]** [*Electronic Device with Irregular Die Perimeter for Increased Beachfront Input/Output Capacity*](https://patents.justia.com/patent/20260182042), US patent application publication **US20260182042A1**, June 2026. A *proposal/disclosure*, not a demonstrated H100 GPU implementation.
- **[R5]** [*Memory Fabric: HBM Die-to-Die Interconnect for GPU Bandwidth Expansion*](https://doi.org/10.1109/LCA.2026.3721758), IEEE Computer Architecture Letters (2026). Uses spare HBM-base-die shoreline and D2D links; distinct from reshaping compute die.
- **[R6]** Z. Wu et al., [*SiPAM: Silicon Photonic Accelerated Memory Pooling Architecture*](https://doi.org/10.1109/MM.2026.3670009), IEEE Micro (2026). Compute-die shoreline with photonic I/O and memory pooling.
- **[R7]** [*Memory-Centric MCM-GPU Architecture*](https://doi.org/10.1109/LCA.2025.3553766), IEEE Computer Architecture Letters (2025). Bridge/interposer-centered memory bandwidth allocation.
- **[R8]** [*Non-adjacent Connection of High-bandwidth Memory Chiplets, I/O Chiplets, and Compute Chiplets through Embedded Logic Bridges*](https://patents.google.com/patent/US20260093634A1/en), US patent application (2026). Alternative geometry/connection approach; patent disclosure not an independently validated packaged GPU.
- **[R9]** K. Mishty and M. Sadi, [*Chiplet-Gym: Optimizing Chiplet-based AI Accelerator Design with Reinforcement Learning*](https://arxiv.org/abs/2406.00858) (2024). PPAC, placement, and packaging-aware accelerator optimization.

**Tools / model and simulator references**

- **[R13]** [Accel-Sim Framework](https://accel-sim.github.io/), including the 2.0 Hopper modeling description. Check trace requirements, validation range, and source version.
- **[R14]** NVIDIA, [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), for TMA, shared memory, clusters/DSM; existing functionality is a critical baseline.
- **[R15]** [CACTI](https://github.com/HewlettPackard/cacti), cache/SRAM area, access-time, and power modeling; process and PHY differences limit absolute physical precision.
- **[R16]** [AccelWattch paper and artifact](https://doi.org/10.1145/3466752.3480063), GPU power modeling; calibration required for nonstandard H100/package components.

---

## 15. Final Research Positioning and Claim Boundaries

**Working title:** *Shoreline-Aware GPU Co-Design: Trading Compute for Deliverable HBM Bandwidth under Die and Package Constraints for LLM Decoding*.

**Core research question:**

> **Under fixed logic-die, reticle, package/route, thermal, cost and latency constraints, when is it beneficial to sacrifice GPU compute resources to create physically usable HBM I/O shoreline, attach more HBM bandwidth, and redesign on-die memory/data-movement structures so that LLM decoding becomes more energy-efficient?**

**Proposed paper narrative (not prematurely asserted results):**

1. Show that decode's low-reuse weight-streaming regimes can give additional **HBM bandwidth** higher marginal value than equal-area cache-capacity increases.
2. Show whether **area-only DSE** produces apparently attractive, physically unplaceable HBM channel allocations, and whether a **shoreline-aware** optimizer identifies a different feasible Pareto frontier.
3. If physically feasible, introduce a **compute-trimmed, PHY/MC-aware GPU floorplan** with optional stepped/notched boundary that attaches more memory *without violating the chosen budgets*.
4. Measure the internal data-supply bottleneck caused by the new HBM system and propose only the needed **L2/NoC/request-generation microarchitecture**.
5. Report **delivered bandwidth**, J/token, latency, physical costs/uncertainties, and comparisons against simpler rectangular and alternative-package designs.

**Critical falsifiable alternative:** If irregular shoreline provides no extra attachable HBM within plausible package/PHY/reticle constraints, or if bridge/chiplet/rectangle solutions dominate, the correct result is to say so and redirect architectural effort to the winning feasible data-movement design. Do not design an irregular boundary merely to match the initial hypothesis.

