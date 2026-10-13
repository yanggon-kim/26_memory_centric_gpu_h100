# Research Handoff: Reallocating GPU Compute Area to On-Chip Memory and Data Movement for Energy-Efficient LLM Decoding

> **Status:** Research concept and executable investigation plan; **not** a validated architecture or experimental result.  
> **Audience:** A Linux-based research/engineering AI agent with access to an NVIDIA GPU, profiling tools, source code, and (potentially) architectural simulators.  
> **Primary objective:** Minimize **joules per generated token** for LLM decode under a realistic, fixed silicon-area budget and explicit performance/latency constraints.  
> **Design philosophy:** Preserve as much of an NVIDIA-like GPU execution and software model as practical; selectively downsize computation hardware, then redesign on-chip storage and data movement. Identify and address the *new bottlenecks* created by this transformation.  
> **Document origin:** Consolidates our prior research discussion and an examination of *Morpheus: Extending the Last Level Cache Capacity in GPU Systems Using Idle GPU Core Resources* (MICRO 2022; arXiv v2, 2023). Hypotheses and architectural proposals below are **ours** unless explicitly attributed to the paper or a cited reference.

---

## 0. Instructions to the Research Agent

Treat this document as a **research brief, not proof of correctness**. Your job is to establish evidence, reject weak hypotheses, and produce reproducible artifacts—not to manufacture a favorable result.

1. **First inventory the available environment:** GPU(s), driver, CUDA toolkit/runtime, profiler availability, CPU/RAM/storage, containers, privileges, simulator installations, accessible RTL/PDK libraries, and relevant model files. Record exact versions.
2. **Do not claim a commercial GPU's physical circuits or SRAM capacity have changed.** A real H100/B200 can be profiled and used for software proxies, but silicon-area reallocation and new links require modeling, RTL/prototyping, or simulation.
3. **Separate measured data from estimates and hypothetical results.** Label figures `measured-real-GPU`, `simulated`, `analytical`, or `illustrative`.
4. **Reproduce a strong baseline before modifying anything.** Prefer existing optimized decode kernels and Hopper memory-movement facilities as controls.
5. **Do not assume that extra SRAM improves LLM decode.** Prove a reusable working set and a plausible access path first.
6. **Do not silently modify system drivers, flash firmware, change cluster-wide configuration, or run disruptive benchmarks.** Request approval before privileged installation, power-limit changes, major downloads, or shared-GPU contention.
7. If the environment lacks an appropriate simulator or PPA model, produce a well-defined trace/analytical study with explicit limitations rather than pretending to have cycle-accurate or physical results.
8. Maintain a daily experiment log and a machine-readable result table. Include unsuccessful and negative results.

**First deliverable:** `environment_inventory.md`, `research_assumptions.md`, and a prioritized `experiment_plan.md`, grounded in the discovered hardware/software environment.

---

## 1. Research Motivation and First-Principles Reasoning

### 1.1 Observation: GPU compute is underutilized during some LLM decode regimes

NVIDIA server GPUs (e.g., H100, B200) devote substantial hardware to high-throughput arithmetic: Tensor Cores, CUDA/SIMT execution lanes, related issue/operand pipelines, and the storage/interconnect needed to feed them. However, **small-batch LLM decoding** often has low arithmetic intensity, repeatedly streams weights and/or attention KV data, and can be limited by memory bandwidth or latency. In such regimes, compute throughput can be overprovisioned relative to achievable data delivery.

Important distinction:

- **Bandwidth-bound:** DRAM/HBM bandwidth is near the attainable limit for the access pattern; peak compute capacity is unnecessary at that operating point.
- **Latency/MLP-bound:** Warps cannot hide memory latency or sustain enough outstanding memory requests; low compute utilization does **not** imply HBM is saturated.
- **Cache/NoC/issue-bound:** L2 bandwidth, memory partition balance, NoC contention, warp scheduling, memory instructions, or address generation can be limiting even when Tensor Core utilization is low.

A low Tensor Core utilization metric by itself does **not** establish removable compute area, available power savings, or cache reuse opportunities.

### 1.2 Hardware resources and expected behavior

| Resource | Role | Possible behavior in bandwidth-limited decode | Caution |
|---|---|---|---|
| Tensor Cores | Matrix multiplication | Underutilized, potential area-reduction candidate | Needed for prefill, larger batches, and GEMM-like decode |
| FP32/INT32 CUDA execution lanes | Arithmetic, indexing, control | Some throughput may be spare | **Do not indiscriminately remove INT/ALU capability** used for address generation |
| Warp schedulers / dispatch / scoreboards | Schedule instructions, track dependencies | Issue slots may be unused during stalls | Necessary for conventional SM semantics unless replaced |
| Load/store units (LSUs), coalescers | Generate/service loads and stores | May be busy even when arithmetic units are idle | Must preserve or redesign for sustained memory throughput |
| Register files | Operands, addresses, state | Occupied despite idle compute pipes | Occupancy, banking, port costs, and RF-to-SRAM tradeoffs matter |
| Shared memory / L1 | SM-local software-managed/cache storage | May be useful for tiles | Larger capacity does not automatically yield hits or remote access |
| L2 / last-level cache | Chip-wide sharing, buffering | May be busy and capacity constrained | Slice/bank/NoC placement matter |
| HBM + controller | Weight/KV transport | Can be bottleneck | Fixed external bandwidth can cap improvements |
| On-chip interconnect | Routes requests and data | Often a hidden cost | Extra sharing/multicast adds traffic/area/latency |
| TMA/async transfers, cluster sharing | Hardware data movement | Existing capabilities to beat | Baseline must include what hardware already provides |

The question is **not** "is an execution unit idle?" but "how many square millimeters and joules can be reallocated *without losing the compute, issue, address-generation, and memory-level parallelism that decode still requires*?"

### 1.3 Primary research question

> **Under a fixed GPU die-area and HBM-interface budget, how should compute pipelines, on-chip SRAM, memory-serving logic, and inter-SM interconnect be co-designed to minimize energy per token for a specified portfolio of LLM decode workloads, while meeting explicit throughput and latency requirements?**

This is a **microarchitectural bottleneck-discovery and optimization study**, not merely a design-space sweep with a larger cache.

### 1.4 Success criteria

- Primary: **joules/token**, including dynamic **and** static/leakage energy across runtime.
- Constraints: die-area parity, reasonable clock/timing assumptions, model accuracy unchanged, throughput target, and p50/p99 inter-token latency/TTFT where relevant.
- Secondary: tokens/s, tokens/s/mm², performance/W, HBM bytes/token, L2 and local SRAM hit rates, on-chip bytes/token and hop-bytes/token, occupancy, memory-level parallelism, and area/power breakdown.
- Results must be compared with **strong existing NVIDIA-like baselines** and with a simple equal-area SRAM-only strawman.

---

## 2. Scope and Non-Goals

### Preserve where feasible

- HBM interface and DRAM technology; no reliance on new 3D-stacked DRAM to justify gains.
- NVIDIA-like SIMT/CTA/warp programming and compute-SM execution path.
- Standard load/store, address translation, synchronization, and memory safety semantics.
- Conventional GPU global L2/cache/coherence behavior, unless a clearly specified mechanism changes it.
- Compatibility path for existing LLM kernels or an explicit transformation/compiler/command API when heterogeneous units are introduced.

### Research freedoms

- Reduce the number/width of Tensor Core pipelines and selected CUDA arithmetic lanes.
- Reallocate *physically freed* area to SRAM macros, tags, banking, queues, routers, interconnect, and controllers.
- Organize storage as SM-local, cluster-shared, distributed, or expanded L2.
- Introduce dedicated memory-serving engines, asymmetric SM roles, local links, multicast, request merging, and scheduling/prefetching mechanisms.
- Investigate compute-near-SRAM as a **stretch** concept if data movement cannot otherwise be reduced.

### Non-goals / avoid conflation

- Disabling an SM at runtime **does not recover its die area**.
- A software cache in idle registers **is not equivalent** to newly fabricated SRAM.
- An SM's local register/shared memory capacity is not freely usable as a globally addressable, coherent cache.
- A fully new chiplet/3D-memory accelerator is a different research direction. It can be related work, but not the central proposal.
- Speedup is **not guaranteed** just because another memory-bound application benefited from more LLC.

---

## 3. Two Related Architectural Paths

Both paths should be modeled rather than prematurely selecting one.

### Path A: Homogeneous, compute-trimmed SMs

Keep the **SM count and memory-execution machinery**. For example, use a Hopper-like SM organization as a reference and hypothetically remove a fraction of its Tensor Core capacity while retaining warp scheduling, LSU, address-generation support, and enough general arithmetic to run decode kernels. Spend the reclaimed area on additional local or shared SRAM and connectivity.

**Pros:** More compatible with existing execution and CTA placement; straightforward compute-vs-storage analysis.  
**Cons:** Smaller incremental per-SM SRAM budget than repurposing an entire SM; internal operand paths and clock distribution may need retiming; adding SRAM may not help streaming weights.

### Path B: Heterogeneous compute and memory-serving SMs

Do **not** delete entire SM tiles. Instead, create two types:

- **Compute SM:** retains the arithmetic throughput and normal CTA execution needed by LLM kernels.
- **Memory-Serving SM (MS-SM):** retains or replaces only the necessary load/store, address generation, request tracking, translation interface, buffering, synchronization, and data delivery machinery; reduces general compute/Tensor pipelines and repurposes area as SRAM banks and local-fabric ports.

An MS-SM supplies one or more nearby Compute SMs. The candidate organization is a **heterogeneous memory–compute cluster**, not a general-purpose all-to-all replacement for the GPU's fabric.

**Pros:** Specialization may reduce memory-control energy and enable SRAM with a deliberate sharing radius; opens clear inter-SM transfer and command scheduling questions.  
**Cons:** A producer hop can add latency; compute blocks cannot run arbitrarily on MS-SMs; request serving may require far more instruction/control/MLP machinery than anticipated; heterogeneous partition can reduce load balance and scheduling flexibility.

**Crucial early comparison:** Is Path B better than Path A or than normal SMs using already available distributed shared memory/TMA multicast? If not, retain Path A and drop architectural heterogeneity.

---

## 4. Related Work and What It Does *Not* Establish

### 4.1 Morpheus — the closest conceptual baseline

**Source:** Sina Darabi et al., *Morpheus: Extending the Last Level Cache Capacity in GPU Systems Using Idle GPU Core Resources*, MICRO 2022, [arXiv:2209.10914](https://arxiv.org/abs/2209.10914). A copy was supplied as `Morpheus.pdf` in the original conversation; make it available to the research agent separately if detailed figures are needed.

**Paper-supported findings** (refer to the paper's printed page numbers):

- **pp. 1–3:** On 14 *non-LLM* memory-bound workloads modeled on RTX 3080-like hardware, about **56% of SMs on average** suffice to reach saturation, motivating reuse of idle SMs. An idealized 4x LLC experiment gives ~**1.57x mean** performance across the evaluated workloads.
- **pp. 3–9:** Hardware/software co-design. SMs dynamically use **compute mode** or **cache mode**; in cache mode a helper kernel reuses **register file, L1 cache, and shared memory** to construct an **extended last-level cache**. Hardware **Morpheus controller** routes requests and uses Bloom-filter hit/miss prediction. It is **not purely software**.
- **pp. 10–12:** Chosen mixed register-file/L1 implementation per cache-mode SM: **328 KiB** useful extended LLC, **185 ns** average access latency, **34 GB/s** average access bandwidth, **61 pJ/byte** in the authors' characterized setup.
- **p. 11:** Register-file extended LLC with a specific 48-warp setup delivered ~**37 GB/s** with the network; an **idealized no-network** experiment gave **290 GB/s** (~7.8x). This identifies a **data movement/access-path bottleneck**, not guaranteed gains from simply wiring adjacent SMs.
- **pp. 12–14:** RTX 3080-derived Accel-Sim/AccelWattch modeling, informed by experiments on a real RTX 3080. On 14 memory-bound GPU benchmarks, the full proposal reports **39% mean performance** and **58% mean performance/W** improvement over the respective baseline; **not** measured as a fully fabricated Morpheus GPU and **not** an LLM decode result.
- **p. 14:** On-chip network injection/load grows by **97%**, with average network latency up **7%**, yet the studied network did not saturate. The authors report LLC MPKI down **47%** versus their improved baseline.

**Morpheus demonstrates:** Idle compute capacity can coexist with cache-reuse opportunities and network costs in some GPU workloads.  
**Morpheus does not demonstrate:** Physical removal of Tensor Cores; area-matched replacement by dedicated SRAM; H100/B200 performance; or guaranteed performance gains in LLM decode.

**Important distinction:** Morpheus reuses *existing* per-SM storage at runtime; our central proposal **changes the physical area allocation and the topology/data movement path**. Morpheus is both motivation and a baseline: compare against a modeled/adapted Morpheus-like cache-mode organization where feasible.

### 4.2 Other research and hardware features to inspect

The following came up in our discussion. **Verify bibliographic metadata and the exact claims from primary papers before asserting novelty.** This document does not establish an exhaustive literature search.

| Work/feature | Why relevant | Gap relative to proposed study |
|---|---|---|
| [LLMCompass (ISCA 2024)](https://doi.org/10.1109/ISCA59077.2024.00082) | LLM-specific accelerator and compute/memory design-space modeling | Not identical to a concrete NVIDIA-like SM compute-to-SRAM and NoC co-design |
| [Adaptive Memory-Side Last-Level GPU Caching (ISCA 2019)](https://research.nvidia.com/publication/2019-06_adaptive-memory-side-last-level-gpu-caching) | Cache placement/topology and NoC-energy tradeoffs | Does not necessarily fund the cache with removed arithmetic execution circuits |
| ChronoStack (previously discussed) | An H100-inspired design involving 3D-stacked memory and compute/memory tradeoffs | Different primary technique; independently retrieve and verify exact claims |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Thread-block clusters, distributed shared memory (DSM), memory-order/synchronization semantics | Any inter-SM sharing proposal must improve on existing capabilities |
| [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) | TMA, cluster execution, async transfers and multicast | Generic DMA or simple cross-SM multicast alone is not novel |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | Optimized GEMM/attention, warp specialization, producer–consumer patterns | Need hardware gain beyond an optimized software pipeline |
| [Accel-Sim](https://github.com/Accel-Sim/accel-sim-framework) + AccelWattch | Candidate simulation/power infrastructure | Verify GPU-generation support, memory-system accuracy, and architectural extensibility |

**Novelty caution:** Existing Hopper hardware has block clusters, DSM and TMA multicast. Novelty requires showing that **heterogeneous memory-serving hardware, physical area reallocation, custom SRAM hierarchy, data movement topology, and decode-specific scheduling** outperform strong implementations of the available mechanisms.

---

## 5. Critical Technical Challenges and Optimization Targets

### C1. Does the extra SRAM actually eliminate off-chip traffic?

**Failure mode:** Large weight and KV working sets swamp the added SRAM; streaming accesses have little reuse before eviction. A BF16 7B-parameter model is roughly **14 GB in weights alone** (illustrative, excluding other state). Adding **100 MiB** does not cache the full model. With plain LRU streaming, some or all added cache lines may see no future hit.

**Targets:**

- Trace **per-tensor, per-layer, per-head, per-token** access/reuse distances, including batching and scheduling.
- Distinguish **temporal** reuse (across steps), **spatial/SM-shared** reuse (concurrent consumers), and **producer–consumer** forwarding.
- Evaluate **pinning/persistent residency** of high-value weight slices and shared-prefix KV pages, not only LRU.
- Model **GQA/MQA KV-head reuse**, shared-prefix requests, grouped decode, and weight-tile multicast separately from one-token batch-1 GEMV.
- Profile or instrument whether existing L2/DSM/TMA already captures the reuse; count **incremental** savings only.

**Go/no-go gate:** Find a meaningful, reproducible theoretical upper bound on HBM bytes/token saved or an independent on-chip movement/energy advantage. If no such opportunity exists, redesign the research target or abandon cache enlargement as the central mechanism.

### C2. SRAM capacity is not SRAM throughput

**Failure modes:** Bank conflicts, insufficient ports, low MSHR/queue capacity, tag lookup latency, high wire delay, occupancy impact, and hotspots when many SMs access a small cluster store.

**Targets:** Banked SRAM; partition/placement policies; address-to-bank interleaving; selective replication; access prioritization; pipelined tag/data paths; a two-level local/global memory hierarchy; measured/modeled bandwidth per mm² and pJ/byte.

**Question:** For the observed request-size distribution, is another MB of SRAM useful, or would another bank, port, request queue, or link produce more benefit?

### C3. Inter-SM movement can erase SRAM savings

**Failure modes:** New remote requests traverse a long round trip, duplicate traffic and buffers; producer funnels become bottlenecks; multicast saves HBM bytes but uses chip-wide wire energy.

**Targets and baselines:**

- Existing L2 + normal NoC.
- Hopper DSM/TMA multicast and software pipeline where accessible.
- Cluster-local direct neighbor links, ring, crossbar, or hierarchical local fabric.
- **Selective** multicast to actual consumers; address-based merging at a point *before* existing L2/MSHR would have merged requests.
- One-hop push/prefetch versus pull/on-demand access; flow control, credit queues, backpressure, and deadlock avoidance.

**Metrics:** on-chip **hop-bytes/token**, injection/throughput, queue occupancy, average/p99 request latency, bandwidth per source/destination, pJ/byte including drivers, repeaters, buffering, and router costs.

**Hypothesis to test (not presume):** A local fabric helps when requests repeatedly share data among spatially close consumer SMs and the avoided global traffic outweighs added local links/coordination.

### C4. A minimal memory-serving SM still needs an execution/control machine

**Failure modes:** Removing all CUDA/INT lanes breaks pointer arithmetic, indexing, descriptor setup, branches, atomics, scatter/gather, or instructions required by the memory-serving software. Aggressively shrinking scheduling/scoreboards reduces memory-level parallelism (MLP), leaving HBM underutilized.

**Candidate minimal engine:**

- Command/descriptor queue and limited control/branch capability.
- Small **address-generation engine (AGE)**: base + offset/stride, indexed gathers, page-table or KV-block-table lookups where applicable.
- LSUs / coalescing, outstanding-request trackers, response buffers, translation and protection interface.
- Limited integer operations (and optional atomics), barriers/events, error/completion handling.
- Dedicated DMA/TMA-like path with explicit descriptors for regular transfers; irregular-page address engine for data TMA cannot directly express without setup.

**Key comparison:** Full SM issuing load/store instructions vs reduced SIMT memory-only SM vs descriptor/DMA-like MS-SM. Evaluate area, power, **maximum memory requests per cycle**, address-pattern coverage, programmability, and synchronization correctness. Do not assume a new AGE is inherently better than existing Hopper TMA.

### C5. Producer–consumer scheduling and correct data delivery

**Failure modes:** Data arrives before buffer space exists, arrives too late and stalls Tensor Cores, or is evicted while consumers are still using it. Dedicated MS-SMs create load imbalance and alter CTA placement.

**Targets:** Double/triple buffering; prefetch distance control; ready/release barriers; credit-based queues; dependency-driven producer–consumer scheduling; cluster-local CTA placement; layout/memory-residency-aware work assignment; safe handoff for inter-SM shared data.

**Correctness:** Track ownership, visibility, fence ordering, atomics, remote shared-memory lifetime, and interactions with CUDA block-cluster semantics. A simulated performance gain is invalid if data races or incoherent cache state are implicitly assumed away.

### C6. Area and power costs are not free

**Failure modes:** Removed Tensor Core gates free less usable floorplan area than assumed; SRAM tags, ECC, ports, decoders, clock tree, routing and NoC take substantial area; added SRAM leaks power; lower compute throughput extends execution time, increasing background energy; removing pipelined compute may affect frequency/timing.

**Targets:** Physically informed SRAM macros and memory-controller estimates; sensitivity to process node, macro density, aspect ratio, ports, ECC; wire/router energy; retained arithmetic and control cost; power-gating assumptions; area-equivalent and iso-frequency comparisons where justified.

**Area accounting:**

\[
A_{\mathrm{new}} = A_{\mathrm{original}} - A_{\mathrm{removed\ compute}} + A_{\mathrm{new\ SRAM\ macros}} + A_{\mathrm{tags/ECC}} + A_{\mathrm{controllers}} + A_{\mathrm{NoC/links}} + A_{\mathrm{extra\ physical\ overhead}}
\]

Require \(A_{\mathrm{new}} \leq A_{\mathrm{original}}\) (or a clearly stated tolerance). Estimate *usable* incremental SRAM, not a fictitious conversion of every removed mm² into ideal bit cells.

### C7. Successfully reducing memory traffic can create a compute bottleneck

Use a first-order roofline estimate:

\[
T_{\mathrm{step}} \gtrsim \max\left(\frac{F}{P_{\mathrm{effective}}},\ \frac{B_{\mathrm{HBM}}}{W_{\mathrm{effective}}},\ T_{\mathrm{NoC}},\ T_{\mathrm{dependencies}}\right)
\]

where \(F\) is operations/step, \(P\) effective compute throughput, \(B\) off-chip bytes, and \(W\) effective HBM bandwidth. This is a **lower-bound conceptual model**, not a validated predictor.

As SRAM hits improve, the bottleneck may move to Tensor Core throughput, address issue, L2 bandwidth, cluster fabric, or synchronization. Therefore sweep **compute fraction × SRAM size × banking × NoC capacity**, rather than simply maximizing SRAM.

---

## 6. Workload Portfolio and Questions to Measure

Do not lump all LLM decode kernels into one behavior.

| Workload segment | Expected patterns | Questions |
|---|---|---|
| Batch-1 dense linear/GEMV | Streaming model weights, low activation reuse | Does added SRAM help at all? Would pinning selected weights help? |
| Larger-batch decode GEMM | Weight reuse across requests, more compute demand | How soon does removing Tensor Cores become harmful? |
| GQA/MQA decode attention | Multiple query heads may share KV heads | Does cache/multicast remove traffic beyond the optimized baseline? |
| PagedAttention / irregular KV | Indexed KV blocks, long contexts | Can an AGE and clustering reduce pointer/translation overhead? |
| Shared-prefix / continuous batching | Cross-request KV reuse, scheduling effects | Can co-location and persistent SRAM residency improve hit rate? |
| Prefill and mixed serving | High arithmetic demand and interference | What is the performance/latency cost when deployment is not decode-only? |

**Sweep at minimum:** models (small, medium, larger subject to available GPU memory); BF16/FP16 and representative quantized weights if implemented correctly; batch size (1, 2, 4, 8, 16, 32+ as feasible); context length (short to long); GQA/attention configurations; request-sharing/prefix scenarios; decode-only versus mixed prefill/decode.

Record architectural and software assumptions: model architecture, precision, KV layout/paging, tensor parallel degree, tensor/core utilization, kernel implementation, batching policy, prefill inclusion, and warmup.

### Upper-bound thought experiment

Suppose only \(C\) bytes of new persistent SRAM can pin weights from a model that reads \(W\) bytes of weights per decode step. For batch-1 sequential weight streaming, an **optimistic** steady-state upper bound from these pinned weight bytes alone is roughly \(C/W\) of weight bytes saved per step if those bytes would otherwise be fetched from HBM each step. This ignores tags, competing consumers, and existing L2 hits. It is **not** a bound for KV, activation, inter-SM sharing, or changed scheduling. Such elementary accounting should be completed **before** detailed RTL/NoC work.

---

## 7. Strawman Designs and Controlled Comparisons

Use the following explicit design ladder so each bottleneck can be attributed to a specific modification.

| ID | Design | What changes | Main question |
|---|---|---|---|
| **B0** | Stock-like H100-class GPU | Original compute, caches, NoC | What exactly limits each decode regime? |
| **B1** | Best software-only baseline | Tuned kernels; TMA/DSM/multicast/warp specialization as supported | What can existing hardware achieve? |
| **S1** | **Naive compute-to-SRAM strawman** | Remove a fraction of arithmetic pipelines; add equal-area SRAM; unchanged interconnect/scheduling | Why is a larger SRAM insufficient? |
| **S2a** | SM-local SRAM | SRAM attached to each trimmed SM | How much local reuse exists? |
| **S2b** | Expanded shared L2 | Same SRAM *area budget*, global banked allocation | How much sharing is worth global movement? |
| **S2c** | Cluster-shared SRAM | Same area budget, regional access | Is sharing radius critical? |
| **H1** | Memory-serving SM + existing NoC | Minimal memory engine, full compute neighbors | Is heterogeneity viable without new links? |
| **H2** | H1 + local links / multicast | New cluster data fabric | Is remote-data movement a bottleneck and can we reduce it? |
| **H3** | H2 + data-placement/scheduling | Reuse-aware pinning, prefetch, CTA placement | Is co-design better than hardware alone? |
| **OPT** | Joint area/energy optimized design | Tune compute/SRAM/NoC/engine fractions | Best Pareto frontier subject to constraints |
| **ORACLE** | Ideal SRAM hit / zero-penalty transport (diagnostic only) | Impossible upper bound | What improvement ceiling exists? |

**Mandatory fairness:** Compare the same total **die area**, operating point, GPU/HBM interface, correctness, and workload. Include controller/tag/links cost. Different layouts may have different achievable clocks: record and compare both iso-frequency sensitivity and plausible timing-constrained values.

**Important design separation:** `S1` tests whether physical compute-to-SRAM reallocation is promising; `H1–H3` test whether **functional disaggregation, inter-SM data movement, and scheduling** create additional advantage. Do not make H3 look good merely by giving it more SRAM than controls.

---

## 8. Step-by-Step Experimental Roadmap

Each phase contains **actions**, **artifacts**, and a **decision gate**. Conduct phase 1 even if the simulator cannot initially model H100 exactly.

### Phase 0 — Environment discovery and feasibility

**Actions**

1. Inventory hardware, OS, driver, toolkit, and tooling. Example *read-only* commands:
   ```bash
   uname -a
   cat /etc/os-release
   nvidia-smi -L
   nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version,power.limit --format=csv
   command -v nvcc || true
   nvcc --version || true
   command -v ncu || true
   command -v nsys || true
   command -v python3 || true
   df -h .
   free -h
   ```
   `nvidia-smi` query fields vary by driver: fall back to `nvidia-smi -q` as needed. Query profiler metric names rather than assuming portability across GPU generations.
2. Establish GPU exclusivity/concurrency constraints, available storage, network availability, permissions, and license/privilege restrictions.
3. Check available decode implementations (PyTorch, vLLM, TensorRT-LLM, llama.cpp, custom CUDA, CUTLASS), simulator checkouts, and trace/profiler permissions.
4. Read available simulator documentation; verify supported compute capabilities and whether new SM types, memory paths, SRAM capacity, or topology can actually be modified.
5. Create a version-locked reproducibility manifest and a lean test workload.

**Artifacts:** `environment_inventory.md`, `versions.json`, `reproduce_baseline.sh`, `experiment_plan.md`.  
**Gate:** At least one real-GPU profiling route and one credible architectural-model route exist, or limitations are explicitly documented with an alternative plan.

### Phase 1 — Establish optimized LLM decode baselines

**Actions**

1. Choose a model that fits comfortably in available HBM and can reproducibly run decode.
2. Measure tokens/s, batch-normalized decode latency, full-system and GPU power, HBM traffic proxy/counters, L2 metrics, SM activity, Tensor Pipe utilization, LSU traffic, and warp-stall reasons. Distinguish TTFT/prefill from decode.
3. Characterize multiple batch/context/head layouts and identify at least **two qualitatively different bottlenecks** (e.g., weight-bandwidth bound and KV/NoC or latency bound).
4. Run Nsight Systems for timelines and Nsight Compute for individual kernel bottlenecks, when permitted. Confirm if profilers change timings; never report profiled latency as unperturbed latency without checking.
5. Build existing-feature baselines: optimized attention, cluster sharing/DSM, TMA/multicast where supported, and kernel fusion or prefetch techniques.

**Artifacts:** `baseline_results.csv`, profiler reports, `kernel_inventory.md`, `bottleneck_classification.md`.  
**Gate:** Clearly identify which decode kernel(s) and data tensor(s) merit new SRAM/fabric hardware.

### Phase 2 — Trace reuse, bound potential gains, and reject weak hypotheses early

**Actions**

1. Collect/derive tensor-level and where possible memory-request-level access traces with timestamp, SM/CTA, address/range, tensor identity, and reuse-sharing group. Use lightweight instrumentation if detailed traces become prohibitive.
2. Compute reuse-distance / working-set curves at candidate SRAM capacities; estimate existing L2 contribution and added-SRAM **incremental** hit rate.
3. Model ideal policies: LRU, pinning, selective bypass, shared-prefix KV residency, and time-/consumer-aware placement.
4. Derive optimistic lower bounds on HBM bytes/token, requests/token, and local/remote byte counts for each scenario.
5. Test idealized multicast/request merging against what existing L2 and TMA already achieve.

**Artifacts:** `trace_schema.md`, trace samples, `reuse_distance.csv`, `traffic_bound_curves.csv`, `opportunity_report.md`.  
**Gate:** At least one kernel/regime displays sufficient addressable reuse or data movement energy/latency opportunity to justify architectural complexity. If no: pivot to reducing on-chip movement cost, improving memory-level parallelism, or right-sizing compute without extra SRAM.

### Phase 3 — Implement the compute-to-SRAM strawman

**Actions**

1. Select a base architecture model (document discrepancies from actual H100/B200).
2. Parameterize Tensor Core/FP/INT throughput separately, not merely total SM count.
3. Preserve and explicitly parameterize LSU count, AGU capacity, warp scheduler/issue limits, register file, L1/shared, L2 banks, MSHRs, and HBM channels.
4. Add equal-area SRAM under alternative allocations: SM-local, cluster-shared, L2; account for tags and SRAM area/power rather than adding ideal megabytes.
5. Use LLM-specific traces or reduced kernels; first validate existing kernels and memory timing against the real-GPU baseline where feasible.
6. Sweep removal fractions (e.g., 0/25/50/75% of a **specified** compute pipeline), storage placement, batch, context.
7. Classify each new bottleneck: no reuse, NoC, SRAM banks, request queues, AGU, compute, occupancy, synchronization, leakage.

**Artifacts:** simulator patches, `s1_sweep.csv`, `strawman_bottleneck_report.md`, area/energy accounting.  
**Gate:** State **which mechanism**, not just more SRAM, is necessary to beat S1 under realistic assumptions.

### Phase 4 — Design the memory-serving execution engine

**Actions**

1. Trace load/store instruction dependence on integer arithmetic, scheduler, registers, synchronization, translation, and coalescing.
2. Define a minimum programming interface: `fetch_tile`, `gather_pages`, `multicast_tile`, `writeback`, `signal`, `wait`, and required semantics (abstract commands, not an asserted existing ISA).
3. Compare (a) stripped SIMT issue, (b) command/descriptor engine, (c) AGU+LSU+queues with limited ALU.
4. Test regular weights, strided tensors, paged KV, and irregular gathers independently.
5. Sweep queue depth, in-flight transactions, AGU throughput, buffer depth, and banks until memory requests/second and MLP match the useful original configuration.
6. Model retained area and energy. The aim is **minimal area per sustained useful byte/s**, not zero ALUs.

**Artifacts:** `memory_serving_isa.md`, `ms_sm_microarchitecture.md`, `engine_sensitivity.csv`, cost model or RTL microblocks.  
**Gate:** It is possible to remove enough compute/control area to pay for proposed SRAM/fabric **without** debilitating data supply.

### Phase 5 — Local fabric and inter-SM data-delivery experiments

**Actions**

1. Implement **existing NoC** remote-SRAM path as baseline; measure request and response hop count.
2. Model neighbor links, crossbar, ring, and hierarchical regional fabric at **same total network area** or provide cost/performance Pareto curves.
3. Explore pull, descriptor-push, multicast, request merging, and selective replication.
4. Add queues, arbitration, congestion, credit/backpressure, timing, deadlock avoidance, coherence/visibility mechanisms.
5. Tune compute-to-memory SM cluster ratio and producer-to-consumer locality; test when all-to-all global connectivity is unnecessary.
6. Evaluate full access/transfer energy, not only raw bandwidth.

**Artifacts:** `fabric_spec.md`, topology simulator/configuration, `noc_results.csv`, end-to-end per-token breakdown.  
**Gate:** New fabric beats existing network/DSM/TMA-capable baseline on **area-normalized energy** and does not violate latency targets.

### Phase 6 — Decode-aware placement and coordinated scheduling

**Actions**

1. Implement placement policies for weights, KV pages, temporaries, shared-prefix data, and staging buffers.
2. Couple memory-serving descriptors with consumer CTA/cluster placement and readiness barriers.
3. Compare local reuse, multicast, pinning, and near-consumer computation/fusion where appropriate.
4. Evaluate buffer stalls, consumer starvation, producer idle time, admission/backpressure, sharing degree, and QoS across continuous-batching traces.
5. Model scheduler metadata, state transitions, synchronization, and changes to kernel execution semantics.

**Artifacts:** `placement_policy.md`, `scheduling_policy.md`, per-workload ablations.  
**Gate:** The policy contributes a measurable gain **beyond** the proposed hardware alone and strong existing-feature software baselines.

### Phase 7 — Area/power/timing-informed optimization

**Actions**

1. Obtain a plausible technology and SRAM macro model; validate source and node assumptions. CACTI is suitable for rough exploration but should not be presented as signoff SRAM PPA.
2. Estimate/synthesize the small AGU/controller/NoC blocks using available open tooling or licensed flow; document limitations of extrapolating older libraries to modern GPUs.
3. Include tag/ECC, buffers, links, clock/control, wire/port overhead, and leakage.
4. Evaluate iso-area, and sensible frequency/timing sensitivity. Separate GPU core dynamic and static energy; estimate board/system energy only when supported by measurement.
5. Optimize or map the Pareto frontier over compute capacity, SRAM capacities/bank counts, MS-SM count, topology/bandwidth, and scheduling policies.
6. Ablate each contribution and stress-test edge cases (compute-heavy prefill, large batch, long context, little shared prefix, irregular accesses).

**Artifacts:** `ppa_assumptions.md`, `area_model.csv`, `energy_model.csv`, `pareto_results.csv`, plots and error bars/sensitivity bands.  
**Gate:** The final design dominates relevant baselines in energy/token for the **specified target regimes**, within an explicit area and latency envelope; otherwise characterize where/why it loses.

### Phase 8 — Research write-up and reproducibility package

Produce a clear experimental narrative: observation → naive equal-area strawman → observed new bottleneck(s) → one mechanism per bottleneck → ablation → area/energy/end-to-end results → limitations. Include a novelty review with verified citations and an artifact README that can recreate plots and model parameters.

**Final artifacts:** paper outline, evaluation scripts/configs, data dictionary, all raw/summarized outputs, deterministic plotting scripts, assumptions/limitations, and negative results.

---

## 9. Measurement Specification and Output Schema

One row per unique `(design, model, kernel, batch_size, context_length, precision, workload_pattern, seed, run_id)` with at least:

```text
run_id,design,classification,hardware_or_simulator_version,model,kernel,
batch_size,context_length,precision,kv_format,attention_type,
latency_ms,decode_tokens_per_s,joules_per_token,gpu_avg_power_w,
hbm_read_bytes_per_token,hbm_write_bytes_per_token,
l2_hit_rate,added_sram_hit_rate,local_bytes_per_token,remote_bytes_per_token,
noc_hop_bytes_per_token,noc_avg_latency_ns,noc_p99_latency_ns,
compute_utilization,tensor_pipe_utilization,lsu_utilization,
agu_req_per_cycle,outstanding_mem_reqs,sm_count,ms_sm_count,
compute_area_mm2,sram_area_mm2,noc_area_mm2,total_area_mm2,
dynamic_energy_j,static_energy_j,notes
```

Where a metric is unavailable, use **`NA` and a reason**, not a guessed value. Metric names and their physical meanings should be documented; the numerator/denominator of each rate must be explicit. For simulation, provide an independent calibration table against real-GPU counters and runtime where possible.

Energy per token should be evaluated consistently:

\[
E_{\mathrm{token}} = E_{\mathrm{HBM}} + E_{\mathrm{cache/SRAM}} + E_{\mathrm{NoC}} + E_{\mathrm{compute}} + E_{\mathrm{control}} + E_{\mathrm{static}} + E_{\mathrm{other}}
\]

For measured GPU-only power, use an explicit measurement window and idle-baseline policy; multiply average GPU power by time and divide by **generated decode tokens**, excluding warmup and stating whether prefill is included. Do **not** confuse performance/W with measured energy/token across workloads that have different scopes or token counts.

### Required plot set

1. Decode roofline/achieved memory bandwidth with evidence of bandwidth vs latency vs cache/NoC limiting factors.
2. Added SRAM size versus **incremental** HBM bytes/token saved and hit/reuse distance.
3. Compute area removed versus **usable** SRAM added, including metadata/NoC/control costs.
4. Energy/token and latency versus compute fraction for each SRAM placement.
5. Remote SRAM path latency, bandwidth, and hop-byte energy across topologies.
6. MS-SM minimal engine: area/energy versus sustainable memory requests/cycle.
7. Bottleneck migration map across batch size and context length.
8. Full-system pareto (energy/token versus decode latency) and contribution ablations.
9. Compare batch-1 weight-streaming vs GQA/paged KV vs shared-prefix vs large-batch decode.
10. Sensitivity to inaccurate physical SRAM/NoC estimates and simulator mismatch.

---

## 10. Practical Simulator/Tooling Strategy

There may be **no ready-made cycle-accurate open simulator reproducing H100/B200** with all Hopper features and useful full-LLM decode speed. Inspect this before spending weeks in simulator bring-up.

Suggested progressive approach:

1. **Real GPU:** Nsight Compute (`ncu`) and Nsight Systems (`nsys`) plus `nvidia-smi` telemetry; optimized CUDA/LLM kernels; measure actual achievable bottlenecks.
2. **Analytical trace model:** Python/C++ reuse-distance, capacity/bandwidth/queue, area/energy budgets; quick rejection of unsuitable data types or cache policies.
3. **GPU architecture simulator:** Accel-Sim/GPGPU-Sim-family if its available model can represent the target; calibrate known behavior and report Hopper-vs-model differences. No false claim of H100 fidelity.
4. **NoC model:** simulator-native NoC or independent network model (e.g., BookSim-like), with explicit link/arbiter/queue behavior and timing integration.
5. **Energy/area model:** AccelWattch, SRAM macro/CACTI estimates, standard-cell synthesis/RTL for new blocks as feasible; separate technology extrapolation uncertainty.
6. **Optional targeted RTL:** Prototype the AGE, command/scoreboard, buffer controller, or multicast router—not a complete NVIDIA SM RTL (which is not publicly available).

### Minimum viable first experiment

A credible fast-start can be performed before building the new GPU simulator:

- Benchmark a real decode kernel and establish its HBM and L2 behavior.
- Collect its tensor-level reuse trace.
- Compare **0 / 8 / 32 / 128 / 512 MiB** additional *idealized* SRAM capacity (purely analytical; **not** an assertion that such area is available).
- Compare LRU vs pinning vs shared-head/prefix-aware policies; calculate incremental HBM traffic savings.
- Add a simple, clearly labeled remote-access cost model (extra hop, local-bank service, 1-to-many sharing) and test break-even conditions.
- Only after observing credible opportunity, impose the **realistic freed-compute area** and begin simulator/RTL work.

The fastest route to a strong paper is to **find and quantify the decisive bottleneck early**, even if that result disproves the initial SRAM hypothesis.

---

## 11. Open Questions and Falsifiable Hypotheses

### H1 — Compute right-sizing

Some decode configurations can tolerate substantially fewer Tensor Core resources without throughput loss **provided LSUs, address generation, and MLP are preserved**. Test batch/context thresholds where this fails.

### H2 — Reuse-aware storage

A carefully placed, banked and managed SRAM pool yields higher energy benefit than the same silicon area distributed naively among SMs or appended to global L2. Test locality and reuse sensitivity.

### H3 — Memory-serving specialization

A dedicated command/AGE/LSU memory-serving engine can deliver the useful bytes/second of a full SM at lower area/energy for a constrained class of data patterns. Test irregular KV, address indirection, and request concurrency.

### H4 — Local data-sharing topology

A regional fabric with selective multicast/request merging and placement-aware scheduling reduces on-chip movement costs beyond existing global L2/NoC and Hopper DSM/TMA features. Measure incremental value and count avoided vs newly added bytes.

### H5 — Co-optimization

Compute-to-SRAM ratio, locality radius, banking, MS-SM count, and scheduling **must be tuned jointly**: independently maximizing SRAM capacity or minimizing compute often worsens overall joules/token or latency.

### H6 — Negative regime

For batch-1 dense weight-streaming with little useful SRAM residency and no cross-SM sharing, new SRAM and remote-serving paths may have negligible benefit or hurt. Treat an observed negative result as a scientifically valuable boundary, not an error to conceal.

**High-priority unresolved questions:**

- How much **removable Tensor Core / FP / INT execution area** exists within an H100/B200-style SM? Vendor functional block-level physical area is not publicly established by the prior conversation; obtain credible floorplan/RTL/proxy bounds, not a single unsupported percentage.
- Is the original decode limit truly HBM bandwidth saturation, memory latency/MLP, L2, NoC, or issue/AGU?
- Which tensor's data sees cache **reuse within realistic incremental SRAM capacity**?
- Does Hopper DSM/TMA already do the desired multicast with low overhead?
- What is the least expensive correct address-generation/control machine for paged and irregular KV access?
- How should MS-SMs relate to the existing CTA/cluster scheduler without violating programmability and memory correctness?
- Can SRAM/control links be added physically without routing/timing costs negating the area/energy savings?
- How much prefill or larger-batch performance may the design sacrifice, and is deployment decode-only?

---

## 12. Candidate Paper Narrative (Provisional)

**Working title:** *Beyond Compute-to-Cache: Area-Constrained Memory-Centric GPU Microarchitecture for Energy-Efficient LLM Decoding*

**Motivation:** Memory-dominated decoding can underutilize arithmetic units.  
**Strawman:** Replace part of the compute area with SRAM while retaining ordinary GPU memory paths.  
**Discovery:** Characterize precisely when this fails—insufficient reuse, poor inter-SM transfer, insufficient bank/AGU throughput, or a shifted bottleneck. **These are hypotheses until measured.**  
**Architecture:** A targeted subset of: trimmed compute SMs; minimal memory-serving engine; area-budgeted regional SRAM; local fabric and multicast; reuse- and producer–consumer-aware placement/scheduling.  
**Evaluation:** Same-area, power-aware, strong software/hardware baselines; detailed performance/energy/area and timing accounting; negative and positive workload regimes.  
**Contribution requirement:** Explain *why* each component exists with a measurable observed bottleneck. Pure parameter sweeps without a resulting novel mechanism should be framed as characterization rather than overclaimed as an architecture breakthrough.

### Proposed contributions, contingent on results

1. A workload-grounded model of useful memory reuse and spare compute capacity during decode, separated by LLM/kernel regime.
2. A concrete **area-constrained** way to preserve LSU/address handling while downsizing unused arithmetic resources.
3. A new SRAM placement/data-delivery fabric and/or minimal memory-serving engine that solves a measured bottleneck beyond existing DSM/TMA.
4. End-to-end, physically informed energy/latency evidence and a clear operating-regime map.

---

## 13. Agent Reporting Cadence, Checkpoints, and Stop Conditions

After the first environment inspection, report in a short checkpoint:

```text
Hardware/GPU:
CUDA/driver/profilers:
Candidate real-GPU decode baseline:
Candidate simulator and GPU model (including unsupported features):
Available PPA/NoC tools:
First two measurements to run:
Principal unknowns and blockers:
Changes/downloads needing permission:
```

After each experiment, record **hypothesis, workload, exact setup, observed result, uncertainty, interpretation, next discriminating test**. Distinguish a software scheduling artifact from an architecture benefit.

**Stop or pivot** if:

- Added SRAM cannot reduce off-chip traffic or on-chip energy meaningfully under realistic area limits.
- H100-like software mechanisms already capture the proposed gain at lower cost.
- A memory-serving SM needs nearly full compute SM control/ALU/MLP footprint, removing area advantage.
- Proposed topology adds more movement energy than it eliminates.
- Energy savings disappear once SRAM leakage, NoC/control, timing, or longer execution time are counted.
- Simulation cannot be calibrated sufficiently to justify quantitative claims.

These are research outcomes, not failures of task execution.

---

## 14. First Concrete Tasks for the Linux AI Agent

**Task 1 — Discover:** Run only read-only environment queries; summarize hardware, drivers, profiler support, resources, simulator compatibility, privileges, and missing dependencies.

**Task 2 — Choose one representative optimized decoder:** Identify a practical, permitted LLM and baseline implementation; keep model/precision/attention format fixed while benchmarking batch and context length. Log warmed-up, unprofiled latency and telemetry.

**Task 3 — Identify bottlenecks:** Profile the main decode kernels, clearly separating GEMV/GEMM weight movement from attention/KV. Produce a table of achieved bandwidth, memory stalls, cache behavior, and compute activity. Explicitly identify ambiguous/non-bandwidth cases.

**Task 4 — Reuse analysis:** Create a trace/analytical model to determine incremental HBM savings from realistic additions of SRAM and from SM-to-SM sharing, above existing L2 cache behavior. Calculate optimistic upper bounds and simple energy break-even points.

**Task 5 — Simulator feasibility:** Check source/build/run of Accel-Sim or an appropriate alternative; run an existing regression and quantify the gap to measured hardware. Identify exact source modules/configurations requiring changes for (a) reduced arithmetic pipelines, (b) new SRAM banking, (c) local inter-SM network, and (d) MS-SM command/AGU/LSU logic. Do not modify everything at once.

**Task 6 — Research decision:** Recommend which single path (homogeneous trimmed SM vs heterogeneous memory-serving cluster) has the strongest evidence-based chance to deliver a paper-worthy energy improvement, what first strawman to implement, and which counterexample would falsify it.

**Requested initial output:** A concise findings report plus a version-controlled experiment repository layout (`configs/`, `scripts/`, `traces/`, `results/`, `plots/`, `docs/`) and reproducible first baseline run. No fabricated benchmark results.

---

## 15. Reference Links

- [Morpheus: Extending the Last Level Cache Capacity in GPU Systems Using Idle GPU Core Resources](https://arxiv.org/abs/2209.10914) — primary source reviewed; MICRO 2022.
- [LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference](https://doi.org/10.1109/ISCA59077.2024.00082) — verify specific methodology and claims.
- [Adaptive Memory-Side Last-Level GPU Caching](https://research.nvidia.com/publication/2019-06_adaptive-memory-side-last-level-gpu-caching) — GPU cache placement/NoC tradeoff.
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass).
- [Accel-Sim Framework](https://github.com/Accel-Sim/accel-sim-framework).

**Integrity note:** References beyond Morpheus were discussed in the conversation but not re-audited in preparing this summary. The research agent should retrieve the relevant primary sources, verify exact names/venues/claims, and update the literature review. This document does not assert priority or absence of all competing patents/papers.

---

## 16. One-Sentence Research North Star

> **Redesign the *balance and physical organization* of arithmetic, address generation, on-chip SRAM, and inter-SM data delivery in an NVIDIA-like GPU—while keeping the die-area and HBM budgets fixed—to discover and remove the bottlenecks that limit energy-efficient LLM decoding.**
