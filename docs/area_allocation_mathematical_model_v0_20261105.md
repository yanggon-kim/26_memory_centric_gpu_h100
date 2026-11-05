# Fixed-area resource allocation for LLM inference — mathematical model V0

Date: 5 November 2026. Status: first analytical formulation, not a calibrated architecture prediction. Owner-requested deliverable: determine component counts and area allocated to arithmetic, non-arithmetic storage/communication/request resources, and external-memory interfaces for (D) decode only and (PD) prefill plus decode. Preserve future versions and alternate formulations rather than silently replacing their assumptions.

## 1. Objective and falsifiable claim

Build a measurement-informed optimizer that returns counts, capacities/bandwidths, logic-area fractions, HBM/package resources, predicted latency/energy, active bottlenecks and uncertainty. Produce separate D and PD solutions at the same declared process and physical budgets.

The hypothesis is that a general-purpose GPU allocation can be suboptimal for a specified LLM workload/objective. Low arithmetic utilization alone does not prove it. We must show a feasible alternative beats a fairly tuned baseline under equal budgets and retained workload constraints. Optimality is conditional on the modeled design family, software mapping, workload distribution and physical assumptions.

The primary objective remains system-relevant J/token subject to performance and physical constraints. Also produce latency/throughput Pareto views. A latency-optimal split need not minimize energy. Decode-only excludes prefill cost from its objective but must still declare prefill feasibility and any guardrail.

## 2. Design variables and component boundaries

Let x be physical design, m software mapping, o operating conditions, w workload and theta uncertain parameters.

| Group | Variables | Accounting |
|---|---|---|
| Arithmetic C | Tensor/MFMA engines, vector arithmetic lanes, precision | Engine counts, area and useful/issued throughput |
| Storage S | RF/L1/shared/LDS/L2 bytes, banks, ports | Data, tags, peripheral/port area, access bandwidth/energy |
| Request/control R | LSU/AGU issuers, schedulers, queues/MSHRs | Issue rate, resident state and outstanding requests |
| Communication N | NoC links/routers, execution-to-L2 ports, D2D links | Topology, bandwidth, latency and physical area |
| Interfaces B | Digital MCs, PHYs, installed/active channels | Separate MC/PHY area, service and placement dimensions |
| Memory H | HBM generation, stack count, capacity/rate | Separate memory silicon, package sites, energy and cost |
| Fixed/physical F | Other I/O, clock/power/control, routing, redundancy | Explicit fixed or scaling terms |

An SM/CU bundles C/S/R/N. Whole-SM removal changes them together. A compute-trimmed structure may reduce C while preserving the feeding resources. Evaluate both families. Address/control integer units belong consistently to R; do not charge a physical unit twice.

Implementable engine/bank/controller/channel/stack counts are discrete. Continuous area variables are screening relaxations. Physical disabled/spare resources consume area even when not contributing throughput; do not transfer one fixed spare ratio to every hypothetical design without justification.

## 3. Logic area, memory capacity and package feasibility

All logic areas below are in mm², without overlap:

\[
A_{logic}=A_C+A_S+A_R+A_N+A_{MC}+A_{PHY}+A_F\le A_0.
\]

Example parameterization:

\[
A_C=\sum_i n_i a_i,\quad
A_S=\sum_{j\in\{RF,L1,SMEM,L2\}}
\left[C_j/\rho_j+A_{tags,j}+A_{periphery,j}(banks_j,ports_j)\right].
\]

Here C_j is bytes and rho_j bytes/mm². Every memory/port type needs its own coefficients. Charge a unified L1/shared array once and constrain its carveouts. RF density is not dense single-port SRAM density. Add explicitly modeled issue/queue, router/link, digital-controller and PHY areas rather than embedding them again in an indivisible SM coefficient.

HBM stacks are separate silicon: reclaimed GPU mm² pays for MC/PHY and delivery, not for HBM die area. Also constrain package footprint/sites, stack capacity, cost, power/thermal envelope, legal PHY width/depth/placement, usable shoreline, stack clearance/routing/reach, compatible channel mapping and reticle bounding box. Irregular perimeter alone is not placeable beachfront. Geometry-only checks remain screens. Apply identical package/cost policies to compared designs or show distinct explicit Pareto scenarios.

For every workload, including fixed batches:

\[
W_{stored}+KV(B,S)+A_{runtime}+Reserve\le Capacity_{HBM}(x).
\]

An infeasible baseline has no valid speedup denominator. Capacity-enabled workloads are a separate benefit.

## 4. Algorithmic demand and on-chip retention

For dense decode, let P_mat be the weight elements in matrix operations actually executed per tick, including the actual LM-head path; W_stream is their bytes. Stored embedding parameters need not all be streamed each tick. With B sequences, layer contexts S_l, query/KV heads H_q/H_kv and dimension d:

\[
F_D\approx2B P_{mat}+4B\sum_l S_l H_{q,l}d_l+F_{other,D},
\]
\[
D_{inputs,D}\approx W_{stream}+2B\sum_l S_l H_{kv,l}d_l b_{KV}.
\]

F includes useful QK/AV but needs separate softmax, padding and issued-work accounting. Add KV writes, activation/reduction/sampling traffic and other route-dependent demand. If weights stream once per batch tick, their bytes per output token are W_stream/B. Initially resident inputs may not require HBM reads. Logical KV sharing does not guarantee the kernel realizes it.

For causal prefill of B prompts of length P, let P_layer exclude the vocabulary head and let F_head,P represent the actual prefill head computation:

\[
F_P\approx2BP P_{layer}+F_{head,P}+2B P(P+1)\sum_l H_{q,l}d_l+F_{other,P}.
\]

Weight reuse and FlashAttention mapping determine traffic; P times W_stream is not an unavoidable HBM requirement. The head may execute only for the selected final prompt position rather than every prompt token; derive F_head,P from that route. These formulas assume dense standard attention; MoE, quantization or local attention require separate models.

For kernel k, mapping m and history h:

\[
D_{HBM,k}=D_{comp,k}(x,h)+D_{reread,k}(C,m,h)
+D_{spill,k}(C_{RF},C_{SMEM},m)+D_{other,k}.
\]

Specify residency, transactions and writebacks consistently. Estimate avoidable terms from validated counters, cache/address studies or simulation; leave unidentifiable terms unknown. Do not assume a universal inverse-capacity law or that extra SRAM reduces one-pass weight traffic. Across-token retention can require model-scale capacity. Single-layer repetition does not preserve that working set.

Maintain distinct D_L1, D_L2 and D_NoC from the dataflow; hits/reuse can make these larger than HBM bytes. Pre-MALL request counts are not physical HBM bytes. Kernel retuning, issued/padded work and load imbalance must be represented or bounded independently of useful FLOPs.

## 5. Service and latency model

Arithmetic service by operation class i is P_eff,i=n_i p_i(f,o) eta_i,k(m,occupancy). Compose different engines according to supported overlap, not one universal utilization multiplier.

External-data service has the screening ceiling

\[
B_{ext,k}\le\min\left\{\eta_{H,k}\sum_h b_h,
B_{PHY,k},B_{MC,k},B_{L2path,k},B_{NoCpath,k},
I_k\bar b_k,Q_{bytes,k}/\ell_k\right\}.
\]

I is transactions/s, b-bar transaction bytes, Q supported outstanding bytes and ell service latency. Q/ell is a Little's-law capacity screen, not a universal fitted hardware constant. A decomposition Q <= N_active q_per_CU + Q_other_engines admits lighter memory-serving structures. Occupancy, placement and queues affect I and Q. More nominal HBM cannot help when another delivery limit stays smaller.

A first per-kernel approximation is

\[
t_k=O_k+\max\{\tau_{compute,k},D_{HBM,k}/B_{ext,k},
D_{L2,k}/B_{L2,k},D_{local,k}/B_{local,k}\}+\delta_{tail,k},
\quad T_{phase}=\sum_{k\in phase}t_k.
\]

O is a separately justified non-scalable cost and delta covers dispatch/tail/dependency effects not already counted. Perfect overlap in the max is optimistic and requires calibration. Do not interpret these as measured compute-only/memory-only intervals. Kernel serialization means sum of kernel maxima generally exceeds a maximum of whole-phase sums. The latter is only a coarse bound. Preserve the measured wvSplitK scheduling effect and graph/eager distinctions. Do not charge the same already-modeled NoC bottleneck again as an independent serial service cost.

## 6. Separate decode-only and prefill-plus-decode schedules

For D, B sequences and G pure-decode ticks after the all-request barrier, with actual contexts S_t:

\[
Y_D=BG,\quad T_D=\sum_{t=1}^G T_{step}(B,S_t),\quad
E_{tok,D}=E_D/Y_D,\quad Throughput_D=Y_D/T_D.
\]

Prefill is excluded from this objective, not claimed free. State whether it runs elsewhere, is amortized away, or remains a constraint.

For PD, a synchronous batch of B requests with P prompt tokens and G total generated outputs normally gets its first output from prefill and then executes G-1 decode ticks:

\[
Y_{PD}=BG,\quad T_{PD}=T_{prefill,batch}(B,P)+\sum_{t=1}^{G-1}T_{step}(B,S_t),
\quad E_{tok,PD}=(E_P+E_D)/Y_{PD}.
\]

Record actual token/barrier/sampling conventions and change the formula explicitly if the producer differs. The prefill term is full-batch cost, not per-request cost accidentally divided by B twice. Preserve context growth; mean-context approximations require checks. Continuous batching, chunked prefill and disaggregation need separate schedules, queues and communication/capacity models. Phase sums alone do not predict p99 TTFT.

For probabilities p_w per request/batch group, the energy objective is

\[
J=\frac{\sum_w p_w E_w}{\sum_w p_w Y_w}.
\]

These are not automatically token weights. Freeze the workload mix and batch policy before final comparisons. Report fixed-feasible-batch and capacity/SLO-selected-batch scenarios separately.

## 7. Energy and the full constrained optimization

Start with non-overlapping component accounting:

\[
E_w=\sum_i e_{op,i}F_{issued,i}+\sum_j e_{access,j}N_{access,j}
+\sum_l e_{bit,l}D_{bit,l}+P_{static}(x,o)T_w+E_{host/board,w}.
\]

Static power depends on retained components, banks/stacks and state, not only total area. SRAM ports/capacity and NoC length affect energy/latency. Use measured DVFS behavior or mark it unknown; a linear board-power cap is not a physical voltage/frequency curve. Physical compute removal differs from a software mask retaining silicon.

Solve min J(x,m,o;theta) separately for D and PD under Section 3 constraints, correctness, decode latency/throughput, declared prefill/TTFT guardrails and peak/thermal power. Also solve minimum-latency variants. Use A_logic <= A0 rather than forced full utilization: unused area may be preferable under power/cost constraints. Use comparable baseline/alternative operating-point policies.

Publish parameter ranges/scenario Pareto results, active constraints and rejected candidates. A nominal winner within overlapping uncertainty is not a robust improvement. Do not claim a global optimum unless the search proves one for its stated model.

## 8. Closed-form first allocation rule

Fix storage, mapping, clocks, traffic and layout. Reserve their area A_fixed and let A=A0-A_fixed be available to independently scalable service resources r. Resource r supplies s_r units/s/mm² and requires d_r units/kernel. Define K_r=d_r/s_r, in mm²*s. With perfect overlap and negligible fixed/tail time:

\[
T=\max_r K_r/A_r,\qquad \sum_r A_r\le A.
\]

Target time T requires A_r >= K_r/T for every resource. Hence

\[
\boxed{T^*=\frac{\sum_r K_r}{A},\qquad A_r^*=A\frac{K_r}{\sum_j K_j}.}
\]

For arithmetic, internal delivery and external interfaces:

\[
A_C:A_N:A_B=(F/\pi):(D_{internal}/\gamma):(D_{HBM}/\beta).
\]

Resource demand per area efficiency determines the ideal fraction. Low FLOP utilization alone does not. Discrete macros, fixed HBM ceilings, shoreline limits and shared resources require a constrained solver. Additional PHY area is useless once its external rate is fixed. This rule does not optimize cache capacity: enumerate storage configurations, reserve their actual area, estimate changed traffic/mapping, solve the remaining allocation and compare feasible candidates. Capacity and access bandwidth are different quantities.

There is also an explicit storage tradeoff in this ideal relaxation. For one variable capacity C, define K(C)=F/pi+D_HBM(C)/beta+K_internal(C) and A_avail(C)=A0-A_fixed-A_S(C). After equalizing the remaining independent resources,

\[
T^*(C)=K(C)/A_{avail}(C),\qquad
\boxed{-K'(C)/A_S'(C)=T^*(C)}
\]

at a differentiable interior optimum with A_S'(C)>0. The left side is the reduction in service-demand coefficient per added storage area; the right side is the opportunity cost of taking area from the remaining services. If extra capacity changes neither traffic nor mapping, K'=0: this ideal latency model favors the minimum feasible storage instead of allocating cache merely because space is available. Real banking, local bandwidth, tile/spill and energy effects can change K and the optimum. Boundaries and discrete banks require enumeration, not this derivative alone.

For a chosen kernel service target t, simple resource-count lower bounds are

\[
n_i\ge\left\lceil F_{i,k}/(\eta_{i,k}p_i t)\right\rceil,\quad
Q_{bytes}\ge B_{target}\ell,\quad
N_{HBM}\ge\max\left\{\left\lceil B_{target}/(\eta_H b_{stack})\right\rceil,
\left\lceil Memory_{required}/Capacity_{stack}\right\rceil\right\}.
\]

Controller, PHY, NoC and issuer service must each reach B_target with their own efficiency/area and compatible channel mapping. These are necessary service/capacity conditions, not sufficient placement or scheduling feasibility. They turn an area fraction into candidate engine, queue and stack counts for the discrete stage.

## 9. Worked mixed-phase example — illustrative only

The following invented coefficients check the mathematics; they are not measurements, physical estimates or recommended GPU fractions. Reserve 20 mm² of a 100 mm² toy die for fixed/local resources, leaving A_C+A_B=80 mm². Suppose internal delivery is adequate and

\[
T_D=\max(8/A_C,72/A_B)\ ms,\quad
T_P=\max(64/A_C,16/A_B)\ ms.
\]

Coefficients have units mm²*ms. Decode-only gives A_C=8, A_B=72 and T_D=1 ms. For one prefill plus four decode ticks, the regime with compute-limited prefill and bandwidth-limited decode gives

\[
T_{PD}=64/A_C+4(72/A_B),\quad
\frac{A_C}{A_B}=\sqrt{\frac{64}{4\cdot72}},\quad
A_C=80\frac{\sqrt{64}}{\sqrt{64}+\sqrt{288}}\approx25.630,
\quad A_B\approx54.370.
\]

Both assumed bottlenecks hold at this solution. Total time is approximately 7.794 ms versus 12 ms for the decode-only split and 19 ms for the prefill-only split (A_C=64, A_B=16). Four post-prefill ticks mean five generated outputs under Section 6's PD convention; normalization must preserve that scope.

More generally, for fixed nonnegative K, mapping and schedule, the continuous problem

\[
\min_{A_r>0}t_P+g t_D,\quad t_P\ge K_{r,P}/A_r,
\quad t_D\ge K_{r,D}/A_r,\quad\sum_r A_r\le A
\]

is convex because K/A is convex on positive A. In the regime above, A_C/A_B=sqrt(K_C,P/(g K_B,D)): more decode ticks favor delivery, while more prefill work favors arithmetic. This is a latency result; discrete macros, area-dependent energy and cache/mapping changes require richer models and may remove convexity. Do not replace sequential phase cost by a max of workload-averaged demands.

## 10. Calibration: what current evidence can identify

| Needed input | Existing evidence | Gap |
|---|---|---|
| Algorithmic demand, actual kernels | Pinned models, inventories and decode records | Issued/padded work and route-specific extras |
| CU/grid/request knee | RTX/H200/AMD restriction/service; balanced sizing | Clock qualification, tuned guardrails and holdouts |
| D_HBM versus storage/history | Selected RTX/H200 physical captures; AMD requests | AMD G4/G7 physical-counter validation |
| Compute/port/NoC/request service | Native controls, selected counters, simulator fits | Matched independent controls; fit versus prediction |
| SRAM/compute/MC/PHY area | Literature/LLMCompass/GPU-1b estimates | Process/port-specific uncertainty and independent checks |
| Energy versus composition | Board energy and operation-based estimates | AMD G3 matched scope; component/static uncertainty |
| Usable shoreline/package | Plan screens and GPU-1b geometry | Placement/routing evidence or conditional scenarios |

Masks do not identify mm² per arithmetic engine. GPU-1b's baseline-area closure is imposed, not independent validation. Its 45 B/cycle/SM and operand-feed constants are simulator-derived. Correct the fixed-batch capacity bypass, analytical/simulated L2 mismatch and completion/cache-provenance issues before trusting its optima. Preserve the imported baseline.

The approximately 42 GB/s advertised-bandwidth/CU alignment on two AMD cohorts is a test target, not measured decoder bandwidth or a universal beta. Failed E7 blind checks require explicit dispatch/mapping effects, not a refit relabeled as validation.

## 11. Ongoing jobs and model versions

The coordinating modeling agent owns these jobs:

1. Maintain this V0, reproducible sanity examples and a unit/source/uncertainty ledger.
2. In a later implementation task, build a continuous screening solver and discrete feasible enumeration; output D and PD solutions with complete area breakdowns and active constraints. This document implements no solver.
3. Map GPU-1b to C/S/R/N/MC/PHY/H without double counting; retain fixed/tuned software and stock-baseline controls.
4. Ingest AMD G1–G8 and RTX/H200 results with separate architecture-specific calibration and frozen validation.
5. Report engine counts; RF/L1/shared/L2 bytes/banks/ports; NoC/request resources; MC/PHY/channel/stack counts; area mm²/fractions; package/cost/capacity; D/PD latency/energy; uncertainty and infeasible candidates.
6. Test stock-allocation suboptimality against a fairly tuned baseline. Accept a negative/unidentifiable result. Do not publish numerical optimal fractions until the parameter evidence supports them.

Version ladder: V0 analytical/ideal allocation; V1 calibrated kernel/mapping and discrete area enumeration; V2 held-out timing/traffic validation; V3 physical/package/component-energy refinement. Alternate versions may coexist with explicit IDs and claim scopes. A fitted surrogate is not automatically a validated physical model.

The AMD agent supplies G1–G8 observables and parameter constraints, including unknowns; it need not invent PHY/SRAM areas or wait for a complete optimizer. Reading this model does not launch experiments or change the separate AMD execution checkpoint.

## 12. Evidence entry points

- AMD measured-evidence baseline: `yanggon-kim/26_memory_centric_gpu_h100`, branch `amd/mi300x-phase0`, commit `5b988063e8391005643351594dd8db0394c4d735`; handbook, per-job reviews and `results_published/COHORTS.md`.
- AMD G1–G8 assignment: `docs/amd/goal_based_followup_20261105.md`, initially committed as `e9c76bc5ec3932f8c04f9668af18cccde59383d7`.
- Coordinating workspace: `reports/gpu_branch_review_20261105.md` and `.sources.json`; 1,208 published acquisition means checked, with raw/readback limits retained; `reports/gpu1b_critical_adoption_review_20261102.md`.
- Physical and energy requirements remain governed by shoreline-aware plan 2.0 and its cited sources. V0 derives equations and reuses campaign facts; it introduces no newly verified physical area coefficients.
