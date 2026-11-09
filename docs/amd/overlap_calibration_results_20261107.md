# OC-v0 results, part 2 — transfer check, OC5 mechanisms, OC7 delivery (MI300X, MI350X)

This document continues `docs/amd/overlap_calibration_results_20261105.md`, which covers OC0–OC4, the OC6 freeze and the
held-out verdict. Those results are unchanged here. It also closes the items that document left open: the Qwen2.5-7B
transfer trace, and the OC5/OC7 synthesis that needed the G1–G8 mains
(`docs/amd/goal_followup_results_20261107.md`).

No new GPU steps were run. All analysis ran on devel CPU nodes or the login node, using published artifacts.

## Transfer check (secondary; not an acceptance criterion)

Tool: `scripts/amd/oc/oc_transfer.py`. Inputs:

- Frozen M1/M6 per cohort.
- The graph-mode kernel trace from the held-out jobs (`oc6_transfer`, Qwen2.5-7B B1, context 4096, full mask, stock
  grid).

Method:

- Each `decode_step_*` range has 28 × (qkv, o_proj, gate_up, down) + lm_head = 113 wvSplitK dispatches.
- All 63 / 63 steps matched that layout on both cohorts.
- Shapes are assigned by position. References are the frozen (n, n) cell. No refit.

| Shape (m × k) | MI300X T (µs) | M1 err | M6 err | MI350X T (µs) | M1 err | M6 err |
|---|---:|---:|---:|---:|---:|---:|
| qkv 4608 × 3584 | 10.3 | +22.0 % | −4.8 % | 7.8 | +20.3 % | −10.4 % |
| o_proj 3584 × 3584 | 8.5 | +31.1 % | 0.0 % | 6.6 | +26.2 % | −6.1 % |
| gate_up 37888 × 3584 | 58.5 | +6.6 % | +13.0 % | 41.6 | +1.1 % | 0.0 % |
| down 3584 × 18944 | 31.8 | +7.6 % | +5.0 % | 22.8 | +3.9 % | −7.0 % |
| lm_head 152064 × 3584 | 224.4 | +3.9 % | +17.8 % | 160.6 | −3.9 % | +3.4 % |
| **projection time per decode step** | | **+9.8 %** | **+8.7 %** | | **+5.2 %** | **−3.1 %** |

The same pattern as the held-out cells appears in real graph decode:

- M1 overpredicts the small, floor-dominated shapes.
- On MI300X, M6 overpredicts the large streaming shapes: in graph mode wvSplitK streams faster than the isolated
  `stream` references imply.

Per-step projection time is within 10 % for both models on both cohorts. The shape-level errors keep either model
from being a validated per-kernel predictor. Results: `results_published/amd_20261105_*_oc_v0_heldout_job*/transfer_qwen7b.json`.

## OC5 — mechanism table

Each claim is tied to its control, observables, confounds and a falsifying outcome. All rows are observational with
respect to clock (G1 clock gate fails on every cohort; clock control is unauthorized). A CU mask removes issue
slots, LDS, L1, request generators and wavefront capacity together, so no row is an arithmetic-only intervention.

| # | Claim | Control / contrast | Observables (cohort values) | Confounds | Falsifier | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Dispatch serialization** explains the stock-kernel loss on reduced masks | Same mask, stock vs sized wvSplitK grid (G1 rows; G2 D vs C) | R stock / sized: MI300X 128 CUs 1.068 / 0.994; MI325X 128 1.136 / 1.005; MI350X 192 1.187 / 0.996. G6 dispatch: stock grid gives ≥ 2 WGs on some CUs | clock (sized and stock share it within a block), output identity (equal) | sized grid not recovering the loss | **supported** (all cohorts) |
| 2 | **Arithmetic shortage** is not what limits sized decode | CU count reduced at sized grid (G1), compared with compute demand | C/M = 0.04–0.07 (OC6 dev); ATT A_C/T 3–20 %; X/A_C 0.94–1.00. R ≤ 1.009 at 96 CUs (MI300X), ≤ 1.005 at 128 (MI325X) | mask also cuts requests in flight; clock rises as CUs are removed (elasticity −0.09 to −0.27) | R rising in proportion to removed MFMA throughput while C/M stays ≪ 1 | **supported** for gfx942 down to 96–128 CUs and MI350X down to 192 |
| 3 | **Per-CU request/delivery ceiling** limits MI350X below 192 CUs and gfx942 at 96 | G6 native service at the same masks (seqread, ILP 1–16, grid sweeps) | seq read per active CU at 96 CUs: 41 (MI300X), 37 (MI325X), 45 GB/s/CU (MI350X); device read at 96 CUs 87 / 71 / 67 % of full. MI350X R(96) = 1.392 vs 1/0.67 = 1.48 memory-only bound | native read ≠ wvSplitK access pattern (OC6 shows wvSplitK can exceed matched read at low grid on MI350X); clock | sized decoder slower than the per-CU ceiling allows, or MI350X R flat below 192 | **supported, conditional** (ceiling from a native proxy, not from the decoder) |
| 4 | Decoder time is **sensitive to memory service**, not to arithmetic competition | G5 competitors at fixed tuned decoder (MI350X, isolation leak_v4 passes) | mem100 1.14–1.26 at 1162–1497 GB/s competitor traffic; arith100 1.006–1.031 | competitor placement; shared clock/power; per-request coverage = 1.0 | arith100 ≥ mem100, or mem ratios flat with competitor GB/s | **supported** on MI350X; on gfx942 **same ordering, observational** (addendum 2026-11-09: mem100 1.16–1.26, arith100 1.03–1.06; leak v5 not established) |
| 5 | **Fitted overlap η** is an architectural constant usable by the area model | OC6 dev fit, held-out, transfer | η profile 1.5–3.0 (> 1, no overlap meaning); η not identifiable (C ∝ M); held-out not accepted | launch floor, memory reference bias | (would need) stable η ∈ [0, 1] passing held-out | **rejected / not identifiable** |
| 6 | **Occupancy / LDS limits** | G6 dispatch probe with and without its LDS reservation, one workgroup per CU (`g6_dispatch` `*_lds0` vs base) | lds0 / base time: 35 of 40 pairs within ±2.5 % across cohorts; outliers 0.83–1.25 (MI300X), 0.94 (MI325X), 1.05 (MI350X) with no consistent sign | probe kernel, not wvSplitK; one WG per CU in both variants | lds0 consistently faster at equal grid | **not supported** at this probe's occupancy; untested for wvSplitK itself |

Not claimed: stall fractions → reclaimable area; any combination of unrelated counters into a time partition; decoder
bandwidth from competitor throughput.

## OC7 — energy, serving transfer and model-input delivery

- **Energy (reused G3, matched scopes).** Decode J/token follows step time. Board power does not change with the CU mask
  (P ratio 0.99–1.00, except MI350X boundary Cb 0.92). The board sensor cannot identify per-component or removal
  savings. Serving J/token at the reduced, sized mask is higher, 1.12–1.28 × A, because prefill slows. No new energy
  case was justified, because the calibrated model was not accepted.
- **Serving transfer.** G2 c2/c8 tables (`goal_followup_results_20261107.md`) are the transfer scope. OC6 adds no
  serving prediction, since neither model passed held-out.
- **Delivery files**, in each held-out folder (`scripts/amd/oc/oc_deliver.py`):

| File | Content |
|---|---|
| `observability.json` | every counter test with validated / measured-unvalidated / invalid / unavailable status, scope and replay; ATT definition; device HBM activity unavailable |
| `overlap_bounds.json` | ATT per-shape A_C, A_M, X, X/A_C (MI300X 0.937–0.999, MI350X 0.938–0.999), coupled-bound rule, η identifiability record |
| `service_references.json` | R_mfma4, B_ref, O_ref per (mask, grid) with 3 acquisitions; pilot references; G6 native service curves |
| `calibration_frozen.json` | copy of `docs/amd/oc/oc6_frozen_<cohort>.json` (sha256 in the dev review) |
| `validation.json`, `transfer_qwen7b.json` | held-out verdict, issued-MFMA check, transfer check |
| `model_inputs.json` | issued MFMA 4·m·⌈k/1024⌉ (exact), references, η not identifiable, X/A_C range, κ, unknowns |
| `oc_status.json` | OC0–OC7 terminal dispositions |

## Dispositions

| Goal | MI300X | MI350X |
|---|---|---|
| OC0 capability | complete | complete |
| OC1 timeline/demand | complete (reused) | complete (reused) |
| OC2/OC3 references, controls | complete | complete |
| OC4 activity | A unavailable; B bounded (per CU) | same |
| OC5 mechanisms | complete (table above) | complete |
| OC6 calibration | frozen; **not accepted** on held-out | frozen; **not accepted** |
| OC7 delivery | complete | complete |

MI325X has no OC jobs, because there is no single-GPU MI325X partition. Its G1–G8 evidence enters OC5 rows 1–3.

**Identifiable:**
- issued MFMA per call;
- per-CU MFMA/load overlap (ATT);
- the independent service references.

**Conditional:**
- the per-CU delivery ceiling, from a native proxy.

**Not identifiable:**
- η;
- device-scope activity;
- physical HBM traffic.

**For the fixed-area model:** decode projection compute is almost fully hidden under memory on every traced CU.
Removing CUs costs nothing until the per-CU delivery ceiling binds, and that ceiling is cohort-specific (37–45 GB/s/CU
for sequential reads). A single advertised-bandwidth-per-CU threshold (G8) does not transfer across cohorts.

AMD acknowledgement for any publication: "This work was supported in part by Advanced Micro Devices, Inc. under the
AMD University Program's AI & HPC Cluster".
