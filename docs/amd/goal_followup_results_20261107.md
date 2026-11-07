# Goal follow-up G1–G8 — results (goal_v1, three cohorts)

Assignment: `docs/amd/goal_based_followup_20261105.md`. Plan and pre-registration:
`docs/amd/goal_followup_execution_plan_20261105.md`. Smoke review: `docs/amd/goal_smoke_review_20261105.md`.
Release fab0ea00 for all three mains (released unchanged after the smoke review). Reports were regenerated after each
job with the fixed reader (`analyze_goal.py report`, commit 2d81405a and later) on a devel CPU node, because the in-job
goal analysis crashed on all three cohorts (`TypeError` in the natives reader); raw roots are unchanged.
No clock, power, driver or firmware change was made; controlled clocks are recorded as **unavailable (not authorized)**.

| Cohort | Job / partition | Wall | Steps (rc 0) | Published |
|---|---|---|---|---|
| MI300X (304 CUs) | 502425 / mi3008x | 3 h 53 min | 593 / 594 (1 timeout) | `results_published/amd_20261107_mi300x_goal_v1_auto_job502425/` |
| MI325X (304 CUs) | 502426 / mi3258x | 3 h 12 min | 590 / 590 | `results_published/amd_20261106_mi325x_goal_v1_auto_job502426/` |
| MI350X (256 CUs) | 502427 / mi3508x | 2 h 37 min | 710 / 710 | `results_published/amd_20261105_mi350x_goal_v1_auto_job502427/` |

The failed MI300X step is `g7_counters:…__n304__absent__stock__eager__FETCH_SIZE` (attempt 1, killed at its 25 min
timeout, rc −15). Hung profilers get no retry under the plan; it is kept as a failed acquisition.

Each folder holds the launch plan, gates, `goal_tables.json` (trial-level acquisitions), `goal_summary.md`,
`goal_status.json`, `model_inputs.json` (new, `scripts/amd/goal/model_inputs.py`), the counter audit, ISA metadata
(`isa_meta_gfx942.json` / `isa_meta_gfx950.json`, from the devel CPU rerun), the cubw-style tables and the raw-root
SHA-256 list. Raw data stays in `$WORK/results_amd/` (no owner-authorized archive destination: delivery gap).

## Gates

| Gate | MI300X | MI325X | MI350X |
|---|---|---|---|
| placement / lifecycle / outputs | pass / pass / pass | pass / pass / pass | pass / pass / pass |
| isolation leak_v4 (G5 release) | **fail** | **fail** | pass |
| clock_ok (any G1 cell) | no | no | no |

Consequences: G1 rows are all `observational_clock_confounded`; G5 is not released on MI300X and MI325X (steps not
judged, results "n/a"), released on MI350X.

## G1 — CU headroom (Qwen2.5-32B graph decode, B1, context 4096; R = step time / stock full device)

| CUs, grid | MI300X | MI325X | MI350X |
|---|---|---|---|
| full, sized below full | 1.000 (288) | 0.997 (288) | 0.989 (192) |
| 256 / 224, sized | 0.998 / 0.999 | 0.998 / 1.002 | — / 0.991 |
| 192, sized | 0.998 | 1.002 | 0.996 |
| 160, sized | 0.997 | 1.001 | 1.055 |
| 128, sized | 0.994 | 1.005 | 1.155 |
| 96, sized | 1.009 | 1.034 | 1.392 |
| reduced, stock kernel | 128: 1.068; 96: 1.091 | 160: 1.021; 128: 1.136 | 192: 1.187; 128: 1.171 |

95 % CIs are ≤ ±0.6 % (8 blocks each; tables in each `goal_summary.md`). Outputs equal stock in every cell.

- With the kernel grid sized to the active CUs, MI300X and MI325X lose ≤ 1 % down to 128 CUs (42 % of the device) and
  1–3 % at 96. MI350X holds to 192 CUs (75 %) and then degrades steeply.
- Keeping the stock kernel on a reduced mask costs 2–19 %; this is a dispatch effect (workgroups serialize), not
  arithmetic shortage (see OC5 in `docs/amd/overlap_calibration_results_20261107.md`).
- Clock confound: gfx clocks ranged 1299–2024 MHz (MI300X), 1582–2068 (MI325X), 1318–2204 (MI350X); the clock rises
  as CUs are removed. The pre-registered sensitivity (elasticity −0.27, −0.09, −0.23) moves R by ≤ 3 %, never enough to
  change a conclusion, but no row is qualified.

## G2 — serving with the tuned kernel (A = full/stock, B = full/sized, C = reduced/sized, Cb = boundary, D = reduced/stock)

Reduced mask: 128 CUs (MI300X, MI325X), 192 (MI350X). Mixed prefill + decode serving, Qwen2.5-32B.

| tok/s ratio vs A | MI300X | MI325X | MI350X |
|---|---|---|---|
| c2: B / C / D | 0.989 / 0.895 / 0.850 | 0.996 / 0.912 / 0.913 | 0.981 / 0.946 / 0.806 |
| c8: B / C / D | 0.994 / 0.719 / 0.712 | 1.009 / 0.782 / 0.776 | 1.003 / 0.801 / 0.772 |
| c8 TTFT p99 (C) | 1.80 | 1.66 | 1.16 |
| c8 J/token (C, g3_serve_energy) | 1.279 | 1.164 | 1.121 |

- Serving loses much more than isolated decode: prefill is compute-bound (isolated prefill C = 1.78 MI300X,
  1.65 MI325X, 1.21 MI350X at 192 CUs) and its time spills into TTFT and throughput.
- The sized kernel at full mask (B) is neutral (±1 %).
- Request outputs agree with A in 94–100 % of requests per cell (BF16 nondeterminism under batching; recorded per cell).
- Route trace: the CU-count override reaches a kernel in every decode case except Qwen2.5-7B B8 (no skinny GEMM at batch
  8); in that case C equals D within CI.

## G3 — matched energy (decode window, board accumulator, Qwen2.5-32B B1)

| J/token ratio vs A | MI300X | MI325X | MI350X |
|---|---|---|---|
| B (full, sized) | 0.993 | 0.988 | 0.993 |
| C (reduced, sized) | 0.989 | 0.984 | 1.001 |
| Cb (boundary) | 1.002 | 0.986 | 1.063 |

Board power is unchanged by masking (P ratio 0.99–1.00 except MI350X Cb 0.92): idle CUs are not power-gated in a way
the board sensor resolves. A CU mask therefore does **not** measure removal savings; J/token follows step time.

## G4 — traffic observation point

No counter is validated as **physical HBM** traffic on any cohort. FETCH_SIZE, TCC_EA0_RDREQ(_DRAM) are classified as
pre-MALL request-side reads (FETCH_SIZE unclassified on gfx950); write counters are streaming-consistent. Full
classification: `counter_audit.json` and `goal_summary.md` per folder.

## G5 — tuned decoder × memory service (MI350X only)

| Competitor | n = 224 | n = 192 | n = 128 |
|---|---|---|---|
| mem25 | 1.038 (295 GB/s) | 1.058 (304) | 1.060 (347) |
| mem50 | 1.111 (559) | 1.123 | 1.057 (737) |
| mem100 | 1.258 (1162) | 1.210 (1298) | 1.142 (1497) |
| arith100 | 1.006 | 1.031 | 1.029 |

Ratios are decoder time vs the competitor-absent cell in the same block. A memory competitor slows the tuned decoder
by up to 26 %; an arithmetic competitor costs at most 3 %. MI300X/MI325X: unavailable (isolation gate failed).

## G6 — dispatch / request-generation knee

- Best wvSplitK grid per mask (five Qwen2.5-32B shapes) tracks the active CU count; full tables per folder.
- Native service curves: sequential read per active CU saturates at 41 GB/s/CU (MI300X, 96 CUs), 37 (MI325X) and
  45 (MI350X, 96–128 CUs). The device then delivers 3964 / 3522 / 4310 GB/s (87 % / 71 % / 67 % of full-device read).
- The MI350X G1 loss below 192 CUs matches this per-CU ceiling. It needs the most bytes per CU (8 TB/s over 256 CUs),
  so it hits the ceiling first.

## G7 — traffic vs logical demand

- Logical compulsory read: 65.04 GB/token (weights 63.97 GB + KV 1.07 GB at 4096, BF16).
- MI300X FETCH_SIZE (KiB, validated on gfx942) at 128 CUs: 63.62 M KiB/token = 65.15 GB, **1.0016 × logical**. Within
  this counter's scope (request-side, pre-MALL), decode reads its compulsory bytes once. The full-mask cell timed out
  (above).
- MI350X: 63.62 M (256 CUs) and 63.62 M (192 CUs) counter units. The gfx950 unit is unresolved (G4 unclassified), so
  no byte ratio is claimed.
- MI325X: no G7 steps (eager pmc unusable on this firmware, pre-registered).

No avoidable-traffic or SRAM-reuse opportunity is visible at this observation point. Physical HBM bytes stay unknown.

## G8 — frozen model R(g) = 1 + k·max(0, peak/g − B0), k = 0.00767, B0 = 40.79 GB/s/CU (fit on MI350X 497082)

Final judge: `docs/amd/goal/g8_judge_final.json` (10 held-out points; frozen sha256 35d451c9…).

| Point | R_pred | R_obs | abs error |
|---|---:|---:|---:|
| MI325X 288 (full) / 256 / 224 / 192 / 160 | 1.000 | 0.997 / 0.998 / 1.002 / 1.002 / 1.001 | ≤ 0.003 |
| MI325X 128 | 1.047 | 1.005 | 0.042 |
| MI325X 96 | 1.167 | 1.034 | **0.133** |
| MI300X 160 | 1.000 | 0.997 | 0.003 |
| MI300X 96 | 1.111 | 1.009 | **0.102** |
| MI350X 96 | 1.326 | 1.392 | **0.066** |

**Verdict: REJECT** (MAE 0.035 > 0.025; three points exceed 0.05). The advertised-bandwidth-per-CU threshold does not
transfer:

- On gfx942, the sized decoder keeps full speed well past the predicted knee.
- On MI350X at 96 CUs, it loses more than predicted.

The G6 per-CU read ceiling, which differs by cohort, is the better candidate for a revised model. It is reported as a
hypothesis and was not tested here.

## Model inputs (area model V0)

`model_inputs.json` per folder: 164–181 entries with units, scope (workload, CUs, grid, clock range), estimate and CI,
qualification and source hashes. It also includes the G8 rows for that cohort and, for MI300X and MI350X, a pointer to
the OC6 frozen model. Explicit unknowns:

- mm² per CU;
- SRAM, MC and PHY areas;
- physical HBM traffic;
- component energy;
- clock-controlled curves;
- G5 on MI300X/MI325X.

## Remaining uncertainty and next actions

- Clock qualification is impossible without authorization. Every G1/G2 effect is observational.
- G5 on gfx942 needs an isolation probe that passes there (leak v4 fails on both gfx942 cohorts).
- No physical HBM counter has been validated, so traffic claims stay at the request side.
- Raw archive destination: still needed from the owner.

AMD acknowledgement for any publication: "This work was supported in part by Advanced Micro Devices, Inc. under the
AMD University Program's AI & HPC Cluster".
