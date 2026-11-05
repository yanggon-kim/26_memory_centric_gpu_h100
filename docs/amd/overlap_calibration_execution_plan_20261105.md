# OC-v0 execution plan — activity, independent service and overlap calibration (AMD)

Date: 5 November 2026. Assignment: [overlap_calibration_followup_20261105.md](overlap_calibration_followup_20261105.md);
framework: [compute_memory_overlap_calibration_framework_v0_20261105.md](../compute_memory_overlap_calibration_framework_v0_20261105.md).
Written before any OC data. Section 6's dev/held-out details are fixed here; the frozen OC6 model file
(`docs/amd/oc/oc6_frozen_<cohort>.json`) is committed after the dev job and before the held-out job.

## 0. Current-job map and coordination (OC0)

| Work | Jobs / release | State at plan time |
|---|---|---|
| G1–G8 smokes | 502423 (MI300X), 502424 (MI350X), release `fab0ea00` | complete; reviewed in [goal_smoke_review_20261105.md](goal_smoke_review_20261105.md) |
| G1–G8 productions | 502425 (MI300X, mi3008x), 502426 (MI325X, mi3258x), 502427 (MI350X, mi3508x), release `fab0ea00` | released unchanged; 502427 running, others queued |
| Frozen G8 | `docs/amd/goal/g8_frozen_model.json` | untouched; judged only on G1–G8 data, original rules |
| OC-v0 | separate plans `$WORK/cubw_plans/oc_*`, release `$WORK/cubw_release/<oc commit>`, roots `results_amd/amd_<date>_<cohort>_oc_v0_<kind>_job<id>` | this plan |

OC runs only on separate single-GPU allocations (`mi3001x` MI300X 0.125/h, `mi3501x` MI350X 0.175/h); no G1–G8 device is
touched and no queued release or plan is modified. Allocation at plan time: 324.9 of 450 units used (project), so OC is
budgeted at ≤ 5 units total. MI325X has no single-GPU partition and its counters were refused before (CP firmware
177 < 186); OC on MI325X is limited to reuse of 502426 (G4 audit, G6 controls) unless that audit shows usable counters,
in which case one ≤ 1.5 h `mi3258x` job with the same dev/held-out design may be added by a later plan revision.

Evidence reused, with scope: G4 counter audit (pre-MALL FETCH_SIZE on gfx942; 0.5× on gfx950 streaming); G6 static ISA
census (CPU re-run, 392 kernels/arch); G6 kernel-shape sweeps and native service curves (502425–502427 pilot stage); G2
kernel-route traces; G1/G5 tuned-grid and memory-competition contrasts; G3 energy. OC does not reacquire any of these.

## 1. Facts that shape the adaptation (measured before this plan)

- vLLM's bf16 decode projection kernel (`wvSplitK_hf*_I14__hip_bfloat16`) on gfx942 and gfx950 issues
  `v_mfma_f32_4x4x4_16b_bf16` (24 static), `global_load_dwordx4` (12–13) and `ds_read_b128` (12): arithmetic is on the
  matrix cores with 4×4×4 bf16 blocks, weights stream through vector memory, the activation is read from LDS.
- Tools: host rocprofv3 1.1.0 (ROCm 7.2.0) without a thread-trace decoder; container rocprofv3 1.3.2 with
  `librocprof-trace-decoder.so.0.2` (copied to `$WORK/lib/rocprof-trace-decoder`, sha256 `9c9203ce…`). Both list
  issued-instruction counters (SQ_INSTS_*), activity-cycle counters (SQ_ACTIVE_INST_*, SQ_BUSY_*, TCC_BUSY,
  GRBM_GUI_ACTIVE) and in-flight level counters (SQ_INST_LEVEL_VMEM, TCC_EA0_RDREQ_LEVEL). PC sampling and ATT are
  advertised; whether they work is OC0's question.
- svc_bench v7 (`scripts/amd/cubw/svc_bench_v7.hip`; binaries `$WORK/bin/svc_bench_gfx9{42,50}_v7`) adds
  route-matched `mfma4` (4×4×4 bf16 MFMA, register operands), `mfma4_lds` (A operand by `ds_read_b128`), and the
  known-schedule controls `split` (even workgroups stream, odd run MFMA, one workgroup per CU) and `fused` (loads in
  flight under independent MFMA in the same wave). Disassembly confirms the intended instructions in each kernel.

## 2. Observables and their status vocabulary

Every metric in `observability.json` carries: definition, units, scope (device / per-XCD / per-CU / per-wave),
time base (dispatch aggregate vs time series), tool+version, replay/serialization behaviour, and one status:
`validated` (passes its known-work test below), `measured-unvalidated`, `unavailable` (tool refuses or not listed),
`invalid` (fails its test). Only `validated` metrics enter a model.

Validation tests (native known-work kernels, `oc_counters.py`, one pass per group):

| Metric | Known-work test | Pass |
|---|---|---|
| SQ_INSTS_VMEM_RD | `read` 1 GiB: n_float4/64 wave-instructions | within 2 % |
| SQ_INSTS_MFMA (and MOPS_BF16) | `mfma4_w1/w4`: 8·iters·waves (MOPS: × the per-instruction MOPS constant, reported) | within 2 % |
| SQ_INSTS_VALU | `fma_w1/w4`: ≥ 8·iters·waves | ≥ 1.0 and ≤ 1.25 × |
| FETCH_SIZE / TCC_EA0_RDREQ_* | `read` streaming 1 GiB: bytes in some declared unit | within 5 % (G4 rule) |
| activity-cycle counters | `*_w1` vs `*_w4`: a per-CU "any active" counter saturates (w4/w1 ≤ 1.1 at equal work per wave × time); a per-wave sum scales with resident waves | classified, not passed/failed |
| SQ_INST_LEVEL_VMEM, TCC_EA0_RDREQ_LEVEL | `ilp1` vs `ilp8`: mean in flight (level/cycles) rises with ILP; Little's-law latency level/requests within 0.5–5 µs | both hold |

## 3. Pilot (this submission): `oc_v0` kind `pilot`, one GPU per cohort

Plans `scripts/amd/oc/plan_oc.py --kind pilot` (45 steps, ~2.6 h estimate, ≤ 4 h limit). Steps (randomized order for
the reference/kernel cells; fixed seed 20261106):

1. Placement at the full mask and n_p (MI300X 304/128, MI350X 256/192) — reuse of the G1–G8 placement rule.
2. `oc0_probe` (`oc_probe.py`): PC sampling host_trap(time) and stochastic(cycles) on the projection route; ATT
   (one CU, SIMD 0xF, `--att-activity 8`) on the projection route (container decoder) and on `split`, `fused`, `mfma4`,
   `read` (host rocprofv3 + copied decoder). Records rc, files, heads.
3. `oc0_counters` (`oc_counters.py`): 11 counter groups × 9 native known-work loads (host) + the same groups on
   `oc_kernels.py --light` Qwen32B shapes (container).
4. `oc23_refs`: svc_bench v7 `mfma4`, `mfma4_lds`, `read` (1 GiB at g workgroups × 1024 threads, and 8 blocks/CU × 256),
   `read` MALL-resident 64 MiB × 16 passes, `read_ilp` 1/2/4/8, `split`, `fused`, at (mask, workgroups) =
   (full, full), (n_p, n_p), (full, n_p). 3 warmups, 10 trials × 20 launches (10 for split/fused).
5. `oc16_kernels` (`oc_kernels.py`): Qwen32B and Qwen1.5B projection shapes (development models only), wvSplitK at the
   stock grid and the reduced grid, histories `stream` and `same`, the tiny-shape overhead probe, and (full mask only)
   hipBLASLt `linear` at 4096 tokens (prefill reference). 3 warmups, 10 trials × 50 calls.
6. `oc1_eager`: Qwen32B eager B1/C4096 under three counter groups (instructions; MFMA/busy; FETCH_SIZE), with kernel and
   marker traces — per-kernel dynamic demand in decode windows (kernels `^(_|triton)` excluded, as in G7).
7. `oc1_graph`: Qwen32B graph B1/C4096 kernel trace (ordinary graph-mode timeline: per-kernel durations, gaps) and one
   unprofiled ordinary run (timing reference for profiler distortion).

Pilot results are labelled pilot; they decide only what the dev job collects (section 6) and the OC4 route.

## 4. OC1 — timeline and demand (method)

Per decode tick (marker ranges): per kernel family (projection wvSplitK; attention; norm/elementwise; reduction/other,
classified by kernel name) — count, summed duration (eager, profiled), issued MFMA/VALU/VMEM/SALU/LDS instructions,
FETCH_SIZE. Useful FLOPs and logical bytes come from geometry (`oc_kernels.py` shapes, KV at context 4096). Graph-mode
timeline (kernel trace) gives ordinary per-kernel durations and inter-kernel gaps; the eager↔graph bridge is accepted
for a family only if kernel names and grid sizes match and the per-family duration ratio graph/eager is reported (no
silent join). Prefill: the pilot's hipBLASLt GEMM reference plus the G2 route traces (`prefill` case).

## 5. OC2/OC3 references (method)

- P_ref (arithmetic, projection route): `mfma4` useful FLOP/s at (mask, g, 1024 threads); `mfma4_lds` as the
  LDS-fed variant. Reported per cohort/mask/grid with trial spread. Not a decoder-derived rate.
- B_ref (memory): `read` at the matched launch (g workgroups × 1024 threads, 1 GiB, streaming) and at 8 blocks/CU;
  MALL-resident 64 MiB; ILP sweep for issue/concurrency (B vs loads in flight per thread). Physical HBM bytes remain NA
  unless a validated counter exists (G4 says FETCH_SIZE is pre-MALL on gfx942).
- O_ref: the tiny wvSplitK (m=64, k=256) per-call time at the same grid and mask.

## 6. OC6 — model, development and held-out sets (fixed now)

Family: decode projection, wvSplitK bf16, N = 1 token. For cell (shape m×k, grid g, mask n):

    C = issued_MFMA(m,k,g) / R_mfma4(n, g)        R_mfma4 = mfma4 wave-instructions per second (OC2)
    M = W(m,k) / B_ref(n, g)                       W = weight bytes, B_ref = matched streaming read (OC3)
    O = O_ref(n, g)                                tiny-shape per-call time (independently measured)
    T_pred = O + C + M − η · min(C, M)

issued_MFMA(m,k,g) is taken from the validated SQ_INSTS_MFMA per call of the projection route (pilot/dev counter
runs), as a function fitted exactly on geometry (expected form a·m·k + b·g; the pilot reports whether it is exact).
If SQ_INSTS_MFMA fails validation, C uses useful FLOPs / the mfma4 useful rate and is labelled a useful-work bound.
One η per cohort for the family, fitted by least squares on log time over the development cells; feasibility
max(C,M) ≤ T−O ≤ C+M is checked per cell and infeasible cells are reported, not forced. Identifiability: report the
η profile likelihood and the (O-scale, η) trade-off over a ±50 % O_ref scale.

- Development cells: Qwen32B and Qwen1.5B shapes (5 each), history `stream`, at (mask, grid) = (full, stock),
  (full, n_p), (n_p, n_p), plus MI300X (192, 192) / MI350X (224, 224).
- Held-out cells (never acquired before the freeze; no G8 exposure): Qwen2.5-7B and Mistral-7B shapes (all 5) and
  Qwen2.5-14B gate_up/down, same history, at the development (mask, grid) pairs plus one new pair per cohort
  (MI300X (224, 224), MI350X (160, 160)).
- Acceptance (pre-declared): per held-out cell |T_pred/T_meas − 1| ≤ 0.10 on the median of ≥ 3 acquisitions × 10
  trials; model accepted for the cohort if ≥ 90 % of held-out cells pass and the median signed error is within ±5 %.
  Also reported: per-model aggregate (sum over the 5 shapes, i.e. projection time per layer-set) error, bias, the worst
  cell. A failed model is a result; no refit on held-out data.
- Secondary transfer check (reported, not an acceptance criterion): predicted vs graph-mode measured projection-kernel
  durations in a Qwen2.5-7B B1 decode kernel trace acquired in the held-out job.

## 7. OC4 — activity (decision rule, fixed now)

Candidate A (per-CU cycle domain, dispatch aggregate): A_C = SQ_ACTIVE_INST_VALU-type cycles (MFMA counted as VALU on
CDNA), A_M = SQ_ACTIVE_INST_VMEM cycles, T = SQ_BUSY_CU_CYCLES (or per-CU cycles), all per CU. Usable for coupled
per-CU bounds max(0, A_C+A_M−T) ≤ X ≤ min(A_C, A_M) only if (i) both counters classify as per-CU "any active" (section 2
saturation test), (ii) on `mfma4` A_M/T ≤ 0.05 and on `read` A_C/T ≤ 0.2 (address arithmetic allowed), (iii) on `split`
(one workgroup type per CU, X = 0 by construction) the lower bound is ≤ 0.05·T and (iv) on `fused` the lower bound is
> 0 (overlap present by construction). Pass → `bounded (per-CU scope)`, never device HBM activity. Fail → unavailable.
Candidate B (ATT time series on one CU): usable only if the probe yields decoded per-instruction timelines on both the
projection route and the controls, and the same (i)–(iv) hold with the ATT-derived c(t), m(t) (m = a VMEM instruction
outstanding on the traced CU) and the estimator error on `split`/`fused` is reported. Scope: one CU/SE; replay and
serialization recorded. Neither candidate is device-level physical HBM activity; that remains unavailable.

## 8. OC5, OC7 (method)

OC5: mechanism table from G1 (tuned grid vs CU count), G5 (decoder × memory competitor), G6 controls and OC2/OC3
references (grid vs mask contrasts at fixed work; ILP response), each with control, observable, confound and falsifier.
OC7: reuse G3 decode/serving energy (D and PD scopes, gates unchanged); `model_inputs.json` and `oc_status.json` per
cohort folder.

## 9. Budget and order

Pilot 2 × ~2.6 h (≈ 0.8 units) → review → dev plan (≈ 2 × 2 h, ≈ 0.6 units) → fit + freeze commit → held-out
(≈ 2 × 1.5 h, ≈ 0.5 units) → OC5/OC7 synthesis after G1–G8 finishes. Two attempts maximum per step; correctness
failures are not retried; failures are preserved. No clock/power/driver/firmware change.

AMD acknowledgement for any publication: "This work was supported in part by Advanced Micro Devices, Inc. under the
AMD University Program's AI & HPC Cluster".
