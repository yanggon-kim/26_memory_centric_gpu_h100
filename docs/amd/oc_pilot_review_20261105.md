# OC-v0 pilot review (OC0–OC4) — MI300X job 502623, MI350X job 502625

Assignment: `docs/amd/overlap_calibration_followup_20261105.md`; pre-registered plan:
`docs/amd/overlap_calibration_execution_plan_20261105.md` (commit d6dced90). Pilot cells are labelled pilot and judge
nothing; this review fixes the development job (plan section 6) and records amendments with their reasons.

| | MI300X | MI350X |
|---|---|---|
| job / partition | 502623 / mi3001x | 502625 / mi3501x |
| release | d6dced90 snapshot | d6dced90 snapshot |
| steps | 45/45 rc 0 | 45/45 rc 0 |
| wall | 23 min 48 s | 17 min 08 s |
| published | `results_published/amd_20261105_mi300x_oc_v0_pilot_job502623/` | `results_published/amd_20261105_mi350x_oc_v0_pilot_job502625/` |

Analysis: `analyze_oc.py` (this commit) on a `$WORK` copy of each raw root, run on a devel CPU node; raw roots are
unchanged (SHA256SUMS copied as `raw_SHA256SUMS`).

## OC0 capability probe

| Route | MI300X | MI350X |
|---|---|---|
| PC sampling host_trap (time), container rocprofv3 1.3.2 | samples (2.98 MB CSV) | samples (1.78 MB CSV) |
| PC sampling stochastic (cycles) | samples (2.07 MB CSV) | samples (1.83 MB CSV) |
| ATT, projection route (wvSplitK), container decoder | decoded, 1 dispatch, 16 waves on CU 1 | decoded, 1 dispatch, 16 waves |
| ATT, native split / fused / mfma4 / read, host rocprofv3 1.1.0 + copied decoder | decoded | decoded |

ATT yields, per traced wave, every issued instruction with issue time, stall and latency (gfx clock), plus the
decoder's sampled SQ counters; this is a per-CU time-domain observable on both architectures.

## Counter validation (plan section 2)

| Metric | Test | MI300X | MI350X | Status |
|---|---|---|---|---|
| SQ_INSTS_VMEM_RD | read 1 GiB, n_float4/64 | 1.000 | 1.000 | validated |
| SQ_INSTS_MFMA | mfma4 w1/w4, 8·iters·waves | 1.000 / 1.000 | 1.000 / 1.000 | validated |
| SQ_INSTS_VALU_MFMA_MOPS_BF16 | per 4x4x4 bf16 instruction | 4 | 4 | measured-unvalidated (constant reported) |
| SQ_VALU_MFMA_BUSY_CYCLES | per 4x4x4 bf16 instruction | 8 | 8 | measured-unvalidated; a fixed charge per instruction (work, not observed time) |
| SQ_INSTS_VALU | fma w1/w4, ≥ 8·iters·waves | 0.5005 | 0.5005 | **invalid as pre-registered**; see amendment A1 |
| FETCH_SIZE | read 1 GiB, KiB | 1.000 | (0.50 scale, G4) | validated (gfx942); gfx950 unit per G4 |
| TCC_EA0_RDREQ / _DRAM | read 1 GiB, 128 B requests | 1.000 | 1.000 | validated |
| TCC_EA0_RDREQ_32B/64B/128B | | not listed | listed | unavailable (gfx942) |
| TCC_EA0_RDREQ_LEVEL | in flight rises ilp1→ilp8; latency | rises; 3155 cycles | rises; 2008 cycles | validated |
| SQ_ACTIVE_INST_VALU | w1 vs w4 | = SQ_INSTS_VALU exactly | same | **instruction count, not a time counter** |
| SQ_ACTIVE_INST_VMEM | | 0 on every load including read | same | not counting (unidentified) |
| SQ_ACTIVE_INST_ANY | | sum of instruction counts | same | instruction count |

A1 (amendment, SQ_INSTS_VALU): the expectation assumed one `v_fma_f32` per `fmaf`. Disassembly of the v7 binary shows
the 8 `fmaf` per iteration compile to 4 `v_pk_fma_f32`; with that expectation the ratio is 1.001 (validated). Both
verdicts are kept in `oc_tables.json` (`status`, `status_isa`). SQ_INSTS_VALU enters no model.

A2 (reader fix): `oc_counters.py` ≤ 6f56b1d9 filtered kernel names by the `k_` prefix and dropped the template kernels
(`void k_read_ilp<K>`); the reader now re-parses the kept raw CSVs, which is how the ilp1/ilp8 level test above was
evaluated. The collector is fixed for later jobs.

## OC4 — activity

Candidate A (dispatch-aggregate SQ_ACTIVE_INST_* over SQ_BUSY_CU_CYCLES): **unavailable** on both cohorts — the
"ACTIVE" counters count instructions (VALU equals SQ_INSTS_VALU to the unit; VMEM is zero), so check (i) fails.

Candidate B (ATT, one CU, all SIMDs; `scripts/amd/oc/oc_att.py`): c(t) = some MFMA executing on the CU, m(t) = some
VMEM load outstanding (issue → end of the retiring `s_waitcnt vmcnt`). Every traced load was retired by a waitcnt that
stalled, so the upper estimate of m is tight on these kernels.

| Route (traced CU) | MI300X A_C/T | A_M/T | X/T | MI350X A_C/T | A_M/T | X/T |
|---|---:|---:|---:|---:|---:|---:|
| mfma4 | 0.944 | 0.000 | 0.000 | 0.952 | 0.000 | 0.000 |
| read | 0.000 | 0.971 | 0.000 | 0.000 | 0.953 | 0.000 |
| split | 0.983 | 0.000 | 0.000 | 0.998 | 0.000 | 0.000 |
| fused | 0.234 | 0.970 | 0.234 | 0.472 | 0.948 | 0.472 |
| wvSplitK (Qwen32B --light dispatch) | 0.043 | 0.954 | 0.043 | 0.131 | 0.907 | 0.131 |

Checks (ii) mfma4 A_M/T ≤ 0.05 and read A_C/T ≤ 0.2, (iii) split X ≤ 0.05 T, (iv) fused X > 0 all hold → **bounded
(per-CU scope)** on both. Caveat: the traced CU ran only MFMA workgroups of `split`, so (iii) holds trivially; the
development job traces split at 4 and 8 workgroups per CU on CUs 0 and 1 so that one traced CU runs both types. On the
projection route every MFMA cycle of the traced CU fell inside a load-outstanding interval (X = A_C), i.e. compute was
fully hidden under outstanding memory on that CU. This is per-CU scope, never device HBM activity.

Cross-check: issued MFMA per decode tick (eager, OC1) is 124.8 M on both cohorts, ≈ 3.9 × the useful FLOPs / 2048
(N = 1 occupies one of the four 4×4×4 columns). With P_ref (mfma4 at the stock grid) and B_ref (matched read), C/M for
the Qwen32B projections is ≈ 0.04 (MI300X), in line with the ATT A_C/T. Consequence for OC6: C ≪ M, so η will be
weakly identified; the plan's profile-likelihood and (O-scale, η) reporting are therefore required outputs, and the
ATT X/A_C per shape is recorded beside the fitted η as an independent per-CU observation.

## OC2/OC3 references and kernel controls (pilot, one acquisition)

| Cell | MI300X | MI350X |
|---|---|---|
| mfma4 (full, full) | 454.6 TFLOP/s | 522.7 TFLOP/s |
| read (full, full) / g8x | 4474 / 4444 GB/s | 6461 / 6509 GB/s |
| read MALL-resident 64 MiB (full, full) | 5824 GB/s | 7503 GB/s |
| ILP 1/2/4/8 (full, full) | 4514 / 4326 / 4169 / 4212 GB/s | 6205 / 6524 / 5885 / 5946 GB/s |

Full tables: `oc_summary.md` per cohort. Qwen32B wvSplitK stream (full, stock): gate_up 137.3 µs, down 72.8 µs,
lm_head 380.2 µs (MI300X); 89.7, 46.6, 228.5 µs (MI350X).

## OC1 timeline and demand (Qwen32B graph B1, context 4096)

| Per decode tick | MI300X | MI350X |
|---|---:|---:|
| projection (wvSplitK) busy, graph | 14.38 ms | 10.03 ms |
| projection, eager profiled | 13.97 ms | 10.03 ms |
| kernel union, graph | 17.34 ms | 12.32 ms |
| inter-kernel gaps, graph | 2.10 ms | 1.14 ms |

The eager↔graph bridge holds for the projection family (ratio 1.03 / 1.00).

## Decisions for the development job (plan section 6)

- Cells unchanged: Qwen32B and Qwen1.5B shapes (5 each), history `stream`, (mask, grid) = (full, stock), (full, n_p),
  (n_p, n_p), and MI300X (192, 192) / MI350X (224, 224); 3 acquisitions × 10 trials in randomized order, with the
  matched `mfma4` and `read` references and the tiny-shape O_ref acquired in the same randomized sequence.
- issued_MFMA(m, k, g): one call per (shape, grid) under `--pmc` with per-dispatch rows (`oc_kernels.py --count`), at the
  full mask for every development grid and again at n_p to confirm mask independence.
- ATT per development shape at (full, full) and (n_p, n_p), plus split traces with both workgroup types per CU.
- Budget: ≈ 0.22 (MI300X) + 0.31 (MI350X) units estimated; pilots took 0.05 + 0.05 units.
- No change to clocks, power, drivers or firmware; frozen G8 predictions untouched.
