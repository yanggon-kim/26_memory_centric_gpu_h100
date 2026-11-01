# Compute/bandwidth follow-up — results (cubw v1–v1.2, 1 November 2026)

## Scope and status

This is the versioned results report required by [compute_bandwidth_followup_20261026.md](compute_bandwidth_followup_20261026.md)
§11. It does not edit older reports or counts; corrections to them are listed in
[cubw_e0_reconciliation_20261026.md](cubw_e0_reconciliation_20261026.md).

- **MI300X and MI325X are separate cohorts.** No cell is pooled across SKUs or allocations.
- **The study is not complete.** Fixed-clock qualification, E6, E7 model fitting, true HBM attribution and the
  durable off-cluster raw archive are unavailable or outstanding (§7).

## 1. Jobs and evidence

| job | SKU | kind | release | outcome | review |
|---|---|---|---|---|---|
| 481167 | MI300X | pilot | v1 (2da9ae35) | ran; v1 lifecycle/outputs gates hit tool defects → v1.1 | [cubw_pilot_review_20261028.md](cubw_pilot_review_20261028.md) |
| 483498 | MI300X | auto (re-pilot → production) | v1.1 (79d8b597) | pilot clean; v1.1 auto-gate defect blocked production except E4 ×3 → v1.2 | [cubw_auto_review_20261030.md](cubw_auto_review_20261030.md) |
| 483499 | MI325X | auto (reduced pilot → production) | v1.1 | same defect; E4 ×3 ran | [cubw_mi325x_auto_review_20261031.md](cubw_mi325x_auto_review_20261031.md) |
| 487341 | MI300X | production | v1.2 (c7cf8aa9) | 126/126 released steps rc=0 | [cubw_mi300x_production_review_20261031.md](cubw_mi300x_production_review_20261031.md) |
| 489594 | MI325X | production | v1.2 (99bcd1ad) | 167/167 released steps rc=0 | [cubw_mi325x_production_review_20261101.md](cubw_mi325x_production_review_20261101.md) |

- Compact tables, journals, plans, environment, gates and per-root `SHA256SUMS` are under
  `results_published/amd_2026102[8]*_cubw_*`, `amd_2026103[01]*_cubw_*` and `amd_20261101_*_cubw_*`.
- Raw roots are in `$WORK/results_amd/`. Every attempt, including failures, is kept.
- Repairs were made only for diagnosed tool and reader defects (v1.1, v1.2), with at most one re-execution per
  step. No gate threshold was changed after data existed.

## 2. Gate outcomes (what passed, failed, was blocked)

| gate | MI300X | MI325X |
|---|---|---|
| placement (8 GPUs × {304, 232, 200, 152, 96, competitor 232–303}, per-XCD balanced) | pass | pass |
| lifecycle (competitor start/stop, PID-gone, per-GPU residuals, tag sweep) | pass | pass |
| outputs (one digest across all unprofiled Qwen32B C4096 runs) | pass (25/25 production) | pass (66/66 production) |
| leak_ok_v11 (decoder probe ≥ 0.85 × the gentlest native co-runner) | **fail** (0.77–0.82) | pass (0.87–0.97) |
| pooled ≤ 5% clock envelope | **fail** (gfx ×1.56) | **fail** (gfx ×1.28) |
| release E2 / E5+E8 | yes / yes | yes / yes |
| release E3-minimal / E3-full | no / no | yes / no |

Because the clock gate failed on both SKUs, **every quantitative result is observational
(clock-confounded)**. Under the predeclared E2 rule (upper 95% CI of R(n) ≤ 1.05 *and* every gate),
**no CU level qualifies as noninferior on either SKU**. The clocks were not changed, as the assignment does not
authorize clock or power changes.

## 3. H1 — Is arithmetic over-provisioned for small-batch decode? (E2)

R(n) = T(n)/T(304). Values are means with 95% intervals from a block-paired bootstrap; all are observational.

| n | MI300X Qwen32B (5 blocks) | MI300X Qwen14B (3) | MI325X Qwen32B (5) | MI325X Qwen14B (3) |
|---:|---|---|---|---|
| 304 | 17.78 ms | 8.64 ms | 15.34 ms | 7.79 ms |
| 232 | 1.024 [1.020, 1.027] | 1.051 [1.047, 1.054] | 1.074 [1.071, 1.077] | 1.133 [1.093, 1.210] |
| 200 | **1.012 [1.010, 1.014]** | **1.027 [1.024, 1.030]** | 1.056 [1.040, 1.085] | **1.047 [1.044, 1.049]** |
| 152 | 1.070 [1.064, 1.075] | 1.107 [1.104, 1.109] | 1.187 [1.165, 1.211] | 1.222 [1.167, 1.254] |
| 96 | 1.083 [1.075, 1.088] | 1.102 [1.096, 1.106] | 1.186 [1.175, 1.204] | 1.170 [1.169, 1.171] |

**Reading (observational):**

- On MI300X, B1 decode loses ≤ 3% with a third of the CUs removed (n=200). H1 is supported in direction, but it is
  not qualified.
- On MI325X, the same cuts cost 2–5× more (+5.6% at 200, +19% at 152). With ~10% more HBM bandwidth, compute is a
  larger share of each decode step, so the arithmetic headroom is SKU-dependent and shrinks as bandwidth grows.
  This is the expected shoreline trade-off, now seen directly.
- **Unexplained anomaly:** n=200 is faster than n=232 on both SKUs, in every production model and in both pilots.
  - Candidate explanations are untested: per-cell clock differences, and the per-XCD CU count (25 vs 29)
    interacting with workgroup scheduling.
  - It weakens any smooth κ(n) model; fine-grained levels between 200 and 304 would be the next measurement.

**Guardrails (E8, n=200 vs 304)** show where the headroom ends:

| workload | MI300X | MI325X |
|---|---:|---:|
| Qwen7B B8 decode | 1.17× | 1.29× |
| Qwen32B prefill | 1.42× | 1.44× |
| C16384 prefill | 1.36× | 1.47× |
| C16384 decode | 1.02× | 1.02× |

Removing CUs is cheap only for latency-bound B1 decode, including at long context. It is not cheap for batched
decode or for prefill.

## 4. H3 — Request capacity (E4, 3 acquisitions per SKU, ratio to 304)

| control | MI300X 232 / 200 / 152 / 96 | MI325X 232 / 200 / 152 / 96 |
|---|---|---|
| streaming read, 9728-block grid | 1.01 / 1.01 / 1.01 / 0.97 | 1.01 / 1.02 / 1.00 / 0.95 |
| read, ILP 1 | 0.84 / 0.88 / 0.72 / 0.57 | 0.83 / 0.89 / 0.70 / 0.55 |
| 8 MiB reuse (cache-resident) | 0.76 / 0.69 / 0.57 / 0.38 | 0.71 / 0.64 / 0.50 / 0.36 |
| FMA | 0.82 / 0.76 / 0.65 / 0.47 | 0.76 / 0.71 / 0.56 / 0.40 |
| MFMA | 0.76 / 0.67 / 0.54 / 0.36 | 0.66 / 0.58 / 0.42 / 0.30 |
| dependent chase latency | within ±7% | within ±6% |

- With enough parallelism, even 96 CUs saturate HBM (≥ 0.95).
- Delivery falls only when requests per CU are limited (ILP 1) or the data is on-chip.
- The decode penalty curve tracks the ILP-1 / reuse rows more closely than the streaming row. The step between
  200 and 152 appears in both, which is consistent with a request/MLP knee in that range.
- This supports H3 qualitatively. No held-out predictive model was fitted (E7 is outstanding), so H3's
  "predicts held-out sensitivity" criterion is **not run**.

## 5. H2 / H4 — Shared memory service (E3)

**MI325X, E3-minimal released.** Ratios to the absent cell at the same n, Qwen32B, 5 blocks:

| treatment | n=232 | n=152 |
|---|---|---|
| mem50 | 1.052 [1.016, 1.083] | 1.090 [1.078, 1.104] |
| mem100 | 1.146 [1.104, 1.186] | 1.182 [1.164, 1.200] |
| arith100 | 1.003 [0.977, 1.024] | 1.018 [1.014, 1.022] |

- **H2 is supported (observational).** At fixed decoder masks:
  - latency rises monotonically with the duty of the memory competitor;
  - an arithmetic competitor on the same 72 CUs costs ≤ 2%;
  - mem100 vs arith100 is 1.14–1.16.
  - The leak gate passed, so this is attributed to shared memory service, not CU leakage. Clocks are not
    controlled, and competitor cells ran at higher clocks, which biases these slowdowns downward.
- **H4 shared-service balance is demonstrated (observational), not a stock-GPU win.**
  - T(152, absent) / T(232, mem100) = 0.924 [0.911, 0.939].
  - T(152, absent) / T(232, mem50) = 1.006 [0.997, 1.015].
  - Restoring the service taken by a full-duty streaming competitor is worth more than 80 CUs.
  - The uncontended 304-CU baseline remains faster than both, so this does not show that a design with more
    bandwidth beats the stock GPU.

**MI300X, E3 not released (leak_ok_v11 failed).** Only pilot single acquisitions exist:

| run | mem100 | arith100 | T(152, absent) / T(232, mem100) |
|---|---:|---:|---:|
| 481167 | 1.27 | 1.06 | 0.83 |
| 483498 | 1.23 | 1.07 | – |

- These match the MI325X pattern and the earlier campaigns, but are **not released results**.
- The leak suite showed the decoder slowing an FMA probe on the reserved CUs about as much as a native streaming
  co-runner does. That is consistent with memory contention, but it does not exclude leakage under the
  predeclared rule.

## 6. Energy (E8, Qwen32B B1, 3 × 10 trials)

| | n=304 | n=200 |
|---|---|---|
| MI300X J/token (mean W) | 13.83 (752) | 13.78 (752) |
| MI325X J/token (mean W) | 15.39 (984) | 15.89 (972) |

- Both SKUs sit at their power cap whether or not CUs are masked.
- Masking CUs saves no measurable board energy on this stack: p ≈ 1, so E'/E ≈ 1/S. Clocks rise to fill the cap.
- Silicon or component savings from physically removed CUs cannot be measured by masking.

## 7. Unavailable, not run, outstanding

| item | status |
|---|---|
| Fixed-clock / clock-qualified comparisons | **unavailable**: changing clocks or power is not authorized; the pooled gate failed on every cohort |
| E3 full grid (mem25/75, n=200/96, Qwen14B) | blocked by the clock gate (both SKUs); MI300X E3-minimal is also blocked by leak_ok_v11 |
| E5 graph-mode hardware counters (byte domains) | **unavailable**: rocprofv3 `--pmc` crashes or hangs on graph decode; kernel-trace timelines only (trace slowdown 1.05–1.09, outputs match) |
| E6 efficient-kernel controls | **not run**: no shape benchmarks were built in this assignment |
| E7 model fit and held-out validation | **not run**: the decisive contrasts exist (§5), but no s/g fit and no held-out prediction |
| True HBM/PHY attribution, a design exceeding stock bandwidth | **unavailable** on stock hardware |
| Durable raw archive off-cluster with checksum readback | **outstanding**: raw roots (~1.2 GB for the four cubw roots after the pilot; the pilot root is 14 GB) and SHA256SUMS exist only on `$WORK`, which is not backed up |
| n=200 vs 232 non-monotonicity | open (§3) |

## 8. Conclusion (bounded)

On both SKUs the observational evidence points the same way:

- small-batch decode tolerates a large CU cut;
- the decoder is sensitive to shared memory service, not to arithmetic competition;
- request-level parallelism, not peak HBM bandwidth, is what CU removal takes away.

The sizes differ by SKU. MI325X, with more bandwidth, has markedly less CU headroom, consistent with the
shoreline argument that compute and bandwidth must be rebalanced together. Batched decode and prefill are clear
negative regimes. Masking does not save board energy at the power cap.

None of these results is clock-qualified, so none is a controlled noninferiority or architecture claim.
