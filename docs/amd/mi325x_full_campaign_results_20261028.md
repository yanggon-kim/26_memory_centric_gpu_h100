# MI325X full campaign (jobs 470176 + 481072): results (28 October 2026)

This is a **separate MI325X cohort**: the same scripts, configs and lanes as the MI300X campaign (job 468885),
repeated on one 8×MI325X node. It extends the MI300X study but does not replace it. The single-MI300X study
in [compute_bandwidth_followup_20261026.md](compute_bandwidth_followup_20261026.md) remains primary.
Almost no cell passes the declared clock gate, so every comparison below is **observational**:
power-managed clocks, with no fixed-clock causality.

## 1. What ran

| phase | job | node | when (CDT) | content | raw root |
|---|---|---|---|---|---|
| 0–1 | 470176 attempt 3 | k002-001 | 2026-10-25 17:57–21:20 (then idle to TIMEOUT 05:57, driver bug) | M1 baselines, GPU-6 repro, text check, M2 counters, M3 rocprof-compute, M4 service, M5 pressure, M6 CU masks + batch sweep, M8 energy, energy-vs-CU, prefill-vs-CU | `$WORK/results_amd/amd_20261025_mi325x_full_job470176` (3.18 GB) |
| 2–4 | 481072 | k002-001 | 2026-10-28 07:20–09:17 | J6 same-GPU competitor; TP2/TP4/TP8 at full and compute-matched CUs; TP at 152-CU-matched totals | `$WORK/results_amd/amd_20261028_mi325x_full_job481072` (0.11 GB) |

Attempts 1 and 2 of 470176 (Oct 20 and Oct 23) ended in NODE_FAIL. Their partial phase-1 roots are kept on disk as
failures and are superseded here. The bug that left 470176 idle (the phase `wait` also waited on the node monitor) was
fixed in d013176 before 481072 ran.

Compact evidence: `results_published/amd_20261025_mi325x_full_job470176/` and
`results_published/amd_20261028_mi325x_full_job481072/`. These hold summaries, configs, preflight, `RUN_STATUS.log`
and a SHA-256 manifest of every raw file left on the cluster (see [artifacts.md](artifacts.md)).

## 2. Completion. Command rc=0 is not qualification

| family | cells / groups | acquisitions | failed attempts retained |
|---|---|---|---|
| M1 baselines | 16 conditions | 3/3 each (10 trials) | 3 |
| M1 GPU-6 repro / text | 8 / 2 conditions | 3/3 each | 0 |
| M6 CU masks (incl. batch sweep) | 56 cells | 3/3 each | 6 |
| M8 prefill vs CU | 4 cases × 4 levels | 3/3 each | 1 |
| M8 energy | 8 cases | **2/3 each**: every `a0` failed twice at start (`AMDSMI_STATUS_FILE_ERROR` reading ASIC info) | 3 dirs + retries |
| M8 energy vs CU | 6 cells | 1–2 of 3 | 1+ |
| M5 pressure | 5 cases | 3, except bridge B1/C1024 and qwen7b at 2 | – |
| M4 service | 140 rows | 3 each | – |
| M2 counters | 12 cases × 4 groups = 48 | 48/48 groups done | 113 failed/interrupted attempts (all kept) |
| M3 rocprof-compute | 13 sets | **5/13 completed**. mem_thruput, roof and three service sets failed (illegal memory access / profiling execution failed) | – |
| J6 competitor | 4 cases × 6 treatments | 3/3 each (72 runs) | 0 |
| TP (2/4/8, full/matched/matched152) | 36 rows | 3/3 each (108 runs) | 0 |

## 3. Clock gate (declared pooled ≤5% envelope, unfiltered)

- M1: **1/16** conditions pass (bridge eager B1/C1024). On MI300X it was 2/16 production conditions.
- M6: **6/56** cells pass. On MI300X it was 19/56.
- M1 GPU-6 repro: 0/8. Text check: 0/2.

GFX clocks ranged about 1405–2095 MHz across cells. The MI325X clock gate was not met, so the fixed-clock conclusion is unavailable for this cohort as well.

## 4. Key tables (MI325X; MI300X published values for like-for-like rows, labelled)

**M1 decode, graph mode, ms/step (HIP-event, mean of 3 acquisitions)**

| case | MI325X | MI300X (468885) | MI325X / MI300X |
|---|---:|---:|---:|
| bridge B1/C1024 | 2.208 | 2.187 | 1.009 |
| qwen7b B1/C4096 | 4.351 | 4.682 | 0.929 |
| qwen7b B8/C4096 | 6.270 | 6.654 | 0.942 |
| mistral7b B1/C4096 | 4.815 | 5.111 | 0.942 |
| qwen14b B1/C4096 | 7.816 | 8.638 | 0.905 |
| qwen32b B1/C4096 | 15.757 | 17.532 | 0.899 |

**M6 decode latency vs CUs, ratio to 304 CUs (graph mode)**

| case | MI325X 232 / 152 / 96 | MI300X 232 / 152 / 96 |
|---|---|---|
| qwen32b B1/C4096 | 1.031 / 1.115 / 1.124 | 1.034 / 1.084 / 1.092 |
| qwen14b B1/C4096 | 1.091 / 1.152 / 1.172 | 1.046 / 1.097 / 1.105 |
| qwen7b B1/C4096 | 1.090 / 1.204 / 1.353 | 1.056 / 1.132 / 1.239 |
| qwen7b B8/C4096 | 1.280 / 1.602 / 1.896 | 1.192 / 1.425 / 1.637 |
| bridge B1/C1024 | 1.045 / 1.167 / 1.260 | 1.015 / 1.128 / 1.209 |

Batch sweep, qwen7b C1024, MI325X, ratio at 96 CUs: 1.314 (B1), 1.562 (B4), 1.989 (B16), 1.596 (B32), 1.716 (B64).

**Prefill vs CUs (ratio to 304), MI325X / MI300X:** qwen32b B1/C4096 at 96 CUs 2.70 / 2.25; qwen7b B8/C4096 at 96 CUs
2.53 / 2.17.

**M4 service at 304 CUs (MI325X / MI300X):**
- 4 GiB streaming read: 5021 / 4549 GB/s.
- MFMA: 1215 / 1071 TFLOP/s.
- FMA: 106.5 / 94.9 TFLOP/s.

At 96 CUs, MI325X keeps 80.8% of read bandwidth and 31.3% of MFMA throughput. MI300X keeps 87.4% and 35.6%.

**M8 energy, graph, 2 acquisitions (MI325X) vs 3 (MI300X), J/token at mean W:**

| case | MI325X | MI300X |
|---|---|---|
| qwen32b B1 | 15.43 at 967 W | 14.00 at 751 W |
| qwen7b B1 | 3.96 at 893 W | 3.59 at 749 W |
| qwen7b B8 | 0.708 at 853 W | 0.659 at 751 W |

Energy vs CUs, MI325X, qwen32b B1, J/token: 15.45 (304, 1 acquisition), 16.89 (152, 2), 16.80 (96, 1).

**J6 same-GPU competitor** (decoder CUs 0–231, competitor 232–303; slowdown vs solo)

| case | read25 | read50 | read75 | read100 | fma100 |
|---|---|---|---|---|---|
| qwen32b B1, MI325X | 1.029 | 1.073 | 1.101 | 1.146 (1233 GB/s) | 1.015 |
| qwen32b B1, MI300X | 1.059 | 1.114 | 1.172 | 1.247 (1094 GB/s) | 1.069 |
| qwen7b B1, MI325X | 1.031 | 1.060 | 1.090 | 1.157 (1492 GB/s) | 0.998 |
| qwen7b B8, MI325X | 1.045 | 1.051 | 1.074 | 1.111 (1623 GB/s) | 1.004 |
| bridge B1, MI325X | 1.051 | 1.085 | 1.090 | 1.159 (1650 GB/s) | 1.001 |

**TP speedup vs TP1, qwen32b B1/C4096** (wall ms/step vs M1 HIP-event; MI325X / MI300X):
- Full CUs: TP2 1.62 / 1.63; TP4 2.34 / 2.43; TP8 2.90 / 3.16.
- Matched to 304 total CUs: TP2 1.36 / 1.48; TP4 1.53 / 1.86; TP8 1.40 / 1.74.
- Matched to 152 total CUs, vs TP1 at 152: TP2 1.24 / 1.39; TP4 1.11 / 1.33.

Output IDs matched TP1 in every row except TP4 qwen7b B8 (full and matched). BF16 reduction order can legitimately
change greedy tokens.

M2 counters: for all 12 counter cases, measured read bytes per decode tick are within 0.14–3.6% of the analytical logical demand. For example,
qwen32b eager B1/C4096 shows 65.14 GB measured vs 65.05 GB logical. FETCH_SIZE is pre-MALL, as on MI300X.
M5 KV-pressure: pressure/normal 1.002–1.011, with identical outputs.

## 5. Caveats

- This is an observational MI325X cohort under power management. Only 1/16 M1 conditions and 6/56 M6 cells pass the
  clock gate. Its CU-mask ratios mix arithmetic restriction with clock changes, because restricted cells run at higher
  GFX clocks.
- MI325X vs MI300X comparisons are cross-node, cross-SKU and cross-date, and both sides are clock-confounded. They
  describe this pair of runs, not a controlled SKU effect.
- Energy has 2 of 3 acquisitions per case, and energy-vs-CU has 1–2. rocprof-compute covers only 5/13 sets.
- The TP rows use phase-1 M1/M6 baselines from job 470176, which ran on the same node 2.5 days earlier. The summary
  was produced through a view directory that links 470176 `m1`/`m6_cumask` with 481072 `tp`.
  The summarizers themselves were not changed.
- All failed and interrupted attempts are kept in the raw roots and indexed by the manifests.
