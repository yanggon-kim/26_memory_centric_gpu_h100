# cubw MI325X production job 489594 review (1 November 2026)

## Run

| | |
|---|---|
| Job | 489594 `cubw-mi325x-production` |
| Partition / node | mi3258x, k002-001 |
| Time | 2026-11-01 10:37–11:28 CST (51 min) |
| Release | 99bcd1ad (v1.2 scripts) |
| Plan | `$WORK/cubw_plans/mi325x_production_20261031_041507_99bcd1ad.json` |
| Gates | v1.2 re-gate of pilot 483499 (copied to the root as `gates.json`) |
| Raw root | `$WORK/results_amd/amd_20261101_mi325x_cubw_v1_production_job489594` (595 files, `SHA256SUMS`) |
| Compact tables | `results_published/amd_20261101_mi325x_cubw_v1_production_job489594/` |

## 1. Completion and gates

**Steps run:** 167/167 released steps returned rc=0 on the first attempt.

| Group | Steps |
|---|---:|
| Placement | 48 |
| E2 (Qwen32B ×5, Qwen14B ×3 blocks) | 40 |
| E3-minimal (Qwen32B; 232/152; absent/mem50/mem100/arith100 ×5, plus the inactive control) | 41 |
| E5 ktrace | 2 |
| E8 | 36 |

- The 152 full-grid E3 steps are journalled not_run, because release_e3_full is false.
- E4 was run in 483499.

**Gates:**
- Placement, lifecycle and outputs pass. Qwen32B C4096 gives 66/66 complete with one digest.
- The clock gate fails: pooled gfx 1630–2083 MHz (×1.28), memory 1227–1500 MHz. **Every ratio below is
  observational_clock_confounded.**

`summary.md` in this root lists `leak_ok_v11`/`release_e3_minimal` as false. That is because the production root
holds no leak suite to recompute them from. The release actually applied is the pilot's v1.2 gate in
`gates.json`, which has release_e3_minimal true.

## 2. E2 — R(n) = T(n)/T(304), block-paired bootstrap (observational)

| n | Qwen32B B1/C4096 (5 blocks) | Qwen14B (3 blocks) |
|---:|---|---|
| 304 | 15.34 ms/step | 7.79 ms/step |
| 232 | 1.074 [1.071, 1.077] | 1.133 [1.093, 1.210] |
| 200 | 1.056 [1.040, 1.085] | 1.047 [1.044, 1.049] |
| 152 | 1.187 [1.165, 1.211] | 1.222 [1.167, 1.254] |
| 96 | 1.186 [1.175, 1.204] | 1.170 [1.169, 1.171] |

- No level has an upper CI ≤ 1.05 for Qwen32B.
- MI325X restriction costs are roughly 2–5× MI300X's at the same n.
- n=200 again beats n=232. For Qwen14B, n=96 beats n=152.

## 3. E3-minimal — T(n, z) ratio to absent at the same n (Qwen32B, 5 blocks, observational)

| treatment | n=232 | n=152 | competitor achieved |
|---|---|---|---|
| mem50 | 1.052 [1.016, 1.083] | 1.090 [1.078, 1.104] | 556 / 652 GB/s |
| mem100 | 1.146 [1.104, 1.186] | 1.182 [1.164, 1.200] | 1175 / 1331 GB/s |
| arith100 | 1.003 [0.977, 1.024] | 1.018 [1.014, 1.022] | 14.4 / 14.8 TFLOP/s |
| inactive (1 block) | 1.021 [0.997, 1.047] | – | – |

**Contrasts:**
- mem100 / arith100 is 1.142 at n=232 and 1.161 at n=152.
- Latency rises monotonically with memory-competitor duty, at both n.

**Shared-service balance** (not a stock-GPU win):
- T(152, absent) / T(232, mem100) = **0.924 [0.911, 0.939]**. A decoder with 80 fewer CUs but uncontended memory
  service beats a 232-CU decoder sharing service with a full-duty streaming competitor.
- T(152, absent) / T(232, mem50) = 1.006 [0.997, 1.015]. Roughly half-duty competition offsets the 232→152 CU
  cut.

## 4. E5 and E8

**E5 kernel trace:**

| | n=304 | n=200 |
|---|---:|---|
| Trace slowdown | 1.076 | 1.087 |
| Outputs | match | match |

Both runs have 13 decode ranges and 731 dispatches per tick.

**E8** (3 acquisitions each):

| case | 304 | 200 | 96 |
|---|---:|---:|---:|
| Qwen7B B8 decode, ms/step | 6.31 | 8.16 (1.29×) | 12.12 (1.92×) |
| Qwen32B prefill wall, ms | 480 | 694 (1.44×) | 1399 (2.91×) |
| Qwen32B C16384 decode | 16.72 | 17.09 (1.02×) | – |
| Qwen32B C16384 prefill, ms | 2470 | 3636 (1.47×) | – |
| Mistral7B B1 decode | 4.96* | 5.00 (1.01×) | – |
| Energy J/token (mean W) | 15.39 (984) | 15.89 (972) (1.03×) | – |

\* One of the three acquisitions at 304 is slow (5.33 vs 4.77 ms/step). Without it the 200/304 ratio is 1.05.
