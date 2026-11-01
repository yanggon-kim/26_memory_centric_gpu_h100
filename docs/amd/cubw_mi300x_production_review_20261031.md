# cubw MI300X production job 487341 review (31 October 2026)

## Run

| item | value |
|---|---|
| job | 487341 `cubw-mi300x-production` |
| partition, node | mi3008x, k002-003 |
| time | 2026-10-31 18:24–18:56 CDT (32 min) |
| release | c7cf8aa9 (v1.2) |
| plan | `$WORK/cubw_plans/mi300x_production_20261030_064338_c7cf8aa9.json` |
| gates | v1.2 re-gate of pilot 483498 (`CUBW_GATES`) |
| raw root | `$WORK/results_amd/amd_20261031_mi300x_cubw_v1_production_job487341` (410 files, `SHA256SUMS`) |
| compact tables | `results_published/amd_20261031_mi300x_cubw_v1_production_job487341/` |

## 1. Completion

- Released steps: 126/126 rc=0 on the first attempt.
  - placement: 48;
  - E2: 40 (Qwen32B 5 blocks, Qwen14B 3 blocks);
  - E5 kernel trace: 2;
  - E8: 36.
- E3 steps are journalled not_run: 193 (release_e3_minimal false, from leak_ok_v11 in 483498).
- E4 is in 483498 (3 acquisitions).

Gates:

- outputs_ok: Qwen32B C4096 runs 25/25 complete, one digest.
- placement_ok: pass.
- lifecycle_ok: false only because this job held no competitor cells. The gate requires at least one; no
  competitor ran, so there is no lifecycle failure.
- clock_ok: fails.
  - Pooled gfx 1294–2015 MHz (×1.56); memory 1068–1300 MHz.
  - Minimum trial coverage 0.495, just below the 0.5 coverage floor.
  - **Every E2 ratio is observational_clock_confounded.** No controlled noninferiority conclusion is published.

## 2. E2 — R(n) = T(n)/T(304), block-paired bootstrap, 95% CI (observational)

| n | Qwen32B B1/C4096 (5 blocks) | Qwen14B (3 blocks) |
|---:|---|---|
| 304 | 17.78 ms/step | 8.64 ms/step |
| 232 | 1.024 [1.020, 1.027] | 1.051 [1.047, 1.054] |
| 200 | **1.012 [1.010, 1.014]** | **1.027 [1.024, 1.030]** |
| 152 | 1.070 [1.064, 1.075] | 1.107 [1.104, 1.109] |
| 96 | 1.083 [1.075, 1.088] | 1.102 [1.096, 1.106] |

- n=200 has an upper CI ≤ 1.05 for both models. n=232 passes for Qwen32B only.
- Under the predeclared criterion (upper CI ≤ 1.05 **and** every gate, including clocks), no level qualifies.
  These are raw observational curves.
- n=200 being faster than n=232 matches both pilots. It is not explained here: clocks differ by cell (gfx ×1.32
  at 232 vs ×1.36 at 200 relative to 304), and per-XCD CU counts are 29 vs 25.

## 3. E5 — kernel-trace timelines (graph mode)

| n | trace slowdown | outputs | decode ranges | dispatches per tick |
|---:|---:|---|---:|---:|
| 304 | 1.055 | match | 13 | 731 |
| 200 | 1.063 | match | 13 | 731 |

Kernel trace is usable. Graph-mode counters remain unavailable (v1.1).

## 4. E8 — guardrails (3 acquisitions, ms/step; 304 / 200 / 96)

| case | 304 | 200 | 96 |
|---|---:|---:|---:|
| Qwen7B B8 C4096 decode | 6.70 | 7.82 (1.17×) | 10.81 (1.61×) |
| Qwen32B prefill wall, ms | 551 | 780 (1.42×) | 1242 (2.25×) |
| Qwen32B C16384 decode | 19.07 | 19.43 (1.02×) | – |
| Qwen32B C16384 prefill, ms | 2810 | 3820 (1.36×) | – |
| Mistral7B B1 decode | 5.12 | 5.32 (1.04×) | – |

Energy (Qwen32B B1, 10 trials × 3):

| n | J/token | power |
|---:|---:|---:|
| 304 | 13.83 | 752 W |
| 200 | 13.78 | 752 W |

Both cells sit at the 750 W cap, so cutting CUs does not reduce power on this stack; it only moves clocks.

## 5. Reading

- Small-batch decode keeps ≤ ~3% latency at 200 of 304 CUs on MI300X (observational).
- The guardrails show where that headroom ends:
  - batch 8: +17% at 200;
  - prefill: +42% at 200;
  - long-context prefill: +36% at 200.
- Removing CUs is therefore only cheap for the latency-bound decode phase.
