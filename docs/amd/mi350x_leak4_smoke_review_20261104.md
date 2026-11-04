# MI350X leak test v4 smoke 499331 — review (4 November 2026)

**Job:** 499331 on `mi3501x` (k007-005-v7, one MI350X), cubw v1.6, release f737d1b1.
- Ran 15 min. All 22 steps completed. Placement, lifecycle and outputs gates passed (4/4 runs, 1 digest).
- Pre-registration: [followups_v16_plan_20261104.md](followups_v16_plan_20261104.md) §A.
- Tables: `results_published/amd_20261104_mi350x_cubw_v1_6_leak4_smoke_job499331/`, with gates in
  `gates_smoke.json`.
- Pilot-labelled: one GPU, releases nothing.

## Result: v4 is sensitive, and the decoder does not leak on this GPU

| | n = 192 (probe 192–255) | n = 128 (probe 128–255) |
|---|---|---|
| probe launches per run (alone / natives / `cuprobe` positive) | ~94–97k | ~93–96k |
| probe launches per run (decoder runs, trial windows only) | 22.6k, 23.4k | 21.1k, 22.1k |
| alone: largest per-CU late fraction | 0.000 | 0.000 |
| native read/FMA on 0..n−1: flagged CUs (largest excess) | 0, 0 (0.0005) | 0, 0 (0.0003) |
| **required positive: v4 full-LDS probe on 0..n+7** | **8 of 8** (excess 0.73–0.75) | **8 of 8** (0.63–0.67) |
| **required positive: the decoder on 0..n+7** | **8 of 8** (0.45–0.49) | **8 of 8** (0.49–0.52) |
| recorded: FMA / read on 0..n+7 | 8 / 8 | 8 / 8 |
| **decoder on 0..n−1: flagged CUs (largest excess)** | **0** (0.0025) | **0** (0.0019) |

**Interpretation:**
- Removing the probe's continuous occupancy gives a clean signal.
- When anything foreign is masked onto a probe CU, the probe's start there is late in 45–75% of launches. That
  includes the real decoder. Otherwise it is late in < 0.3% of launches.
- So the test now detects the case that matters: decoder work leaking onto the competitor partition.
- The masked decoder does not leak, on this one GPU.

## Decision

Per the pre-registered rule, the smoke shows v4 sensitive at both levels. The held 8-GPU job **499332** (same
release) was released unchanged. Its gate requires a pass on all 16 decoder tests (8 GPUs × 2 levels) before
E3-minimal runs.
