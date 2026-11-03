# MI350X smoke runs — review (2 November 2026)

These are one-GPU smoke runs on `mi3501x` (k007-005-v7, one MI350X slice) before the 8-GPU campaign. They are
pilot-labelled, single acquisitions, and release nothing. Plan: [mi350x_campaign_plan_20261102.md](mi350x_campaign_plan_20261102.md).
The compact tables are in `results_published/amd_20261102_mi350x_cubw_v1_4_smoke_job496073/`.

| job | release | outcome | cause / fix |
|---|---|---|---|
| 496056 | 8ba209a8 | stopped in preflight after 12 s | FETCH_SIZE read 32780 KB for a 64 MiB copy (0.50×). Preflight now records this on non-gfx942 GPUs instead of aborting (36198d84). |
| 496059 | 36198d84 | 29 steps: every vLLM step failed; everything else passed | gfx942-only assert in `scripts/gpu_target.py`. It now also accepts MI350X gfx950 (e7116c1f). |
| 496060 (8 GPUs) | 36198d84 | pilot stage only; the gate released nothing; finished by itself in 6.6 min | the same assert. Placement passed on all 8 GPUs. Kept as a failed acquisition. |
| 496073 | e7116c1f | **29/29 steps rc = 0** in 20.5 min | — |

## What 496073 establishes on gfx950

- **Placement:** exact counts with even per-XCD balance:

  | mask | CUs | per XCD |
  |---|---:|---:|
  | full device | 256 | 32 |
  | 0–191 | 192 | 24 |
  | 0–127 | 128 | 16 |
  | competitor 192–255 | 64 | 8 |

- **Decode works:** Qwen32B graph decode, B1/C4096. 5 runs gave 1 output digest, identical to the MI325X digest
  (`8fb99b67…`).
- **Memory clock was constant** at 2000 MHz across the samples. The gfx clock varied from 1547 to 2202 MHz
  (pooled ratio 1.42), so timing is observational.
- **Decode kernels:** `wvSplitK` takes 81.5% of decode kernel time (MI300X/MI325X: about 84%). vLLM launches
  **256 workgroups, each with 160 KiB of LDS**, which is all of a gfx950 CU's LDS. So vLLM keeps one workgroup per
  CU on gfx950 too. The plan's hypothesis that two would fit per CU does not hold.
- **Single-acquisition decode times** (pilot values; not a production estimate):

  | n | 256 | 192 | 128 | 128, kernel sized for 128 |
  |---|---:|---:|---:|---:|
  | ms/step | 12.23 | 14.73 | 14.51 | 15.40 |

- **E4 rows at 128 vs 256 CUs:**

  | row | ratio |
  |---|---:|
  | streaming read | 0.68 |
  | ILP-8 read | 0.75 |
  | FMA | 0.63 |
  | pointer-chase latency | 0.99 |

- **Counters:** FETCH_SIZE and TCC_EA0_RDREQ_LEVEL runs completed. FETCH_SIZE is **not validated** on gfx950
  (0.50× in preflight). The campaign's E0 native counter validation must back any MI350X counter claim.

## Leak test v2 is insensitive on MI350X

The positive control shares 8 CUs with the probe. It found **0** CUs with q < 0.85. On MI210 and MI325X it found
exactly 8. Every q was 1.000 at both n = 192 and n = 128.

Under the pre-registered rule, an insensitive test fails `leak_ok_v2`. So the campaign's E3-minimal will **not** be
released on MI350X, and its other experiments run unaffected.

A likely cause, **not tested**:
- With 160 KiB per probe workgroup, probe and foreign workgroups cannot be on the same CU at the same time.
- If gfx950 hands out each launch's workgroups to CUs in a fixed pattern, each probe CU completes the same count
  and the whole launch waits for the slowest CU.
- Completion counts per CU would then be uniform by construction.

A probe that measures each workgroup's duration per CU, or that uses a smaller LDS so foreign work can share the
CU, would be needed. It would have to be pre-registered and smoke-tested before any MI350X E3.
