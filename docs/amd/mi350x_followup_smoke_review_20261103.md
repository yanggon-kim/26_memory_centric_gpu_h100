# MI350X follow-up smoke 497081 — review (3 November 2026)

**Job:** 497081 on `mi3501x` (k007-005-v2, one MI350X slice), cubw v1.5, release 09178c91.
- Ran 21.5 min. All 58 steps completed: 37 with a passing check, the others without one.
- Pilot-labelled: one acquisition per cell, releases nothing.
- Pre-registration: [mi350x_followup_plan_20261103.md](mi350x_followup_plan_20261103.md).
- Tables: `results_published/amd_20261103_mi350x_cubw_v1_5_followup_smoke_job497081/`.
- Every timing is clock-confounded (gfx 1659–2189 MHz; memory clock held at 2000 MHz).

## 1. Leak test v3: insensitive, so E3 stays blocked

`leak_ok_v3` = false at n = 192 and 128. Each of the three positive controls flagged **0** of the 8 shared CUs,
against the ≥ 4 required. Per the pre-registered rule, the test is insensitive. The threshold is not changed.

What the raw per-CU residency shows at n = 192, probe on CUs 192–255:

| run | probe workgroup residency, median over CUs | on the 8 shared CUs | co-runner on the shared CUs |
|---|---:|---:|---|
| alone | 438 µs | — | — |
| read co-runner on 0–199 | 451 µs | 479–490 µs (about +7%) | — |
| FMA co-runner on 0–199 | 453 µs | not among the slowest | — |
| full-LDS probe on 0–199 | 438 µs | unchanged | **starved**: 30k–44k workgroups vs a 496k median |

**Interpretation:**
- The duration probe sees the memory co-runner on exactly the shared CUs, but only by about 7%, under the 15%
  flag threshold.
- Against a full-LDS workgroup, which is how `wvSplitK` occupies a CU, the probe wins the CU. The foreign
  kernel is the one that slows down, so the probe cannot see it.
- This is the same asymmetry as v2. On gfx950, a probe that sits on its CUs does not reveal a `wvSplitK`-like
  intruder; the intruder pays instead.
- A decoder leaking onto probe CUs would therefore show up as **decoder** slowdown, which a probe cannot measure.

E3 on MI350X stays blocked. The 8-GPU follow-up still runs its leak-v3 suite on two GPUs, as pre-registered.

## 2. Shader-engine balance (H_SE): P1 holds on this GPU

The dispatch probe launches 16-wave workgroups with 160 KiB LDS and 1 GiB streamed in total. Medians over 5
trials:

| mask n | active CUs per SE (XCD0) | grid = n: max WG/CU, serialized WGs, kernel ms | grid = balanced(n) | grid = 256 (device size) |
|---:|---|---|---|---|
| 256 | 8,8,8,8 | 1, 0, 0.198 | — | — |
| 232 | 8,7,7,7 | **2, 6.4, 0.346** | 224: 1, 0, **0.205** | 2, 24, 0.322 |
| 208 | 7,7,6,6 | **2, 8, 0.372** (only 200 CUs used) | 192: 1, 0, **0.239** | 2, 48, 0.316 |
| 192 | 6,6,6,6 | 1, 0, 0.234 | — | 2, 64, 0.314 |
| 184 | 6,6,6,5 | **2, 4.8, 0.409** | 160: 1, 0, **0.264** | 2, 72, 0.319 |
| 168 | 6,5,5,5 | **2, 6.4, 0.449** | 160: 1, 0, **0.264** | 2, 88, 0.314 |
| 152 | 5,5,5,4 | **2, 4.8, 0.484** | 128: 1, 0, **0.313** | 2, 104, 0.314 |
| 128 | 4,4,4,4 | 1, 0, 0.317 | — | 2, 128, 0.317 |

- **The dispatcher does not follow the active CUs per shader engine.** At 208, the workgroups go 6,7,6,7 per SE
  while the active CUs are 7,7,6,6. At 232 they go 7,7,8,7 against 8,7,7,7. One CU per XCD gets two
  workgroups, which run one after the other.
- **Every unbalanced level** has 8 CUs with two workgroups, and the kernel takes 1.5–2× the balanced time.
- **Every balanced grid** gives one workgroup per CU and the short time.
- **With no LDS (control)**, the same 8 CUs still get two workgroups, but they co-reside instead of
  serializing (0 serialized), and the kernel is about 15% faster than with full LDS.

**P1 criteria:**
- balanced levels: 1 workgroup per CU and 0 serialized — met;
- ≥ 3 of the 4 strongly unbalanced levels with 2 per CU and serialized > 0 — met at all 4 (and at 232);
- grid = balanced(n) gives 1 per CU at every level — met.

**P1 holds on this single GPU.** The judged run is the 8-GPU job.

## 3. Balanced-size decode and shapes: pilot preview of P2 and P3

| Qwen32B decode, 1 acquisition | ms per step | vs 256 |
|---|---:|---:|
| 256, default kernel | 12.48 | 1.000 |
| 208, default kernel | 15.15 | 1.214 |
| **208, kernel sized for 192 (`e6_balanced`)** | **12.60** | **1.009** |
| 192, kernel sized for 192 (`e6_decode`) | 12.66 | 1.014 |

- Under mask 208, `wvsplitk_c192` takes 10.1–225 µs per shape. That is the same as the 192-sized kernel at
  mask 192 and the full device at 256, and 22–45% faster than the 208-sized kernel. All 5 shapes are faster.
- P2 and P3 are judged on the 8-GPU job's 3 shape acquisitions and 8 decode blocks.

## 4. Decision

The smoke found no port bug, so the held 8-GPU job **497082** (same release 09178c91) was released unchanged.
Its gate will not release E3, because leak v3 is insensitive. Its balanced-size decode, shapes, traces and
dispatch probe are the judged tests of H_SE.

**If H_SE holds, it is practical:** on MI350X, sizing `wvSplitK` to the largest shader-engine-balanced CU count
≤ the active CUs (a multiple of 32) recovers full-device decode speed.
