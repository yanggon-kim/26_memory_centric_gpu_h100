# MI300X shader-engine smoke 499333 — review (4 November 2026)

**Job:** 499333 on `mi3001x` (k002-004-v8, one MI300X), cubw v1.6, release f737d1b1.
- Ran 17 min. All 52 steps had rc = 0.
- Pre-registration: [followups_v16_plan_20261104.md](followups_v16_plan_20261104.md) §B.
- Tables: `results_published/amd_20261104_mi300x_cubw_v1_6_se_smoke_job499333/`.
- Pilot-labelled: one GPU, one acquisition per cell, judges nothing. Timings are clock-confounded.

## 1. Dispatch probe (5 trials, 64 KiB LDS, 16 waves)

| mask n | grid = n: CUs used, max WG/CU, serialized, ms | grid = balanced(n) | stock grid 304 under mask n |
|---:|---|---|---|
| **304** (9,10,10,9) | 304, **1, 0**, 0.259 | 288: 1, 0, 0.326 | — |
| 288 | 288, 1, 0, 0.327 | — | 2, 16, 0.427 |
| 272 | 264, **2, 8**, 0.372 | 256: 1, 0, 0.341 | 2, 32, 0.412 |
| 256 | 256, 1, 0, 0.346 | — | 2, 48, 0.399 |
| 232 | 232, **2, 6.4**, 0.395 | 224: 1, 0, 0.329 | 2, 72, 0.380 |
| 224 | 224, 1, 0, 0.257 | — | 2, 80, 0.372 |
| 200 | 200, **2, 6.4**, 0.481 | 192: 1, 0, 0.336 | 2, 104, 0.287 |
| 192 | 192, 1, 0, 0.267 | — | 2, 112, 0.283 |
| 152 | 152, **2, 4.8**, 0.490 | 128: 1, 0, 0.360 | 3, 152, 0.362 |
| 128 | 128, 1, 0, 0.354 | — | 3, 176, 0.360 |

- **At reduced masks, gfx942 behaves like gfx950.** Every unbalanced mask serializes about 8 workgroups, and
  every balanced grid gives one per CU.
- **At the full device it does not.** With 304 workgroups on 9,10,10,9 CUs per SE, the dispatcher places exactly
  9,10,10,9 per SE: one per CU, none serialized.
- **Pre-registered prediction P1m(b) therefore looks likely to fail** at 304 on the judged 8-GPU job. P1m(b)
  required serialization at the full device as well as at ≥ 3 of the reduced unbalanced levels.

## 2. Decode and shapes (one acquisition each)

| cell | ms per step | vs 304 stock |
|---|---:|---:|
| 304 stock | 17.96 | 1.000 |
| 304, kernel sized for 288 | 18.01 | 1.002 |
| 272 stock | 19.03 | 1.060 |
| **272, kernel sized for 256** | **18.01** | **1.003** |
| 256, kernel sized for 256 (`e6_decode`) | 18.00 | 1.002 |

- **Full device:** sizing for 288 does not help, consistent with no serialization there. P3m(b) looks likely to
  fail too.
- **At 272 CUs, sizing for 256 brings decode back to full-device speed.** The MI350X pattern appears on MI300X at
  a reduced level.
- **Shapes at 272:** the 256-sized kernel is fastest or tied for all 5 shapes (down 69.2, gate_up 135.9,
  lm_head 371.0, o_proj 14.5, qkv 17.7 µs). At 304, the 288-sized and device-sized kernels are equal within 1%,
  except qkv (19.1 vs 17.8).

## 3. Decision

- No port bug was found, so the held 8-GPU job **499334** was released unchanged. The pre-registered predictions
  and verdict rule stay as written.
- If the 8-GPU job reproduces the smoke, the pre-registered verdict will be **"H_SE refuted on MI300X"**, because
  P1m fails at 304.
- The reduced-level serialization and the balanced-size recovery will be reported separately, labelled as not
  pre-registered.
