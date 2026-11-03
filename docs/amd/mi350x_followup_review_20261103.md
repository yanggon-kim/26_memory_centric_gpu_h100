# MI350X follow-up 497082 — review (3 November 2026)

**Job:** 497082 on `mi3508x` (k007-004, 8× MI350X), cubw v1.5, release 09178c91. Smoke:
[mi350x_followup_smoke_review_20261103.md](mi350x_followup_smoke_review_20261103.md).
- Ran 48 min. All 269 steps that report a return code had rc = 0. The other 84 journal records are co-run
  entries, which carry no return code.
- Judged against the pre-registration [mi350x_followup_plan_20261103.md](mi350x_followup_plan_20261103.md).
- Tables: `results_published/amd_20261103_mi350x_cubw_v1_5_followup_job497082/`. The scored predictions are in
  `hse_judgment.json` there.
- **Cohort: MI350X only.** Every timing is observational and clock-confounded: pooled gfx ratio 1.58, gfx
  1395–2198 MHz, memory clock held at 2000 MHz. R is relative to n = 256 in the same job.

## 1. Gates

| gate | value |
|---|---|
| placement_ok | true: all 8 GPUs, all 8 levels, exact per-XCD counts |
| lifecycle_ok | true |
| outputs_ok | true: 84/84 runs complete, 1 digest |
| leak_ok_v3 | **false**: insensitive on both GPUs, as in the smoke |
| release_e3_minimal_v3 | **false**, so E3-minimal did not run (0/5 in every cell, by design) |
| clock_ok | false |

**Leak v3 replication:**
- Insensitive at n = 192 and 128 on GPUs L0 and L1.
- Every positive control flagged 0 of the 8 shared CUs, against the ≥ 4 required.
- Natives and decoder runs had min q ≥ 0.96.

E3 stays blocked on MI350X. A gfx950 leak test needs a different observable: the smoke showed that the intruder,
not the probe, pays.

## 2. Hypothesis H_SE: supported

### P1: dispatch probe (lane L2, 5 trials per row, 160 KiB LDS, 16 waves) — pass

| mask n | CUs per SE (XCD0) | grid = n: CUs used, max WG/CU, serialized, ms | grid = balanced(n) | ms ratio |
|---:|---|---|---|---:|
| 256 | 8,8,8,8 | 256, 1, 0, 0.199 | — | — |
| 232 | 8,7,7,7 | 227, **2, 6.4**, 0.352 | 224: 1, 0, 0.208 | 0.59 |
| 208 | 7,7,6,6 | 202, **2, 8.0**, 0.383 | 192: 1, 0, 0.238 | 0.62 |
| 192 | 6,6,6,6 | 192, 1, 0, 0.232 | — | — |
| 184 | 6,6,6,5 | 180, **2, 5.6**, 0.421 | 160: 1, 0, 0.270 | 0.64 |
| 168 | 6,5,5,5 | 164, **2, 6.6**, 0.460 | 160: 1, 0, 0.267 | 0.58 |
| 152 | 5,5,5,4 | 148, **2, 5.4**, 0.504 | 128: 1, 0, 0.320 | 0.63 |
| 128 | 4,4,4,4 | 128, 1, 0, 0.320 | — | — |

**P1 criteria:**
- Balanced levels: 1 workgroup per CU and 0 serialized — met.
- Unbalanced levels with 2 per CU and serialized > 0: 4 of 4 (≥ 3 required) — met.
- Balanced grid gives 1 per CU at every level — met.

**Observations beyond the criteria:**
- At every unbalanced level, a grid of n workgroups lands on only n − 4 to n − 6 CUs. Some active CUs get none,
  and others get two.
- With no LDS, the same CUs double up but co-reside, so 0 workgroups serialize. That accounts for the 15–20% gap
  in kernel time between the full-LDS and no-LDS runs.

### P2: GEMM shapes (3 acquisitions; median µs, size balanced(n) vs size n under mask n) — pass

| n (balanced size) | down | gate_up | lm_head | o_proj | qkv | faster |
|---|---|---|---|---|---|---:|
| 232 (224) | 45.7 vs 45.5 | 83.4 vs 115.4 | 224.6 vs 314.0 | 9.8 vs 9.9 | 13.2 vs 13.4 | 4/5 |
| 208 (192) | 44.3 vs 65.6 | 83.3 vs 120.1 | 223.7 vs 329.0 | 10.0 vs 14.8 | 13.0 vs 16.2 | 5/5 |
| 184 (160) | 50.0 vs 68.5 | 89.6 vs 128.0 | 238.4 vs 350.8 | 10.7 vs 15.5 | 14.0 vs 20.1 | 5/5 |
| 168 (160) | 51.9 vs 51.8 | 90.0 vs 140.5 | 237.1 vs 377.3 | 10.7 vs 11.0 | 14.1 vs 18.1 | 4/5 |
| 152 (128) | 51.5 vs 73.1 | 102.3 vs 145.2 | 273.4 vs 406.4 | 11.7 vs 16.5 | 15.6 vs 22.4 | 5/5 |

The two misses (down at 232 and 168) are ties within 0.2 µs, at levels where the size-n kernel happened to
avoid serialization for that shape. The full-device times for comparison are down 44.9, gate_up 83.4,
lm_head 229.1, o_proj 10.5 and qkv 13.4 µs.

### P3: Qwen32B decode (8 blocks, paired with E2 at n and 256) — pass

| mask n | stock kernel R_e2(n) | kernel sized to balanced(n): R | balanced size |
|---:|---|---|---:|
| 232 | 1.234 [1.230, 1.236] | **1.004** [1.002, 1.006] | 224 |
| 208 | 1.209 [1.206, 1.212] | **1.009** [1.007, 1.010] | 192 |
| 192 | 1.199 [1.196, 1.202] | 1.009 [1.007, 1.011] (`e6_decode`) | 192 |
| 184 | 1.198 [1.194, 1.201] | **1.068** [1.065, 1.071] | 160 |
| 168 | 1.191 [1.187, 1.194] | **1.066** [1.064, 1.068] | 160 |
| 152 | 1.183 [1.179, 1.187] | **1.168** [1.164, 1.171] | 128 |
| 128 | 1.185 [1.182, 1.187] | 1.168 [1.165, 1.172] (`e6_decode`) | 128 |

**P3 criteria:**
- R_balanced(n) < R_e2(n) at all 5 unbalanced levels, with confidence intervals that do not overlap — met.
- R_balanced(208) = 1.0086 vs R_e6_decode(192) = 1.0089, a difference of −0.0003, within ±0.03 — met.

**Kernel traces, balanced size** (`e6_kernels_balanced.json`):
- `wvSplitK` is 80.8% of decode kernel time.
- Its time relative to the 496074 full-device trace is 0.97 (224, 192), 1.04 (160, 160) and 1.17 (128).
- This is a cross-job ratio under trace distortion, so treat it as indicative only.

### Verdict

**H_SE is supported:** P1 and P3 pass, and P2 passes too. It remains observational and clock-confounded.

On MI350X, `wvSplitK` launches one 160 KiB workgroup per CU. When the active CUs per shader engine are unequal,
the dispatcher gives some CUs two workgroups, which serialize. That roughly doubles the time of the kernel that
carries 81% of decode.

## 3. What this means: the B_CU picture without the dispatch artefact

With the dispatch artefact removed, decode time depends only on the **balanced CU count actually used**:

| CUs doing the work | fraction | B_CU needed (8.0 TB/s peak) | R |
|---:|---:|---:|---:|
| 256 | 100% | 31.2 GB/s | 1.000 |
| 224 | 88% | 35.7 GB/s | 1.004 |
| 192 | 75% | 41.7 GB/s | 1.009 |
| 160 | 63% | 50.0 GB/s | 1.067 |
| 128 | 50% | 62.5 GB/s | 1.168 |

- **MI350X keeps decode within 1% of full speed down to 75% of its CUs**, when the kernel is sized to a
  shader-engine-balanced count.
- Below that, the cost rises steadily with the bandwidth each CU must carry. This is the clean B_CU curve that
  the stock kernel's 16–25% step had hidden.
- **Practical rule (MI350X):** set `wvSplitK`'s CU count to the largest multiple of 32 ≤ the active CUs. For
  example, restrict to 208 and size for 192.
- **Earlier result revised:** the 496074 finding that "the sized kernel fails at 208/184/168/152" was a
  dispatch artefact of sizing to n, not a bandwidth limit.

## 4. Open items

- **E3 on MI350X:** still blocked. Neither completion counts (v2) nor residency (v3) detect a full-LDS intruder
  on gfx950. A v4 would have to watch the decoder's own CUs, for example with per-workgroup CU keys on a
  `wvSplitK`-shaped canary. That is not pre-registered and has not been started.
- **Generality of H_SE:** tested on one GPU (dispatch) and one model (decode). MI300X is unbalanced at the full
  device (9,10,10,9 per SE), which is a candidate explanation for its non-monotone R(n). It is untested.
- **Clock qualification and the off-cluster raw archive:** unchanged.

If any of this is presented or published, the AMD acknowledgement is: "This work was supported in part by
Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
