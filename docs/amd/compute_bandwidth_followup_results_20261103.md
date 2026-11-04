# Compute vs bandwidth follow-up — results, version 2 (3 November 2026)

This report supersedes nothing silently. Version 1 is
[compute_bandwidth_followup_results_20261101.md](compute_bandwidth_followup_results_20261101.md) and is left
unchanged. Version 2 adds three jobs:

| job | cohort | GPUs | release | run time | review / tables |
|---|---|---|---|---|---|
| 492957 | MI325X extension | 8 | de2baecb (v1.3) | — | [cubw_mi325x_extension_review_20261102.md](cubw_mi325x_extension_review_20261102.md) |
| 492956 | MI300X extension (this report, §2) | 8 (k002-002) | de2baecb (v1.3), read with v1.4 | 1 h 13 min | `results_published/amd_20261103_mi300x_cubw_v1_3_auto_job492956/` |
| 496074 | MI350X campaign | 8 (k007-004) | e7116c1f (v1.4) | 1 h 09 min | [mi350x_campaign_review_20261103.md](mi350x_campaign_review_20261103.md) |

**Rules that apply throughout:**
- Cohorts are never pooled.
- Every timing comparison is **observational, clock-confounded**: no job passed the pooled clock gate. The
  pooled gfx-clock ratio was 1.36–1.53, against a gate of ≤ 1.05.
- No level is a qualified non-inferiority result.

## 1. The question, and the short answer

Plan 2.0 asks how much compute memory-bound LLM decode needs relative to HBM bandwidth.

- **With vLLM's stock kernels:**
  - MI300X keeps decode within 1–2% of full-device speed down to about 60–66% of its CUs.
  - MI325X behaves similarly but less consistently.
  - MI350X loses 16–25% at any reduction.
- **With the dominant skinny-GEMM kernel (`wvSplitK`) resized to the active CUs:** MI350X runs at full speed on
  75% of its CUs. On MI300X and MI325X, resizing never helped.
- **Mechanism:** every level moves the same bytes. Decode time tracks how many HBM reads are in flight. A CU cut
  costs more the more HBM bandwidth each CU must sustain: MI300X 17, MI325X 20, MI350X 31 GB/s per CU.

> ## ★ Key concept: HBM bandwidth per CU (B_CU)
>
> An AMD compute unit (CU) is the counterpart of an NVIDIA streaming multiprocessor (SM), so this is
> "bandwidth per SM" in NVIDIA terms.
>
> **Definition.** **B_CU = HBM bandwidth / CUs in use.** Each CU of a memory-bound kernel must carry B_CU of
> HBM traffic. By Little's law, it must keep **B_CU × memory latency** bytes of reads in flight to do so.
> Removing CUs raises the share each remaining CU must carry by FULL/n.
>
> | cohort | CUs | HBM peak | peak B_CU | measured B_CU (E4 seqread at full) | decode R at ~75% of CUs, stock kernel |
> |---|---:|---:|---:|---:|---:|
> | MI300X | 304 | 5.3 TB/s | 17.4 GB/s | 15.0 GB/s | 1.02–1.03 |
> | MI325X | 304 | 6.0 TB/s | 19.7 GB/s | 16.3 GB/s | 1.05–1.07 |
> | MI350X | 256 | 8.0 TB/s | 31.2 GB/s | 25.1 GB/s | 1.18 |
>
> **Why it explains the results:**
> 1. **The penalty for a CU cut ranks with B_CU** at every matched per-XCD fraction (§3).
> 2. **A CU cut moves the same bytes; it lowers the reads in flight** (FETCH_SIZE/RDREQ equal,
>    RDREQ_LEVEL/GRBM lower), and time rises in step.
> 3. **Positive control (MI350X):** a kernel sized for 192 CUs raised in-flight reads per CU and recovered
>    full-device speed.
> 4. **E4:** one load in flight per thread (ilp1) loses about a third of the bandwidth at reduced CU counts;
>    eight in flight (ilp8) keeps 90–100%.
>
> **What it means for the shoreline question:** the compute that memory-bound decode needs near HBM is set by
> how many reads each CU can keep in flight against B_CU, not by FLOP/s. A design that moves compute toward
> memory should be sized by **in-flight capacity per unit of bandwidth**. Raising bandwidth without raising
> per-CU in-flight capacity makes every CU more critical, as MI350X shows.
>
> The per-cohort values are in [results_published/COHORTS.md](../../results_published/COHORTS.md).

## 2. MI300X extension 492956 (new in this version)

**Execution:**
- 267 of the 269 steps with a return code had rc = 0, and every check passed. The other 16 journal records are
  leak-v2 co-run entries, which use a different record format.
- Two eager-mode FETCH_SIZE counter steps (n = 304 and 200) hit their timeout (rc = −15). They are kept as failed
  acquisitions. The eager RDREQ_LEVEL steps completed.
- Outputs: 108/108 runs complete, 1 digest. Placement was exact.

**Gates:**
- **leak_ok_v2 passed:** the positive control flagged 8 of 8 CUs, and the decoder flagged 0, at 232 and 152.
  So E3-minimal was released and run.
- leak_ok_v11 was false: it had no v1.1 cells in this plan.
- clock_ok was false.

**E2-fine, Qwen32B** (8 blocks per level):

| n | 272 | 248 | 232 | 216 | 200 | 184 | 152 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R | 1.068 | 1.053 | 1.033 | 1.019 | 1.017 | **1.010** | 1.085 |
| 95% CI | 1.064–1.071 | 1.048–1.056 | 1.030–1.036 | 1.016–1.023 | 1.015–1.019 | 1.007–1.013 | 1.081–1.090 |

- It has the same non-monotone shape as MI325X 492957: worst near full, lowest at 184, rising again at 152.
- The 232 and 200 values agree with production 487341 (1.024 and 1.012) within about 1%.

**E6:**
- **Kernel time:** `wvSplitK` (304 WGs × 64 KiB LDS; 82.4% of decode kernel time) has ratios to 304 of:
  272 1.111, 248 1.072, 232 1.051, 216 1.034, 200 1.033, 184 1.031, 152 1.130. This is the R(n) shape again.
  The other kernels are each ≤ 4.4% of time.
- **Resized kernel (`e6_decode`) is slower at every level:** 272 1.043, 248 1.114, 232 1.075, 216 1.127,
  200 1.158, 184 1.141, 152 1.202. MI325X behaved the same way.
- **Counters:** FETCH_SIZE and RDREQ are identical across 304/232/200/152 for each GEMM shape. The extra time
  shows up as busy cycles at constant bytes.

**E3-minimal: fixed partition.** The decoder runs on n CUs and an owned competitor on CUs 232–303; 5 blocks per
cell. Values are decode time relative to no competitor at the same n:

| n | arithmetic 100% | memory 50% | memory 100% |
|---|---:|---:|---:|
| 232 | 1.054 (10.9 TFLOP/s) | 1.106 (540 GB/s) | 1.246 (1082 GB/s) |
| 152 | 1.044 (11.8 TFLOP/s) | 1.113 (632 GB/s) | 1.267 (1253 GB/s) |

- Memory 100% vs arithmetic 100%: **1.18× at 232** and **1.21× at 152**.
- Decoder alone on 152 CUs vs decoder on 232 CUs beside a memory-100% competitor: **0.846**, so the 152-CU
  decoder is 15% faster. Fewer CUs with uncontested HBM beat more CUs with contested HBM.
- MI325X 489594 is the same direction, a bit smaller: memory/arithmetic 1.14 and 1.16, the same comparison 0.924.
- Placement isolation is proven for this cohort (leak v2), so the contention is through shared memory service
  (HBM, MALL, fabric), not through stolen CUs.

**E7 r2 blind check** (`e7_mi300x_r2_blindcheck.json`):
- The constrained seqread4g model predicted 272/248/216/184 = 1.013/1.023/1.026/1.039.
- Observed: 1.068/1.053/1.019/1.010. Only 1 of 4 is within tolerance, and Qwen14B at n = 152 also missed.
- **E7 is not validated on MI300X**, as on MI325X. A smooth bandwidth-scaling model cannot produce the
  kernel-geometry curve.

## 3. Three cohorts at matched per-XCD fractions (Qwen32B E2, production)

| fraction per XCD | MI300X 487341 / 492956 | MI325X 489594 / 492957 | MI350X 496074 |
|---|---:|---:|---:|
| 34/38 ≈ 29/32 | — / 1.068 | — / 1.116 | 1.251 |
| 31/38 ≈ 26/32 | — / 1.053 | — / 1.080 | 1.177 |
| 29/38 ≈ 24/32 | 1.024 / 1.033 | 1.074 / 1.052 | 1.180 (sized kernel: **0.989**) |
| 27/38 ≈ 23/32 | — / 1.019 | — / 1.026 | 1.197 |
| 25/38 ≈ 21/32 | 1.012 / 1.017 | 1.056 / 1.021 | 1.191 |
| 23/38 ≈ 19/32 | — / 1.010 | — / 1.008 | 1.180 |
| 19/38 = 16/32 | 1.070 / 1.085 | 1.187 / 1.141 | 1.159 |
| 12/38 ≈ 10/32 | 1.083 / — | 1.186 / — | 1.858 |

**Findings that hold in all three cohorts** (all observational):
1. **`wvSplitK` sets the shape of R(n).** It runs one workgroup per CU, sized to the full device, and is 81–84%
   of decode kernel time. Its kernel-time ratio reproduces R(n) in every cohort.
2. **Same bytes, fewer reads in flight.** Restricting CUs does not change FETCH_SIZE or RDREQ. Time goes up
   when in-flight HBM reads go down.
   - MI350X provides the positive control: the sized-192 kernel raises in-flight reads to 0.73–0.89 of the
     full-device value and recovers full speed.
3. **The penalty rises with bandwidth per CU:** MI300X < MI325X < MI350X at almost every matched fraction.
4. **Memory contention, not CU count, is what hurts a co-located decoder** (E3, MI300X and MI325X; MI350X
   added by 499332 after this version, memory/arithmetic 1.23–1.25).
5. **E7's smooth service model fails** on both cohorts where it had a blind test.

**What differs:**
- On MI300X/MI325X the stock kernel is near its best. Resizing it only hurts, and a 34–40% CU cut costs about
  1–6%.
- On MI350X the stock kernel is far from its best below 256 CUs. A kernel resized for 192 CUs recovers
  full-device speed, but the same resizing fails at 208/184/168/152.

## 4. Guardrails (E8)

E8 was run on MI350X and in v1 on the other cohorts.
- On MI350X at 168 CUs, prefill, 16k-context decode, B8 decode and Mistral-7B are 14–21% slower, and J/token is
  about 9% higher.
- A CU reduction that looks free for one decode shape is not free for the rest of the workload mix unless it is
  checked there.

## 5. Open items

- **gfx950 leak probe: resolved after this version** by job 499332
  ([mi350x_leak4_review_20261104.md](mi350x_leak4_review_20261104.md)). The pre-registered start-delay probe (leak
  v4) is sensitive on all 8 GPUs and the decoder flags 0 CUs in 16/16 tests. E3-minimal then ran on MI350X:
  memory 100% / arithmetic 100% = 1.25 at 192 and 1.23 at 128, the same direction as MI300X and MI325X and larger.
- **The MI350X sizing pattern: resolved after this version** by job 497082
  ([mi350x_followup_review_20261103.md](mi350x_followup_review_20261103.md)). Unequal active CUs per shader
  engine make the dispatcher serialize two `wvSplitK` workgroups on some CUs. Sizing to the largest multiple of
  32 restores R 1.004 at 232 CUs and 1.009 at 208 CUs (pre-registered H_SE; observational).
- **Clock qualification:** any non-inferiority claim needs a run that passes the pooled clock gate. Clock and
  power changes remain unauthorized.
- **Off-cluster raw archive:** the destination is still to be named by the owner.

If any of this is presented or published, the AMD acknowledgement is: "This work was supported in part by
Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
