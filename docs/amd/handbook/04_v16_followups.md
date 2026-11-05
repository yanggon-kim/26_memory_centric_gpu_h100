# 04 — cubw v1.6 follow-ups: leak test v4 on MI350X and the shader-engine study on MI300X (jobs 499331–499334)

This chapter covers the last four AMD jobs of the cubw ("compute vs bandwidth") follow-up, run on 4–5 November 2026.
Acronyms are defined on first use here; the shared glossary is in [README.md](README.md).

Short definitions used throughout:

- **CU** = compute unit. MI300X has 304 CUs, MI350X has 256. Both have 8 **XCDs** (accelerator complex dies), so
  38 CUs per XCD on MI300X and 32 on MI350X. Each XCD has 4 **SEs** (shader engines).
- **Mask** = `HSA_CU_MASK`, the ROCm environment variable that restricts a process to a set of CUs. "Mask n" or
  "0..n−1" means the contiguous mask of n CUs, which spreads evenly over the 8 XCDs.
- **LDS** = local data share, the per-CU scratchpad (64 KiB per CU on gfx942/MI300X, 160 KiB on gfx950/MI350X;
  `results_published/*/cohort.json`). A workgroup that reserves the full LDS cannot share its CU with any other
  workgroup that also holds LDS.
- **WG** = workgroup. **`wvSplitK`** = vLLM's skinny GEMM kernel, about 83% of decode kernel time on MI300X
  (`e6_kernels_se.json`). It launches one full-LDS WG per CU, with grid = the device CU count, under any mask.
- **R(n)** = decode time per step at n CUs / decode time at the full device, in the same block (same GPU).
- **B_CU** = HBM bandwidth per active CU (peak HBM / CUs doing the work). Peak HBM is 5.3 TB/s on MI300X and
  8.0 TB/s on MI350X (`cohort.json`).
- **E2 / E3 / E6** are cubw experiment names. E2 = decode time vs CU count with the stock kernel. E3 = a decoder
  on 0..n−1 beside an owned "competitor" process on the remaining CUs. E6 = decode or kernel shapes with `wvSplitK`
  resized via `CUBW_CU_COUNT_OVERRIDE`. `e6_decode` sizes it to n, `e6_balanced` to balanced(n).
- **balanced(n)** = the largest CU count ≤ n that gives every SE of every XCD the same number of active CUs under
  a contiguous mask. It is `n − n mod 32` on both cohorts (8 XCDs × 4 SEs; `plan.py` `balanced()`).
- **H_SE** = the shader-engine hypothesis from job 497082 (MI350X): when active CUs per SE are unequal, the
  dispatcher does not follow them, so some CUs get two full-LDS WGs that run one after the other ("serialized").

## Why these two follow-ups were run

At the end of the MI350X follow-up 497082 two questions were open
([mi350x_followup_review_20261103.md](../mi350x_followup_review_20261103.md)):

- **E3 was blocked on MI350X.** E3 may only run once a *leak test* shows that the masked decoder does not spill
  onto the competitor's CUs. The v2 (per-CU completion count) and v3 (per-CU residency time) leak probes were
  both insensitive on gfx950: their own positive controls flagged nothing.
- **H_SE had only been shown on gfx950.** Did it also apply to MI300X (gfx942)?

The owner's instruction on 2026-11-04 was "Do Two follow-ups"
([followups_v16_plan_20261104.md](../followups_v16_plan_20261104.md), first section):

- **A** — a leak test that works on gfx950, so E3 can run on MI350X. MI350X only.
- **B** — does shader-engine imbalance (H_SE) also shape MI300X? MI300X only.

The cohorts are never pooled (`cohort.json` rule: "cohorts are never pooled").

**Pre-registration.** [docs/amd/followups_v16_plan_20261104.md](../followups_v16_plan_20261104.md) was written and
committed before any data. It fixes the probe, the leak rule and its 0.10 threshold ("will not be relaxed after
data"), the predictions P1m–P3m for B, and the verdict rule. It is in release commit `f737d1b1` (4 Nov 2026
08:58 −0600), the same commit as the code.

**Code changes in cubw v1.6** (all in `scripts/amd/cubw/`, commit `f737d1b1`):

| file | change |
|---|---|
| `svc_bench_v6.hip` | new competitor mode `gapprobe` (start-delay probe). The v5 kernels are unchanged. |
| `plan.py` | `--leak4` (`build_leak4`, `leak_suite_v4`) and `--se-study` (`build_se_study`, `SE_LEVELS`); both take `--smoke` for the one-GPU version. |
| `cubw.py` | treatment `gapprobe` runs `$WORK/bin/svc_bench_gfx950_v6` (`CUBW_BENCH6`) with `GAPPROBE = --iters 600 --threads 1024 --gap-us 200 --late-ns 5000` and `--blocks` = probe-mask size. |
| `analyze.py` | `gap_counts`, `late_fraction`, `leak_v4_eval`; gates `leak_ok_v4` and `release_e3_minimal_v4`. |

Binaries: `svc_bench_gfx950_v6`, `svc_bench_gfx942_v6` and `svc_bench_gfx942_v5` (v5 source unchanged, newly built
for gfx942). Hashes are in `$WORK/bin/svc_bench.sha256` (plan §0).

No clock, power or cluster setting was changed. Every timing in this chapter is **observational and
clock-confounded**: the pooled clock gate (`clock_ok`: max/min gfx and memory clock ≤ 1.05 over the compared
trial windows) failed in every job.

**Statistics** (`analyze.py` docstring). One acquisition's value is the mean of its trials' HIP-event ms per
decode step. A "block" is one GPU lane holding one acquisition of every cell. Ratios get a 95% CI from 10,000
bootstrap draws (seed 20261018) that resample blocks, then trials inside each chosen acquisition. "Paired within
GPU" means numerator and denominator come from the same block.

## The leak test v4 ("gapprobe") method in detail

### What a leak test asks

E3 puts the decoder on CUs 0..n−1 and a competitor on n..255. If the decoder's work leaked onto the competitor's
CUs, an E3 slowdown could be "stolen CUs", not memory contention. The leak test places a *probe* on the competitor
CUs n..255 and asks whether the masked decoder disturbs it.

### Why v2 and v3 were blind on gfx950

(Plan §A; jobs 496074, 497081, 497082.)

- v2 (`cuprobe`, `svc_bench_v4`) counted completed probe WGs per physical CU. v3 (`cuprobe_t`, `svc_bench_v5`)
  measured how long each probe WG stayed resident.
- A `wvSplitK`-like intruder holds all 160 KiB of a gfx950 CU's LDS.
- When the probe runs **continuously**, it keeps re-occupying its CUs. It wins the CU and the intruder starves.
  So the probe's completions (v2) and residency (v3) do not change. In 496074, v2 per-CU counts stayed within 18
  of 506,270 alone and beside an overlapping probe (`svc_bench_v6.hip` header comment, v5 note).
- Result: both positive controls flagged 0 of 8 shared CUs in 497082
  ([experiment_log.md](../experiment_log.md), 2026-11-03).

v4 changes the question from "how fast does the probe run" to "**when can the probe start**".

### Launch shape (`svc_bench_v6 --mode gapprobe`, kernel `k_gapprobe`)

- Each launch has **one WG per probe CU**: `--blocks` = size of the probe mask (256 − n). `cubw.py` sets it from
  the mask range.
- Each WG has 1024 threads = **16 waves**, and reserves the **full LDS** of a CU (160 KiB on gfx950). So it
  cannot share a CU with any other LDS-holding WG.
- Each WG writes to LDS, then runs **600 iterations** of 4 independent FMA chains (about 20 µs, plan §A).
- Thread 0 records three values per WG: the physical **CU key** (`cu_key()`, the HW_ID/XCC_ID key), the **start**
  time (`wall_clock64()` read at kernel entry) and the **end** time.

### The 200 µs gap

- The host launches, synchronizes, copies back the records, then **sleeps 200 µs** (`--gap-us 200`).
- During the gap the probe holds no CU. A foreign full-LDS WG masked onto a probe CU can take it.
- At the next launch, the probe's WG for that CU must either **wait** for the foreign WG to finish (a late start),
  or be **placed on another probe CU** (the original CU gets no probe WG; another CU gets two).

### Per-launch, per-CU bookkeeping (host loop in `svc_bench_v6.hip`)

For each launch, with t_min = the earliest start of any WG in that launch:

| counter | meaning |
|---|---|
| `cu_blocks` ("seen") | +1 if CU c ran at least one probe WG in this launch |
| `cu_late` | +1 if CU c's first WG started more than `--late-ns 5000` (5 µs) after t_min |
| `cu_multi` | +1 if CU c ran two or more probe WGs in this launch |

These counters are printed per ~0.25 s window as one JSON line with `gap_launches` (launches in the window).

**Why 5 µs:** in 497082 the start skew within one launch of a balanced full-LDS grid was ≤ 0.81 µs in every trial,
so 5 µs is more than 6× that (plan §A).

### From windows to a verdict (`analyze.py`)

1. **Which windows count.** Probe-only and co-run runs: the steady windows (first and last dropped), and for co-runs
   only windows inside the co-runner's start/stop. Decoder runs: only windows fully inside decode trial windows
   (so decoder runs have ~21–24k launches vs ~92–100k for 30 s probe runs; `gates.json` `launches`).
2. **`gap_counts`** sums `launches`, `seen`, `late`, `multi` per CU over those windows.
3. **`late_fraction`**: f_c = (late launches + launches with no probe WG on c) / launches.
   A missing WG counts the same as a late one, because a displaced WG is also a sign of a foreign occupant.
4. **Excess**: e_c = f_c(beside X) − f_c(alone), same GPU (lane) and same probe mask.
5. **Flag**: CU c is flagged if e_c ≥ **0.10**.
6. **Coverage**: the alone run must see all 256 − n probe CUs (`alone_cus == expected_cus`).

### The runs in one leak test (`plan.py` `leak_suite_v4`; per GPU and per level n ∈ {192, 128})

Eight steps in random order (seeded per lane):

| run | co-runner | mask of co-runner | role |
|---|---|---|---|
| alone | none | — | baseline f_c(alone) |
| native read | `mem100` (read, 1 GiB) | 0..n−1 | native: does not overlap the probe |
| native FMA | `arith100` (fma) | 0..n−1 | native |
| **positive 1** | `cuprobe` (v4 full-LDS probe) | 0..n+7 | **required positive**: overlaps 8 probe CUs (one per XCD) |
| positive (info) | `arith100` | 0..n+7 | recorded, not required |
| positive (info) | `mem100` | 0..n+7 | recorded, not required |
| **positive 2** | **the Qwen32B decoder itself** | 0..n+7 | **required positive** |
| **decoder test** | the Qwen32B decoder | 0..n−1 | the run being judged |

- The decoder is `qwen32b_graph_b1_c4096` (Qwen 32B, CUDA-graph mode, batch 1, context 4096).
- FMA and read co-runners on 0..n+7 are recorded only. They hold no LDS and can share a CU with the probe, so they
  were not expected to delay its start (plan §A). In practice they did flag CUs (see results).

### Pass rule (`leak_v4_eval`, pre-registered)

- **Sensitive** iff every required positive (the `cuprobe` positive and the decoder on 0..n+7) flags **≥ 4** CUs.
  If either misses, the test is insensitive and `leak_ok_v4` is false.
- **Pass** (per GPU and level) iff coverage holds, the test is sensitive, **and** the decoder on 0..n−1 flags no
  more CUs than the most flagged by any native run on that GPU and level.
- **`leak_ok_v4`** = every test passes. On the 8-GPU job that is 8 GPUs × 2 levels = **16 tests**.
- **`release_e3_minimal_v4`** = `placement_ok` and `lifecycle_ok` and `outputs_ok` and `leak_ok_v4`. Only then do the
  E3-minimal cells run.
- Rule for A (plan §C): if the smoke showed v4 insensitive, the 8-GPU job would be **cancelled, not released**,
  and no threshold changed.

---

### #25 MI350X leak v4 smoke — job 499331

- **At a glance:** `mi3501x` (one MI350X slice), node k007-005-v7. 2026-11-04 08:58:41–09:14:02 (15 min; RUN_STATUS.log).
  Release `f737d1b1`. Status: **complete**, 22/22 steps. Pilot-labelled: one GPU, releases nothing.
- **Question:** Is the v4 start-delay probe sensitive on gfx950 (do both required positives flag ≥ 4 CUs at both
  levels)? If not, the 8-GPU job 499332 is cancelled.
- **Design:** placement at 256/200/192/136/128 and the competitor mask 192–255 (6 steps); the leak v4 suite at
  n = 192 (probe 192–255, 64 CUs) and n = 128 (probe 128–255, 128 CUs), 8 runs each (16 steps). The levels 200 and
  136 are the n+8 positive masks.
- **How it was run:** `plan.py --kind pilot --cohort mi350x --leak4 --smoke`; plan
  `$WORK/cubw_plans/mi350x_pilot_20261104_085838_f737d1b1.json`. One lane L0. The 8-GPU job 499332 was submitted
  **held** behind it.
- **What was measured:** per-CU seen/late/multi counts; gates written to `gates_smoke.json`.
- **Results** (`gates_smoke.json` → `leak_v4`; [mi350x_leak4_smoke_review_20261104.md](../mi350x_leak4_smoke_review_20261104.md)):

| | n = 192 (probe 192–255) | n = 128 (probe 128–255) |
|---|---|---|
| probe launches (alone / natives / `cuprobe` positive) | 94–97k | 93–96k |
| probe launches (decoder runs, trial windows only) | 22.6k, 23.4k | 21.1k, 22.1k |
| alone: coverage / largest late fraction | 64/64 / 0.000 | 128/128 / 0.000 |
| natives: flagged CUs (largest excess) | 0, 0 (0.0005) | 0, 0 (0.0003) |
| **required: `cuprobe` on 0..n+7** | **8 of 8** (excess 0.73–0.75) | **8 of 8** (0.63–0.67) |
| **required: decoder on 0..n+7** | **8 of 8** (0.45–0.49) | **8 of 8** (0.49–0.52) |
| recorded: FMA / read on 0..n+7 | 8 / 8 | 8 / 8 |
| **decoder on 0..n−1: flagged (largest excess)** | **0** (0.0025) | **0** (0.0019) |

  Gates: `placement_ok`, `lifecycle_ok`, `outputs_ok` (4/4 runs, 1 digest), `leak_ok_v4`, `release_e3_minimal_v4`
  all true; `clock_ok` false.
- **What it means:** Removing the probe's continuous occupancy gives a clean signal. Anything foreign on a probe CU
  makes the probe's start there late (or missing) in 45–75% of launches, including the real decoder. Otherwise it is
  late in < 0.3%. Each positive flagged exactly the 8 shared CUs. The masked decoder did not leak on this GPU.
  Raw check (this handbook, `corun_cuprobe_0-199_gapprobe_192-255_L0`): the 8 shared CUs got a probe WG in only
  ~35–37k of ~97k launches, and exactly 8 CUs show `multi` (the displaced WGs).
- **Decision:** v4 sensitive at both levels, so 499332 was **released unchanged**.
- **Failures and caveats:** one GPU only. No clock data in this job (`pooled clocks` samples 0).
- **Files:** review [mi350x_leak4_smoke_review_20261104.md](../mi350x_leak4_smoke_review_20261104.md);
  `results_published/amd_20261104_mi350x_cubw_v1_6_leak4_smoke_job499331/` (`summary.md`, `gates_smoke.json`,
  `cubw_tables.json`, `launch_plan.json`); raw `$WORK/results_amd/amd_20261104_mi350x_cubw_v1_pilot_job499331/`
  (`e1_leak_v4/<cell>/a0/attempt1/competitor.jsonl`); commits `f737d1b1` (code), `043afe8c` (review).

### #26 MI350X leak v4 + E3-minimal — job 499332

- **At a glance:** `mi3508x` (8× MI350X), node k007-003. 2026-11-04 11:37:10–12:15:06 (38 min: pilot 11:37–11:52,
  production 11:52–12:14). Release `f737d1b1`. Status: **complete**. 153/153 steps that report a return code had
  rc = 0; the other 112 journal records are co-run entries with no return code.
- **Question:** Does the masked decoder leak onto the competitor CUs on any of 8 GPUs at n = 192 and 128? If not,
  run E3-minimal on MI350X and check the expected direction: memory competition slows decode more than arithmetic
  competition, as on MI300X (1.18×, 1.21×) and MI325X (1.14×, 1.16×).
- **Design:**
  - Pilot stage: placement at 256/200/192/136/128 + competitor mask 192–255 on all 8 GPUs (48 steps); the leak v4
    suite on every GPU, both levels (128 steps).
  - Gate: `analyze.py gate` → `gates.json` → `release_e3_minimal_v4`.
  - Production: 8 blocks, one per GPU. Each block holds E2 at 256 and E3-minimal: Qwen32B at n ∈ {192, 128} ×
    {absent, mem50, mem100, arith100}; plus one `inactive` control (an idle competitor process) at 192 in block 0.
    65 E3 + 8 E2 steps, shuffled per block (seed `20261026:mi350x:leak4:block<b>`).
  - Competitor treatments (`cubw.py` `TREATMENTS`): `mem50`/`mem100` = streaming read of a 1 GiB buffer (above the
    256 MiB MALL) at duty 0.5/1.0; `arith100` = FMA at duty 1.0. Competitor mask = n..255.
- **How it was run:** `plan.py --kind auto --cohort mi350x --leak4` (249 steps; plan
  `$WORK/cubw_plans/mi350x_auto_20261104_085839_f737d1b1.json`), via `submit.sh auto mi350x …`, held until the smoke
  was reviewed, then released. One lane per GPU (L0–L7).
- **What was measured and how it was analysed:** leak v4 as above, 16 tests. E3: decode ms/step; ratio to `absent`
  at the same n, block-paired bootstrap 95% CI over 8 blocks; competitor GB/s or TFLOP/s during trial windows.
- **Results — gates** (`gates.json`):

| gate | value |
|---|---|
| placement_ok | true (8 GPUs, 5 levels + competitor mask, exact per-XCD counts) |
| lifecycle_ok | true |
| outputs_ok | true (105/105 runs complete, 1 digest) |
| **leak_ok_v4** | **true: 16 of 16 tests pass** |
| **release_e3_minimal_v4** | **true** |
| clock_ok | false (pooled gfx 1583–2172 MHz, ratio 1.37; memory clock 2000 MHz) |

- **Results — leak v4** (`gates.json` → `leak_v4`):

| | n = 192 (probe CUs 192–255) | n = 128 (probe CUs 128–255) |
|---|---|---|
| alone: coverage / largest late fraction | 64/64 on every GPU / ≤ 0.0042 | 128/128 / ≤ 0.0058 |
| natives: flagged CUs | 0 in all 16 runs | 0 in all 16 runs |
| **required: `cuprobe` on 0..n+7** | 25–33 flagged (top-8 excess 0.68–0.77) | 25–36 (0.60–0.67) |
| **required: decoder on 0..n+7** | 28 flagged (0.42–0.46) | 28–33 (0.49–0.54) |
| recorded: FMA / read on 0..n+7 | 24–41 / 25–28 | 24–29 / 25–43 |
| **decoder on 0..n−1: flagged (largest excess)** | **0** on all 8 GPUs (≤ 0.0020) | **0** on all 8 GPUs (≤ 0.0017) |

  Every positive cleared ≥ 4 on every GPU, so the test was sensitive everywhere. Every decoder run flagged 0 = the
  natives' maximum of 0. The largest decoder excess (0.002) is 50× below the 0.10 threshold.

- **Unregistered observation: positives flag ~25 CUs here vs 8 in the smoke.** Not pre-registered. From the raw
  per-CU counts on GPU L0, n = 192, `cuprobe` positive (re-checked for this handbook from
  `e1_leak_v4/corun_cuprobe_0-199_gapprobe_192-255_L0/a0/attempt1/competitor.jsonl`):
  - **The 8 shared CUs** (one per XCD): the probe got a WG there in only ~38.4k–41.1k of ~98k launches (~40%). The
    displaced WG ran as a second WG on another CU: exactly 8 CUs show large `multi` counts (~57–60k each). Same as
    the smoke.
  - **About 17 unshared probe CUs**: a WG landed there in every launch, but started > 5 µs late in most of them
    (late fraction mostly 0.67–0.71). The co-runner cannot reach these CUs, yet their starts are delayed.
  - Likely cause given in the review: in-order dispatch. A WG waiting for a busy shared CU holds back the WGs
    queued behind it. **Why this shows on the 8-GPU node and not on the `mi3501x` slice is not known.**
  - Effect: it only makes positives stronger. No flag, threshold or verdict changed, and decoder runs show no such
    spread (0 flagged).

- **Results — E3-minimal** (8 blocks; ratio to no competitor at the same n; `summary.md` E3 table):

| n | arithmetic 100% | memory 50% | memory 100% |
|---:|---:|---:|---:|
| 192 | 1.019 [1.015, 1.023] (29.4 TFLOP/s) | 1.116 [1.115, 1.118] (1183 GB/s) | 1.273 [1.269, 1.278] (2536 GB/s) |
| 128 | 1.022 [1.018, 1.027] (29.6 TFLOP/s) | 1.107 [1.104, 1.109] (1269 GB/s) | 1.253 [1.248, 1.258] (2683 GB/s) |

  - Decoder alone: 14.75 ms/step at 192, 14.57 ms at 128. E2 at 256 in the same job: 12.28 ms.
  - Inactive control at 192: 1.0001 [0.9992, 1.0010] (1 block): an idle process alone costs nothing.
  - memory100 / arith100: **1.2495** [1.2464, 1.2532] at 192, **1.2257** [1.2218, 1.2301] at 128.
  - 128 alone / 192 beside mem100: **0.7754** [0.7723, 0.7782]; 128 alone / 192 beside mem50: 0.8845.

  Cross-cohort (review §3):

| | MI300X 492956 | MI325X 489594 | **MI350X 499332** |
|---|---:|---:|---:|
| memory 100% / arithmetic 100% | 1.18 (232), 1.21 (152) | 1.14, 1.16 | **1.25 (192), 1.23 (128)** |
| decoder alone on fewer CUs / decoder beside memory 100% | 0.846 | 0.924 | **0.775** |

- **What it means:**
  - Pre-registered expectation (direction only): **met**. Memory competition hurts much more than arithmetic.
  - On MI350X a co-runner holding 64 CUs of arithmetic costs the decoder ~2%. A co-runner streaming 2.5 TB/s costs
    it ~27%.
  - **B_CU reading:** the decoder's cost tracks the HBM bandwidth taken from it, not the CUs taken from it. 128 CUs
    with uncontested HBM decode 22% faster than 192 CUs beside a memory-bound neighbour.
  - The leak test shows the contention is not through stolen CUs. It goes through the shared memory system (HBM,
    MALL = the Infinity Cache, fabric).
  - All three cohorts now have E3-minimal, with the same ordering; the gap is largest on MI350X.
  - Follow-up A is **done**: E3 is no longer blocked on MI350X.
- **Failures and caveats:**
  - Observational and clock-confounded (`clock_ok` false).
  - E3 used the stock `wvSplitK` (grid = 256). The decoder's absent time at 192 and 128 carries the ~1.20
    dispatch cost found in 497082. The E3 ratios are relative to that same stock decoder, so they are not affected.
  - Inactive control has 1 block only.
  - The ~25-CU positive spread is unexplained (above).
- **Files:** review [mi350x_leak4_review_20261104.md](../mi350x_leak4_review_20261104.md);
  `results_published/amd_20261104_mi350x_cubw_v1_6_leak4_job499332/` (`gates.json` has every leak test,
  `summary.md`, `cubw_tables.json`, `launch_plan.json`); raw
  `$WORK/results_amd/amd_20261104_mi350x_cubw_v1_auto_job499332/` (`e1_leak_v4/`, `e3/`, `e2/`, `RUN_STATUS.log`,
  `lane_L*_{pilot,production}.log`); commits `f737d1b1`, `4c9ea53f` (review).

---

## Follow-up B: shader-engine balance on MI300X — background

What was known before these jobs (plan §B):

- **SE layout under contiguous masks** (placement keys from 492956). Active CUs per SE on each XCD:

| n | 304 | 288 | 272 | 256 | 232 | 224 | 200 | 192 | 152 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|
| CUs per SE | **9,10,10,9** | 9,9,9,9 | 9,9,8,8 | 8,8,8,8 | 8,7,7,7 | 7,7,7,7 | 7,6,6,6 | 6,6,6,6 | 5,5,5,4 | 4,4,4,4 |

  288, 256, 224, 192, 128 are balanced. 304, 272, 232, 200, 152 are not. **The full MI300X device itself is
  unbalanced** (9,10,10,9). (`plan.py` `SE_LEVELS`.)
- `wvSplitK` on gfx942: one 64 KiB (full-LDS) WG per CU, grid = 304 under any mask.
- Stock E2 R(n) on MI300X is non-monotone (272 1.068, 232 1.033, 184 1.010, 152 1.085). 492956 found the kernel
  sized to n slower at every level.

**H_SE on MI300X** (as pre-registered): the gfx942 dispatcher, like gfx950's, does not follow unequal active-CU
counts per SE. Because the full device is unbalanced, **the stock full-device decode is itself affected**.

**Pre-registered predictions** (judged on the 8-GPU job only):

| id | test | pass condition |
|---|---|---|
| P1m | dispatch probe (`svc_bench_v5 --mode dispatch`), 2 GPUs, 5 trials, 16 waves, 64 KiB LDS | (a) balanced masks 288/256/224/192/128 with grid = n: max 1 WG/CU and 0 serialized in every trial, both GPUs. (b) full device 304 **and** ≥ 3 of 272/232/200/152 with grid = n: max ≥ 2 WG/CU and serialized > 0, both GPUs. (c) grid = balanced(n) under mask n: max 1 WG/CU at every level. |
| P2m | GEMM shapes (`e6_shapes`), 3 acquisitions | at every unbalanced level including 304, size balanced(n) faster than size n for ≥ 4 of 5 Qwen32B GEMM shapes |
| P3m | Qwen32B decode, 8 blocks, paired within GPU | (a) R_balanced(n) < R_e2(n) at each of 272, 232, 200, 152 (R vs stock 304 in the same block). (b) T(304, sized 288) / T(304, stock) < 1 with upper 95% CI < 1. |

**Verdict rule:** holds if P1m and P3m pass; **refuted on MI300X if P1m fails**; if P1m passes and P3m fails,
the artefact exists but does not drive decode.

The dispatch probe (`svc_bench_v5 --mode dispatch`) launches `--blocks` WGs of 1024 threads (16 waves), each
holding the full LDS (or `--lds 1` for the no-LDS control) and streaming its own chunk of a 1 GiB buffer. Each WG
records its physical CU, start and end. Outputs: CUs used, max WGs on one CU, "serialized" = WGs that started only
after another WG on the same CU finished, and WGs per SE of XCD 0. `dispatch_suite` runs, for each mask n, grids
{304, n, balanced(n)} plus a no-LDS control at grid n.

### #27 MI300X shader-engine smoke — job 499333

- **At a glance:** `mi3001x` (one MI300X), node k002-004-v8. 2026-11-04 09:03:08–09:20:32 (17 min). Release
  `f737d1b1`. Status: **complete**, 52/52 steps rc = 0. Pilot-labelled; judges nothing.
- **Question:** port check before the 8-GPU job, and a first look: does the gfx942 dispatcher serialize at the
  unbalanced masks, including the full device?
- **Design:** placement at the 10 SE levels + competitor mask (11 steps); the dispatch suite at the 10 levels
  (34 steps); E2 (stock) and `e6_balanced` at 304 and 272; `e6_decode` at 256; `e6_shapes` at 304 and 272 with
  `--cu-counts balanced(n)`. One acquisition per cell.
- **How it was run:** `plan.py --kind pilot --cohort mi300x --se-study --smoke`; plan
  `$WORK/cubw_plans/mi300x_pilot_20261104_085841_f737d1b1.json`. 499334 was held behind it.
- **Results — dispatch** (5 trials; [mi300x_se_smoke_review_20261104.md](../mi300x_se_smoke_review_20261104.md)):

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

- **Results — decode and shapes** (one acquisition each):

| cell | ms per step | vs 304 stock |
|---|---:|---:|
| 304 stock | 17.96 | 1.000 |
| 304, sized for 288 | 18.01 | 1.002 |
| 272 stock | 19.03 | 1.060 |
| **272, sized for 256** | **18.01** | **1.003** |
| 256, sized for 256 (`e6_decode`) | 18.00 | 1.002 |

  Shapes at 272: the 256-sized kernel was fastest or tied for all 5 shapes. At 304, 288-sized and device-sized were
  equal within 1% except qkv (19.1 vs 17.8 µs).
- **What it means:**
  - At reduced unbalanced masks gfx942 behaves like gfx950: ~8 WGs serialize; every balanced grid gives one per CU.
  - **At the full device it does not.** With 304 WGs on 9,10,10,9 CUs per SE, the dispatcher places exactly
    9,10,10,9 WGs per SE. So P1m(b), and likely P3m(b), looked set to fail.
  - At 272, sizing for 256 brings decode back to full-device speed.
- **Decision:** no port bug found, so **499334 was released unchanged**. Predictions and verdict rule untouched.
  The reduced-level findings would be reported separately as not pre-registered.
- **Failures and caveats:** one acquisition per cell; clock-confounded (pooled gfx 1328–1668 MHz, ratio 1.26). The
  smoke ran the dispatch suite on one lane only, so it could not reveal the two-lane cell collision found in 499334.
- **Files:** review [mi300x_se_smoke_review_20261104.md](../mi300x_se_smoke_review_20261104.md);
  `results_published/amd_20261104_mi300x_cubw_v1_6_se_smoke_job499333/`; raw
  `$WORK/results_amd/amd_20261104_mi300x_cubw_v1_pilot_job499333/` (`e6_dispatch/`, `e6_balanced/`, `e6_decode/`,
  `e6_shapes/`); commits `f737d1b1`, `a5d56bad` (review).

### #28 MI300X shader-engine study — job 499334

- **At a glance:** `mi3008x` (8× MI300X), node k002-003. 2026-11-05 10:11:19–11:11:58 (61 min: pilot 10:11–10:18,
  production 10:18–11:10). Release `f737d1b1`. Status: **complete with one lane crash** (L1's pilot lane, see
  below). 337/337 steps with a return code had rc = 0; 15 journal records carry no return code. Outputs 98/98,
  1 digest. Placement exact on all 8 GPUs at all 10 levels.
- **Question:** the pre-registered H_SE test on gfx942 (P1m–P3m), with emphasis on whether the full 304-CU device
  is hurt by its own 9,10,10,9 layout.
- **Design** (`plan.py` `build_se_study`, 371 steps):
  - Pilot: placement at 10 levels on 8 GPUs (88 steps); dispatch suite on L0 and L1 (68 steps); E2 pilot at 304,
    288, 272 on L2–L7 (18 steps).
  - Gate `release_e2` (= placement_ok and outputs_ok).
  - Production, 8 blocks (one per GPU), shuffled per block: E2 (stock) at all 10 levels (80); `e6_balanced`
    (size balanced(n)) at the 5 unbalanced levels incl. 304 → 288 (40); `e6_decode` (size n) at the 5 balanced
    levels (40); kernel traces (`e6_timeline`) of stock 304 and 272 and of balanced-sized cells at the 5
    unbalanced levels (7); `e6_shapes` × 3 acquisitions at 10 levels (30).
- **How it was run:** `plan.py --kind auto --cohort mi300x --se-study`; plan
  `$WORK/cubw_plans/mi300x_auto_20261104_085842_f737d1b1.json`; held behind 499333, then released. Sizing via
  `CUBW_CU_COUNT_OVERRIDE=<count>`.
- **What was measured and how it was analysed:** dispatch probe rows; GEMM shape means (µs); decode ms/step and
  R with block-paired bootstrap CI. The predictions were scored in `hse_judgment.json`, which reads the raw dispatch
  files and uses L0 only. It was written by `scripts/amd/cubw/judge_se_mi300x.py` (added 2026-11-05; rerunning it reproduces the published file byte for byte).

**Port bug: lane L1's dispatch probe collided with L0's.**

- The `--se-study` plan ran `dispatch_suite` on L0 and L1 with **the same cell names** (`n<n>_g<g>`). Attempt
  directories are per cell (`e6_dispatch/<cell>/a0`), so L1's first dispatch step found L0's directory and exited
  with `FileExistsError`. That ended L1's pilot lane after its 11 placement steps. Its production steps ran
  normally.
- Effect on tables: the dispatch table in `summary.md` lists every row twice, both copies L0's data.
- Effect on verdict: none. P1m needed the full-device prediction on both GPUs; it already fails on L0, in
  agreement with the smoke on a third GPU.
- **Fix (v1.6.1, `plan.py`, commit `8b648879`):** `dispatch_suite` takes a `tag`; the 8-GPU study tags cells
  `_L0` and `_L1`. The regenerated plan has 371 unique cells, the 497082 follow-up plan regenerates unchanged, and
  `test_analyze.py` has 0 failures. Nothing was rerun.

**P1m — dispatch probe, GPU L0 (5 trials per row) — FAIL** (`hse_judgment.json` → `P1m`):

| mask n | CUs per SE | grid = n: CUs used, max WG/CU, serialized (mean), ms | grid = balanced(n) | stock grid 304 under mask n |
|---:|---|---|---|---|
| **304** | **9,10,10,9** | 304, **1, 0**, 0.257 | 288: 1, 0, 0.259 | — |
| 288 | 9,9,9,9 | 288, 1, 0, 0.258 | — | 2, 16, 0.408 |
| 272 | 9,9,8,8 | 264, **2, 8**, 0.368 | 256: 1, 0, 0.278 | 2, 32, 0.415 |
| 256 | 8,8,8,8 | 256, 1, 0, 0.272 | — | 2, 48, 0.329 |
| 232 | 8,7,7,7 | 232, **2, 6.4**, 0.393 | 224: 1, 0, 0.258 | 2, 72, 0.380 |
| 224 | 7,7,7,7 | 224, 1, 0, 0.258 | — | 2, 80, 0.307 |
| 200 | 7,6,6,6 | 200, **2, 6.4**, 0.409 | 192: 1, 0, 0.263 | 2, 104, 0.289 |
| 192 | 6,6,6,6 | 192, 1, 0, 0.263 | — | 2, 112, 0.353 |
| 152 | 5,5,5,4 | 152, **2, 4.8**, 0.463 | 128: 1, 0, 0.299 | 3, 152, 0.361 |
| 128 | 4,4,4,4 | 128, 1, 0, 0.300 | — | 3, 176, 0.361 |

- Serialization at the reduced unbalanced masks is per trial either 0 or 8 (for example 152: 0, 8, 8, 8, 0; 272:
  8 in all 5), so the means 6.4 and 4.8 are averages of such trials (`hse_judgment.json` `trials`).
- (a) met. (c) met. (b) **not met**: the reduced levels serialize (4/4), but the full device does not. The
  dispatcher places 9,10,10,9 WGs per SE, matching the active CUs.

**P2m — GEMM shapes (3 acquisitions; µs, sized balanced(n) vs sized n under mask n) — FAIL:**

| n (balanced size) | down | gate_up | lm_head | o_proj | qkv | faster |
|---|---|---|---|---|---|---:|
| **304 (288)** | 67.0 vs 71.0 | 132.3 vs 131.6 | 366.0 vs 364.8 | 14.0 vs 14.4 | 17.8 vs 19.2 | **3/5** |
| 272 (256) | 68.0 vs 67.1 | 128.6 vs 138.9 | 360.7 vs 388.0 | 14.3 vs 14.5 | 17.6 vs 18.1 | 4/5 |
| 232 (224) | 67.2 vs 67.6 | 128.5 vs 143.4 | 361.5 vs 397.0 | 14.3 vs 14.5 | 17.4 vs 18.0 | 5/5 |
| 200 (192) | 67.1 vs 82.2 | 128.7 vs 144.0 | 360.0 vs 405.7 | 14.5 vs 16.7 | 17.5 vs 23.8 | 5/5 |
| 152 (128) | 72.3 vs 83.8 | 127.6 vs 150.8 | 360.4 vs 423.3 | 14.5 vs 16.8 | 17.4 vs 23.3 | 5/5 |

Fails only at 304, where the two sizes are within noise (size-304 repeat noise about ±4%; review §2).

**P3m — Qwen32B decode (8 blocks, paired within GPU) — (a) PASS, (b) FAIL:**

| mask n | stock R_e2(n) | sized to balanced(n): R | balanced size |
|---:|---|---|---:|
| 304 | 1 | **0.998** [0.997, 1.000] | 288 |
| 272 | 1.066 [1.061, 1.070] | **0.996** [0.995, 0.998] | 256 |
| 232 | 1.032 [1.029, 1.035] | **0.997** [0.995, 0.999] | 224 |
| 200 | 1.017 [1.014, 1.020] | **0.997** [0.995, 1.000] | 192 |
| 152 | 1.082 [1.078, 1.086] | **0.994** [0.991, 0.996] | 128 |

- (a) R_balanced < R_e2 at all four reduced unbalanced levels, non-overlapping CIs: met.
- (b) T(304, sized 288) / T(304, stock) = **0.9984**, upper 95% CI **1.00002**. The rule needs upper CI < 1:
  **not met**, by 0.00002, and recorded as a fail as written.

Recorded, not judged (balanced masks):

| n | 288 | 256 | 224 | 192 | 128 |
|---|---:|---:|---:|---:|---:|
| stock R_e2(n) | 1.045 | 1.062 | 1.023 | 1.013 | 1.069 |
| sized to n (`e6_decode`) | **0.998** | **0.996** | **0.998** | **0.996** | **0.993** |

**Verdict (pre-registered): "refuted on MI300X"** (`hse_judgment.json` `verdict_pre_registered`). P1m fails. The
claim that failed is specifically that the **full** MI300X device is hurt by its 9,10,10,9 layout. It is not:
the gfx942 dispatcher follows that layout, the stock kernel does not serialize at 304, and sizing for 288 gains at
most 0.2%.

**Beyond the verdict (not pre-registered as a whole; observational):**

- **At reduced CU counts the stock decode penalty on MI300X is a dispatch artefact, and sizing removes it.** Under
  any mask below 304, stock `wvSplitK` still launches 304 WGs, and 16–176 of them serialize behind another WG on
  the same CU (right-hand dispatch column).
- Kernel traces (`e6_kernels_se.json`; `wvSplitK` = 83.2% of decode kernel time): mean `wvSplitK` time vs stock
  304 is **1.090 at 272 stock**, against 0.981–0.990 when sized to balanced(n) at 272, 232, 200, 152 (and 0.983
  for 304 sized 288). Trace timings carry trace distortion; the ratios share it.
- **Balanced-sizing result:** with `wvSplitK` sized to a balanced count, decode stays at full-device speed at
  every level tested, **down to 128 of 304 CUs (42%)**. All 9 sized cells below 304 (4 `e6_balanced` at
  272/232/200/152 and 5 `e6_decode` at 288/256/224/192/128) sit at **R 0.993–0.998**.
- **Revision of 492956.** The MI300X extension 492956 reported "the resized kernel is slower at every level"
  (272 1.043 … 152 1.202). Every size it used (272, 248, 232, 216, 200, 184, 152) is SE-unbalanced. That slowdown
  was the same dispatch artefact, not a bandwidth limit, as on MI350X in 497082. **492956's conclusion is revised.**
- **Cross-cohort B_CU** (balanced-size kernel; B_CU = peak HBM / CUs doing the work):

| cohort | CUs doing the work | B_CU needed | R |
|---|---:|---:|---:|
| MI300X | 288 / 256 / 224 / 192 | 18.4 / 20.7 / 23.7 / 27.6 GB/s | 0.996–0.998 |
| MI300X | 128 | 41.4 GB/s | 0.993 |
| MI350X (497082) | 192 | 41.7 GB/s | 1.009 |
| MI350X (497082) | 160 | 50.0 GB/s | 1.067 |
| MI350X (497082) | 128 | 62.5 GB/s | 1.168 |

  Both cohorts decode at full speed up to about **42 GB/s of HBM bandwidth per CU**; MI350X shows the cost rising
  above that. The two cohorts meet at the same B_CU (MI300X at 128 CUs, MI350X at 192 CUs).
- **Practical rule, now on both cohorts:** size `wvSplitK` to the largest SE-balanced CU count ≤ the active CUs (a
  multiple of 32 on MI300X and MI350X). On MI300X the full device needs no change.

- **What it means:** Follow-up B is **done**. H_SE as pre-registered is refuted on MI300X because the full device
  is unaffected. But the reduced-CU serialization and its removal by balanced sizing replicate on gfx942, and this
  revises the 492956 reading.
- **Failures and caveats:**
  - L1 dispatch lane crashed (port bug above); P1m judged on one GPU, not two. `summary.md` dispatch rows are
    duplicated.
  - `lifecycle_ok` is false by construction: the plan has no E3 competitor runs. It gates only E3, unused here.
  - Clock-confounded: pooled gfx 1304–1921 MHz (ratio 1.47), memory clock 1051–1300 MHz. R slightly below 1 is
    probably clock: fewer active CUs run at higher gfx clocks (pooled ratio up to 1.49 at 128; review §3). None of
    this is a non-inferiority claim.
  - P3m(b) failed by 0.00002 on the CI bound; recorded as a fail.
  - MI325X (gfx942, 304 CUs) is presumably the same as MI300X but untested.
- **Files:** review [mi300x_se_review_20261105.md](../mi300x_se_review_20261105.md);
  `results_published/amd_20261105_mi300x_cubw_v1_6_se_job499334/` (`hse_judgment.json` = scored predictions,
  `e6_kernels_se.json` = kernel traces, `gates.json`, `summary.md`, `cubw_tables.json`, `launch_plan.json`); raw
  `$WORK/results_amd/amd_20261105_mi300x_cubw_v1_auto_job499334/` (`e6_dispatch/`, `e6_timeline/`, `e2/`,
  `e6_balanced/`, `e6_decode/`, `e6_shapes/`); commits `f737d1b1` (code), `8b648879` (review + plan.py v1.6.1).

---

## Summary of the v1.6 follow-ups

| # | job | partition | question | outcome |
|---|---|---|---|---|
| 25 | 499331 | `mi3501x` | is leak v4 sensitive on gfx950? | yes: positives 8/8 at both levels; decoder 0 → 499332 released |
| 26 | 499332 | `mi3508x` | leak v4 on 8 GPUs, then E3-minimal | `leak_ok_v4` 16/16; E3 memory100/arith100 = 1.25 (192), 1.23 (128) |
| 27 | 499333 | `mi3001x` | H_SE smoke on MI300X | full device does not serialize; reduced unbalanced masks do → 499334 released |
| 28 | 499334 | `mi3008x` | H_SE on MI300X (P1m–P3m) | **refuted on MI300X** (P1m fail at 304); balanced sizing R 0.993–0.998 down to 128 CUs; revises 492956; ~42 GB/s per CU on both cohorts |

Open items (review §4): clock qualification; the off-cluster raw archive destination; MI325X untested for H_SE.

If any of this is presented or published, the AMD acknowledgement is: "This work was supported in part by Advanced
Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
