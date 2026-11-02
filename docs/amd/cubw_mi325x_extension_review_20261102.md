# cubw v1.3 extension — MI325X job 492957 review (2 November 2026)

This reviews the MI325X job of the pre-registered extension
([cubw_extension_plan_20261101.md](cubw_extension_plan_20261101.md)), release de2baecb.

- **Job:** 492957, `cubw-mi325x-auto`, k002-001, 07:35–08:44 CST, 1 h 09 min.
- **Raw data:** `$WORK/results_amd/amd_20261102_mi325x_cubw_v1_auto_job492957`. 757 files are hashed in
  `SHA256SUMS`. The raw data is not in git.
- **Compact tables:** [results_published/amd_20261102_mi325x_cubw_v1_3_auto_job492957/](../../results_published/amd_20261102_mi325x_cubw_v1_3_auto_job492957/).
  They were re-read with reader v1.4 (§7).

All timing here is **observational**, because the pooled clock gate failed:
- gfx clock ranged 1450–2087 MHz;
- memory clock ranged 1257–1500 MHz.

Clock and power changes are not authorized, so this cannot be fixed within scope.

## 1. Gates

| gate | result |
|---|---|
| placement_ok | True: all 8 levels on all 8 GPUs, exact count and balanced per XCD |
| lifecycle_ok / outputs_ok | True: 66/66 ordinary runs, 1 output digest |
| leak_ok_v2 (pre-registered) | **True** at n = 232 and n = 152 (§5) |
| release_e3_minimal_v2 | True. MI325X E3-minimal was already released; no E3 was planned in this job |
| clock_ok | False, so every timing result is observational |

**Step outcomes:** 235 steps ran. All E2, `e6_decode`, timeline and shape steps completed. The counter steps failed
in two groups, and the runner made no retries:
- **4/4 eager-decode counter runs failed.**
  - vLLM eager decode under `rocprofv3 --pmc` raised `hipErrorLaunchFailure` (one run raised
    `hipErrorIllegalAddress`).
  - rocprofv3 then aborted with "retired dangling correlation IDs".
  - rocprofv3 warns that this node's CP firmware (177) is below the minimum it requires (186), and that older
    firmware "can cause device resets and VM Page Faults on MI3XX".
  - These are recorded as failed acquisitions. Firmware is out of scope.
- **4/16 shape-counter runs failed:**
  - n304 TCC_EA0_RDREQ: SIGTERM after a hang;
  - n200 FETCH_SIZE and n200 TCC_EA0_RDREQ: SIGABRT;
  - n232 TCC_EA0_RDREQ_LEVEL: rc = 1.

## 2. E2-fine: the 200-vs-232 anomaly is part of a non-monotone curve

Qwen32B graph decode, B1, context 4096. 8 blocks, with level order randomized independently per GPU.

| n | 304 | 272 | 248 | 232 | 216 | 200 | 184 | 152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ms/step | 15.58 | 17.39 | 16.83 | 16.39 | 15.98 | 15.90 | 15.71 | 17.77 |
| R(n) | 1 | 1.116 | 1.080 | 1.052 | 1.026 | 1.021 | 1.008 | 1.141 |
| 95% CI | | 1.088–1.144 | 1.056–1.100 | 1.029–1.071 | 1.003–1.044 | 0.998–1.039 | 0.986–1.027 | 1.116–1.162 |

- **The effect is per level, not drift.** Every GPU, in its own random order, gives the same per-level time
  (272: 17.2–18.1 ms; 184: 15.6–15.8 ms).
- **The anomaly reproduces.** R(200) < R(232), as in production job 489594 (1.056 vs 1.074).
- **Across jobs, absolute R differs.** This job's R(232) = 1.052 vs 1.074 in 489594, and R(152) = 1.141 vs 1.187.
  The two jobs ran on different days with unpinned clocks.
- **The curve is not monotone.** It *falls* from 272 to 184 and then rises sharply at 152. Removing the first
  32 CUs costs more than removing the next 88.

## 3. Mechanism: one kernel, `wvSplitK`

Kernel traces at all 8 levels (`e6_kernels_all_levels.json`; kernel time ratio to n=304):

| kernel (share at 304) | 272 | 248 | 232 | 216 | 200 | 184 | 152 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wvSplitK (83.5%) | 1.162 | 1.125 | 1.082 | 1.049 | 1.047 | 1.035 | 1.203 |
| paged attention (4.3%) | 0.96 | 1.11 | 1.13 | 1.12 | 1.10 | 1.10 | 1.06 |
| all other top kernels (~2% each) | 0.90–0.98 at every level | | | | | | |

`wvSplitK` alone reproduces the E2 curve's shape. Every other kernel is flat or slightly faster.

**Measured facts:**
- In decode, vLLM launches `wvSplitK` with **304 workgroups**, one per device CU, whatever the mask.
- Each workgroup uses all 64 KiB of LDS, so only one fits on a CU.
- Under a mask of n CUs, 304 − n workgroups wait for a second round. FETCH_SIZE is unchanged.

**Hypothesis (not tested here):** the decode cost is set by how the 304 workgroups split into two rounds on n
CUs, where per-workgroup memory parallelism limits each round. Near 272, a small second round runs at low
bandwidth. Near 184 the two rounds are better balanced. At 152 there are two full rounds of half the
memory-level parallelism. This would explain the shape but it is not a fitted model.

**Kernel sized for the mask (`e6_decode`, `CUBW_CU_COUNT_OVERRIDE`=n):**
- 5 blocks, same GPUs and blocks as E2. Outputs are bit-identical to ordinary decode (35/35, same digest).
- This *removes* the second round but makes decode **slower** at every level.
- The CIs exclude the ordinary kernel's R at every level except 272, where the gap is 0.06 and the CIs touch.

| n | 272 | 248 | 232 | 216 | 200 | 184 | 152 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R, kernel sized for n | 1.056 | 1.187 | 1.108 | 1.210 | 1.255 | 1.238 | 1.332 |
| R, ordinary (sized for 304) | 1.116 | 1.080 | 1.052 | 1.026 | 1.021 | 1.008 | 1.141 |

So having fewer workgroups resident in flight costs more than the second round does. This is consistent with
decode GEMV being limited by memory-level parallelism, not by HBM bandwidth.

## 4. Counters on the shape controls

These are pre-MALL counters on the Qwen32B weight shapes. Each is one acquisition in light mode, averaged over
3 timed calls, so the precision is about ±5–8% (304 has the same grid for both variants).

- **Traffic is unchanged by CU count.** FETCH_SIZE equals the weight bytes and is identical at every level and
  variant. RDREQ = 8 × FETCH_SIZE, so there are no extra or partial 128 B requests.
- **Mean outstanding reads drop.** The measure is TCC_EA0_RDREQ_LEVEL / GRBM_GUI_ACTIVE, the mean number of
  in-flight HBM read requests (Little's law; from separate runs), shown as a ratio to n=304:

| shape | kernel | 304 (abs) | 200 | 152 |
|---|---|---:|---:|---:|
| gate_up | sized for 304 | 4607 | 0.76 | 0.78 |
| gate_up | sized for n | 4697 | 0.63 | 0.57 |
| down | sized for 304 | 4267 | 0.72 | 0.68 |
| down | sized for n | 4089 | 0.71 | 0.64 |
| lm_head | sized for 304 | 4668 | 0.82 | 0.72 |
| lm_head | sized for n | 4305 | 0.65 | 0.60 |
| o_proj | sized for 304 / n | 2955 / 2864 | 0.71 / 0.77 | 0.57 / 0.55 |
| qkv | sized for 304 / n | 2747 / 2991 | 0.91 / 0.66 | 1.06 / 0.92 |

Fewer CUs means fewer HBM reads in flight for the same bytes, and the kernel sized for n has fewer than the one
sized for 304. This is direct counter evidence that **the CU-restriction penalty here is a memory-level
parallelism loss, not a byte-volume or bandwidth-ceiling effect.**

The isolated shape timings (`e6_shapes`, 3 acquisitions) are noisier than decode. They are not monotone in n
either, and they should not be read cell by cell.

## 5. Leak test v2 (pre-registered rule)

| suite | probe CUs | native min q | positive control: CUs < 0.85 | decoder min q | threshold | ok |
|---|---:|---|---:|---:|---:|---|
| n = 232 | 72 | 1.000, 1.000 | 8 | 1.000 | 0.950 | **True** |
| n = 152 | 152 | 0.885, 0.881 | 8 | 0.890 | 0.831 | **True** |

At n = 232 every q is 1.000. All 72 probe CUs lost rate equally under every co-runner, so contention was
chip-wide. The positive control still flagged exactly its 8 shared CUs, so the test was sensitive.

## 6. E7 blind check (pre-registered predictions)

Model r2, constrained: control = 4 GiB streaming read, s = 0.916. A point passes if |obs − pred| ≤ 0.02 + the
CI half-width.

| n | predicted | observed | residual | within tolerance |
|---:|---:|---:|---:|---|
| 272 | 1.025 | 1.116 | +0.092 | **no** |
| 248 | 1.043 | 1.080 | +0.037 | yes |
| 216 | 1.047 | 1.026 | −0.021 | yes |
| 184 | 1.082 | 1.008 | −0.074 | **no** |

**The model is not validated on MI325X:** 2 of 4 blind points miss, and 1 of 5 held-out points missed earlier.
Any model that is monotone in n cannot fit a non-monotone curve. §3 indicates the curve comes from the
kernel's round count and memory-level parallelism, not from scaling with a bandwidth control. The tables are in
`e7_mi325x_r2_blindcheck.json`.

## 7. Reader change (v1.4)

v1.3 labelled `e6_decode` rows "pilot … never promoted", although they are a blocked 5-acquisition comparison.
v1.4 applies the ordinary disposition, so they are now `observational_clock_confounded`. No values changed. The
tables in the job's own `analysis/` (written by the v1.3 reader inside the job) are left as they are.

## 8. Consequences for the research question

- Under a CU mask, the decode penalty of this vLLM build on MI325X is set by one software kernel's launch
  geometry and per-CU memory-level parallelism, not by bandwidth demand. The penalty ranges from 1–14% depending
  on n, with no monotone trend.
- A fixed CU-to-bandwidth "shoreline" ratio is therefore not identifiable from these decode measurements alone.
  At least two kernel configurations change the answer in opposite directions.
- Not explained yet: why paged attention is *faster* at 272 than at 304 but 10–13% slower at 248–184.
