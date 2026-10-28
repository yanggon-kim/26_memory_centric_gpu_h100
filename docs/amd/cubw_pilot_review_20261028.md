# cubw_v1 MI300X pilot review and v1.1 repair (28 October 2026)

Pilot of [the compute/bandwidth follow-up](compute_bandwidth_followup_20261026.md), job 481167 (mi3008x,
k002-002, 2026-10-28 08:34–09:06 CDT, 32 min), release 2da9ae35. Raw root:
`$WORK/results_amd/amd_20261028_mi300x_cubw_v1_pilot_job481167` (643 files, `SHA256SUMS`); compact tables in
`results_published/amd_20261028_mi300x_cubw_v1_pilot_job481167/`. Every number below is **one pilot
acquisition**: pilot rows are never promoted to production comparisons, and none of them is clock-qualified.

## 1. Gates as declared in the v1 plan (unchanged record)

| gate | v1 result | diagnosis |
|---|---|---|
| placement_ok | **pass** | 8 GPUs × {304, 232, 200, 152, 96, competitor 232–303}: distinct physical CUs equal the request and every XCD gets the same count (38/29/25/19/12/9 per XCD). |
| lifecycle_ok | fail | **Tool defect.** amdsmi lists every KFD process on the node for every GPU (an L3 competitor PID appears in GPU 2's list). The v1 residual check therefore never could be empty. Competitor started/stopped events and PID-gone checks all passed; no own-lane stopped PID ever reappeared. |
| outputs_ok | fail | **Reader defect.** The gate counted two rocprofv3 runs although its declared definition is unprofiled runs. Unprofiled Qwen32B C4096 runs: 25/25 complete, one output digest across all CU levels and treatments. |
| leak_ok | fail | FMA probe during decoder trials / probe alone = 0.58–0.61 on all four probes (n=232 and 152 arith100; n=152 and 96 leakprobe). Clocks explain only ~8% (probe alone 2088 MHz, beside the decoder 1923 MHz at 232-303). Leakage vs inter-process contention is not identifiable from these probes. |
| clock_ok | fail | Pooled Qwen32B E2 gfx clocks 1319–2021 MHz (×1.53); memory 1059–1300 MHz. Decode-alone cells run ~1250–1400 MHz at 260–300 W, competitor cells ~1920 MHz. Fixed-clock qualification is unavailable on this stack (clocks/power not changed, per the assignment). |
| E5 graph counters | failed | `rocprofv3 --pmc FETCH_SIZE` on graph-mode decode: illegal memory access, then a hang to the 1200 s timeout; the Triton-excluded attempt also hung. rocprofv3 left PPID=1 processes that survived the group stop until job end (they held no VRAM on the target GPU: later runs' isolation check saw none). Kernel-trace timelines worked: slowdown ×1.076 (304) and ×1.099 (152), outputs match. |

Release under v1 rules: nothing. Re-gating this root with the v1.1 reader gives placement, outputs → **release_e2
and release_e5e8 true**; lifecycle still fails (old records hold only unfiltered PID lists) and leak_ok_v11 has no
references, so E3 stays unreleased until the re-pilot.

## 2. Pilot observations (one acquisition each, clock-confounded, not conclusions)

- E2 R(n)=T(n)/T(304), Qwen32B B1/C4096, two GPUs: 232 → 1.028 / 1.034; 200 → 1.015 / 1.023; 152 → 1.084 / 1.087;
  96 → 1.095 / 1.102. Qwen14B: 1.045, 1.024, 1.106, 1.109. Bridge: 1.046, 1.048, 1.151, 1.218.
  **Boundary for E5/E8 set to n=200** (smallest pilot level with point R ≤ 1.05 on both Qwen32B GPUs);
  this is a planning choice, not a noninferiority result.
- E3 Qwen32B (ratio to absent at the same n): n=232 mem50 1.122, mem100 1.269, arith100 1.062, inactive 0.998;
  n=152 mem50 1.127, mem100 1.306, arith100 1.062. Competitor achieved 515–1188 GB/s (memory) and 10.5–10.7 TFLOP/s
  (arith) during trials, full trial coverage.
- E4 service at reduced CUs (ratio to 304): seqread 4 GiB 0.95 (232), 0.96 (200), 0.88 (152), 0.87 (96); ILP-1 read
  0.84/0.88/0.72/0.61; FMA 0.81/0.74/0.64/0.46; MFMA 0.75/0.66/0.53/0.35; 8 MiB reuse 0.73/0.66/0.54/0.37.
- Mask revalidation: comma list `0-7,16-23` was accepted on this runtime (16 CUs, balanced) — opposite of the
  historical observation; a mask naming another GPU's index did not restrict this GPU (304 CUs); the empty-XCD
  mask `0-6` placed work on 45 CUs including 38 on XCD 7 (unsafe, as before).
- E8: energy 14.08 J/token at 752 W (one acquisition); C16384 B1 19.74 ms/step at 304.

## 3. v1.1 repair (commit 79d8b597)

- Residual-process check applies the existing `scripts/gpu_target.py` rule (VRAM ≥ 64 MiB or GFX time on the
  target GPU). Every owned process carries a per-attempt `CUBW_STEP_TAG`; escaped processes are found by tag,
  killed and recorded (`orphans`), and lifecycle requires them gone.
- outputs_ok counts only unprofiled runs (as declared).
- Leak reference: the same FMA probe beside placement-proven native co-runners (svc_bench fma and read) on the
  decoder mask, same GPU. `leak_ok_v11`: every decoder probe ratio ≥ 0.85 × the highest native co-runner ratio.
  Predeclared before any co-run data exists; the v1 leak gate is still computed and reported.
- Graph-mode counters recorded `unavailable`; E5 production keeps kernel-trace timelines only.
- MI210 smoke (devel, job 483497): co-runner start/stop and tag sweep work; native co-runners left the probe at
  0.90/0.91 of alone on that GPU.

## 4. Submissions after review

| job | plan | content |
|---|---|---|
| 483498 `cubw-mi300x-auto` (mi3008x, 2:45:00) | `$WORK/cubw_plans/mi300x_auto_20261028_091901_79d8b597.json` | minimal re-pilot (placement ×8, leak suites on 2 GPUs, lifecycle at n=232, Qwen32B E2 on 2 GPUs) → gates → production (Qwen32B ×5, Qwen14B ×3 blocks; E3 gated; E4 ×3; E5 ktrace; E8), boundary 200 |
| 483499 `cubw-mi325x-auto` (mi3258x, 3:15:00, `afterany:481072`) | `$WORK/cubw_plans/mi325x_auto_20261028_091903_79d8b597.json` | reduced pilot incl. leak suite → gates → production (Qwen32B ×3, Qwen14B ×3), boundary 200 (taken from the MI300X pilot) |

481168 (v1 release) was cancelled before it ran. 481072 completed (see the experiment log).
