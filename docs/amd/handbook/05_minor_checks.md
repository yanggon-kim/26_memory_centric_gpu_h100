# 05 — Minor checks: failed MI325X attempts, smoke tests, MI350X access and failed MI350X smokes

This chapter covers checks that until now had only a line or two in
[experiment_log.md](../experiment_log.md). None of them produced a headline result. They matter because they explain:

- why the MI325X raw data comes from one specific root;
- which tools were proven on a cheap MI210 node before an expensive 8-GPU job;
- how the MI350X port failed twice before it worked.

Most facts here were reconstructed from raw files in `$WORK` and from Slurm accounting (`sacct`), not from a
results document. Where a fact could not be recovered, the text says **not recorded**.

Acronyms are defined on first use here. The shared glossary is in [README.md](README.md).

Short definitions used throughout:

- **CU** = compute unit. MI300X and MI325X have 304 CUs, MI350X has 256, MI210 has 104. **XCD** = accelerator
  complex die (8 per MI300X/MI325X/MI350X).
- **Mask** = `HSA_CU_MASK`, the ROCm environment variable that restricts a process to a set of CUs. `0:52-103`
  means "GPU 0, CUs 52 to 103".
- **gfx90a / gfx942 / gfx950** = the GPU architecture names of MI210, MI300X/MI325X and MI350X.
- **Lane** = one sequence of runs bound to one GPU. A job runs several lanes in parallel, one per GPU.
- **Acquisition** = one complete repeat of a measurement cell (`a0`, `a1`, `a2`).
- **cubw** = the "compute vs bandwidth" follow-up code in `scripts/amd/cubw/` (`plan.py` builds a plan,
  `cubw.py` runs lanes and writes a journal, `analyze.py` computes tables and gates).
- **Probe** = a small microbenchmark (from `svc_bench`) placed on CUs that the process under test should not use.
  If its rate drops, something is interfering with it.
- **Co-runner** = a second, native (non-vLLM) `svc_bench` process placed on the CUs a decoder would use.
- **Leak test** = a check that the masked decoder does not run work on the competitor's or probe's CUs.
- **FETCH_SIZE** = a rocprofv3 hardware counter: kilobytes fetched from memory by the L2 cache.
- **NODE_FAIL** = the Slurm job state for "the node stopped responding". **TIMEOUT** = the job hit its time limit.
- **rc** = return code of a command (0 = success).

## Where the raw files are (read this first)

**Except for #16, the raw files of these checks are only in `$WORK`, not in git.** `$WORK` is
`/work1/proj_blaisetine/yanggon`. It is not backed up and not snapshotted. The repo's `results/` symlinks that
point to some of these roots are git-ignored (`.gitignore`: `results/*`).

| # | check | job(s) | partition / GPU | date (CDT/CST) | status | raw files (absolute paths) | in git? |
|---|---|---|---|---|---|---|---|
| 4 | MI325X full campaign, attempts 1–2 | 470176 (attempts 1 and 2) | `mi3258x` / 8× MI325X, node k002-001 | 2026-10-20 and 2026-10-23 | NODE_FAIL both times; partial phase 1; superseded by attempt 3 | `/work1/proj_blaisetine/yanggon/results_amd/amd_20261020_mi325x_full_job470176` (588 MB), `/work1/proj_blaisetine/yanggon/results_amd/amd_20261023_mi325x_full_job470176` (2.3 GB) | no |
| 10 | cubw v1.1 co-runner and orphan-sweep smoke | 483497 | `devel` / 1× MI210, node k006-004-v2 | 2026-10-28 | completed (28 s) | `/work1/proj_blaisetine/yanggon/cubw_smoke/v11/` | no |
| 16 | Leak-probe v2 (`cuprobe`) smoke | 492951 | `devel` / 1× MI210, node k006-007-v2 | 2026-11-01 | completed (50 s) | `/work1/proj_blaisetine/yanggon/smoke/cuprobe_492951/`, `/work1/proj_blaisetine/yanggon/smoke/cuprobe_smoke.sbatch`, `/work1/proj_blaisetine/yanggon/smoke/cuprobe_smoke.492951.out` | **yes, a full copy:** `results_published/amd_20261101_cubw_extension_prep/mi210_cuprobe_smoke_job492951/` |
| 19 | MI350X access toy (vector add) | 496003 | `mi3501x` / 1× MI350X, node k007-005-v7 | 2026-11-02 | completed (1 s) | `/work1/proj_blaisetine/yanggon/mi350x_toy/` (`toy.sbatch`, `vadd.hip`, `vadd_gfx950`, `mi350x_toy.496003.out`) | no |
| 20 | MI350X failed smoke 1 | 496056 | `mi3501x` / 1× MI350X, k007-005-v7 | 2026-11-02 | FAILED in preflight (12 s, exit 2) | `/work1/proj_blaisetine/yanggon/results_amd/amd_20261102_mi350x_cubw_v1_pilot_job496056`, `/work1/proj_blaisetine/yanggon/cubw_496056.out` | no |
| 20 | MI350X failed smoke 2 | 496059 | `mi3501x` / 1× MI350X, k007-005-v7 | 2026-11-02 | ran to the end; every vLLM step failed | `/work1/proj_blaisetine/yanggon/results_amd/amd_20261102_mi350x_cubw_v1_pilot_job496059`, `/work1/proj_blaisetine/yanggon/cubw_496059.out` | no |
| 20 | MI350X failed 8-GPU run | 496060 | `mi3508x` / 8× MI350X, node k007-003 | 2026-11-02 | ran to the end; every vLLM step failed; gate released nothing | `/work1/proj_blaisetine/yanggon/results_amd/amd_20261102_mi350x_cubw_v1_auto_job496060`, `/work1/proj_blaisetine/yanggon/cubw_496060.out` | no |

Related files that are in `$WORK` only and are mentioned below:

- `/work1/proj_blaisetine/yanggon/mi325x_full_470176.out` — Slurm output of job 470176. It holds **only attempt 3**
  (see #4).
- `/work1/proj_blaisetine/yanggon/cubw_smoke/smoke.sh`, `comp.jsonl`, `svc_bench_gfx90a_v3` — an earlier E0
  smoke of `svc_bench_v3` (devel job 481164, 2026-10-26; from `sacct`). It is not one of the checks in this chapter,
  but it shares the folder with #10.
- Plans and frozen code releases for #20: `/work1/proj_blaisetine/yanggon/cubw_plans/mi350x_*_20261102_23*.json`
  and `/work1/proj_blaisetine/yanggon/cubw_release/<commit>/`.

Timing sources: start, end, node and state come from `sacct` (queried 2026-11-05). Slurm prints times in
the cluster's local time (CDT until 2026-11-01, CST after).

---

### #4 MI325X full campaign, attempts 1–2 — job 470176

- **At a glance:**

  | | attempt 1 | attempt 2 |
  |---|---|---|
  | partition / GPU / node | `mi3258x` / 8× MI325X / k002-001 | same |
  | submitted | 2026-10-19 11:40:49 | requeued 2026-10-20 15:26:53 (eligible 15:28:54) |
  | ran | 2026-10-20 14:18:49 – 15:26:53 | 2026-10-23 10:16:28 – 13:42:31 |
  | run time | 1 h 08 min | 3 h 26 min |
  | Slurm state | NODE_FAIL | NODE_FAIL |
  | driver version | `scripts/amd/mi325x_full.sbatch` at commit e79a7aae (committed 2026-10-19 11:40:47, 2 s before submission) | same batch script (same job) |
  | status | partial phase 1; superseded | partial phase 1 (nearly complete); superseded |

  Attempt 3 of the same job (2026-10-25 17:57 – 2026-10-26 05:57, TIMEOUT) is the one that counts. It is covered
  in the MI325X campaign chapter and in
  [mi325x_full_campaign_results_20261028.md](../mi325x_full_campaign_results_20261028.md).

- **Question:** the same as the whole MI325X campaign. Repeat the first MI300X campaign (job 468885: baselines,
  counters, CU masks, energy, tensor parallelism) on an 8×MI325X node, unattended, as a separate cohort. The
  driver has five phases: phase 0 preflight; phase 1 eight single-GPU lanes; phase 2 TP4 (tensor parallelism over
  4 GPUs) + TP2 and same-GPU competitor lanes; phase 3 TP8; phase 4 TP at about 160 total CUs (header of
  `scripts/amd/mi325x_full.sbatch`).

- **Design:** identical to attempt 3 (same script, configs and lanes). Phase 1 runs eight lanes in parallel, one
  per GPU. Most lanes run twice (`_p1`, `_p2`); the second pass fills runs that failed sporadically. Lane roles,
  from `RUN_STATUS.log`:

  | GPU | lane(s) | content |
  |---|---|---|
  | 0 | `g0_m1_p1/p2` | M1 baselines |
  | 1 | `g1_counters_p1/p2` | M2 rocprofv3 counters (bridge, qwen7b eager/graph) |
  | 2 | `g2_service`, then `g2_batch_p1/p2` | M4 service sweep, then qwen7b batch × CU sweep |
  | 3 | `g3_cumask_p1/p2`, then `g3_prefill_p1/p2` | M6 CU masks (bridge, qwen7b), then prefill vs CU |
  | 4 | `g4_energy_p1/p2`, then `g4_energy_cu` | M8 energy, then energy vs CU |
  | 5 | `g5_counters_p1/p2` | M2 counters (mistral7b, qwen14b, qwen32b, …) |
  | 6 | `g6_rocprof_compute`, then `g6_repro_p1/p2` | M3 rocprof-compute, then M1 repeat on GPU 6 |
  | 7 | `g7_pressure`, then `g7_cumask_p1/p2`, then `g7_text` | M5 pressure, M6 CU masks (larger models), text check |

  There were no pre-registered gates in this driver. The only automatic check was phase 0 preflight
  (`scripts/amd/preflight_node.py`): 8 GPUs, container, models, a FETCH_SIZE check on a 64 MiB copy, and an FMA
  rate at full and 152 CUs.

- **How it was run:** `sbatch scripts/amd/mi325x_full.sbatch` (12 h limit). The exact submit line is not in
  `sacct` for this job (**not recorded**). A dry run was made just before submission
  (`/work1/proj_blaisetine/yanggon/results_amd/dryrun_mi325x_full_20261019_113920`). The driver names its root
  `amd_$(date +%Y%m%d)_mi325x_full_job$SLURM_JOB_ID`. Slurm requeued the job after each NODE_FAIL under the same
  job ID (`sacct --duplicates` shows three records; the second and third have Submit = previous End and Reason
  `BeginTime`). Each restart therefore created a **new, dated root**: `amd_20261020_…`, `amd_20261023_…`,
  `amd_20261025_…`. Nothing was resumed; each attempt started phase 1 from scratch. Whether the requeue was
  automatic or requested by the user is not recorded; the `sacct` pattern is consistent with automatic requeue.

  The driver `cd`s into the live repo and runs lane scripts from there (`scripts/amd/lane_*.sh`). The lane
  scripts were therefore not frozen; the exact lane-script version used by each attempt is not recorded beyond
  the date.

- **What was measured and how it was analysed:** the same per-run `run.json` files and lane logs as attempt 3.
  These two roots were **never analysed**: no `analysis/` folder, no summary, no `SHA256SUMS`, and they are not
  indexed in [artifacts.md](../artifacts.md).

- **Results:** what each attempt reached before the node failed.

  Preflight (from `RUN_STATUS.log`, both passed):

  | | attempt 1 | attempt 2 | attempt 3 (for comparison) |
  |---|---:|---:|---:|
  | GPUs | 8 | 8 | 8 |
  | FETCH_SIZE for 64 MiB copy | 65548 KB | 65548 KB | 65548 KB |
  | FMA TFLOP/s, full device | 102.956 | 102.353 | 102.932 |
  | FMA TFLOP/s, 152 CUs | 64.86 | 64.326 | 64.532 |

  Lanes that finished (`end rc=0` in `RUN_STATUS.log`):

  | attempt | lanes finished | lanes still running when the node failed | phase-1 lanes never started |
  |---|---|---|---|
  | 1 | `g7_pressure` (14:28), `g2_service` (15:20) | g0 M1 p1, g1 counters p1, g2 batch p1 (started 15:20), g3 cumask p1, g4 energy p1, g5 counters p1, g6 rocprof-compute, g7 cumask p1 | all `_p2` passes, prefill, energy vs CU, GPU-6 repro, text |
  | 2 | 19 lane passes: pressure, service, M1 p1+p2, cumask g3 p1+p2, cumask g7 p1+p2, energy p1+p2, rocprof-compute, GPU-6 repro p1+p2, text, counters g1 p1+p2, counters g5 p1, prefill p1+p2 | `g2_batch_p1` (on its last cell, qwen7b B64 at 96 CUs, acquisition a2), `g4_energy_cu` (on qwen32b at 152 CUs, a2), `g5_counters_p2` (started 13:10) | `g2_batch_p2` |

  Raw run counts (number of `run.json` files, and how many contain `"status": "complete"`; counted on
  2026-11-05 with `find`/`grep`; counter folders hold one `run.json` per attempt, so their counts are not cells):

  | family | attempt 1 | attempt 2 | attempt 3 |
  |---|---|---|---|
  | `m1` baselines | 45 / 43 | 50 / 48 | 51 / 48 |
  | `m1_gpu6` repro | — | 24 / 24 | 24 / 24 |
  | `m1_text` | — | 6 / 6 | 6 / 6 |
  | `m6_cumask` (incl. batch sweep) | 77 / 75 | 172 / 166 | 174 / 168 |
  | `m8_energy` | 16 / 14 | 26 / 24 | 19 / 16 |
  | `m8_energy_cu` | — | 16 / 16 | 10 / 9 |
  | `m8_prefill` | — | 48 / 48 | 49 / 48 |
  | `m2_counters` (attempt-level) | 56 / 10 | 149 / 42 | 161 / 48 |
  | root size | 588 MB | 2.3 GB | 3.0 GB |

  Attempt 2 got almost as far as attempt 3 did at the end of phase 1 (attempt 3 finished all phase-1 lanes in
  3 h 22 min; attempt 2 failed at 3 h 26 min with three lanes still running).

- **What it means:** nothing scientific. These attempts are infrastructure failures. The campaign's phase-1 data
  come only from attempt 3 (`amd_20261025_…`), and phases 2–4 from the resume job 481072. The two partial roots
  were kept so that nothing is deleted, but they are not pooled with attempt 3 and are not cited by any results
  document ([mi325x_full_campaign_results_20261028.md](../mi325x_full_campaign_results_20261028.md) §1;
  [cubw_e0_reconciliation_20261026.md](../cubw_e0_reconciliation_20261026.md): "Superseded by attempt 3 for
  phase 1; kept as failures").

- **Failures and caveats:**
  - **Cause of NODE_FAIL: not recorded.** No document gives a cause, and no node-side log is available to users.
    The raw files show only that all writing stopped a few minutes before Slurm declared the failure:
    - attempt 1: last node-monitor sample 15:22:06 (`node_monitor.jsonl` `t_unix_ns` 1792527726…); last vLLM log
      line 15:22:02; Slurm end 15:26:53.
    - attempt 2: last node-monitor sample 13:38:15; last vLLM log line 13:38:05; Slurm end 13:42:31.
    - In attempt 2's last monitor sample, six of eight GPUs were idle (132 MHz) and GPUs 4 and 5 were busy, which
      matches the lanes still running.
  - The Slurm output file `/work1/proj_blaisetine/yanggon/mi325x_full_470176.out` was rewritten by attempt 3 and
    holds only attempt 3's log (it ends with `CANCELLED AT 2026-10-26T05:57:51 DUE TO TIME LIMIT`). The driver log
    of attempts 1 and 2 survives only as each root's `RUN_STATUS.log`.
  - `Killed` / `Aborted` lines in the lane logs (for example `lane_g5_counters_p2.log`, `lane_g3_cumask_p1.log`)
    are the lanes' own per-run timeouts and watchdogs, the same as in attempt 3. They are not the node failure.
  - Even if attempts 1 or 2 had survived phase 1, the batch script they ran (commit e79a7aae) ended phase 1 with a
    bare `wait`, which also waits for the background node monitor that never exits. Attempt 3 hit exactly this
    and idled about 8.6 h until TIMEOUT. The fix (`wait_lanes`, plus `PHASES` selection) is commit d0131769
    (2026-10-26). This is an inference from the script; attempts 1–2 never reached that point.

- **Files:**
  - Raw: `/work1/proj_blaisetine/yanggon/results_amd/amd_20261020_mi325x_full_job470176`,
    `/work1/proj_blaisetine/yanggon/results_amd/amd_20261023_mi325x_full_job470176` (each has `RUN_STATUS.log`,
    `preflight.json`, `gpus.tsv`, `lane_*.log`, `m*/`, `node_monitor.jsonl`, `monitor.log`).
  - Driver: `scripts/amd/mi325x_full.sbatch` (e79a7aae as run; fixed in d0131769).
  - Docs: [experiment_log.md](../experiment_log.md) (2026-10-26 entry);
    [mi325x_full_campaign_results_20261028.md](../mi325x_full_campaign_results_20261028.md) §1;
    [cubw_e0_reconciliation_20261026.md](../cubw_e0_reconciliation_20261026.md) (table of prior jobs).

---

### #10 MI210 devel smoke of the cubw v1.1 co-runner and orphan sweep — job 483497

- **At a glance:** partition `devel`, 1× MI210 (gfx90a, 104 CUs per `plan.json` `full_cus`), node k006-004-v2,
  2026-10-28 09:18:18–09:18:46 (28 s), `COMPLETED`. Code: the live repo working tree at commit 79d8b597
  (cubw v1.1, committed 09:17:56, 22 s before the job started). Binary:
  `/work1/proj_blaisetine/yanggon/cubw_smoke/svc_bench_gfx90a_v3` (built 2026-10-26). Status: complete.

- **Question:** the MI300X pilot 481167 failed two of its gates on tool defects
  ([cubw_pilot_review_20261028.md](../cubw_pilot_review_20261028.md) §1). Version v1.1 added three things that
  had never run on a GPU:
  1. A **native co-runner**: `cubw.py` can start a second `svc_bench` process on the decoder's CUs during a
     step, and stop it cleanly afterwards.
  2. A **tag-based orphan sweep**: every process a step starts carries `CUBW_STEP_TAG`. After the step,
     `cubw.sweep_tag()` scans `/proc/*/environ` for the tag and kills any process that escaped its process group
     (the pilot had left rocprofv3 processes with PPID 1).
  3. A **leak reference** (`leak_ok_v11`): the probe's slowdown beside the decoder is compared with its slowdown
     beside native co-runners on the same CUs, instead of with the probe alone.

  The question was: do the co-runner start/stop, the per-step residual-process check and the sweep work, before
  they are used in the 8-GPU jobs 483498 and 483499? An MI210 `devel` node is the cheapest GPU (0.1×).

- **Design:** one GPU, one lane (`L0`), three steps of kind `competitor_only`, one acquisition each, 8 s each, no
  retries (`/work1/proj_blaisetine/yanggon/cubw_smoke/v11/plan.json`):

  | step | probe (FMA, `svc_bench --mode fma`) | co-runner |
  |---|---|---|
  | `t:alone:a0` | CUs 52–103 | none |
  | `t:co_fma:a0` | CUs 52–103 | FMA (`arith100`) on CUs 0–51 |
  | `t:co_read:a0` | CUs 52–103 | streaming read of 1 GiB (`mem100`) on CUs 0–51 |

  In addition, before the lane, `run.sh` started a decoy process `CUBW_STEP_TAG=probe-tag setsid sleep 300` that
  escapes its group. After the lane it called `cubw.sweep_tag('probe-tag')` and printed the result. No gates or
  predictions were pre-registered for this smoke.

- **How it was run:** `srun -p devel -N1 -t 00:10:00 /work1/proj_blaisetine/yanggon/cubw_smoke/v11/run.sh`
  (submit line from `sacct`). `run.sh` sources `scripts/amd/env.sh`, writes `gpus.tsv`, runs
  `cubw.py lane --plan …/plan.json --root …/root --lane L0` with `CUBW_BENCH` pointing at the v3 binary, then runs
  the sweep and prints a short summary of each journal record.

- **What was measured and how it was analysed:** the journal `root/journal/L0.jsonl` records, per step, the
  probe's and co-runner's start/stop (`stopped_by`, `returncode`, `gone`), `orphans` and `residual_gpu_pids`.
  Each `svc_bench` process writes 0.25 s interval records (`competitor.jsonl`, `corunner.jsonl`). The probe ratio
  is mean probe TFLOP/s beside the co-runner divided by mean probe TFLOP/s alone (recomputed here from the
  interval records, excluding the final partial interval).

- **Results:**

  | step | probe TFLOP/s (mean) | ratio to alone | co-runner rate (median) | co-runner stop | orphans | residual GPU PIDs |
  |---|---:|---:|---:|---|---|---|
  | alone | 16.74 | 1 | — | — | none | none |
  | beside FMA co-runner | — | **0.90** | 14.98 TFLOP/s | SIGTERM, rc 0, gone | none | none |
  | beside read co-runner | — | **0.91** | 1035 GB/s | SIGTERM, rc 0, gone | none | none |

  The ratios match the review ("native co-runners left the probe at 0.90/0.91 of alone",
  [cubw_pilot_review_20261028.md](../cubw_pilot_review_20261028.md) §3).

- **What it means:**
  - The co-runner mechanism works: both processes start, are placed on their masks, and stop on SIGTERM with
    rc 0, with no residual processes on the GPU.
  - A native co-runner on **disjoint** CUs still costs the probe about 10% on MI210. This is chip-wide contention
    (shared memory, power, clocks), not leakage, because the masks do not overlap. That is why v1.1 compares the
    decoder's effect with native co-runners instead of with "alone".
  - Later, `leak_ok_v11` failed on MI300X (decoder probe ratio 0.77–0.82 relative, below 0.85; job 483498) and
    passed on MI325X (0.87–0.97; job 483499) ([experiment_log.md](../experiment_log.md)). Because the aggregate
    probe could not separate leakage from contention on MI300X, it was replaced by the per-CU probe v2 (#16).

- **Failures and caveats:**
  - **The sweep's printed result was not saved.** `srun` sent stdout to the terminal, and no copy is in `$WORK`.
    The journal shows `orphans: []` for every step, which is expected because the decoy had a different tag.
    So the only record that the sweep found and killed the decoy is the review's sentence "tag sweep work[s]".
    It cannot be checked from the raw files.
  - The run used the repo working tree, not a frozen release folder. That it matched commit 79d8b597 exactly is
    likely from the timestamps but not recorded.
  - MI210 is a different architecture (gfx90a, 104 CUs, one die) from the cohorts. The numbers are tool checks,
    not cohort data.

- **Files:** raw `/work1/proj_blaisetine/yanggon/cubw_smoke/v11/` (`run.sh`, `plan.json`, `root/journal/L0.jsonl`,
  `root/t/{alone,co_fma,co_read}/a0/attempt1/*.jsonl`); code `scripts/amd/cubw/cubw.py` (`sweep_tag`,
  `gpu_processes`), commit 79d8b597; review [cubw_pilot_review_20261028.md](../cubw_pilot_review_20261028.md) §3;
  [experiment_log.md](../experiment_log.md) (2026-10-28 entry).

---

### #16 MI210 smoke of leak-probe v2 (`cuprobe`) — job 492951

- **At a glance:** partition `devel`, 1× MI210 (gfx90a), node k006-007-v2, 2026-11-01 15:54:48–15:55:38 (50 s),
  `COMPLETED`, all four runs rc 0. Binary `/work1/proj_blaisetine/yanggon/bin/svc_bench_gfx90a_v4` (built
  2026-11-01 15:53). Its source `scripts/amd/cubw/svc_bench_v4.hip` was committed in de2baecb (cubw v1.3,
  15:58:28), four minutes **after** the smoke. Status: complete.

- **Question:** the v1.1 leak test (#10) failed on MI300X, because the decoder slowed an aggregate FMA probe about
  as much as a native streaming co-runner did. An aggregate probe cannot tell CU leakage from chip-wide
  contention. Before MI300X E3 could run in the extension job 492956, a new test was needed that is (a) sensitive
  to a foreign workgroup on a probe CU and (b) not triggered by contention. This smoke checked both properties on
  a cheap GPU before the rule was pre-registered
  ([cubw_extension_plan_20261101.md](../cubw_extension_plan_20261101.md) §2).

- **Design:** the probe is `svc_bench_v4 --mode cuprobe`. Each FMA workgroup reserves all 64 KiB of LDS (local
  data share, the per-CU scratchpad), so only one probe workgroup fits on a CU, the same footprint as vLLM's
  `wvSplitK`. Each workgroup reads its physical CU ID (HW_ID/XCC_ID) and the probe counts completions per CU. A
  foreign workgroup on a probe CU lowers that CU's count; contention lowers all CUs together.

  Four runs on GPU 0, probe always on CUs 52–103 (52 CUs), 12 s each; co-runner started 1 s after the probe,
  10 s long (`cuprobe_smoke.sbatch`):

  | run | co-runner | role |
  |---|---|---|
  | `alone` | none | reference |
  | `read_disjoint` | streaming read, 1 GiB, CUs 0–51 | negative control (contention, no shared CUs) |
  | `fma_disjoint` | FMA, CUs 0–51 | negative control |
  | `cuprobe_overlap8` | another `cuprobe`, CUs 0–59 | positive control: shares 8 CUs (52–59) with the probe |

- **How it was run:** `sbatch /work1/proj_blaisetine/yanggon/smoke/cuprobe_smoke.sbatch` (`-p devel -N 1 -n 4
  -t 00:15:00`, `HIP_VISIBLE_DEVICES=0`). No cubw lane or journal; a plain shell function `run()` starts probe
  and co-runner with `HSA_CU_MASK=0:<range>`.

- **What was measured and how it was analysed:** per-CU completion counts (`cu_blocks`) in each 0.25 s window.
  The rule later pre-registered in the plan (`analyze.leak_v2_eval`):
  - r_c = per-CU rate beside X / per-CU rate alone (same GPU, same probe mask);
  - q_c = r_c / median over CUs of r_c (removes chip-wide slowdown);
  - a CU is flagged if q_c < 0.85; the positive control must flag ≥ 4 CUs or the test is "insensitive";
  - a decoder run passes if its min q_c ≥ (lowest min q_c of the native co-runner runs) − 0.05.

- **Results** (from [cubw_extension_plan_20261101.md](../cubw_extension_plan_20261101.md) §2; reproduced from the
  raw JSONL for this chapter using only windows while the co-runner ran, with the same outcome):

  | run | CUs with q < 0.85 | min q | lowest q values |
  |---|---:|---:|---|
  | native read co-runner on 0–51 | 0 | 0.95 | – |
  | native FMA co-runner on 0–51 | 0 | 0.95 | – |
  | positive control on 0–59 (shares 8 CUs) | **8** | 0.46 | 0.46 ×4, 0.64–0.66 ×4 |

- **What it means:** on MI210 the per-CU probe is sensitive (it flags exactly the 8 shared CUs) and specific
  (contention from disjoint co-runners does not flag anything). On this basis the leak v2 rule was pre-registered
  and used in jobs 492956 (MI300X, passed, so E3-minimal ran) and 492957 (MI325X, passed at 232 and 152).
  **Revision:** on MI350X (gfx950) v2 turned out to be **insensitive**. Its positive control flagged 0 of 8 CUs in
  smoke 496073 ([mi350x_smoke_review_20261102.md](../mi350x_smoke_review_20261102.md)), because completion counts
  per CU are fixed per launch there. That led to v3 (residency, also insensitive on gfx950) and v4 (start-delay
  probe, sensitive, passes 16/16 on MI350X).

- **Failures and caveats:** none in the run. The smoke ran before the source commit, so the binary's exact source
  is "the v4 source as committed in de2baecb four minutes later" (likely identical, not proven). Single
  acquisition on a non-cohort GPU.

- **Files:** raw `/work1/proj_blaisetine/yanggon/smoke/cuprobe_492951/*.jsonl|*.err`,
  `/work1/proj_blaisetine/yanggon/smoke/cuprobe_smoke.sbatch`, `/work1/proj_blaisetine/yanggon/smoke/cuprobe_smoke.492951.out`;
  **git copy** `results_published/amd_20261101_cubw_extension_prep/mi210_cuprobe_smoke_job492951/` (same 13 files);
  plan [cubw_extension_plan_20261101.md](../cubw_extension_plan_20261101.md) §2; code
  `scripts/amd/cubw/svc_bench_v4.hip`, `scripts/amd/cubw/analyze.py` (`leak_v2_eval`), commit de2baecb.

---

### #19 MI350X access toy (vector add) — job 496003

- **At a glance:** partition `mi3501x`, 1× MI350X slice, node k007-005-v7. Submitted 2026-11-02 22:50:04, ran
  22:50:18–22:50:19 (1 s), `COMPLETED`, rc 0. No repo commit; a standalone HIP program. Status: complete.

- **Question:** access to `mi3501x`/`mi3508x` was granted on 2026-11-02. Before porting cubw, does the hardware
  match expectations, and does a gfx950 binary built on the login node run on it?
  ([mi350x_campaign_plan_20261102.md](../mi350x_campaign_plan_20261102.md), "Access").

- **Design:** `vadd.hip` queries device properties, then runs a vector add `c = a + b` on n = 2^24 floats
  (64 MiB per array), 256 threads per block. One warm-up launch, then 20 timed launches between two HIP events.
  It checks every element (`c[i] == 2i`) and reports bandwidth as 3 arrays × 64 MiB × 20 / elapsed time.

- **How it was run:** `sbatch toy.sbatch` from `/work1/proj_blaisetine/yanggon/mi350x_toy/`
  (`-p mi3501x -N 1 -n 1 -t 00:10:00`). The script prints hostname and date, loads `rocm/7.1.0`, runs
  `rocm-smi --showproductname --showmeminfo vram` and `rocminfo | grep gfx9`, then `./vadd_gfx950`. The binary was
  built on the login node at 22:49. **The exact compile command is not recorded** (an offload arch of gfx950 is
  implied by the file name and by the fact that it ran).

- **What was measured:** device name, architecture, CU count, memory; vector-add errors and GB/s.

- **Results** (`mi350x_toy.496003.out`):

  | item | value |
  |---|---|
  | device (HIP) | AMD Instinct MI350X |
  | arch | gfx950:sramecc+:xnack- |
  | CUs | 256 |
  | memory | 288.0 GiB (rocm-smi VRAM total 309,220,868,096 B) |
  | vector add | n = 16,777,216, **errors = 0** |
  | bandwidth | **6453.1 GB/s** (≈ 6.45 TB/s) |

- **What it means:** the hardware is as specified (256 CUs = 8 XCDs × 32, 288 GiB), and the login-node toolchain
  produces working gfx950 code. 6.45 TB/s is about 81% of the 8.0 TB/s vendor peak quoted in the campaign plan.
  The later calibrated 4 GiB streaming read (E4 `seqread4g`) at 256 CUs gave 6427–6430 GB/s in smokes 496059 and
  496073 (`analysis/summary.md` of each), consistent with the toy.

- **Failures and caveats:**
  - This is not a bandwidth benchmark: 20 launches of about 31 µs each, one run, no clock record. Use E4 numbers
    for any claim.
  - `rocm-smi` printed "AMD GPU device(s) is/are in a low-power state" and reported "VRAM Total Used Memory
    65,048,813,568 B" at job start, before the toy allocated anything. Neither was investigated (**cause not
    recorded**). `rocm-smi` also named the card "AMD Radeon Graphics"; HIP named it MI350X.

- **Files:** raw `/work1/proj_blaisetine/yanggon/mi350x_toy/` (`toy.sbatch`, `vadd.hip`, `vadd_gfx950`,
  `mi350x_toy.496003.out`); docs [mi350x_campaign_plan_20261102.md](../mi350x_campaign_plan_20261102.md),
  [experiment_log.md](../experiment_log.md) (2026-11-02 entry).

---

### #20 MI350X failed smokes — jobs 496056, 496059, 496060

These three jobs were the first runs of cubw v1.4 (the MI350X port). Each was stopped by a port bug. Each bug was
fixed in a commit within minutes, and a new frozen release was submitted. The working pair was 496073 (one-GPU
smoke, 29/29 steps rc 0) and 496074 (8-GPU campaign). All three failed jobs are **kept as failed acquisitions**:
their roots are not deleted, overwritten or reused, and no published table uses them
([mi350x_smoke_review_20261102.md](../mi350x_smoke_review_20261102.md)).

Every job was submitted as a pair: a one-GPU pilot on `mi3501x` and an 8-GPU auto job on `mi3508x`, both from the
same release.

| release (frozen in `$WORK/cubw_release/`) | commit time | one-GPU job (`mi3501x`, 1:15:00) | 8-GPU job (`mi3508x`, 3:45:00) |
|---|---|---|---|
| 8ba209a8 (cubw v1.4: MI350X port) | 2026-11-02 23:22:56 | **496056** — FAILED in preflight | 496057 — cancelled before it started (23:23:47, `sacct`) |
| 36198d84 (preflight fix) | 23:23:45 | **496059** — vLLM steps failed | **496060** — vLLM steps failed |
| e7116c1f (gpu_target fix) | 23:32:46 | 496073 — 29/29 rc 0 | 496074 — the campaign |

Submit lines (from `sacct`) all have this form:

```
sbatch -p mi3501x -t 01:15:00 -J cubw-mi350x-pilot \
  --export=ALL,CUBW_PLAN=$WORK/cubw_plans/mi350x_pilot_<stamp>_<commit8>.json,CUBW_RELEASE=$WORK/cubw_release/<commit> \
  $WORK/cubw_release/<commit>/scripts/amd/cubw/cubw.sbatch
```

The 8-GPU form uses `-p mi3508x -t 03:45:00 -J cubw-mi350x-auto` and an `mi350x_auto_…` plan. Plans used:
`mi350x_pilot_20261102_232303_8ba209a8.json` (496056), `mi350x_pilot_20261102_232349_36198d84.json` (496059),
`mi350x_auto_20261102_232355_36198d84.json` (496060), all in `/work1/proj_blaisetine/yanggon/cubw_plans/`.

#### #20a MI350X smoke 1 — job 496056

- **At a glance:** `mi3501x`, 1× MI350X, k007-005-v7, 2026-11-02 23:23:06–23:23:18 (12 s), `FAILED` exit 2,
  release 8ba209a8. Status: failed in preflight; no step ran.
- **Question:** the first test of the v1.4 port on one MI350X GPU before the 8-GPU campaign: placement, decode,
  `wvSplitK` geometry on gfx950, leak test v2, E4 and counters (37 planned steps, of which 29 have a return code;
  the other 8 are `competitor_only` probe runs).
- **Design and how it was run:** the planned content is the same as 496073 (same 37-step plan shape:
  e0_capability 3, e1_placement 4, e1_leak_v2 10, e1_e2_pilot 3, e4_pilot 10, e5_counter_probe 1, e6_decode 1,
  e6_timeline 1, e6_shapes 2, e6_counters 2; from `launch_plan.json`). `cubw.sbatch` runs
  `preflight_node.py --expect-arch gfx950 …` first and exits 2 if preflight reports anything fatal.
- **Results** (`preflight.json`, `cubw_496056.out`):

  | check | value |
  |---|---|
  | GPU | AMD Instinct MI350X, gfx950, 256 CUs |
  | container | vLLM 0.23.1.dev1+g9ddef7117 OK |
  | models | none missing |
  | FMA full / 128 CUs | 93.199 / 64.763 TFLOP/s |
  | **FETCH_SIZE for a 64 MiB copy** | **32780 KB** (expected 65536 KB; accepted window 0.9–1.2×) |
  | fatal | `counter check failed (FETCH_SIZE 32780.0 KB for a 64 MiB copy)` |

- **What it means:** the preflight counter check was written for gfx942, where FETCH_SIZE was validated on a
  known copy. On gfx950 it reads about 0.50× of the bytes copied. This is not a node fault; it means FETCH_SIZE is
  not validated on gfx950.
- **Fix:** commit **36198d84** (`scripts/amd/preflight_node.py`): the FETCH_SIZE check is fatal only when
  `--expect-arch` is gfx942. On another architecture the result is recorded as `unvalidated_on_arch` and the job
  continues. Every MI350X counter claim must instead rest on that cohort's own E0 native counter validation.
- **Failures and caveats:** the paired 8-GPU job 496057 (same release) was cancelled before it started, so no
  8-GPU time was spent on this bug.

#### #20b MI350X smoke 2 — job 496059

- **At a glance:** `mi3501x`, 1× MI350X, k007-005-v7, 2026-11-02 23:23:56–23:30:30 (6 min 34 s), Slurm
  `COMPLETED` (the driver finished normally), release 36198d84. Status: **failed acquisition** — every vLLM step
  failed.
- **Question / design:** as 496056 (same 37-step plan shape, new plan file).
- **How it was run:** preflight passed (FETCH_SIZE 32780 KB now recorded, not fatal; FMA 92.849 / 63.723 TFLOP/s).
  Stage `pilot` ran 23:24:08–23:30:30, then `analyze.py` (rc 0), then hashing (240 files).
- **Results** (journal `journal/*.jsonl`, `analysis/summary.md`):

  | step group | steps | outcome |
  |---|---:|---|
  | e0_capability, e5_counter_probe, e6_shapes, e6_counters (shell) | 8 | rc 0 |
  | e1_placement (native) | 4 | rc 0; 256/192/128/64 CUs exactly, 32/24/16/8 per XCD |
  | e4_pilot (native) | 10 | rc 0 |
  | e1_leak_v2 probe runs (`competitor_only`) | 8 | completed |
  | **vLLM steps** (e1_e2_pilot ×3, e1_leak_v2 decoder ×2, e6_decode ×1, e6_timeline ×1) | 7 | **rc 1**, 2 attempts each (e6_timeline 1 attempt) |

  Every vLLM failure has the same traceback:
  ```
  File ".../scripts/gpu_target.py", line 74, in verify_cuda
    assert arch=='gfx942' and 'MI3' in props.name, 'AMD port requires a verified MI300X/MI325X gfx942 allocation'
  AssertionError: AMD port requires a verified MI300X/MI325X gfx942 allocation
  ```
  Gates: placement_ok true; every other gate false (outputs: 5 runs, 0 complete).
- **What it means:** the target-GPU guard in `scripts/gpu_target.py`, shared with the older H100 package, only
  allowed gfx942. The vLLM runner refused to run on gfx950 before loading a model. All native steps
  (placement, E4, shapes, counters) worked, so the port's native side was sound.
- **Fix:** commit **e7116c1f** (`scripts/gpu_target.py`): the guard now also accepts
  `arch=='gfx950' and 'MI35' in props.name`.
- **Failures and caveats:** the E4 rows from this job (for example FMA 128/256 = 0.634, streaming read 4 GiB
  0.888) are close to those of 496073 (0.631, 0.891), but they are not used anywhere. The review's "29 steps"
  counts steps with a return code; the plan has 37 steps including the 8 `competitor_only` runs.

#### #20c MI350X failed 8-GPU run — job 496060

- **At a glance:** `mi3508x`, 8× MI350X, node k007-003, 2026-11-02 23:26:01–23:32:38 (6 min 37 s), Slurm
  `COMPLETED` (finished by itself), release 36198d84. Status: **failed acquisition** — pilot vLLM steps failed;
  the gate released nothing.
- **Question:** the full MI350X campaign (727 planned steps: E1 pilot, then gated E2, E3, E6, E8, and ungated
  E4/shapes/counters), submitted together with 496059 so it could start as soon as a node was free.
- **How it was run:** preflight passed on 8 GPUs (FMA 93.326 / 63.142 TFLOP/s, FETCH_SIZE 32780 KB recorded).
  Pilot stage 23:26:12–23:30:50 on lanes L0–L7 (one per GPU); gates computed at 23:30:50; production stage
  23:30:50–23:32:37; analysis rc 0; 1339 files hashed (`RUN_STATUS.log`).
- **Results:**

  | stage | steps | outcome |
  |---|---:|---|
  | pilot: e1_placement | 82 | rc 0 on all 8 GPUs |
  | pilot: e0_capability, e5_counter_probe | 4 | rc 0 |
  | pilot: probe runs (`competitor_only`: e1_leak, e1_leak_v2, e1_e3_pilot) | 25 | completed |
  | pilot: vLLM steps (e1_e2_pilot 10, e1_e3_pilot 4, e1_leak 2, e1_leak_v2 4, e1_timeline 4) | 24 | all rc 1 (same `gpu_target.py` assertion) |
  | gates | — | `placement_ok` true; lifecycle, outputs (22 runs, 0 complete), leak, clock false; **every release false** |
  | production: e4 (ungated) | 339 | rc 0 |
  | production: e6_shapes, e6_counters (ungated) | 24 + 16 | rc 0 |
  | production: e2, e3, e6_decode, e6_timeline, e6_eager_counters, e8 | 213 | `not_run` ("gate release_e2 not passed", etc.) |

- **What it means:** the same bug as 496059, at 8-GPU scale. The gate logic did its job: with no complete decode
  run, nothing that depends on decode was released, and the job ended after 6.6 min instead of using its 3:45:00
  limit. Placement on all 8 MI350X GPUs was proven here for the first time. The review's "pilot stage only"
  should be read as "only the pilot and ungated production steps ran": the journal shows 379 ungated production
  steps with rc 0.
- **Fix:** the same commit e7116c1f. The campaign was then rerun from scratch as 496074 (release e7116c1f), held
  until smoke 496073 was reviewed, then released.
- **Failures and caveats:** this root holds real E4, shape and counter data from 8 MI350X GPUs. It is **not** used
  and not published; the campaign's numbers come only from 496074. Its counters carry the same gfx950 FETCH_SIZE
  caveat as above.

#### Common to #20

- **Files:** raw roots and `.out` files as in the table at the top; frozen releases
  `/work1/proj_blaisetine/yanggon/cubw_release/{8ba209a8…,36198d84…,e7116c1f…}`; plans in
  `/work1/proj_blaisetine/yanggon/cubw_plans/`; review
  [mi350x_smoke_review_20261102.md](../mi350x_smoke_review_20261102.md); plan
  [mi350x_campaign_plan_20261102.md](../mi350x_campaign_plan_20261102.md); commits 8ba209a8, 36198d84, e7116c1f;
  [experiment_log.md](../experiment_log.md) (2026-11-02 entries). The successful smoke's compact files are in
  `results_published/amd_20261102_mi350x_cubw_v1_4_smoke_job496073/`.
- The `.out` files of 496059 and 496060 end with an "Omnistat Report Card" that the cluster appends to every job
  (GPU utilisation, power, energy). It is cluster monitoring, not project data.
