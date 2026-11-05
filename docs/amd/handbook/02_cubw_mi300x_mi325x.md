# 02 — The cubw follow-up on MI300X and MI325X (E0–E8, jobs 481167 → 492956)

This chapter covers the "compute vs bandwidth" follow-up ("cubw") on the two gfx942 cohorts, MI300X and MI325X.
It runs from the E0 reconciliation on 2026-10-26 to the v1.3 extension jobs on 2026-11-02/03. The MI350X port
(v1.4–v1.6) and the MI300X shader-engine study (499333/499334) are in later chapters. They are mentioned here
only where they revise a result in this chapter.

Every number below comes from a file in the repository or in `$WORK`. The source is named next to key facts.
Paths are relative to the repository root (`/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100`, branch
`amd/mi300x-phase0`) unless they start with `$WORK` (`/work1/proj_blaisetine/yanggon`).

## Glossary used in this chapter

The handbook-wide glossary is meant to be in `docs/amd/handbook/README.md`. The terms this chapter relies on:

| term | meaning |
|---|---|
| CU | Compute unit, AMD's counterpart of an NVIDIA SM. MI300X and MI325X have 304 (8 XCDs × 38). |
| XCD | Accelerator complex die. Each MI300-class GPU has 8. Physical CU indices interleave across XCDs. |
| SE | Shader engine. 4 per XCD on gfx942 (`scripts/amd/cubw/plan.py`, `HW[...]['se_per_xcd']`). |
| HBM | High-bandwidth memory (MI300X 5.3 TB/s peak, MI325X 6.0 TB/s peak; `results_published/COHORTS.md`). |
| MALL | Infinity Cache (memory-attached last-level cache), 256 MiB. Counters before it are "pre-MALL". |
| LDS | Local data share, 64 KiB per CU on gfx942. |
| FMA / MFMA | Scalar fused multiply-add / matrix fused multiply-add (the matrix cores). |
| `HSA_CU_MASK` | ROCm environment variable that restricts a process's kernels to a set of physical CUs, e.g. `3:0-199`. |
| n | Number of CUs the decoder is allowed to use (its mask is the contiguous range `0..n-1`). |
| R(n) | T(n)/T(304): decode time per step at n CUs divided by decode time at the full device. R > 1 is slower. |
| B1/C4096 | Batch size 1, context length 4096 tokens. |
| graph / eager | vLLM with HIP graphs captured (production serving mode) / without graphs. |
| ILP, MLP | Instruction-level parallelism (independent loads per thread) / memory-level parallelism (reads in flight). |
| `wvSplitK` | vLLM's ROCm "skinny GEMM" kernel used for batch-1 linear layers; ~82–84% of decode kernel time here. |
| block | One GPU lane holding one acquisition of every cell of a comparison, in random order. The pairing unit. |
| acquisition | One fresh decoder process: 3 warmups, then 10 timed trials of 64 generated tokens (63 decode ticks). |
| pilot | A single-acquisition check. Pilot rows are never promoted to production comparisons. |
| KFD | The kernel fusion driver (ROCm's kernel driver). |
| PMC | Performance-monitoring counters, collected with `rocprofv3 --pmc`. |

Cohort labels: **MI300X** runs on partition `mi3008x` (8 GPUs, 1.0× charge), **MI325X** on `mi3258x` (8 GPUs,
1.2× charge). The two are never pooled.

---

## Part A. The cubw design (E0–E8)

### A.1 Why the follow-up exists

The research question (plan 2.0, `docs/shoreline_aware_research_plan_2.0.md`) is whether GPU compute can be
traded for memory (SRAM, HBM "shoreline") to improve LLM decode. On AMD the proxy is: restrict compute units with
`HSA_CU_MASK` and see how much decode slows, and measure the memory system at the same time.

The first MI300X campaign (job 468885, 2026-10-18) gave the motivating numbers. They are reproduced in
`docs/amd/compute_bandwidth_followup_20261026.md` §1 (audited snapshot commit `5fc029aa`):

| MI300X quantity | 304 CUs | 232 | 152 | 96 |
|---|---:|---:|---:|---:|
| Native BF16 MFMA, TFLOP/s | 1071.12 | 791.62 | 566.43 | 380.97 |
| 4 GiB useful-read stream, TB/s | 4.5485 | 4.3164 | 4.0049 | 3.9744 |
| Qwen32B B1/C4096 graph, ms/step | 17.4868 | 18.0823 | 18.9548 | 19.0918 |
| Qwen7B B8/C4096 graph, ms/step | 6.6539 | 7.9305 | 9.4793 | 10.8926 |

At 96 CUs streaming kept 87% and MFMA kept 36%. Qwen32B B1 slowed only 9%, but Qwen7B B8 slowed 64%.

That campaign also had weaknesses, which the follow-up was written to fix (same doc, §1):
- Only 2 of 16 production M1 conditions passed the ≤5% clock envelope. The published "2/17" counted a smoke.
- The old competitor summary averaged overlapping windows without duration weighting and gave no ratio CIs.
- `FETCH_SIZE`/`WRITE_SIZE` counters are pre-MALL. Runtime Triton kernels were excluded from counters.
- CU masks keep the silicon. Clocks rise when CUs are masked, so masking is not "removing" CUs.

The follow-up was an owner-requested assignment dated 2026-10-26. It may not change clocks, power caps,
firmware, drivers or partition modes (§0). That rule matters everywhere below: no job could pass a clock gate by
fixing clocks, so every timing result ended up **observational**.

### A.2 Hypotheses and falsification criteria (§2 of the assignment)

| id | hypothesis | supported if | falsified or limited if |
|---|---|---|---|
| H1 | Arithmetic provision exceeds small-batch need. | Reduced-CU graph cells pass a predeclared noninferiority gate. | Latency regresses at moderate restriction, or padding/kernels/clocks explain the headroom. |
| H2 | Reduced-CU decode stays memory-service-sensitive. | At a fixed decoder mask, latency improves as shared memory service is restored, with arithmetic controls. | No reproducible response, or one explained by clocks/cache/host effects. |
| H3 | Request capacity is an enabling constraint. | Service-vs-CU curves locate a request/MLP knee and predict held-out decoder sensitivity. | Removing CUs collapses delivery that extra HBM cannot use, or the controls do not transfer. |
| H4 | More available service offsets a CU penalty. | A lower-CU/higher-service cell beats a higher-CU/lower-service cell with matched work. | Added service cannot overcome the restriction cost in range. |

H4 has two levels. A **same-chip shared-service demonstration** (removing a competitor restores service on the
stock chip) is not the same as a future design that beats the uncontended stock GPU. The docs keep these apart.

### A.3 Run contract and statistics (§3)

- TP1 (one GPU), dense BF16, fixed model revisions, seed 20261013, fixed prompts, greedy decoding with EOS
  ignored, and a one-token prefill barrier for all requests.
- Primary workload: Qwen2.5-32B B1/C4096, then Qwen2.5-14B B1/C4096. Graph mode is primary.
- ≥3 warmups, 10 timed trials per acquisition, ≥3 independent acquisitions per quantitative cell (5 for
  decisive comparisons if the budget allows). 64 generated tokens, so 63 pure decode ticks.
- Treatment order is randomized within blocks with a recorded seed. Trials are nested inside acquisitions and
  are never treated as independent processes.
- Outputs must be identical across baseline, restricted and competitor cells. Mismatches are logged, never
  hidden.

Implemented statistics (`scripts/amd/cubw/analyze.py`, header and `ratio_ci`):
- An acquisition's value is the mean over its 10 trials of HIP-event ms per decode step.
- Ratios use a **block-paired bootstrap**: 10,000 draws, seed 20261018. Each draw resamples blocks with
  replacement, then resamples trials inside each chosen acquisition. The 95% CI is the 2.5th–97.5th percentile.
- Cells without blocks fall back to resampling acquisitions per cell.

### A.4 The experiment families

| id | name | what it does | primary output |
|---|---|---|---|
| E0 | Reconciliation and adapters | Read real job state, audit old scripts, build new versioned adapters, CPU fixtures, dry runs, immutable manifests. No GPU. | `cubw_e0_reconciliation_20261026.md`, `scripts/amd/cubw/` |
| E1 | Correctness, mask, clock and instrumentation pilots | Prove each mask places work on exactly the intended CUs; revalidate mask syntax; pilot the competitor lifecycle; measure trace distortion. | `placement_ok`, `lifecycle_ok`, `outputs_ok`, leak gates |
| E2 | Fine CU noninferiority boundary | Qwen32B, then Qwen14B, at n = 304, 232, 200, 152, 96. Fresh own-cohort 304 baseline. | R(n) with 95% CI |
| E3 | Decoder CU allocation × shared-service treatment | Decoder on n ≤ 232 inside CUs 0–231; an owned competitor on fixed CUs 232–303. | T(n, z) and contrasts |
| E4 | Request/service curves | Native microbenchmarks (stream, ILP sweep, grid sweep, stride, gather, pointer chase, cache reuse, FMA, MFMA) at the same masks. | service ratio to 304 |
| E5 | Graph traffic and timeline | Kernel-trace timelines and (if possible) graph-mode counters at the baseline and boundary. | trace slowdown, coverage |
| E6 | Kernel efficiency | Identify dominant kernels; reproduce their shapes; try a kernel sized for the mask. | kernel time ratios |
| E7 | Decisive comparisons and a model | Fit R(n) = (1 − s) + s·P_x(n) on training points; check held-out and blind points. | validated or not |
| E8 | Guardrails and energy | Qwen7B B8, prefill, C16384, Mistral-7B; long-window J/token. | negative regimes |

**E2 criterion** (assignment §6). Noninferior at n means upper 95% CI of R(n) ≤ 1.05 **and** every gate passes,
including clocks. This is a ≤5% *latency* criterion. The plan text in `plan.py` `finalize()` repeats it:
`upper 95% CI of R(n)=T(n)/T(304) <= 1.05 (latency)`.

**E3 design** (§7). The competitor always holds CUs 232–303 (72 CUs, 9 per XCD). The decoder uses n = 232, 200,
152 or 96 inside 0–231. Six primary treatments: `absent`, `mem25`, `mem50`, `mem75`, `mem100`, `arith100`. An
`inactive` process (competitor started but idle) is a named validation control, not a seventh treatment.
- Full grid: 2 models × 4 masks × 6 treatments × 3 acquisitions = 144 acquisitions, only if clocks qualify.
- **E3-minimal** (the default if clocks are confounded): Qwen32B only, n = 232 and 152,
  absent/mem50/mem100/arith100 (`plan.py` `E3_MINIMAL`).
- The competitor is `svc_bench_v3` with a 1 GiB read footprint (`--mode read`) or FMA (`--mode fma`) and a
  requested duty (`scripts/amd/cubw/cubw.py`, `TREATMENTS`). Its achieved GB/s or TFLOP/s is joined to each
  decoder trial window, weighted by overlap duration. The achieved rate is a response as well as an
  intervention. It is never subtracted from decoder bandwidth.

**E7 model** (§10). For a restriction penalty κ = T_R/T_0, a bandwidth-scalable fraction s and a bandwidth
multiplier g: T'/T_0 = κ[(1 − s) + s/g]. The assignment says s and g must be fitted on a preregistered training
set and checked on held-out points. The illustration (κ = 1.09179, s = 0.8, g = 1.2 → 0.946) is not a measurement.

### A.5 Gates and dispositions

Each gate is a separate boolean. Command completion (rc = 0) is never treated as scientific success. Gate text
is copied into every launch plan (`plan.py` `finalize()`, key `gates`); the code is `analyze.py` `gates_from()`.

| gate | definition (as implemented) | introduced |
|---|---|---|
| `placement_ok` | For every lane and every planned level plus the competitor mask, the `svc_bench_v3 --mode placement` probe saw exactly the requested number of distinct physical CUs, and per-XCD counts differ by ≤ 1 across all 8 XCDs. | v1 |
| `lifecycle_ok` | Every competitor had a `started` event, a `stopped` event, its PID is gone, ≥ 95% trial coverage, no residual owned GPU process, and every tag-swept orphan is gone. Requires at least one competitor cell. | v1 (residual check fixed in v1.1) |
| `outputs_ok` | All Qwen32B C4096 unprofiled timing runs are complete and share **one** output digest across levels and treatments. | v1 (fixed in v1.1, v1.2, v1.3) |
| `leak_ok` (v1) | FMA probe TFLOP/s on the reserved CUs during decoder trials ≥ 0.85 × the probe alone. Recorded only after v1. | v1 |
| `leak_ok_v11` | Every decoder probe ratio ≥ 0.85 × the highest ratio of the same probe beside a placement-proven native co-runner (fma or read) on the decoder mask, same GPU. | v1.1 |
| `leak_ok_v2` | Per-CU probe (one full-LDS workgroup per CU). See #15/#17/#18. | v1.3 |
| `clock_ok` | Pooled in-window gfx and memory clock samples over every Qwen32B E2 trial: max/min ≤ 1.05 for each, no missing readings, per-trial sample coverage ≥ 0.5 of the monitor's median cadence. | v1 |
| `release_e2` | `placement_ok and outputs_ok` (observational label if clocks fail). | v1 |
| `release_e3_minimal` | `placement_ok and lifecycle_ok and outputs_ok and leak_ok_v11`. | v1.1 |
| `release_e3_full` | `release_e3_minimal and clock_ok`. | v1 |
| `release_e5e8` | `outputs_ok`. | v1 |
| `release_e3_minimal_v2` | `placement_ok and lifecycle_ok and outputs_ok and leak_ok_v2`. | v1.3 |

Later versions add `leak_ok_v3`, `leak_ok_v4` and matching releases for MI350X. They are not used in this chapter.

**Why the leak gate exists.** E3 attributes a decoder slowdown to shared memory service. That is only valid if
the decoder's kernels really stay on CUs 0..n−1 and do not "leak" onto the competitor's CUs. A leak would make
the competitor steal CUs, not bandwidth.

**Dispositions** (`analyze.py` `disposition()`), applied per cell:

| disposition | when |
|---|---|
| `not_run` | No acquisitions (gate not passed, or deadline). |
| `failed` | Any correctness/output failure. |
| `unavailable` | Fewer than 3 good acquisitions (1 for pilots), or a placement/isolation/lifecycle gate failed. |
| `observational_clock_confounded` | Everything passes except the pooled clock envelope. |
| `qualified` | Everything passes, including clocks. |
| `pilot` | An `e1_e2_pilot` row. Never promoted. |

In this chapter **no cell was ever `qualified`**. The pooled gfx clock ratio was 1.28–1.56 in every job, against
a limit of 1.05.

### A.6 How the software runs it

All code is in `scripts/amd/cubw/`. The flow is: commit → `submit.sh` → frozen release → `plan.py` → dry run →
`sbatch cubw.sbatch` → lanes run `cubw.py` → `analyze.py gate` (auto plans) → `analyze.py report`.

**`submit.sh <pilot|production|auto> <mi300x|mi325x> <HH:MM:SS> [plan.py args] [-- sbatch args]`**
1. Refuses to run if `scripts/` has uncommitted changes.
2. Freezes a release: `git archive <HEAD> scripts docs/amd/compute_bandwidth_followup_20261026.md` into
   `$WORK/cubw_release/<full commit sha>/`, writes `RELEASE_COMMIT`, and makes `scripts/` read-only. A queued job
   therefore cannot be changed by later edits.
3. Writes the plan to `$WORK/cubw_plans/<cohort>_<kind>_<YYYYmmdd_HHMMSS>_<sha8>.json` and makes it read-only.
4. Dry-runs `cubw.sbatch` with `DRY_RUN=1` (no GPU, no scheduler job). It prints every step command into
   `$WORK/results_amd/dryrun_cubw_<cohort>_<kind>_<timestamp>/DRYRUN_COMMANDS.txt`.
5. Submits `cubw.sbatch` with the plan's partition, the given time limit, `-J cubw-<cohort>-<kind>`, and
   `CUBW_PLAN`, `CUBW_RELEASE` and optionally `CUBW_GATES` exported.

**`plan.py`** builds an immutable, declarative plan. Every step names its lane (= one GPU), stage, experiment,
cell, acquisition, block, CU count, treatment, config, time estimate and timeout. Orders are randomized with
`random.Random` seeded from `SEED = 20261026` plus the cohort/model/block. Kinds and flags used in this chapter:

| invocation | builds | used by |
|---|---|---|
| `--kind pilot` | Full pilot: placement ×8 GPUs, capability audit, mask-syntax revalidation, E4 pilot, E2 pilots (bridge, Qwen32B, Qwen14B, second Qwen32B GPU), E3 pilots at 232/152, leak probes, timeline/counter pilots, E8 pilots. | 481167 |
| `--kind auto` (default `--pilot reduced`) | Reduced pilot → gate → production with Qwen32B ×3 and Qwen14B ×3 blocks. | 481168 (unrun), 483499 |
| `--kind auto --pilot minimal --boundary 200` | v1.1 re-pilot (placement, leak suites on L0/L1, lifecycle on L2, Qwen32B E2 on L3/L4) → gate → production with Qwen32B ×5 and Qwen14B ×3 blocks. | 483498 |
| `--kind production --boundary 200 --e4-done-in <root>` | Production only (placement ×8 then blocks); E4 omitted because it already exists. | 487341, 489594 |
| `--kind auto --extension` | v1.3: placement at fine levels, leak v2, gate, E2-fine, `e6_decode`, timelines, shapes, counters, MI300X E3-minimal. | 492956, 492957 |

Production blocks (`production()`): lane i holds block i of one model. Each block has one acquisition of every E2
level and every E3 cell, shuffled. E3 cells carry `requires: release_e3_minimal` or `release_e3_full`; E2 cells
carry `requires: release_e2`. After the blocks come E4 (3 acquisitions, spread over lanes), E5 kernel traces at 304
and the boundary, and E8. Timeouts: `max(900 s, 3 × estimate)` for decoder steps; profiled steps are capped at
1200 s and never retried. `reserve_s` is 1800.

**`cubw.sbatch`** (runs from the frozen release, never the live repo):
- Step 0: fresh result root `$WORK/results_amd/amd_<date>_<cohort>_cubw_v1_<kind>_job<id>`, symlinked into
  `results/`. Copies the plan to `launch_plan.json`, copies `CUBW_GATES` to `gates.json` if set, copies configs
  and writes derived configs (C16384, 10-trial energy, eager profile). Checks that the `svc_bench` binaries exist.
- Preflight (`scripts/amd/preflight_node.py`): expected GPU count, name, arch and CU count.
- `manifest.py` writes `environment.json` (node, GPUs, ROCm/tool versions, container digest, bench SHA-256s),
  `source_manifest.json` (SHA-256 of every release file, configs, model revisions; weights are not rehashed) and
  `case_manifest.json` (every planned step).
- Deadline: `CUBW_DEADLINE` = Slurm end time − `reserve_s`.
- A node monitor (`scripts/amd/gpu_monitor.py`, requested 0.05 s cadence) runs in the background. It is never a
  lane-completion prerequisite (`wait_lanes` skips it — the lesson of the old MI325X 470176 hang).
- For each stage: in an `auto` plan, before any non-pilot stage, `analyze.py gate` writes `gates.json`. Then one
  `cubw.py lane` per GPU runs in parallel.
- Closeout: stop the monitor, `analyze.py report`, then `SHA256SUMS` of the whole root.

**`cubw.py lane`** executes one lane's steps in order and appends every record to
`<root>/journal/<lane>.jsonl`:
- If the step has `requires` and that key in `gates.json` is not true: journal `disposition: not_run` with the
  reason, and skip.
- If `now + est_s > CUBW_DEADLINE`: journal `not_run` with reason `deadline`.
- Each attempt gets its own directory `<root>/<experiment>/<cell>/a<acq>/attempt<k>`, created with
  `exist_ok=False`, so nothing is overwritten.
- At most two attempts, and a retry only if the failure class starts with `execution`. Correctness failures
  (`differ`, `barrier`, `decode did not progress`) are never retried. Profiled, shell and competitor-only steps
  are `no_retry`.
- Step kinds: `vllm` (a decoder acquisition via `scripts/controlled_vllm_experiments.py` in the vLLM container,
  optionally beside a competitor, optionally under `rocprofv3`), `native` (one `svc_bench` process), `shell`
  (capability commands and E6 scripts), `competitor_only` (probe alone or beside a native co-runner).
- Competitor lifecycle: start in its own process group with `HSA_CU_MASK=<gpu>:232-303`; wait up to 90 s for the
  `started` event (written after the first completed launch); pass its PID to the decoder via
  `RESEARCH_ALLOWED_PIDS`; at the end SIGTERM, 15 s grace, then SIGKILL; verify the PID is gone.
- v1.1 additions: each owned process carries a per-attempt `CUBW_STEP_TAG`. `sweep_tag()` finds escaped
  processes by that tag in `/proc/*/environ`, kills and records them. `gpu_processes()` applies the
  `scripts/gpu_target.py` rule (VRAM ≥ 64 MiB or GFX engine time on the target GPU).
- The decoder mask is `<physical index>:0-<n-1>`; no mask at n = 304.

**`analyze.py`** has two commands:
- `gate --root <pilot root> --out gates.json` computes every gate and release from what has run.
- `report --root <root>` writes `analysis/cubw_tables.json` and `summary.md`: placement table, E2 R(n) per
  model with pooled clocks and dispositions, E3 T(n, z) with competitor joins and contrasts, E4 service ratios,
  E5/timeline summaries, E8 tables, and (v1.3) E6 tables.
- `test_analyze.py` holds the CPU fixtures: 17/17 at E0, 23/23 at v1.1, 25/25 at v1.2 (experiment log).

**Where things live**

| item | path |
|---|---|
| frozen releases | `$WORK/cubw_release/<commit>/` (2da9ae35…, 79d8b597…, c7cf8aa9…, 99bcd1ad…, de2baecb… for this chapter) |
| plans | `$WORK/cubw_plans/*.json`; also the gate files `gates_483498_pilot_v12.json`, `gates_483499_pilot_v12.json` |
| raw roots | `$WORK/results_amd/amd_<date>_<cohort>_cubw_v1_<kind>_job<id>/` (journals, attempt dirs, `RUN_STATUS.log`, `SHA256SUMS`) |
| dry runs | `$WORK/results_amd/dryrun_cubw_*` |
| compact copies | `results_published/amd_<date>_<cohort>_cubw_*_job<id>/` (summary, tables, gates, plan, journal, environment) |
| binaries | `$WORK/bin/svc_bench_gfx942_v3`, `_v4` (sources `scripts/amd/cubw/svc_bench_v3.hip`, `_v4.hip`) |

### A.7 Reader and tool versions in this chapter

| label | commit | what broke | fix |
|---|---|---|---|
| v1 | b428ec06, 2da9ae35 | — | First release (E0). 2da9ae35 caps profiled decoder steps at 1200 s. |
| v1.1 | 79d8b597 | Pilot 481167: lifecycle could never pass (amdsmi lists every KFD process on the node for every GPU); outputs gate counted profiled runs; rocprofv3 left PPID=1 orphans; graph-mode PMC crashed/hung; the v1 leak ratio could not separate leakage from contention. | Per-GPU residual rule, tag sweep, outputs gate on unprofiled runs only, native co-runner leak references (`leak_ok_v11`), graph-mode counters dropped. |
| v1.2 | c7cf8aa9 | Auto jobs 483498/483499: the gate ran after the pilot stage and counted the not-yet-attempted production steps as incomplete runs, so `outputs_ok` and every release were false. | `outputs_ok` ignores `not_run` steps; `plan.py --e4-done-in`. 99bcd1ad has identical scripts (`git diff c7cf8aa9 99bcd1ad -- scripts` is empty). |
| v1.3 | de2baecb (+ 363c4f05 reader/pre-registration) | `leak_ok_v11` could not separate leakage from contention on MI300X; E6/E7 not run; n = 200 < 232 anomaly unexplained. | Per-CU probe `svc_bench_v4 --mode cuprobe` and `leak_ok_v2`; fine levels; `e6_decode` with `CUBW_CU_COUNT_OVERRIDE`; E6 shapes and counters; separate output gate for the sized kernel (363c4f05). |
| reader v1.4 | 4f346b44 | v1.3 labelled every non-`e2` E2-table row "pilot … never promoted", including the 5-block `e6_decode` comparison. | Only `e1_e2_pilot` rows are labelled pilot; `e6_decode` gets the ordinary disposition (`observational_clock_confounded`). No values changed. |

Note on "v1.4": the label is used twice. Reader v1.4 (4f346b44, 2026-11-02) is the one-line disposition fix
above. The "cubw v1.4" MI350X port (8ba209a8) is a separate, later change covered in the MI350X chapter.

---

## Part B. Experiments

### #7 E0 reconciliation, capability audit and adapters — no job (CPU only)

- **At a glance:** login node / no GPU job / 2026-10-26 (checked 14:10–16:30 CDT) / commits b428ec06 and
  2da9ae35 / complete.
- **Question:** before launching anything, what is the true state of earlier jobs and data, which old scripts
  can be reused, and what new adapters are needed? (assignment §4)
- **Design:** read scheduler state and result roots; check old published counts against summaries; audit each
  old script; build versioned adapters; write CPU fixtures; publish dry-run plans and a budget.
- **How it was run:** CPU work only. `svc_bench_v3` was smoke-tested on an MI210 `devel` node (gfx90a build):
  placement counted 104/104 CUs unmasked and 52/52 under `0:0-51`; competitor SIGTERM produced
  `started`/`stopped` events with rc = 0. Dry runs enumerated 216 steps (MI300X pilot) and 763 (MI325X auto).
- **What was measured and how it was analysed:** fixture checks in `test_analyze.py`: status/correctness
  separation, nested-trial bootstrap units, missing clocks, coverage gaps, pooled vs per-cell clock gate,
  overlapping windows, gate logic.
- **Results:**

| item | finding |
|---|---|
| MI325X 470176 attempt 3 | Phase 1 rc = 0 by 21:20 on Oct 25, then TIMEOUT; phases 2–4 never started (monitor in the lane wait). |
| MI325X resume 481072 | PENDING at the time; not interrupted or duplicated. |
| Old M1 clock count | 2/16 production triplets pass, not "2/17" (the 17th was a smoke). |
| Old M6 | 19/56 cells pass individually; no pooled comparison. |
| Old competitor reader | Unweighted overlap; the new fixture reproduces old 200 GB/s vs duration-weighted 120 GB/s. |
| Allocation | 286.38 / 450 units used; ~8 units per MI300X node-hour, ~12.8 per MI325X node-hour. |
| Fixtures | 17/17 pass after one fixture repair (an overlap fixture gave equal old/new means by chance; a realtime-step check read clipped stamps). |

Budget table (estimates): MI300X pilot ~0.9 node-h (~7 units); MI300X production ~2.7 (~22); MI325X auto ~3.0
(~39).

- **What it means:** the old scripts were reusable components but did not implement the follow-up. New
  adapters replaced `lane_cumask.sh` (no placement proof), `lane_competitor.sh` (no start proof, unverified kill)
  and the old summarizers. Release rules were predeclared in every plan.
- **Failures and caveats:** none in E0 itself. GPU-side E0 items (environment, counter list, manifests) are
  produced at the start of every job, not here.
- **Files:** `docs/amd/cubw_e0_reconciliation_20261026.md`; `docs/amd/experiment_log.md` (2026-10-26 entries);
  `scripts/amd/cubw/` at b428ec06/2da9ae35; dry runs `$WORK/results_amd/dryrun_cubw_mi300x_pilot_20261026_144349`,
  `dryrun_cubw_mi325x_auto_20261026_144357`.

### #8 MI300X pilot — job 481167

- **At a glance:** `mi3008x` / 8× MI300X / k002-002 / 2026-10-28 08:34–09:06 CDT (32 min, after ~42 h in queue)
  / release 2da9ae35 (v1) / complete; v1 gates failed on tool/reader defects; nothing released.
- **Question:** do the new tools work on gfx942 (placement, lifecycle, outputs, clocks, leak, counters), and
  where is the E2 boundary, before spending on production?
- **Design:** plan `$WORK/cubw_plans/mi300x_pilot_20261026_144349_2da9ae35.json` (216 steps, `--kind pilot`).
  - Placement on all 8 GPUs at 304, 232, 200, 152, 96 and the competitor 232–303.
  - Mask revalidation: comma list `0-7,16-23`, a mask naming another GPU's index, the empty-XCD mask `0-6`.
  - One-acquisition E2 pilots: bridge, Qwen32B (two GPUs), Qwen14B.
  - E3 pilots at n = 232 (absent/inactive/mem50/mem100/arith100) and n = 152 (absent/mem50/mem100/arith100).
  - Leak probes, kernel-trace and graph-mode counter pilots, E4 pilot suite, E8 pilots.
- **How it was run:** `submit.sh pilot mi300x 01:45:00`. One pilot stage, 8 lanes.
- **What was measured and how it was analysed:** per-cell ms/step; gates as in A.5; leak v1 ratio = probe
  TFLOP/s during decoder trials / probe alone.
- **Results — gates under v1 rules:**

| gate | result | diagnosis |
|---|---|---|
| placement_ok | pass | 8 GPUs × all levels; per XCD 38/29/25/19/12/9. |
| lifecycle_ok | fail | Tool defect: amdsmi lists every KFD process on the node for every GPU, so the residual list was never empty. Started/stopped events and PID-gone checks all passed. |
| outputs_ok | fail | Reader defect: two rocprofv3 runs were counted. Unprofiled Qwen32B runs: 25/25 complete, one digest. |
| leak_ok (v1) | fail | Probe beside decoder / alone = 0.58–0.61; clocks explain only ~8% (2088 vs 1923 MHz). |
| clock_ok | fail | Pooled gfx 1319–2021 MHz (×1.53); memory 1059–1300 MHz. |
| E5 graph counters | failed | `rocprofv3 --pmc FETCH_SIZE` on graph decode: illegal memory access, then hang to the 1200 s timeout; the Triton-excluded retry also hung; PPID=1 rocprofv3 orphans survived until job end. |

- **Results — pilot observations (one acquisition each, clock-confounded):**

| measure | 232 | 200 | 152 | 96 |
|---|---:|---:|---:|---:|
| Qwen32B R(n), GPU A / GPU B | 1.028 / 1.034 | 1.015 / 1.023 | 1.084 / 1.087 | 1.095 / 1.102 |
| Qwen14B R(n) | 1.045 | 1.024 | 1.106 | 1.109 |
| E4 4 GiB stream (ratio to 304) | 0.95 | 0.96 | 0.88 | 0.87 |
| E4 ILP-1 read | 0.84 | 0.88 | 0.72 | 0.61 |
| E4 FMA / MFMA | 0.81 / 0.75 | 0.74 / 0.66 | 0.64 / 0.53 | 0.46 / 0.35 |

E3 Qwen32B (ratio to absent at same n): n = 232 mem50 1.122, mem100 1.269, arith100 1.062, inactive 0.998;
n = 152 mem50 1.127, mem100 1.306, arith100 1.062. Kernel-trace slowdown ×1.076 (304), ×1.099 (152), outputs
match. Energy 14.08 J/token at 752 W; C16384 B1 19.74 ms/step.

Mask revalidation: the comma list was **accepted** (16 CUs, balanced), the opposite of the 468885 observation; a
mask naming another GPU's index did not restrict this GPU; `0-6` placed work on 45 CUs including 38 on XCD 7
(unsafe, as before).

- **What it means:** the boundary for E5/E8 was set to **n = 200** (smallest pilot level with point R ≤ 1.05 on
  both Qwen32B GPUs). This is a planning choice, not a noninferiority result. The 200 < 232 pattern first
  appeared here.
- **Failures and caveats:** led to **v1.1** (see A.7). Re-gating this root with the v1.1 reader gives
  `release_e2` and `release_e5e8` true, but lifecycle still fails (old records hold only unfiltered PID lists) and
  `leak_ok_v11` has no references, so E3 stayed unreleased. Two `runner.log` files changed after hashing because
  orphaned rocprofv3 processes kept writing (`cubw_extension_plan_20261101.md` §5). Raw root is 14 GB.
- **Files:** `docs/amd/cubw_pilot_review_20261028.md`; `docs/amd/interim_synthesis_20261028.md` (cross-run
  reading); `results_published/amd_20261028_mi300x_cubw_v1_pilot_job481167/` (incl. `gates_regated_v11.json`);
  raw `$WORK/results_amd/amd_20261028_mi300x_cubw_v1_pilot_job481167` (643 files); v1.1 commit 79d8b597.

### #9 MI325X auto (v1) — job 481168 (cancelled, never ran)

- **At a glance:** `mi3258x` / 8× MI325X / no node / submitted 2026-10-26 / release 2da9ae35 (v1) / cancelled
  before it ran.
- **Question:** run a separate MI325X cohort in one allocation: reduced pilot → gate → production.
- **Design:** plan `$WORK/cubw_plans/mi325x_auto_20261026_144357_2da9ae35.json`, `--kind auto` (reduced pilot),
  763 steps, boundary 152 (the `plan.py` default; the MI300X pilot had not yet run).
- **How it was run:** `submit.sh auto mi325x 03:45:00 -- --dependency=afterany:481072`, with a user hold until
  the MI300X pilot validated the tools.
- **What was measured and how it was analysed:** nothing.
- **Results:** none.
- **What it means:** the pilot found v1 tool/reader defects, so a v1 release would have produced unusable gates.
  It was cancelled and replaced by 483499 on release 79d8b597 (v1.1).
- **Failures and caveats:** none recorded beyond the cancellation.
- **Files:** `docs/amd/experiment_log.md` (2026-10-26 and 2026-10-28 entries); `docs/amd/cubw_pilot_review_20261028.md` §4.

### #11 MI300X auto (v1.1) — job 483498

- **At a glance:** `mi3008x` / 8× MI300X / k002-003 / 2026-10-30 06:26–06:39 CDT (13 min, after ~45 h in queue)
  / release 79d8b597 (v1.1) / complete; production blocked by a reader defect except E4.
- **Question:** re-prove the v1.1 fixes on MI300X (placement, lifecycle, leak references, outputs, clocks), then
  release production automatically in the same allocation.
- **Design:** plan `$WORK/cubw_plans/mi300x_auto_20261028_091901_79d8b597.json` (691 steps):
  - Pilot stage (`--pilot minimal`): placement ×8; leak suites on L0 and L1 (FMA probe on n..303 alone, beside
    native fma and read co-runners on 0..n−1, beside the decoder at n; n = 232 and 152); lifecycle at n = 232
    (absent/inactive/mem100/arith100); Qwen32B E2 on two GPUs.
  - Production: Qwen32B ×5 and Qwen14B ×3 blocks; E3 gated; E4 ×3; E5 ktrace; E8; boundary 200.
- **How it was run:** `submit.sh auto mi300x 02:45:00 --pilot minimal --boundary 200`. Gate at 06:38:27, then
  production.
- **What was measured and how it was analysed:** `leak_ok_v11` = decoder probe ratio ÷ highest native co-runner
  ratio, must be ≥ 0.85 (predeclared before any co-run data).
- **Results — gates:**

| gate | v1.1 as run (`gates.json`) | v1.2 re-gate (`gates_regated_v12.json`) |
|---|---|---|
| placement_ok / lifecycle_ok | pass / pass | pass / pass |
| outputs_ok | **fail** (164 runs, 18 complete) | pass (18/18, 1 digest) |
| leak_ok_v11 | fail | fail |
| clock_ok | fail (pooled gfx 1300–1964 MHz, ×1.51) | fail |
| release_e2 / e5e8 | false | **true** |
| release_e3_minimal / full | false | false |

| leak cell | decoder ratio | native arith100 | native mem100 | relative |
|---|---:|---:|---:|---:|
| n232 arith100 L0 | 0.614 | 0.749 | 0.592 | 0.820 |
| n232 arith100 L1 | 0.609 | 0.762 | 0.590 | 0.799 |
| n152 leakprobe L0 | 0.622 | 0.811 | 0.650 | 0.766 |
| n152 leakprobe L1 | 0.667 | 0.870 | 0.701 | 0.766 |

Production: 271 gated steps journalled `not_run` (E2 40, E3 193, E5/E8 38). Ungated E4 ×3 ran: 339/339 rc = 0.

| E4 (3 acq., ratio to 304) | 232 | 200 | 152 | 96 |
|---|---:|---:|---:|---:|
| FMA | 0.82 | 0.76 | 0.65 | 0.47 |
| MFMA | 0.76 | 0.67 | 0.54 | 0.36 |
| ILP-1 read | 0.84 | 0.88 | 0.72 | 0.57 |
| 8 MiB reuse | 0.76 | 0.69 | 0.57 | 0.38 |
| 9728-block streaming | 0.97–1.01 at every level | | | |

Pilot E2 Qwen32B: 232 → 1.022; 200 → 1.009; 152 → 1.064; 96 → 1.078. Lifecycle cells at 232: mem100 1.231
(competitor 1043 GB/s), arith100 1.066, inactive 0.994.

- **What it means:** the decoder slows the probe more than a native FMA co-runner and about as much as a native
  streaming-read co-runner (decoder/mem100 0.95–1.04). That fits a memory-bound decoder contending for shared
  memory service, but under the predeclared rule it fails, so **E3 stays unreleased on MI300X**. The threshold was
  not changed after the fact. E4 shows that with enough parallelism even 96 CUs nearly saturate HBM; loss
  appears when requests per CU are limited (ILP 1) or data is on-chip.
- **Failures and caveats:** **v1.1 reader defect → v1.2.** An auto plan's gate runs when all 146 production
  decoder steps are still unattempted; v1.1 counted them as incomplete runs. Fixed in c7cf8aa9 (`outputs_ok`
  ignores `not_run`). Attempted runs that fail still fail the gate. Two fixtures added; 25/25 pass.
- **Files:** `docs/amd/cubw_auto_review_20261030.md`; `results_published/amd_20261030_mi300x_cubw_v1_auto_job483498/`;
  raw `$WORK/results_amd/amd_20261030_mi300x_cubw_v1_auto_job483498` (968 files); re-gate copy
  `$WORK/cubw_plans/gates_483498_pilot_v12.json`.

### #12 MI325X auto (v1.1) — job 483499

- **At a glance:** `mi3258x` / 8× MI325X / k002-001 / 2026-10-31 03:57–04:13 CDT (17 min) / release 79d8b597
  (v1.1) / complete; same reader defect blocked production except E4.
- **Question:** the MI325X cohort's pilot, gates and (auto) production, using the boundary from the MI300X pilot.
- **Design:** plan `$WORK/cubw_plans/mi325x_auto_20261028_091903_79d8b597.json` (759 steps, `--kind auto`
  with the reduced pilot): placement, capability audit, mask revalidation, E4 pilot, E2 pilots (Qwen32B two GPUs,
  Qwen14B), E3 pilots at 232 and 152, leak suite with native references, timeline, E8 pilots; then production
  with Qwen32B ×3 and Qwen14B ×3 blocks; boundary 200.
- **How it was run:** `submit.sh auto mi325x 03:15:00 --boundary 200 -- --dependency=afterany:481072`. Left in the
  queue after 483498 exposed the defect, to keep its priority age.
- **What was measured and how it was analysed:** as #11.
- **Results — gates:**

| gate | v1.1 as run | v1.2 re-gate (`gates_regated_v12.json`) |
|---|---|---|
| placement_ok / lifecycle_ok | pass / pass | pass / pass |
| outputs_ok | fail (190 unattempted steps counted) | pass (21/21, 1 digest) |
| leak_ok (v1) | fail | fail (probe 0.75–0.84 of alone) |
| leak_ok_v11 | pass | **pass**, relative 0.87 / 0.91 / 0.92 / 0.97 |
| clock_ok | fail | fail: pooled gfx 1494–2076 MHz (×1.39), mem 1285–1500 MHz |
| release_e2 / e5e8 / e3_minimal | false | **true / true / true** |
| release_e3_full | false | false |

Pilot E2 Qwen32B R(n) (full-CU baseline 16.23 ms/step):

| GPU | 232 | 200 | 152 | 96 |
|---|---:|---:|---:|---:|
| GPU 1 | 1.069 | 1.014 | 1.153 | 1.108 |
| GPU 2 | 1.012 | 1.038 | 1.100 | 1.135 |

Qwen14B: 1.178 / 1.053 / 1.172 / 1.210. E3 pilot (ratio to absent): n = 232 mem100 1.169 (1170 GB/s), arith100
1.002, inactive 1.010; n = 152 mem100 1.194 (1299 GB/s), arith100 1.045. E4 ×3 (ratio to 304, 232/200/152/96):
FMA 0.76/0.71/0.56/0.40; MFMA 0.66/0.58/0.42/0.30; ILP-1 0.83/0.89/0.70/0.55; 8 MiB reuse 0.71/0.64/0.50/0.36;
9728-block stream 1.01/1.02/1.00/0.95.

- **What it means:** on MI325X the decoder's slowdown of the probe stays within the native co-runner envelope,
  so **E3-minimal is released for this cohort** as predeclared. The cohorts differ on this gate. Compute-bound
  E4 rows lose more on MI325X than on MI300X.
- **Failures and caveats:** the only non-zero step was the intentional unsafe empty-XCD probe `0-6` (rc = 3), a
  revalidation item. Single-run spread is large, so the boundary stayed 200.
- **Files:** `docs/amd/cubw_mi325x_auto_review_20261031.md`; `results_published/amd_20261031_mi325x_cubw_v1_auto_job483499/`;
  raw `$WORK/results_amd/amd_20261031_mi325x_cubw_v1_auto_job483499` (1311 files); re-gate copy
  `$WORK/cubw_plans/gates_483499_pilot_v12.json`.

### #13 MI300X production — job 487341

- **At a glance:** `mi3008x` / 8× MI300X / k002-003 / 2026-10-31 18:24–18:56 CDT (32 min) / release c7cf8aa9
  (v1.2) / complete (126/126 released steps rc = 0, first attempt).
- **Question:** the production E2 curve (H1) for both models with ≥3 blocks, plus E5 timelines and E8 guardrails
  at the boundary.
- **Design:** plan `$WORK/cubw_plans/mi300x_production_20261030_064338_c7cf8aa9.json` (319 steps). Placement 48;
  E2 Qwen32B 5 blocks and Qwen14B 3 blocks at 304/232/200/152/96 (40 steps); E5 kernel traces at 304 and 200;
  E8 36 steps (Qwen7B B8, prefill, C16384, Mistral-7B, energy at 304 and 200). E3 cells journalled `not_run`
  (193). No E4 (`--e4-done-in` 483498).
- **How it was run:** `CUBW_GATES=$WORK/cubw_plans/gates_483498_pilot_v12.json submit.sh production mi300x 01:45:00
  --boundary 200 --e4-done-in $WORK/results_amd/amd_20261030_mi300x_cubw_v1_auto_job483498`.
- **What was measured and how it was analysed:** R(n) by block-paired bootstrap; pooled clock gate; E8 as
  ms/step and J/token over 10 trials × 2047 decode ticks.
- **Results:**

| n | Qwen32B (5 blocks) | Qwen14B (3 blocks) |
|---:|---|---|
| 304 | 17.78 ms/step | 8.64 ms/step |
| 232 | 1.024 [1.020, 1.027] | 1.051 [1.047, 1.054] |
| 200 | **1.012 [1.010, 1.014]** | **1.027 [1.024, 1.030]** |
| 152 | 1.070 [1.064, 1.075] | 1.107 [1.104, 1.109] |
| 96 | 1.083 [1.075, 1.088] | 1.102 [1.096, 1.106] |

| E8 (3 acq.) | 304 | 200 | 96 |
|---|---:|---:|---:|
| Qwen7B B8 C4096 decode, ms/step | 6.70 | 7.82 (1.17×) | 10.81 (1.61×) |
| Qwen32B prefill wall, ms | 551 | 780 (1.42×) | 1242 (2.25×) |
| Qwen32B C16384 decode | 19.07 | 19.43 (1.02×) | – |
| Qwen32B C16384 prefill, ms | 2810 | 3820 (1.36×) | – |
| Mistral7B B1 decode | 5.12 | 5.32 (1.04×) | – |
| Energy, J/token (W) | 13.83 (752) | 13.78 (752) | – |

E5: trace slowdown 1.055 (304) and 1.063 (200), outputs match, 13 decode ranges, 731 dispatches per tick.

- **What it means:** small-batch decode keeps ≤ ~3% latency at 200 of 304 CUs (observational). n = 200 has upper
  CI ≤ 1.05 for both models, but under the predeclared rule (CI **and** every gate including clocks) **no level
  qualifies**. Guardrails show where the headroom ends: batch 8 +17%, prefill +42%, long-context prefill +36%.
  Both energy cells sit at the 750 W cap, so masking moves clocks, not power. n = 200 being faster than 232
  matched both pilots and was unexplained here (clocks differ by cell: gfx ×1.32 at 232 vs ×1.36 at 200; per-XCD
  29 vs 25 CUs). #18 later traced it to `wvSplitK`.
- **Failures and caveats:** clock gate fails (pooled gfx 1294–2015 MHz, ×1.56; memory 1068–1300 MHz; minimum
  trial coverage 0.495, just under 0.5). Every E2 ratio is `observational_clock_confounded`. `lifecycle_ok` is
  false only because this job ran no competitor (the gate needs at least one).
- **Files:** `docs/amd/cubw_mi300x_production_review_20261031.md`; `docs/amd/compute_bandwidth_followup_results_20261101.md`;
  `results_published/amd_20261031_mi300x_cubw_v1_production_job487341/`; raw
  `$WORK/results_amd/amd_20261031_mi300x_cubw_v1_production_job487341` (410 files).

### #14 MI325X production (incl. E3-minimal) — job 489594

- **At a glance:** `mi3258x` / 8× MI325X / k002-001 / 2026-11-01 10:37–11:28 CST (51 min) / release 99bcd1ad
  (v1.2 scripts) / complete (167/167 released steps rc = 0, first attempt).
- **Question:** MI325X E2 curve, plus the first released E3 (H2, H4), E5 and E8.
- **Design:** plan `$WORK/cubw_plans/mi325x_production_20261031_041507_99bcd1ad.json` (319 steps). Placement 48;
  E2 40 (Qwen32B ×5, Qwen14B ×3 blocks); **E3-minimal** 41 (Qwen32B; n = 232/152; absent/mem50/mem100/arith100
  × 5 blocks, plus the inactive control in block 0); E5 ktrace 2; E8 36. The 152 full-grid E3 steps journalled
  `not_run`. No E4 (done in 483499).
- **How it was run:** `CUBW_GATES=$WORK/cubw_plans/gates_483499_pilot_v12.json submit.sh production mi325x 02:30:00
  --boundary 200 --e4-done-in …483499`. In each block the E2 and E3 cells are shuffled together on one GPU.
- **What was measured and how it was analysed:** R(n); T(n, z)/T(n, absent); competitor GB/s or TFLOP/s joined
  to decoder trials by duration-weighted overlap; contrasts mem100/arith100 and T(152, absent)/T(232, mem·).
- **Results — E2:**

| n | Qwen32B (5 blocks) | Qwen14B (3 blocks) |
|---:|---|---|
| 304 | 15.34 ms/step | 7.79 ms/step |
| 232 | 1.074 [1.071, 1.077] | 1.133 [1.093, 1.210] |
| 200 | 1.056 [1.040, 1.085] | 1.047 [1.044, 1.049] |
| 152 | 1.187 [1.165, 1.211] | 1.222 [1.167, 1.254] |
| 96 | 1.186 [1.175, 1.204] | 1.170 [1.169, 1.171] |

- **Results — E3-minimal (Qwen32B, 5 blocks, ratio to absent at the same n):**

| treatment | n = 232 | n = 152 | competitor achieved (232 / 152) |
|---|---|---|---|
| mem50 | 1.052 [1.016, 1.083] | 1.090 [1.078, 1.104] | 556 / 652 GB/s |
| mem100 | 1.146 [1.104, 1.186] | 1.182 [1.164, 1.200] | 1175 / 1331 GB/s |
| arith100 | 1.003 [0.977, 1.024] | 1.018 [1.014, 1.022] | 14.4 / 14.8 TFLOP/s |
| inactive (1 block) | 1.021 [0.997, 1.047] | – | – |

Contrasts: mem100/arith100 = 1.142 (232), 1.161 (152). **T(152, absent)/T(232, mem100) = 0.924 [0.911, 0.939]**;
T(152, absent)/T(232, mem50) = 1.006 [0.997, 1.015].

- **Results — E5/E8:** trace slowdown 1.076 (304), 1.087 (200), outputs match. E8 at 304/200/96: Qwen7B B8 6.31 /
  8.16 (1.29×) / 12.12 (1.92×); prefill 480 / 694 (1.44×) / 1399 (2.91×) ms; C16384 decode 16.72 / 17.09 (1.02×);
  C16384 prefill 2470 / 3636 (1.47×); Mistral7B 4.96* / 5.00; energy 15.39 J/token (984 W) / 15.89 (972 W). (*One
  304 acquisition is slow, 5.33 vs 4.77; without it 200/304 = 1.05.)
- **What it means:**
  - H1: MI325X restriction costs are roughly 2–5× MI300X's at the same n; no level has upper CI ≤ 1.05 for Qwen32B.
    With ~10% more HBM bandwidth, compute is a larger share of each step (results v1 §3). n = 200 again beats 232.
  - H2 (observational): latency rises monotonically with memory-competitor duty at both n; an arithmetic
    competitor on the same 72 CUs costs ≤ 2%. Because the leak gate passed, this is attributed to shared memory
    service, not CU leakage.
  - H4 shared-service balance (not a stock-GPU win): a decoder with 80 fewer CUs but uncontended memory service
    beats a 232-CU decoder beside a full-duty streaming competitor. The uncontended 304 baseline is faster than both.
- **Failures and caveats:** clock gate fails (pooled gfx 1630–2083 MHz, ×1.28; mem 1227–1500 MHz); every ratio is
  `observational_clock_confounded`. Competitor cells run at higher clocks, which biases their slowdowns
  downward (interim synthesis). The job's own `summary.md` lists `leak_ok_v11`/`release_e3_minimal` as false
  because the production root has no leak suite to recompute them; the release actually applied is the pilot's
  v1.2 gate copied to `gates.json`.
- **Files:** `docs/amd/cubw_mi325x_production_review_20261101.md`; `docs/amd/compute_bandwidth_followup_results_20261101.md`;
  `results_published/amd_20261101_mi325x_cubw_v1_production_job489594/`; raw
  `$WORK/results_amd/amd_20261101_mi325x_cubw_v1_production_job489594` (595 files).

**Status after #13/#14** (results report v1, `compute_bandwidth_followup_results_20261101.md` §7): fixed clocks
unavailable; E3 full grid blocked (both SKUs) and MI300X E3-minimal blocked by `leak_ok_v11`; graph-mode counters
unavailable; E6 and E7 not run; n = 200 vs 232 open; off-cluster raw archive outstanding.

### #15 E6 kernel identification and E7 model (CPU) — no job

- **At a glance:** login node / CPU only / 2026-11-01 / commits de2baecb and 363c4f05 (pre-registration) /
  complete; E7 r1 rejected, r2 fitted with blind predictions registered.
- **Question:** (E6) which kernels carry the CU penalty, and why? (E7) can a one-parameter service-response model
  predict R(n) at held-out and unmeasured levels? Also: build a leak test that can separate leakage from
  contention for MI300X.
- **Design and how it was run:**
  - **E6 step 1:** `scripts/amd/cubw/e6_kernels.py --trace n304=<dir> --trace n200=<dir>` on the graph-mode
    kernel traces from 487341 and 489594 (Qwen32B B1/C4096). Keeps dispatches inside the `decode` NVTX ranges and
    aggregates per kernel.
  - **E7:** `scripts/amd/cubw/e7_model.py --cohort … --e4 <auto tables> --e2 <production tables>`. Model
    R(n) = (1 − s) + s·P_x(n), where P_x is the native E4 penalty of one control x. Training: Qwen32B production
    at n = 232, 152, 96. Held-out: Qwen32B n = 200 and Qwen14B at every level. Blind: Qwen32B n = 272, 248, 216,
    184 (P_x interpolated). Acceptance: |obs − pred| ≤ 0.02 + CI half-width for **every** held-out and blind point.
  - **Leak test v2 smoke:** MI210 `devel` job 492951 with `svc_bench_v4 --mode cuprobe` on CUs 52–103.
- **Results — E6 kernels:**

| kernel | share of decode kernel time (MI300X / MI325X) | per tick | n200/n304 time |
|---|---:|---:|---|
| `wvSplitK_hf_sml_<bf16,64,2,16,8,2,1>` | 83.5% / 84.0% | 257 (64 layers × 4 GEMMs + lm_head) | 1.034 / 1.059 |
| `paged_attention_ll4mi_QKV_mfma16` | 4.4% | 64 | 1.073 / 1.112 |
| reshape/cache, RMS norm, SiLU, attention reduce | ~1.7–2.1% each | 64 | 0.95–1.03 |

`wvSplitK` workgroups are 64 threads with 64 KiB of LDS (all of a CU's LDS), so at most one is resident per CU.
vLLM sizes it with `num_compute_units()`, which returns 304 even under `HSA_CU_MASK`; the grid is 19456 = 64 × 304
at every mask. `scripts/sitecustomize.py` adds a hook: when `CUBW_CU_COUNT_OVERRIDE` is set, vLLM sizes
`wvSplitK` with that count instead (`e6_decode`).

- **Results — E7:**

| revision | cohort | selected control | s | held-out within tolerance |
|---|---|---|---:|---|
| r1 (unconstrained) | MI300X | `chase_c4864` (a latency row) | −1.30 | 1/5 |
| r1 | MI325X | `stride4` | 2.91 | 2/5 |
| r2 (throughput rows, 0 ≤ s ≤ 1) | MI300X | `seqread4g` | 0.50 | 4/5 (miss Qwen14B n = 152, +0.035) |
| r2 | MI325X | `seqread4g` | 0.92 | 4/5 (miss Qwen14B n = 96, −0.035) |

Blind predictions (registered before the extension jobs): MI300X 272/248/216/184 = 1.013/1.023/1.026/1.039;
MI325X = 1.025/1.043/1.047/1.082.

- **Results — leak v2 smoke (MI210):** native read and FMA co-runners on 0–51 flagged 0 CUs (min q 0.95); a
  positive-control probe on 0–59 (sharing 8 CUs) flagged **8** (q 0.46 ×4, 0.64–0.66 ×4).
- **What it means:** one kernel dominates decode and its launch geometry ignores the mask. r1 was rejected: three
  training points cannot choose among 22 controls, and s outside [0, 1] has no meaning. r2 was revised **after**
  seeing r1, so its held-out checks on existing data are not blind; only the fine-level predictions are. Even r2
  cannot produce n = 200 faster than 232. The leak v2 probe is sensitive to shared CUs and specific against
  contention, so its rule was pre-registered as `leak_ok_v2`:
  - r_c = per-CU probe rate beside X / alone (same GPU, same probe mask); q_c = r_c / median_c(r_c).
  - Every run must cover all 304 − n probe CUs.
  - Positive control must show ≥ 4 CUs with q_c < 0.85, else the test is insensitive and fails.
  - Pass iff every decoder run has min_c q_c ≥ (lowest min_c q_c of native read/FMA co-runners) − 0.05.
- **Failures and caveats:** kernel-trace timings carry the trace distortion (E5 slowdown 1.05–1.09); level
  ratios share it. Raw-data readback on 2026-11-01: all data files in every cubw root match `SHA256SUMS`; only
  logs appended after hashing differ. The off-cluster copy was not done (needs an owner-chosen destination).
- **Files:** `docs/amd/cubw_extension_plan_20261101.md`; `results_published/amd_20261101_cubw_extension_prep/`
  (`e6_kernels_mi300x.json`, `e6_kernels_mi325x.json`, `mi210_cuprobe_smoke_job492951/`);
  `results_published/amd_20261101_cubw_e7_model/` (`e7_{mi300x,mi325x}_{r1,r2}.json`); scripts `e6_kernels.py`,
  `e7_model.py`, `svc_bench_v4.hip`.

### #17 MI325X extension (v1.3) — job 492957

- **At a glance:** `mi3258x` / 8× MI325X / k002-001 / 2026-11-02 07:35–08:44 CST (1 h 09 min) / release de2baecb
  (v1.3), tables re-read with reader v1.4 / complete; 8 counter acquisitions failed and were preserved.
- **Question:** is n = 200 < 232 part of a larger pattern (finer levels)? Does `wvSplitK` explain it? Does sizing
  the kernel for the mask help? Do the E7 blind predictions hold? (Leak v2 here is a cross-check only; MI325X
  E3-minimal was already released.)
- **Design:** plan `$WORK/cubw_plans/mi325x_auto_20261101_155837_de2baecb.json` (235 steps):
  - Pilot: placement at 304/272/248/232/216/200/184/152 on all 8 GPUs (per XCD 38/34/31/29/27/25/23/19); leak v2
    suite on L0 at n = 232 and 152.
  - Gate.
  - Production: **E2-fine** Qwen32B, 8 blocks, level order randomized independently per GPU; **`e6_decode`**
    (`CUBW_CU_COUNT_OVERRIDE=n`) at 7 levels on blocks 3–7 (5 blocks, same GPUs as E2-fine); kernel traces at all 8
    levels plus the sized kernel at 232/200; E6 shape controls (`e6_shapes.py`) ×3 per level; pre-MALL counters on
    the shapes (`FETCH_SIZE`, `TCC_EA0_RDREQ`, `TCC_EA0_RDREQ_LEVEL`, `GRBM_GUI_ACTIVE`) at 304/232/200/152; eager
    decode counters (`FETCH_SIZE`, `TCC_EA0_RDREQ_LEVEL`) at 304/200.
- **How it was run:** `submit.sh auto mi325x 02:30:00 --extension`. Node k002-001 was down from 11:48 on Nov 1, so
  it ran the next morning. Pilot 07:36–07:44, gate 07:44, production 07:44–08:43.
- **What was measured and how it was analysed:** R(n) by block bootstrap; per-kernel time ratios from traces;
  mean outstanding HBM reads = `TCC_EA0_RDREQ_LEVEL / GRBM_GUI_ACTIVE` (Little's law; from separate runs);
  `e6_decode` vs the same-block E2 304 cell, with its own output gate.
- **Results — gates:** placement, lifecycle, outputs (66/66, 1 digest) pass; **`leak_ok_v2` True** at both
  levels; `release_e3_minimal_v2` True (no E3 planned); clock fails (gfx 1450–2087 MHz, mem 1257–1500 MHz).
- **Results — E2-fine and the sized kernel:**

| n | 304 | 272 | 248 | 232 | 216 | 200 | 184 | 152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ms/step | 15.58 | 17.39 | 16.83 | 16.39 | 15.98 | 15.90 | 15.71 | 17.77 |
| R(n), stock kernel | 1 | 1.116 | 1.080 | 1.052 | 1.026 | 1.021 | 1.008 | 1.141 |
| 95% CI | | 1.088–1.144 | 1.056–1.100 | 1.029–1.071 | 1.003–1.044 | 0.998–1.039 | 0.986–1.027 | 1.116–1.162 |
| `wvSplitK` time ratio | | 1.162 | 1.125 | 1.082 | 1.049 | 1.047 | 1.035 | 1.203 |
| R(n), kernel sized for n | | 1.056 | 1.187 | 1.108 | 1.210 | 1.255 | 1.238 | 1.332 |

- **Results — counters (shape controls, ratio of outstanding reads to n = 304):** FETCH_SIZE equals the weight
  bytes at every level and variant; RDREQ = 8 × FETCH_SIZE. Outstanding reads drop with n, and more for the
  kernel sized for n (e.g. `gate_up` sized for 304: 0.76 at 200, 0.78 at 152; sized for n: 0.63, 0.57).
- **Results — leak v2:** n = 232: 72 probe CUs, native min q 1.000/1.000, positive control flagged 8, decoder min q
  1.000 (threshold 0.950) → True. n = 152: native min q 0.885/0.881, positive 8, decoder 0.890 (threshold 0.831) → True.
- **Results — E7 r2 blind check (s = 0.916):** 272 pred 1.025 / obs 1.116 (miss); 248 1.043 / 1.080 (ok); 216
  1.047 / 1.026 (ok); 184 1.082 / 1.008 (miss). **Not validated on MI325X.**
- **What it means (as concluded at the time):**
  - The 200-vs-232 anomaly is part of a **non-monotone curve**: worst near full, lowest at 184, sharp rise at 152.
    Each GPU in its own random order gives the same per-level time, so it is per level, not drift.
  - `wvSplitK` alone reproduces the curve; other kernels are flat or slightly faster.
  - Sizing the kernel for the mask removed the "second round" of workgroups but made decode **slower at every
    level** (CIs exclude the stock R except at 272), with bit-identical outputs (35/35). The review read this as
    decode GEMV being limited by memory-level parallelism, not by HBM bandwidth.
  - A monotone model cannot fit a non-monotone curve, hence the E7 failure.
- **Later revision (stated explicitly):** the conclusion "the kernel sized for n is slower at every level" was
  **revised** by MI350X job 497082 (`docs/amd/mi350x_followup_review_20261103.md`) and MI300X job 499334
  (`docs/amd/mi300x_se_review_20261105.md`). Every count it resized to (272, 248, 232, 216, 200, 184, 152) leaves
  the shader engines with unequal CU counts. With unequal active CUs per SE, the dispatcher serializes two
  full-LDS workgroups on some CUs. Sizing to a **shader-engine-balanced** count (largest multiple of 32 ≤ n) gave
  MI300X decode R 0.993–0.998 down to 128 of 304 CUs (499334). The "slower at every level" result was this
  dispatch artefact, not a bandwidth limit. MI325X itself was never retested: 499334 says MI325X (also gfx942,
  304 CUs) "is presumably the same as MI300X but untested".
- **Failures and caveats:**
  - 4/4 eager-decode counter runs failed: `hipErrorLaunchFailure` (one `hipErrorIllegalAddress`), then rocprofv3
    aborted with "retired dangling correlation IDs". rocprofv3 warns this node's CP firmware (177) is below its
    minimum (186). Firmware is out of scope.
  - 4/16 shape-counter runs failed (n304 RDREQ SIGTERM after a hang; n200 FETCH_SIZE and RDREQ SIGABRT; n232
    RDREQ_LEVEL rc = 1). Shape counters are one acquisition each, ±5–8%.
  - Absolute R differs from 489594 (R(232) 1.052 vs 1.074; R(152) 1.141 vs 1.187): different days, unpinned clocks.
  - **Reader v1.4** fixed the `e6_decode` label (A.7). The in-job `analysis/` tables (reader v1.3) are left as written.
  - Unexplained at the time: paged attention faster at 272 than 304 but 10–13% slower at 248–184.
- **Files:** `docs/amd/cubw_mi325x_extension_review_20261102.md`; `docs/amd/compute_bandwidth_followup_results_20261103.md`;
  `results_published/amd_20261102_mi325x_cubw_v1_3_auto_job492957/` (`cubw_tables.json`, `e6_kernels_all_levels.json`,
  `e6_counters_by_variant.json`, `e7_mi325x_r2_blindcheck.json`); raw
  `$WORK/results_amd/amd_20261102_mi325x_cubw_v1_auto_job492957` (757 files); reader v1.4 commit 4f346b44.

### #18 MI300X extension (v1.3, incl. E3 and leak v2) — job 492956

- **At a glance:** `mi3008x` / 8× MI300X / k002-002 / 2026-11-03 08:08–09:21 CST (1 h 13 min) / release de2baecb
  (v1.3), read with reader v1.4 / complete; two eager counter steps timed out and were preserved.
- **Question:** the MI300X counterpart of #17, plus the key MI300X gap: with a leak test that can tell leakage
  from contention, can E3-minimal finally run on MI300X?
- **Design:** plan `$WORK/cubw_plans/mi300x_auto_20261101_155830_de2baecb.json` (285 steps). As #17, with two
  differences: leak v2 suites on **two** GPUs (L0, L1), and **E3-minimal** (Qwen32B, n = 232/152,
  absent/mem50/mem100/arith100, 5 blocks on lanes L0–L4) gated by `release_e3_minimal_v2`.
- **How it was run:** `submit.sh auto mi300x 02:45:00 --extension`. Pilot 08:08–08:17, gate 08:17 (all of
  placement, lifecycle, outputs, `leak_ok_v2`, `release_e3_minimal_v2` true), production 08:17–09:19.
- **What was measured and how it was analysed:** as #17, plus E3 T(n, z) and contrasts as in #14.
- **Results — gates and execution:** 267 of the 269 steps with a return code had rc = 0 (the other 16 journal
  records are leak-v2 co-run entries in a different format). Outputs 108/108, 1 digest. Placement exact.
  `leak_ok_v2` True: positive control flagged 8 of 8 shared CUs and the decoder 0, at 232 and 152 on both GPUs
  (e.g. L0 n = 152: native min q 0.887/0.857, decoder 0.856, threshold 0.807). `leak_ok_v11` false (no v1.1 cells in
  this plan). Clock fails: pooled gfx 1294–1959 MHz (×1.51), mem 1069–1300 MHz.
- **Results — E2-fine and the sized kernel (Qwen32B; 304 = 17.54 ms/step):**

| n | 272 | 248 | 232 | 216 | 200 | 184 | 152 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R, stock (8 blocks) | 1.068 | 1.053 | 1.033 | 1.019 | 1.017 | **1.010** | 1.085 |
| 95% CI | 1.064–1.071 | 1.048–1.056 | 1.030–1.036 | 1.016–1.023 | 1.015–1.019 | 1.007–1.013 | 1.081–1.090 |
| `wvSplitK` time ratio | 1.111 | 1.072 | 1.051 | 1.034 | 1.033 | 1.031 | 1.130 |
| R, sized for n (5 blocks) | 1.043 | 1.114 | 1.075 | 1.127 | 1.158 | 1.141 | 1.202 |

`wvSplitK` is 304 workgroups × 64 KiB LDS and 82.4% of decode kernel time. 232 and 200 agree with 487341 (1.024,
1.012) within about 1%.

- **Results — E3-minimal (decoder on n, competitor on 232–303, 5 blocks, ratio to absent at same n):**

| n | arithmetic 100% | memory 50% | memory 100% |
|---|---|---|---|
| 232 | 1.054 [1.046, 1.062] (10.9 TFLOP/s) | 1.106 [1.095, 1.118] (540 GB/s) | 1.246 [1.226, 1.265] (1082 GB/s) |
| 152 | 1.044 [1.038, 1.050] (11.8 TFLOP/s) | 1.113 [1.104, 1.121] (632 GB/s) | 1.267 [1.248, 1.286] (1253 GB/s) |

Contrasts: mem100/arith100 = **1.18 [1.16, 1.20] at 232** and **1.21 [1.20, 1.23] at 152**.
T(152, absent)/T(232, mem100) = **0.846 [0.832, 0.862]**; T(152, absent)/T(232, mem50) = 0.953 [0.942, 0.965].
(CIs from the job's `summary.md`.)

- **Results — counters and E7:** FETCH_SIZE and RDREQ are identical across 304/232/200/152 for each GEMM shape; the
  extra time shows up as busy cycles at constant bytes. E7 r2 blind (seqread4g, s = 0.50): 272 pred 1.013 / obs
  1.068 (miss), 248 1.023 / 1.053 (miss), 216 1.026 / 1.019 (ok), 184 1.039 / 1.010 (miss) → 1/4, and Qwen14B
  n = 152 had already missed. **Not validated on MI300X.**
- **What it means (as concluded at the time):**
  - MI300X has the same non-monotone R(n) as MI325X, set by `wvSplitK`.
  - "The resized kernel is slower at every level", as on MI325X.
  - With isolation now proven by leak v2, **memory contention, not CU count, is what hurts a co-located decoder**
    on MI300X: a 152-CU decoder alone is 15% faster than a 232-CU decoder beside a memory-100% competitor. Same
    direction as MI325X (1.14/1.16, 0.924), a bit larger. This is shared-service balance, not a stock-GPU win.
  - A smooth bandwidth-scaling model (E7) cannot produce the kernel-geometry curve.
- **Later revision (stated explicitly):** "the resized kernel is slower at every level" was **revised** by
  499334 (MI300X, `docs/amd/mi300x_se_review_20261105.md`) and, on MI350X, 497082
  (`docs/amd/mi350x_followup_review_20261103.md`). All resize targets in this job (272, 248, 232, 216, 200, 184,
  152) are shader-engine-unbalanced (e.g. 272 → 9,9,8,8 CUs per SE; 152 → 5,5,5,4). Unbalanced counts make the
  dispatcher serialize two full-LDS workgroups on some CUs. Sizing `wvSplitK` to the balanced count (multiple of
  32: 288, 256, 224, 192, 128) gave R 0.993–0.998 at every level from 288 down to 128 CUs. The stock
  non-monotone curve at reduced n is the same dispatch artefact (the stock 304-workgroup grid serializes 16–176
  workgroups under every reduced mask). The pre-registered H_SE prediction for the full device was refuted on
  MI300X: at 304 the dispatcher places 9,10,10,9 workgroups per SE with no serialization. Results report v2
  (`compute_bandwidth_followup_results_20261103.md` §1 and §3) carries the same revision note.
- **Failures and caveats:** two eager `FETCH_SIZE` counter steps (n = 304 and 200) hit their timeout (rc = −15) and
  are kept as failed acquisitions; the eager `RDREQ_LEVEL` steps completed. All timings
  `observational_clock_confounded`. Kernel-trace slowdown in this job was 1.09–1.11 (stock), 1.16 and 1.26 with
  the sized kernel at 232/200 (`summary.md`).
- **Files:** `docs/amd/compute_bandwidth_followup_results_20261103.md` §2–§3 (the MI300X extension has no separate
  review doc); `results_published/amd_20261103_mi300x_cubw_v1_3_auto_job492956/` (`summary.md`, `gates.json`,
  `cubw_tables.json`, `e6_kernels_all_levels.json`, `e6_counters_by_variant.json`, `e7_mi300x_r2_blindcheck.json`);
  raw `$WORK/results_amd/amd_20261103_mi300x_cubw_v1_auto_job492956` (1001 files hashed); release
  `$WORK/cubw_release/de2baecbfb1eed2da0bf091de2eabec3869a57ca`.

---

## Part C. What the two cohorts show together (as of 492956, with later revisions marked)

| finding | MI300X | MI325X | status |
|---|---|---|---|
| B1 decode tolerates a large CU cut (stock kernel) | R ≤ ~1.02 at 216–184; 1.07–1.09 at 152 | R ≤ ~1.03 at 216–184; 1.14–1.19 at 152 | observational |
| R(n) is non-monotone and tracks `wvSplitK` | yes (492956) | yes (492957) | explained by SE-unbalanced dispatch (499334, MI300X) |
| Kernel resized to n | slower at every level | slower at every level | **revised**: balanced sizing gives R 0.993–0.998 to 128 CUs on MI300X (499334); MI325X untested |
| Memory competitor hurts more than arithmetic | 1.18 / 1.21 (leak v2 passed) | 1.14 / 1.16 (leak v1.1 passed) | observational |
| Fewer CUs + uncontended HBM beat more CUs + contended HBM | 0.846 | 0.924 | shared-service balance, not a stock-GPU win |
| Batched decode and prefill are negative regimes | B8 +17%, prefill +42% at 200 | B8 +29%, prefill +44% at 200 | observational |
| Masking saves board energy | no (13.83 vs 13.78 J/token at the cap) | no (15.39 vs 15.89) | observational |
| E7 smooth model | not validated (1/4 blind) | not validated (2/4 blind) | refuted as predictive |
| HBM bandwidth per CU (B_CU, peak) | 17.4 GB/s | 19.7 GB/s | `results_published/COHORTS.md` |

No result here is clock-qualified, so none is a controlled noninferiority or architecture claim. Clock and power
changes were never authorized.

## Part D. Facts not recorded

- The exact `submit.sh` command lines are not logged. The ones shown are reconstructed from the plan contents
  (kind, cohort, boundary, `e4_done_in`, extension flag), the time limits in the experiment log and the
  `CUBW_GATES` lines in `RUN_STATUS.log`.
- The scheduler record of how and when 481168 was cancelled is not in the docs; only "cancelled before it ran".
- Allocation units actually charged per cubw job are not recorded in the reviews (only plan estimates and the
  E0 snapshot of 286.38 / 450).
