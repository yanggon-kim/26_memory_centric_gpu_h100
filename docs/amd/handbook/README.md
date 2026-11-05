# AMD experiment handbook — start here

**Next assignment (prepared 2026-11-05):** read [goal-based follow-up G1–G8](../goal_based_followup_20261105.md) for the next research goals and delivery contract. Pulling/reading does not authorize execution; the owner will order the experiments separately. This handbook describes the historical evidence below and adds no measurements for that next assignment.

This handbook explains every AMD experiment on branch `amd/mi300x-phase0`, from the first MI300X campaign
(2026-10-18) to the MI300X shader-engine study (2026-11-05). It covers 28 experiments, numbered #1–#28 in the
order they happened.

The existing review documents in `docs/amd/` are short and assume you already know the story. This handbook
spells the story out. It does not replace those documents, and it adds no new data. Every number here was copied
from a file in the repository, and the source is named next to it. When a fact was not recorded anywhere, the
handbook says "not recorded".

## 1. How to read this handbook

**If you are a new agent, read in this order:**
1. Sections 2–4 of this file: the question, the hardware and the vocabulary.
2. Section 5: how one experiment is run, from plan to commit. Read this before you touch any script.
3. Section 6: the master index. Use it to find the experiment you need.
4. Section 7: what we currently believe, and which job showed it. Many early conclusions were later revised.
   Section 7 lists the revisions, so you do not repeat a superseded claim.
5. Then the per-experiment chapter you need:

| file | experiments | what it covers |
|---|---|---|
| [01_early_campaigns.md](01_early_campaigns.md) | #1–#3, #5, #6 | first MI300X campaign (M1–M8, J0–J9, TP, J6 competitor), MI325X comparison, compute-matched and full MI325X campaigns |
| [02_cubw_mi300x_mi325x.md](02_cubw_mi300x_mi325x.md) | #7–#9, #11–#15, #17, #18 | the "compute vs bandwidth" (cubw) follow-up E0–E8 on MI300X and MI325X, v1 to v1.3 |
| [03_mi350x.md](03_mi350x.md) | #21–#24 | MI350X port (v1.4), MI350X campaign, shader-engine follow-up (v1.5) |
| [04_v16_followups.md](04_v16_followups.md) | #25–#28 | v1.6: leak test v4 + E3 on MI350X; shader-engine study on MI300X |
| [05_minor_checks.md](05_minor_checks.md) | #4, #10, #16, #19, #20 | failed attempts, MI210 smoke tests, the MI350X toy job, failed MI350X smokes |

Each chapter uses the same headings for every experiment: at a glance, question, design, how it was run, what
was measured, results, what it means, failures and caveats, files.

**Path conventions used everywhere:**
- Repository root: `/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100` (called "the repo").
- `$WORK` = `/work1/proj_blaisetine/yanggon` (`scripts/amd/cubw/cubw.sbatch`, `docs/AMD_PORT_PLAN.md`).
- Raw data: `$WORK/results_amd/<run>/` for jobs from 470176 onward. The first three jobs keep raw data under
  `<repo>/results/<run>/` (`docs/amd/artifacts.md`). Raw data is read-only. It is not in git.
- Compact, tracked evidence: `<repo>/results_published/<folder>/`.
- Links in this file are relative to `docs/amd/handbook/`.

## 2. The research question and the AMD proxy

### 2.1 The question (plan 2.0)

The governing plan is `docs/shoreline_aware_research_plan_2.0.md` (plan 2.0). `docs/AMD_PORT_PLAN.md` quotes its
question:

> Can some GPU compute resources be reallocated to on-chip SRAM, to external-memory PHY/controller capacity and
> delivered HBM bandwidth, or to both, to improve LLM decode latency/throughput and energy per token under explicit
> silicon, shoreline, package, thermal and cost constraints?

In plain terms:
- LLM decode at small batch reads every weight from memory once per generated token. It is usually limited by
  memory, not by arithmetic.
- A chip has a fixed area and a fixed edge ("shoreline") where memory interfaces (HBM PHYs) can sit.
- So: would a chip that spends less area on compute and more on SRAM, or on more HBM bandwidth, decode faster or
  with less energy per token?

Real silicon cannot be resized. The AMD work measures how decode responds when compute is taken away or when
memory bandwidth is taken away on real GPUs. Those measurements feed the later modelling stages of plan 2.0.
Nothing measured here is a physical architecture result (`docs/amd/shoreline_opportunity_report.md`, labels).

### 2.2 The AMD proxy

- **Take compute away:** restrict a process to n of the GPU's compute units (CUs) with the `HSA_CU_MASK`
  environment variable. Then time decode. The ratio R(n) = T(n) / T(full device) says how much the lost compute
  costs.
- **Take bandwidth away at fixed compute:** run an owned "competitor" process on other CUs of the same GPU that
  streams HBM (or does arithmetic, as a control). This is experiment J6 in the first campaigns and E3 in the
  cubw follow-up.
- **Measure the memory system directly:** microbenchmarks (`svc_bench_v3`..`v6.hip`) for streaming bandwidth,
  latency and how many reads are in flight, and hardware counters (`rocprofv3 --pmc`) for bytes moved.
- **Compare SKUs:** MI300X and MI325X have the same compute and different HBM. MI350X has fewer CUs and much more
  HBM bandwidth per CU. Each SKU is a separate cohort (Section 3).

Why AMD: MI300X has 8 HBM stacks, a 256 MB memory-side Infinity Cache (MALL) and per-process CU masks. CUDA green
contexts on H100/H200 could not restrict graph kernels; `HSA_CU_MASK` can (`docs/AMD_PORT_PLAN.md`, "Why AMD").

### 2.3 Branch rules

- Work **only** on branch `amd/mi300x-phase0`. Check with `git branch --show-current` before you edit anything.
- **Never** commit to, push to, rebase or edit branch `codex/h200-phase01`. Another person owns it. Plan 2.0 was
  copied from that branch (`docs/AMD_PORT_PLAN.md`); copying text from it is fine, changing it is not.
- H100/H200/RTX evidence on other branches is not AMD evidence and must not be relabelled as such
  (`docs/AMD_PORT_PLAN.md`).
- The remote is `origin` (`git remote -v`). Push with `git push origin amd/mi300x-phase0`.

## 3. Hardware cohorts

A **cohort** is one GPU model. Every result belongs to exactly one cohort. Cohorts are never pooled. A comparison
across cohorts is allowed only at matched per-XCD CU fractions and must be labelled
(`results_published/COHORTS.md`).

Values below come from `results_published/*/cohort.json` (written by `scripts/amd/cohort_index.py`) and from the
`HW` table in `scripts/amd/cubw/plan.py`.

| | MI300X | MI325X | MI350X |
|---|---|---|---|
| architecture | gfx942 | gfx942 | gfx950 |
| CUs (full device) | 304 | 304 | 256 |
| XCDs (compute dies) | 8 | 8 | 8 |
| CUs per XCD | 38 | 38 | 32 |
| shader engines (SEs) per XCD | 4 (plan.py `se_per_xcd`) | 4 (plan.py; not verified by a probe) | 4 (plan.py) |
| CUs per SE at the full device (XCD 0) | 9, 10, 10, 9 (499333, 499334) | not measured | 8, 8, 8, 8 (497082) |
| LDS per CU | 64 KiB | 64 KiB | 160 KiB |
| HBM capacity | 192 GiB | 256 GiB | 288 GiB |
| HBM peak bandwidth | 5.3 TB/s | 6.0 TB/s | 8.0 TB/s |
| measured streaming read at full device (E4 `seqread4g`) | 4563.5 GB/s (483498) | 4950.1 GB/s (483499) | 6416.2 GB/s (496074) |
| B_CU peak (HBM peak / CUs) | 17.4 GB/s | 19.7 GB/s | 31.2 GB/s |
| B_CU measured (seqread / CUs) | 15.0 GB/s | 16.3 GB/s | 25.1 GB/s |
| partitions | `mi3008x` (8 GPUs), `mi3001x` (1 GPU) | `mi3258x` (8 GPUs) | `mi3508x` (8 GPUs), `mi3501x` (1 GPU) |
| allocation units per node-hour, as used by plan.py | 8.0 | 12.8 | 11.2 |

Notes:
- The 9,10,10,9 layout follows from 38 CUs per XCD over 4 SEs. Contiguous masks fill the XCDs in turn and then the
  SEs of each XCD (`plan.py` `balanced()` docstring, `SE_LEVELS` comment).
- MI325X has the same layout on paper (same gfx942 die, 304 CUs). No probe has measured it; see Section 8.
- Charge multipliers per partition change over time. Check `sinfo -s` and the cluster docs, not this table, before
  you estimate cost.

## 4. Glossary

Terms are grouped. Each entry gives where the term is defined in the code or docs.

### 4.1 Hardware and software

| term | meaning |
|---|---|
| **CU** (compute unit) | AMD's basic compute block. NVIDIA's equivalent is the streaming multiprocessor (SM). A CU runs workgroups and issues memory requests. |
| **XCD** (accelerator complex die) | One compute die. MI300X, MI325X and MI350X each have 8. Each XCD has its own L2 cache. |
| **SE** (shader engine) | A group of CUs inside one XCD, with its own workgroup dispatcher. 4 per XCD here (`plan.py`). |
| **LDS** (local data share) | Fast on-CU scratch memory shared by one workgroup, the counterpart of CUDA shared memory. 64 KiB per CU on gfx942, 160 KiB on gfx950. A workgroup that uses all of it occupies the whole CU. |
| **MALL** / Infinity Cache | 256 MB memory-side cache in front of HBM on MI300X/MI325X. No counters for it are exposed (`docs/amd/experiment_log.md`, tool findings). |
| **HBM** | High-bandwidth memory stacks attached to the GPU package. |
| **workgroup (WG)** | A block of threads launched together onto one CU (CUDA: thread block). The grid is the set of workgroups of one kernel launch. |
| **`HSA_CU_MASK`** | ROCm environment variable that restricts a process to a set of CUs, written `<physical GPU index>:<lo>-<hi>`. Findings from job 468885: the index is the physical, not the HIP-visible, GPU index; comma lists are ignored; ranges work; CU indices interleave across the 8 XCDs; a mask that leaves an XCD empty is ignored (`docs/amd/experiment_log.md`). |
| **contiguous per-XCD mask** | A mask `0-(n-1)` with n a multiple of 8. Because indices interleave across XCDs, it gives every XCD n/8 CUs. All decoder masks are of this form (`cubw.py` `decoder_mask`). The competitor gets the top range, e.g. `232-303` on MI300X, `192-255` on MI350X. |
| **wvSplitK** | vLLM's ROCm "skinny GEMM" kernel (`wvSplitK_hf_sml_...`). It carries 81–84% of decode kernel time in every cohort. Each workgroup uses the full LDS, so one workgroup fits per CU. vLLM sizes its grid with the device CU count (304 or 256) even under a mask (`docs/amd/cubw_extension_plan_20261101.md` §1; `docs/amd/mi350x_smoke_review_20261102.md`). |
| **`CUBW_CU_COUNT_OVERRIDE`** | Environment variable read by `scripts/sitecustomize.py` inside the container. It makes vLLM size `wvSplitK` for a chosen CU count. Used by the `e6_decode` and `e6_balanced` cells (`cubw.py` docstring, v1.3). |
| **graph mode vs eager mode** | vLLM can record the decode step as a HIP graph and replay it (graph mode, the primary timing mode), or launch each kernel from Python (`enforce_eager`, eager mode). Graph mode is faster and is the baseline; eager mode is used where profilers crash in graph mode (`docs/AMD_PORT_PLAN.md`; `cubw_pilot_review_20261028.md`). |
| **TP** (tensor parallelism) | Splitting each layer across N GPUs (TP2/TP4/TP8). Owner-authorized on 2026-10-18 as a labelled extension (`docs/AMD_PORT_PLAN.md`, "Multi-GPU extension"). |
| **B1/C4096** | Batch 1, context 4096 tokens. The primary cubw workload is Qwen2.5-32B B1/C4096 graph mode (`plan.py` `CASE`). |

### 4.2 Measurement and statistics

| term | meaning |
|---|---|
| **R(n)** | T(n) / T(full), where T is decode time per step (HIP-event ms) on n CUs and "full" is 304 (MI300X/MI325X) or 256 (MI350X). R > 1 means slower. Defined in `docs/amd/compute_bandwidth_followup_20261026.md` §6. |
| **noninferiority** | The pre-declared test that a reduced-CU level is "not worse": the upper end of the 95% CI of R(n) must be ≤ 1.05, **and** every source, output, isolation, mask and clock gate must pass (`NONINF = 1.05` in `analyze.py`). No AMD result has passed it, because the clock gate failed in every job. |
| **trial** | One timed repetition inside one process. 10 timed trials per acquisition after ≥ 3 warmups; 64 generated tokens = 63 decode steps (`docs/AMD_PORT_PLAN.md`, statistics). |
| **acquisition** | One independent process run of one cell. Its value is the mean of its trials. At least 3 per quantitative cell. Trials are never treated as independent (`analyze.py` docstring). |
| **block** | One GPU lane holding one acquisition of every cell of a comparison, in randomized order. Ratios are paired within a block. CIs come from a bootstrap that resamples blocks, then trials inside each acquisition: 10,000 draws, seed 20261018 (`analyze.py`). |
| **cell** | One condition: model, CU level n, treatment, and kernel sizing. |
| **lane** | The ordered list of plan steps that runs on one GPU (`L0`..`L7`). One `cubw.py lane` process per GPU. A one-GPU smoke has one lane. |
| **pilot / production / auto** | Plan kinds (`plan.py --kind`). *pilot*: one acquisition per cell, labelled pilot, never promoted to a result. *production*: randomized complete blocks. *auto*: one allocation that runs a pilot stage, computes gates (`analyze.py gate`), then runs the production steps those gates release. `--pilot minimal` or `reduced` picks the size of an auto plan's pilot stage. |
| **smoke** | A one-GPU pilot (`--smoke`, on `mi3001x` or `mi3501x`) run before an 8-GPU job. It releases nothing. |
| **gate** | A pass/fail check computed by `analyze.py gates_from()` and written to `gates.json`. Production steps carry `requires: <release gate>` and are skipped (`not_run`) if it is false. Gates are listed in 4.3. |
| **disposition** | The scientific status of a cell (`analyze.py disposition()`): `qualified` (all gates and the pooled clock gate pass, ≥ 3 good acquisitions), `observational_clock_confounded` (everything passes except the clock gate), `unavailable` (too few acquisitions, or a placement/isolation/lifecycle gate failed), `failed` (an output/correctness failure), `not_run`. A return code of 0 is command completion, not qualification. |
| **clock confounding** | MI300X-class GPUs change their clocks with load: fewer active CUs run at higher gfx clock (e.g. 1650 → 1950 MHz for 7B B1 in 468885). The pre-declared clock gate requires the pooled max/min of gfx and memory clocks over the compared trial windows to be ≤ 1.05. It failed in every job (pooled gfx ratio 1.28–1.58). Clock and power changes are not authorized, so every timing comparison is **observational, clock-confounded**. |
| **B_CU** | HBM bandwidth per CU = HBM bandwidth / CUs in use. By Little's law each CU must keep B_CU × memory latency bytes of reads in flight. Introduced in `compute_bandwidth_followup_results_20261103.md` (key-concept box). |
| **H_SE** | The shader-engine hypothesis, pre-registered in `docs/amd/mi350x_followup_plan_20261103.md`: when a mask gives the SEs of an XCD unequal CU counts, the dispatcher puts two full-LDS `wvSplitK` workgroups on some CUs, they run one after the other, and the kernel slows. Supported on MI350X (497082). On MI300X, refuted as pre-registered because the full device does not serialize, although reduced unbalanced masks do (499334). |
| **balanced(n)** | The largest CU count ≤ n that gives every SE of every XCD the same CU count: n − n mod 32 on all three cohorts (8 XCDs × 4 SEs; `plan.py balanced()`). Examples: balanced(232) = 224, balanced(304) = 288. |
| **P1–P3, P1m–P3m** | The scored predictions of H_SE: P1 dispatch probe, P2 GEMM-shape timing, P3 decode R. "m" marks the MI300X versions (`followups_v16_plan_20261104.md` §B). |

### 4.3 Gates (from `analyze.py gates_from()`)

| gate | passes when |
|---|---|
| `placement_ok` | Every lane's placement probe proved every decoder level and the competitor mask, with an exact per-XCD balance. |
| `lifecycle_ok` | Every owned competitor started, was stopped, was verified gone, covered ≥ 95% of the trial windows, and left no residual GPU process. False by construction in a plan with no competitor runs (499334). |
| `outputs_ok` | Every attempted, unprofiled timing run completed, and all runs produced one output digest (exact greedy tokens). Unattempted steps are ignored from v1.2 on. |
| `clock_ok` | The pooled clock envelope (≤ 1.05) passed over all compared E2 trials. |
| `leak_ok` (v1) | The competitor's FMA probe keeps ≥ 0.85 of its alone rate during decoder trials. |
| `leak_ok_v11` | That ratio is ≥ 0.85 × the gentlest placement-proven native co-runner's ratio. |
| `leak_ok_v2` / `_v3` / `_v4` | The per-CU leak tests below pass on every tested GPU and level. |
| `release_e2` | placement_ok and outputs_ok. |
| `release_e5e8` | outputs_ok. |
| `release_e3_minimal` | placement, lifecycle, outputs and leak_ok_v11. |
| `release_e3_minimal_v2` / `_v3` / `_v4` | placement, lifecycle, outputs and the matching leak gate. |
| `release_e3_full` | release_e3_minimal and clock_ok. Never true. |

### 4.4 Leak tests

A leak test asks: does the masked decoder run any work on the competitor's CUs? If it did, E3 would measure stolen
CUs rather than memory contention. Each version was added because the previous one failed or was blind.

| version | first used | probe | rule | outcome |
|---|---|---|---|---|
| v1 | 481167 | FMA probe on the competitor CUs; rate beside the decoder / rate alone | ≥ 0.85 | failed on MI300X; cannot tell CU leakage from chip-wide contention (`cubw_pilot_review_20261028.md`) |
| v1.1 | 483498, 483499 | same, compared with native read/FMA co-runners on the decoder mask | decoder ratio ≥ 0.85 × gentlest native ratio | MI300X fail (0.77–0.82); MI325X pass (0.87–0.97) |
| v2 | 492951 (MI210), 492956, 492957 | `svc_bench_v4 cuprobe`: one full-LDS probe workgroup per CU, completions counted per physical CU | positive control (probe overlapping 8 CUs) must flag ≥ 4 CUs with q < 0.85; pass if every decoder min q ≥ lowest native min q − 0.05 (`cubw_extension_plan_20261101.md` §2) | passed on MI300X and MI325X at 232 and 152; insensitive on MI350X (positive control 0 of 8; 496073) |
| v3 | 497081, 497082 | `svc_bench_v5 cuprobe_t`: per-CU workgroup residency (duration) | same rule as v2 | insensitive on MI350X: on gfx950 the intruder, not the probe, loses the shared CU |
| v4 | 499331, 499332 | `svc_bench_v6 gapprobe`: start-delay probe, one 16-wave full-LDS workgroup per CU, 200 µs host gaps; late fraction per CU | flag CU if excess late fraction ≥ 0.10; required positives (full-LDS probe and the decoder itself on 0..n+7) must flag ≥ 4 (`followups_v16_plan_20261104.md` §A) | sensitive; decoder flags 0; passes 16/16 on MI350X |

### 4.5 Experiment families and labels

| label | meaning | defined in |
|---|---|---|
| **M1–M8** | The eight required motivation measurements of the common contract: M1 baselines and timelines, M2 logical vs physical bytes, M3 per-kernel evidence, M4 service microbenchmarks, M5 cache-history pressure, M6 CU-allocation sweep, M7 service under the M6 allocations, M8 energy and prefill counterexamples. | `docs/shoreline_hopper_measurements.md` §3; AMD mapping in `docs/AMD_PORT_PLAN.md` |
| **A–F** | Six additional controls: A dependent-latency pointer chase, B compute service, C memory service, D concurrent compute+memory, E layer timing and instrumentation controls, F profiler pass-layout and model validation. | `docs/shoreline_hopper_measurements.md` §4 |
| **J0–J11** | The job order of the contract, adapted to AMD: J0 inventory, J1–J3 baselines/bytes/kernels, J4 service, J5 CU sweep, J6 same-GPU competitor, J7 retention, J8 layer timing, J9 compute/SRAM/bandwidth sensitivity, J10 energy and negative regimes, J11 traces. Campaign 468885 ran J0, the M1–M8 and A–D families, E/F + J9 (analytical) and J6 (`shoreline_opportunity_report.md` §1). | `docs/h200_shoreline_jobs.md`; `docs/AMD_PORT_PLAN.md` "Phased execution" |
| **E0–E8** | The cubw follow-up: E0 reconciliation and adapters (CPU), E1 correctness/mask/clock/instrumentation pilots, E2 fine CU noninferiority boundary, E3 decoder CUs × shared-service competitor, E4 service curves at the same CU levels, E5 graph traffic and timelines, E6 kernel efficiency (which kernel sets R(n); kernel resized), E7 service-response model with held-out checks, E8 guardrails (prefill, B8, long context, energy). | `docs/amd/compute_bandwidth_followup_20261026.md` §4–§10 |
| **E2-fine** | E2 at the finer levels 304/272/248/232/216/200/184/152 (v1.3). | `plan.py` `FINE` |
| **E3-minimal** | The pre-declared E3 subset: Qwen32B at two decoder levels (232 and 152 on MI300X; 192 and 128 on MI350X) with competitor absent, memory 50%, memory 100% and arithmetic 100%. | `plan.py` `E3_MINIMAL` |
| **`e6_decode` / `e6_balanced`** | Decode with `wvSplitK` sized for n (`e6_decode`) or for balanced(n) (`e6_balanced`). | `plan.py`, `analyze.py` `SIZED` |
| **cubw versions v1–v1.6.1** | Code versions of `scripts/amd/cubw/`. Each repair is described in `experiment_log.md` and in the docstrings of `plan.py`, `cubw.py`, `analyze.py`. | see table below |
| **release commit** | The git commit a job ran from. `submit.sh` exports it with `git archive` into a read-only snapshot `$WORK/cubw_release/<sha>/` and the job runs only from there, so later edits cannot change a queued job. | `scripts/amd/cubw/submit.sh` |

**cubw versions and their release commits** (from `experiment_log.md` and the reviews):

| version | release commit | what changed | jobs |
|---|---|---|---|
| v1 | 2da9ae35 | first release (E0) | 481167, 481168 (cancelled) |
| v1.1 | 79d8b597 | per-GPU residual check, orphan sweep, leak v1.1, graph-mode counters dropped | 483498, 483499 |
| v1.2 | c7cf8aa9 (487341), 99bcd1ad (489594) | outputs gate ignores unattempted steps; `--e4-done-in` | 487341, 489594 |
| v1.3 | de2baecb | leak v2, E2-fine, E6 controls and counters | 492956, 492957 |
| v1.4 | 8ba209a8 → 36198d84 → e7116c1f | MI350X port; two port fixes found by smokes | 496056, 496059, 496060, 496073, 496074 |
| v1.5 | 09178c91 | leak v3, dispatch probe, `e6_balanced`; cohort index | 497081, 497082 |
| v1.6 | f737d1b1 | leak v4 (`gapprobe`), MI300X SE study | 499331–499334 |
| v1.6.1 | 8b648879 | `dispatch_suite` `tag` fixes the L0/L1 cell collision | none yet |

## 5. How an experiment is run, end to end (cubw pipeline)

This is the pipeline for every cubw job (#8 onward). The first campaigns (#1–#6) used older per-lane scripts
(`scripts/amd/lane_*.sh`, `scripts/amd/mi325x_full.sbatch`); see chapter 01.

### 5.1 Cluster rules you must follow

These come from the cluster's terms of use (`/etc/claude-code/CLAUDE.md`) and from the project's own pre-declared
rules (`compute_bandwidth_followup_20261026.md` §3, `followups_v16_plan_20261104.md` §C).

- **Login node:** never run anything on its GPUs, not even a one-kernel test. Use at most 4 cores (`-j4`). Check
  with `hostname -s` (contains `login`) or an empty `$SLURM_JOB_ID`. Plan generation, dry runs, analysis and
  publishing are CPU-only and belong on the login node.
- **8-GPU partitions:** use every GPU on the node. cubw plans have one lane per GPU (8 lanes).
- **Every submission** needs a partition, a node or task count and a time limit. `submit.sh` sets `-p`, `-t`, and
  `cubw.sbatch` sets `-N 1`.
- **No clock, power, firmware or driver changes.** Fixed-clock qualification is therefore unavailable.
- **At most 2 attempts per step,** and a retry only for a diagnosed execution or tool failure, never for a
  correctness failure or to get different timings (`cubw.py`). Every attempt keeps its own directory.
- **Pre-register before data.** Hypotheses, predictions, thresholds and the plan are committed before the job
  runs (e.g. `mi350x_followup_plan_20261103.md`, `followups_v16_plan_20261104.md`).
- **Never relax a gate after seeing data.** A failed prediction is recorded as failed, even by 0.00002 (499334 P3m
  (b)).
- **Cohorts are never pooled.**
- **Preserve failed acquisitions.** Nothing is deleted or overwritten; failures stay in the raw root and are
  reported.
- **Smoke first:** each 8-GPU job is submitted held and released only after its one-GPU smoke is reviewed
  (`followups_v16_plan_20261104.md` §C).
- **AMD acknowledgement** in any presentation or paper: "This work was supported in part by Advanced Micro
  Devices, Inc. under the AMD University Program's AI & HPC Cluster".

### 5.2 The steps

**Step 0 — write and commit the pre-registration.** A plan document in `docs/amd/` (e.g.
`followups_v16_plan_20261104.md`) states the question, levels, predictions, thresholds and job list. Commit it,
and any code change, before submitting. `submit.sh` refuses to run if `scripts/` has uncommitted changes.

**Step 1 — build the binaries (login node, CPU only).** `cubw.sbatch` aborts unless
`$WORK/bin/svc_bench_<arch>`, `_v2`, `_v3` and `_v4` exist, and it exports `CUBW_BENCH`..`CUBW_BENCH6` pointing
at `$WORK/bin/svc_bench_<arch>_v3`..`_v6`. `<arch>` is `gfx942` (MI300X/MI325X) or `gfx950` (MI350X). The
binaries are compiled from `scripts/amd/cubw/svc_bench_v*.hip` with `hipcc --offload-arch=<arch>` (the exact build
command line is not recorded in the repo; hashes are in `$WORK/bin/svc_bench.sha256`).

**Step 2 — generate the plan, dry-run and submit: `scripts/amd/cubw/submit.sh`.**

```
scripts/amd/cubw/submit.sh <pilot|production|auto> <mi300x|mi325x|mi350x> <HH:MM:SS> [plan.py args] [-- sbatch args]
```

What it does, in order:
1. Checks `git diff --quiet HEAD -- scripts`, takes `sha = git rev-parse HEAD`.
2. Creates the release snapshot `$WORK/cubw_release/<sha>/` with `git archive` (scripts plus the follow-up
   handoff doc), writes `RELEASE_COMMIT`, makes it read-only.
3. Runs `plan.py --kind <kind> --cohort <cohort> --out $WORK/cubw_plans/<cohort>_<kind>_<date>_<sha8>.json
   [plan args]` from the snapshot and makes the plan read-only.
4. Dry-runs every lane: `DRY_RUN=1 CUBW_PLAN=... CUBW_RELEASE=... bash cubw.sbatch`. This writes
   `$WORK/results_amd/dryrun_cubw_<cohort>_<kind>_<date>/DRYRUN_COMMANDS.txt` without touching a GPU.
5. Reads the partition from the plan and calls
   `sbatch -p <partition> -t <time> -J cubw-<cohort>-<kind> --export=ALL,CUBW_PLAN=...,CUBW_RELEASE=...[,CUBW_GATES=...] <sbatch args> cubw.sbatch`.

Real examples (from the `submit.sh` header, `experiment_log.md` and `followups_v16_plan_20261104.md` §C):

```
scripts/amd/cubw/submit.sh pilot mi300x 01:45:00                                   # 481167
scripts/amd/cubw/submit.sh auto mi325x 03:45:00 -- --dependency=afterany:481072    # 481168
scripts/amd/cubw/submit.sh auto mi300x 02:45:00 --pilot minimal --boundary 200      # 483498 style
CUBW_GATES=/path/gates.json scripts/amd/cubw/submit.sh production mi300x 01:45:00 --boundary 200 --e4-done-in <483498 root>
scripts/amd/cubw/submit.sh auto mi300x 02:45:00 --extension                        # 492956 (v1.3)
scripts/amd/cubw/submit.sh pilot mi350x <time> --smoke                             # 496073 (v1.4, mi3501x)
scripts/amd/cubw/submit.sh auto mi350x 03:45:00 --campaign                         # 496074
scripts/amd/cubw/submit.sh pilot mi350x <time> --followup --smoke                  # 497081
scripts/amd/cubw/submit.sh auto mi350x <time> --followup                           # 497082
scripts/amd/cubw/submit.sh pilot mi350x <time> --leak4 --smoke                     # 499331
scripts/amd/cubw/submit.sh auto mi350x <time> --leak4                              # 499332
scripts/amd/cubw/submit.sh pilot mi300x <time> --se-study --smoke                  # 499333 (mi3001x)
scripts/amd/cubw/submit.sh auto mi300x <time> --se-study                           # 499334
```

The flags are real (`plan.py` argparse: `--kind`, `--cohort`, `--pilot`, `--boundary`, `--e4-done-in`,
`--extension`, `--smoke`, `--campaign`, `--followup`, `--leak4`, `--se-study`, `--out`). The exact time limits and
the exact command lines of most jobs are not recorded; where a review gives the time limit, chapter files quote
it. To hold a job behind its smoke, pass an sbatch hold after `--` and release it later with `scontrol release
<job>` (the docs say jobs were "held"; the exact flag used is not recorded).

**Step 3 — the batch job: `scripts/amd/cubw/cubw.sbatch`.** It runs from the release snapshot, never from the
live repo.
1. Creates a fresh root `$WORK/results_amd/amd_<date>_<cohort>_cubw_v1_<kind>_job<id>/` and a symlink under
   `<repo>/results/`. Note: the raw root always says `cubw_v1_<kind>`; the published folder name carries the real
   version (e.g. raw `..._cubw_v1_auto_job499334` → published `..._cubw_v1_6_se_job499334`).
2. Copies the plan to `launch_plan.json` and, if given, `CUBW_GATES` to `gates.json`. Copies configs.
3. Preflight: `scripts/amd/preflight_node.py --expect-gpus <lanes> --expect-name <GPU> --expect-arch <arch>
   --expect-cus <full>`. Then `manifest.py` writes `environment.json`, `source_manifest.json`,
   `case_manifest.json`.
4. Sets a deadline (job end minus the plan's `reserve_s`) so no step starts that cannot finish.
5. Starts `gpu_monitor.py` (clock and power samples every 0.05 s into `node_monitor.jsonl`). The monitor is
   never a lane-completion prerequisite.
6. For each stage in the plan: for an `auto` plan, before each non-pilot stage it runs
   `analyze.py gate --root <root> --out <root>/gates.json`. Then it starts one `cubw.py lane` per GPU in parallel
   and waits for the lanes only.
7. Stops the monitor, runs `analyze.py report --root <root>` (writes `analysis/summary.md` and
   `analysis/cubw_tables.json`), and writes `SHA256SUMS` for every file in the root.

**Step 4 — the lanes: `scripts/amd/cubw/cubw.py lane --plan <plan> --root <root> --lane L<i> --stage <stage>`.**
- Runs that lane's steps in their randomized plan order. Step kinds: `vllm` (one decoder acquisition in the vLLM
  container under a CU mask, optionally beside an owned competitor, optionally under `rocprofv3`), `native`
  (`svc_bench` microbenchmark or probe), `shell` (recorded command), `competitor_only`.
- A step with `requires: <gate>` is journalled `not_run` if that gate is false in `gates.json`.
- Each attempt gets `<root>/<step out>/attempt1/` (and `attempt2/` on a diagnosed execution failure). The
  directory must not exist (`mkdir(exist_ok=False)`); this is what caught the 499334 L0/L1 cell collision.
- Every attempt is appended to the journal `<root>/journal/<lane>.jsonl`.
- Owned processes run in their own process groups, carry a per-attempt `CUBW_STEP_TAG`, and are verified gone by
  PID and tag.

**Step 5 — review (login node).** Read `RUN_STATUS.log`, `gates.json`, `analysis/summary.md` and the journals.
Re-run the reader if needed: `python3.12 scripts/amd/cubw/analyze.py report --root <raw root> --out <dir>`.
CPU-only extra analyses: `e6_kernels.py --trace n304=<attempt dir> --trace n200=<attempt dir> --out <json>`,
`e6_counters.py <root> <out.json>`, `e7_model.py`. Unit fixtures: `scripts/amd/cubw/test_analyze.py`. Judge the
pre-registered predictions exactly as written.

**Step 6 — publish compact evidence.** Copy into `results_published/amd_<date>_<cohort>_cubw_v<ver>_<campaign>_job<id>/`
the files a review needs: `summary.md`, `cubw_tables.json`, `gates.json`, `launch_plan.json`, plus any extra
judgment files (e.g. `hse_judgment.json`, written by `scripts/amd/cubw/judge_se_mi300x.py`, and `e6_kernels_se.json`,
written by `e6_kernels.py`, in 499334; commit 8b648879). No dedicated
publish script for cubw folders was found; `scripts/amd/publish_results.py` served the first campaigns. Raw data
stays in `$WORK`.

**Step 7 — index cohorts.** `python3.12 scripts/amd/cohort_index.py --repo .` (CPU only). It writes
`results_published/<folder>/cohort.json`, `results_published/cohorts.json` and `results_published/COHORTS.md`.
The cohort comes from `launch_plan.json` or the folder name. Do not edit `COHORTS.md` by hand.

**Step 8 — write the review and the log entry.** A review `docs/amd/<topic>_review_<date>.md` (gates, each
prediction with its verdict, tables, caveats, the acknowledgement), and a dated entry at the end of
`docs/amd/experiment_log.md`. If a conclusion changes, mark the old statement in the results report as revised
and say by which job; do not delete it (see `compute_bandwidth_followup_results_20261103.md`).

**Step 9 — commit and push** on `amd/mi300x-phase0` only: `git add` the review, log, published folder, cohort
index and any script fix; `git commit`; `git push origin amd/mi300x-phase0`.

## 6. Master index of all 28 experiments

Paths: methodology and results docs are in `docs/amd/` unless they start with `docs/`. Published folders are in
`results_published/`. "—" means none exists. Dates are run dates where recorded, otherwise the folder or log date.

| # | name (job) | GPU / partition | date | one-line purpose | handbook | methodology doc | results doc | results_published folder |
|---|---|---|---|---|---|---|---|---|
| 1 | MI300X first campaign (468885) | 8× MI300X, `mi3008x` | 2026-10-18 | J0 inventory, M1–M8, A–D, TP2/4/8, J6 competitor: where decode time goes and what CU masks cost | 01 | `docs/AMD_PORT_PLAN.md` | `shoreline_opportunity_report.md`, `experiment_log.md` | `amd_20261018_mi300x` |
| 2 | MI325X comparison (468949) | MI325X, `mi3258x` | 2026-10-18 | M4 full-CU service sweep and 8 graph baselines on the higher-bandwidth SKU | 01 | `docs/AMD_PORT_PLAN.md` | `shoreline_opportunity_report.md` | `amd_20261018_mi325x_job468949` |
| 3 | MI325X compute-matched (469084) | MI325X, `mi3258x` | 2026-10-18 | 264 vs 304 CUs to match MI300X compute and isolate bandwidth (did not hold bandwidth fixed) | 01 | `docs/AMD_PORT_PLAN.md` ("Added 2026-10-18") | `shoreline_opportunity_report.md`, `experiment_log.md` | `amd_20261018_mi325x_matched_job469084` |
| 4 | MI325X full, attempts 1–2 (470176) | 8× MI325X, `mi3258x` | 2026-10-20, 2026-10-23 | first two tries of the full MI325X campaign; NODE_FAIL, kept as failures | 05 | `docs/AMD_PORT_PLAN.md`, `scripts/amd/mi325x_full.sbatch` | `mi325x_full_campaign_results_20261028.md` §1, `cubw_e0_reconciliation_20261026.md` | — (raw `$WORK/results_amd/amd_20261020_…`, `amd_20261023_…`) |
| 5 | MI325X full phase 1 (470176 att. 3) | 8× MI325X, `mi3258x` | 2026-10-25/26 | repeat of the MI300X M1–M8 campaign on MI325X | 01 | as #4 | `mi325x_full_campaign_results_20261028.md` | `amd_20261025_mi325x_full_job470176` |
| 6 | MI325X phases 2–4 (481072) | 8× MI325X, `mi3258x` | 2026-10-28 | J6 competitor, TP2/4/8 full and compute-matched | 01 | as #4 | `mi325x_full_campaign_results_20261028.md` | `amd_20261028_mi325x_full_job481072` |
| 7 | E0 reconciliation (CPU) | none | 2026-10-26 | reconcile old results, build and test the cubw scripts | 02 | `compute_bandwidth_followup_20261026.md` §4 | `cubw_e0_reconciliation_20261026.md` | — |
| 8 | MI300X pilot (481167) | 8× MI300X, `mi3008x` | 2026-10-28 | first cubw pilot: placement, lifecycle, leak v1, clocks, E1/E2 pilot | 02 | `compute_bandwidth_followup_20261026.md` | `cubw_pilot_review_20261028.md` | `amd_20261028_mi300x_cubw_v1_pilot_job481167` |
| 9 | MI325X auto (481168, cancelled) | 8× MI325X, `mi3258x` | submitted 2026-10-26, cancelled 2026-10-28 | v1 auto job held behind the pilot; cancelled unrun | 02 | `compute_bandwidth_followup_20261026.md` | `cubw_pilot_review_20261028.md` §4, `experiment_log.md` | — |
| 10 | MI210 smoke (483497) | MI210, `devel` | 2026-10-28 | check v1.1 co-runner start/stop and tag sweep | 05 | `cubw_pilot_review_20261028.md` §3 | `cubw_pilot_review_20261028.md` | — |
| 11 | MI300X auto (483498) | 8× MI300X, `mi3008x` | 2026-10-30 | v1.1 minimal re-pilot → gates → production (blocked by a reader defect; only E4 ran) | 02 | `cubw_pilot_review_20261028.md` §4 | `cubw_auto_review_20261030.md` | `amd_20261030_mi300x_cubw_v1_auto_job483498` |
| 12 | MI325X auto (483499) | 8× MI325X, `mi3258x` | 2026-10-31 | same on MI325X; same defect; E4 ran | 02 | `cubw_pilot_review_20261028.md` §4 | `cubw_mi325x_auto_review_20261031.md` | `amd_20261031_mi325x_cubw_v1_auto_job483499` |
| 13 | MI300X production (487341) | 8× MI300X, `mi3008x` | 2026-10-31 | E2 (Qwen32B, Qwen14B), E5 traces, E8 guardrails; E3 not released | 02 | `cubw_auto_review_20261030.md` | `cubw_mi300x_production_review_20261031.md`, `compute_bandwidth_followup_results_20261101.md` | `amd_20261031_mi300x_cubw_v1_production_job487341` |
| 14 | MI325X production (489594) | 8× MI325X, `mi3258x` | 2026-11-01 | E2, E3-minimal, E5, E8 | 02 | `cubw_mi325x_auto_review_20261031.md` | `cubw_mi325x_production_review_20261101.md`, `compute_bandwidth_followup_results_20261101.md` | `amd_20261101_mi325x_cubw_v1_production_job489594` |
| 15 | E6 kernel id + E7 model (CPU) | none | 2026-11-01 | find the kernel behind R(n); fit E7 and register blind predictions | 02 | `compute_bandwidth_followup_20261026.md` §10 | `cubw_extension_plan_20261101.md` §1, §3 | `amd_20261101_cubw_extension_prep`, `amd_20261101_cubw_e7_model` |
| 16 | MI210 leak v2 smoke (492951) | MI210, `devel` | 2026-11-01 | prove the per-CU probe is sensitive and specific | 05 | `cubw_extension_plan_20261101.md` §2 | `cubw_extension_plan_20261101.md` §2 | `amd_20261101_cubw_extension_prep/mi210_cuprobe_smoke_job492951` |
| 17 | MI325X extension (492957) | 8× MI325X, `mi3258x` | 2026-11-02 | v1.3: leak v2 cross-check, E2-fine, E6 sized kernel, shapes, counters, E7 blind check | 02 | `cubw_extension_plan_20261101.md` | `cubw_mi325x_extension_review_20261102.md` | `amd_20261102_mi325x_cubw_v1_3_auto_job492957` |
| 18 | MI300X extension (492956) | 8× MI300X, `mi3008x` | 2026-11-03 | v1.3: leak v2 → E3-minimal, E2-fine, E6, E7 blind check | 02 | `cubw_extension_plan_20261101.md` | `compute_bandwidth_followup_results_20261103.md` §2 | `amd_20261103_mi300x_cubw_v1_3_auto_job492956` |
| 19 | MI350X toy (496003) | 1× MI350X, `mi3501x` | 2026-11-02 | confirm MI350X access, gfx950 build, vector add | 05 | `mi350x_campaign_plan_20261102.md` | `mi350x_campaign_plan_20261102.md` (top) | — (raw `$WORK/mi350x_toy/`) |
| 20 | MI350X failed smokes (496056, 496059, 496060) | `mi3501x` ×2, `mi3508x` | 2026-11-02 | first v1.4 runs; found two port bugs | 05 | `mi350x_campaign_plan_20261102.md` | `mi350x_smoke_review_20261102.md` | — (raw `$WORK/results_amd/amd_20261102_mi350x_cubw_v1_*_job4960{56,59,60}`) |
| 21 | MI350X smoke (496073) | 1× MI350X, `mi3501x` | 2026-11-02 | one-GPU smoke of the v1.4 port | 03 | `mi350x_campaign_plan_20261102.md` | `mi350x_smoke_review_20261102.md` | `amd_20261102_mi350x_cubw_v1_4_smoke_job496073` |
| 22 | MI350X campaign (496074) | 8× MI350X, `mi3508x` | 2026-11-03 | E2, E4, E6, E8 on the third cohort | 03 | `mi350x_campaign_plan_20261102.md` | `mi350x_campaign_review_20261103.md`, `compute_bandwidth_followup_results_20261103.md` | `amd_20261103_mi350x_cubw_v1_4_auto_job496074` |
| 23 | MI350X follow-up smoke (497081) | 1× MI350X, `mi3501x` | 2026-11-03 | leak v3 and dispatch probe on one GPU | 03 | `mi350x_followup_plan_20261103.md` | `mi350x_followup_smoke_review_20261103.md` | `amd_20261103_mi350x_cubw_v1_5_followup_smoke_job497081` |
| 24 | MI350X follow-up (497082) | 8× MI350X, `mi3508x` | 2026-11-03 | test H_SE (P1–P3); leak v3 | 03 | `mi350x_followup_plan_20261103.md` | `mi350x_followup_review_20261103.md` | `amd_20261103_mi350x_cubw_v1_5_followup_job497082` |
| 25 | leak v4 smoke (499331) | 1× MI350X, `mi3501x` | 2026-11-04 | is the start-delay probe sensitive? | 04 | `followups_v16_plan_20261104.md` §A | `mi350x_leak4_smoke_review_20261104.md` | `amd_20261104_mi350x_cubw_v1_6_leak4_smoke_job499331` |
| 26 | leak v4 + E3 (499332) | 8× MI350X, `mi3508x` | 2026-11-04 | leak v4 on 8 GPUs, then E3-minimal on MI350X | 04 | `followups_v16_plan_20261104.md` §A | `mi350x_leak4_review_20261104.md` | `amd_20261104_mi350x_cubw_v1_6_leak4_job499332` |
| 27 | MI300X SE smoke (499333) | 1× MI300X, `mi3001x` | 2026-11-04 | dispatch probe and balanced decode on gfx942, one GPU | 04 | `followups_v16_plan_20261104.md` §B | `mi300x_se_smoke_review_20261104.md` | `amd_20261104_mi300x_cubw_v1_6_se_smoke_job499333` |
| 28 | MI300X SE study (499334) | 8× MI300X, `mi3008x` | 2026-11-05 | test H_SE on MI300X (P1m–P3m) at 10 levels | 04 | `followups_v16_plan_20261104.md` §B | `mi300x_se_review_20261105.md` | `amd_20261105_mi300x_cubw_v1_6_se_job499334` |

Cross-job reports: `interim_synthesis_20261028.md` (cross-run reading of #1–#8),
`compute_bandwidth_followup_results_20261101.md` (cubw results v1: #8–#14) and
`compute_bandwidth_followup_results_20261103.md` (v2: adds #17, #18, #22, with revision notes up to #28).

## 7. What we know now

Every timing claim below is **observational and clock-confounded** (the pooled clock gate failed in every job).
None is a qualified noninferiority result. Each claim names its cohort; cohorts are never pooled.

### 7.1 Current conclusions

| # | conclusion | cohort(s) | established by | source |
|---|---|---|---|---|
| C1 | Batch-1 decode streams every weight and KV byte once per step; L2→memory read bytes equal logical demand within 0.2–0.5%. Evicting L2 and the 256 MB MALL between tokens costs only 0.2–1.4%. | MI300X | 468885 | `shoreline_opportunity_report.md` bottom line 1–2 |
| C2 | CU trimming costs batch-1 decode little but prefill a lot: at 32% of CUs, B1 decode +9–24%, B8–64 decode +40–70%, prefill 2.1–2.3×. Power does not fall (clocks rise), J/token rises 10–19%. | MI300X | 468885 | same, lines 4–5 |
| C3 | With vLLM's stock kernels, R(n) is non-monotone on MI300X and MI325X: worst near full, lowest near 184, rising again at 152. MI300X 272: 1.068, 184: 1.010, 152: 1.085. MI325X 272: 1.116, 184: 1.008, 152: 1.141. | MI300X, MI325X | 492956, 492957 | `compute_bandwidth_followup_results_20261103.md` §2–§3 |
| C4 | On MI350X the stock R(n) is a step: 1.16–1.25 from 232 down to 128, and 1.86 at 80. It is larger than MI300X/MI325X at every matched per-XCD fraction. | MI350X | 496074 | `mi350x_campaign_review_20261103.md`; results v2 §3 |
| C5 | `wvSplitK` sets the shape of R(n). It is 81–84% of decode kernel time, uses full LDS (one workgroup per CU), and launches a grid sized for the full device under any mask. Its kernel-time ratio reproduces R(n) in every cohort. | all three | 487341/489594 traces (#15), 492957, 492956, 496073, 496074 | results v2 §3 finding 1 |
| C6 | A CU cut moves the same bytes (FETCH_SIZE and RDREQ unchanged) but fewer reads are in flight; time rises in step. | all three | 492957, 492956, 496074 | results v2 §3 finding 2 |
| C7 | **Mechanism of the stock penalty at reduced CUs:** when the SEs have unequal active CUs, the dispatcher puts two full-LDS workgroups on some CUs and they serialize. Sizing `wvSplitK` to balanced(n) removes this. | MI350X (H_SE supported), MI300X (reduced masks) | 497082, 499334 | `mi350x_followup_review_20261103.md`; `mi300x_se_review_20261105.md` §3 |
| C8 | With balanced sizing, MI350X decode R is 1.004 (232→224), 1.009 (208→192), 1.07 (184/168→160), 1.17 (152→128); stock is 1.18–1.23. | MI350X | 497082 | `experiment_log.md` 2026-11-03 |
| C9 | With balanced sizing, MI300X decode stays at full speed at every level tested: R 0.993–0.998 from 288 down to 128 of 304 CUs (42%). | MI300X | 499334 | `mi300x_se_review_20261105.md` §2 P3m, §3 |
| C10 | The full MI300X device does **not** serialize: the dispatcher places 9,10,10,9 workgroups per SE, matching the active CUs. Sizing for 288 at 304 gains at most 0.2% (0.9984, upper CI 1.00002). H_SE is refuted on MI300X as pre-registered. | MI300X | 499333 (smoke), 499334 | `mi300x_se_review_20261105.md` §2 |
| C11 | **B_CU threshold:** with the dispatch artefact removed, both cohorts decode at full speed up to about **42 GB/s of HBM bandwidth per CU** (MI300X at 128 CUs = 41.4 GB/s, R 0.993; MI350X at 192 CUs = 41.7 GB/s, R 1.009). MI350X cost rises above it: 50.0 GB/s → 1.067, 62.5 GB/s → 1.168. | MI300X, MI350X | 497082 + 499334 | `mi300x_se_review_20261105.md` §3 |
| C12 | **E3: memory contention, not CU loss, hurts a co-located decoder.** Memory-100% / arithmetic-100% competitor: MI300X 1.18 (232), 1.21 (152); MI325X 1.14, 1.16; MI350X 1.25 (192), 1.23 (128). Fewer CUs with uncontested HBM beat more CUs beside a memory-bound neighbour: 0.846 (MI300X), 0.924 (MI325X), 0.775 (MI350X). | all three, separately | 489594, 492956, 499332 | `mi350x_leak4_review_20261104.md` §3 |
| C13 | The masked decoder does not leak onto competitor CUs: leak v2 passed on MI300X/MI325X; leak v4 passed 16/16 on MI350X. So E3 contention goes through the shared memory system (HBM, MALL, fabric). | all three | 492956, 492957, 499332 | results v2 §2; leak4 review §2 |
| C14 | **E7 is not validated.** The smooth service-response model (R = (1 − s) + s·P_x) missed the blind points: 2/4 missed on MI325X, only 1/4 within tolerance on MI300X. A smooth bandwidth-scaling model cannot produce the kernel-geometry curve. | MI300X, MI325X | 492957, 492956 (fit in #15) | results v2 §2 |
| C15 | Guardrails (E8): a CU cut that looks free for one decode shape is not free for the rest of the mix. MI350X at 168 CUs: prefill, 16k-context, B8 and Mistral-7B 14–21% slower, ~9% more J/token. MI300X at n=200: B8 decode +17%, prefill +42%. | MI300X, MI350X | 487341, 496074 | `experiment_log.md`; results v2 §4 |
| C16 | Practical rule: size `wvSplitK` to the largest SE-balanced count ≤ the active CUs (a multiple of 32 on MI300X and MI350X). On MI300X the full device needs no change. | MI300X, MI350X | 497082, 499334 | `mi300x_se_review_20261105.md` §3 |

### 7.2 Revisions over time

| earlier claim | job(s) | revised by | current reading |
|---|---|---|---|
| "The resized kernel (`e6_decode`) is slower at every level" on MI300X/MI325X (MI300X 272 1.043 … 152 1.202); "on MI300X and MI325X, resizing never helped". | 492957, 492956 | 499334 | Every size tried (272, 248, 232, 216, 200, 184, 152) is SE-unbalanced. Resizing to a balanced count gives R 0.993–0.998 on MI300X. MI325X with balanced sizing is untested. |
| MI350X: the sized kernel is fast at 192 (R 0.989) but worse at 208/184/168/152 — pattern unexplained. | 496074 | 497082 | Explained by H_SE: 256/192/128 are the only SE-balanced levels. |
| "Leak test v2 is insensitive on MI350X, so E3 cannot run there." | 496073 | 497081/497082 (v3 also blind), 499331/499332 (v4) | Leak v4 is sensitive and passes; E3 ran on MI350X. |
| E3 direction shown on MI300X and MI325X only. | 489594, 492956 | 499332 | Same ordering in all three cohorts, largest on MI350X. |
| H_SE prediction that the full MI300X (9,10,10,9) serializes. | pre-registration `followups_v16_plan_20261104.md` §B | 499333, 499334 | Refuted; reduced unbalanced masks do serialize. |
| MI300X E3 blocked: leak v1.1 failed (0.77–0.82). | 483498 | 492956 (leak v2 passes) | E3-minimal ran on MI300X. |
| M1 clock gate "2/17" in the first report. | 468885 | E0 (#7) | 2/16 production triplets; the "2/17" counted a smoke. The old report is not edited. |
| The MI325X compute-matched pairing would isolate bandwidth. | 469084 | 469084 itself | 264 CUs also cut HBM stream bandwidth 9–11%, so it did not hold bandwidth fixed. |

## 8. Open items

| item | status | source |
|---|---|---|
| **Clock qualification** | Any noninferiority claim needs a run that passes the pooled clock gate (≤ 1.05). Clock and power changes are not authorized, so this is unavailable. Every result stays observational. | results v2 §5; `mi300x_se_review_20261105.md` §4 |
| **Off-cluster raw archive** | Raw data lives only in `$WORK` (not backed up) and `<repo>/results/`. A checksum readback was done; the off-cluster destination has not been named by the owner. | results v2 §5; `cubw_extension_plan_20261101.md` |
| **MI325X balanced sizing** | MI325X (gfx942, 304 CUs) is presumably the same as MI300X but no SE study or balanced-size decode has run on it. Its "resizing slower" result from 492957 is therefore unrevised by measurement. | `mi300x_se_review_20261105.md` §4 |
| **Leak v4 on-node spread** | On the 8-GPU node (499332) each positive control flagged the 8 shared CUs plus about 17 unshared ones (late starts > 5 µs in ~67% of launches); on the one-GPU slice (499331) only the 8. A likely cause is in-order dispatch; why it shows only on the 8-GPU node is not known. It changes no verdict. | `mi350x_leak4_review_20261104.md` §2 |
| Missing generator scripts | 497082's `hse_judgment.json` and `e6_inflight.json` were computed by ad-hoc CPU scripts that were not saved; their inputs are published, so they can be recomputed. 499334's `hse_judgment.json` is reproducible with `scripts/amd/cubw/judge_se_mi300x.py`. `e6_kernels_*.json` come from `e6_kernels.py`. | `03_mi350x.md`; `04_v16_followups.md` |
| v1.6.1 fix not yet exercised on a GPU | The `tag` fix for the dispatch-suite collision regenerates a plan with 371 unique cells and passes `test_analyze.py`; no job has run it. | `mi300x_se_review_20261105.md` §1 |

---

If any of this work is presented or published, use the AMD acknowledgement: "This work was supported in part by
Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
