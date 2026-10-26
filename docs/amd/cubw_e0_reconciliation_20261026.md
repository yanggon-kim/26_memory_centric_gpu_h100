# cubw_v1 E0 — reconciliation, capability audit and adapters (26 October 2026)

E0 of [the compute/bandwidth follow-up](compute_bandwidth_followup_20261026.md). This note contains no new GPU
measurements. GPU-side E0 items (environment, counter list, source/model manifests) are produced at the start of
every cubw job in its campaign root (`environment.json`, `source_manifest.json`, `case_manifest.json`,
`launch_plan.json`, `preflight.json`).

## 1. Current scheduler and result state (checked 2026-10-26 14:10–16:30 CDT)

| Item | Actual state | Disposition |
|---|---|---|
| MI325X job 470176, attempt 3 (Oct 25 17:57 – Oct 26 05:57, k002-001) | All phase-1 lane commands ended rc=0 by 21:20; then TIMEOUT. Phases 2–4 never started (the phase-level `wait` also waited for the node monitor). | Raw root `$WORK/results_amd/amd_20261025_mi325x_full_job470176`; rc=0 is command completion only — not yet read or qualified. Not published. |
| 470176 attempts 1–2 (Oct 20, Oct 23) | NODE_FAIL; partial phase-1 roots kept (`amd_20261020_…`, `amd_20261023_…`). | Superseded by attempt 3 for phase 1; kept as failures. |
| Driver fix | `scripts/amd/mi325x_full.sbatch`: lane waits exclude the monitor; `PHASES` selection (commit d013176). | Dry run and a CPU reproduction of the hang were checked; not yet exercised on a node. |
| MI325X resume job 481072 (phases 2–4, TP/J6) | PENDING (Priority) on mi3258x. | Not interrupted or duplicated. Its results stay a separately labelled MI325X extension. |
| MI300X campaign 468885 | Complete; raw run JSONs on cluster storage (`results/amd_20261018_mi300x`: m1 51, m6 187, j6 72, m2 122, m8 53 run.json; m4 as service.jsonl). | Historical; not rewritten. |
| Allocation | 286.38 / 450 units used (submission filter, 2026-10-26). Observed rates: ~8 units per MI300X node-hour, ~12.8 per MI325X node-hour. | Budget table in §5. |

## 2. Corrections carried forward (verified against the published summaries)

- M1: 16 production triplets plus one single-acquisition smoke (17 conditions). Clock envelope passes **2/16**
  production triplets; the published "2/17" counted the smoke. `qualified_acquisitions` in the old reader means
  completed acquisitions, not clock-qualified ones.
- M6: 19/56 cells pass individually; no pooled cross-condition clock comparison was demonstrated.
- The old competitor summary averages every window overlapping any trial without duration weighting (and can
  count a window once per overlapping trial); it gives no ratio CIs or clock gates. The new reader's fixture
  reproduces the difference (old 200 GB/s vs duration-weighted 120 GB/s on overlapping windows).
- FETCH_SIZE/WRITE_SIZE are pre-MALL; runtime Triton kernels were excluded; latency-bound residuals mix graph T
  with eager traffic; the Mistral graph classification kept one of three ticks. These remain corrections to the
  old report, which is not edited; a new versioned report will carry the follow-up results.

## 3. Audit of existing components (reuse vs. new adapter)

| Component | Finding | Use in cubw_v1 |
|---|---|---|
| `controlled_vllm_experiments.py` | Graph timing, one-token prefill barrier, exact-output gates, per-trial realtime windows, env record incl. `HSA_CU_MASK`/`RESEARCH_ALLOWED_PIDS`. | Reused unchanged (also used by pending job 481072). |
| `preflight_node.py`, `env.sh`, `vllm_in_container.sh`, `gpu_monitor.py` | Working on both SKUs. | Reused unchanged; monitor cadence requested 0.05 s, actual cadence measured in analysis. |
| `lane_cumask.sh` | Fixed 304/232/152/96; no 200; no placement proof (FMA rate only). | Replaced by plan steps + `svc_bench_v3 --mode placement`. |
| `lane_competitor.sh` | 232/72 split only; no start proof, no stopped event, kill without verification. | Replaced by `cubw.py` competitor lifecycle (started event, SIGTERM→SIGKILL, PID-gone and residual-GPU-process checks). |
| `summarize_competitor.py`, `summarize_timing.py` | Unweighted overlap; completion conflated with qualification; per-cell clock flags only. | Replaced by `cubw/analyze.py` (old readers untouched). |
| `svc_bench.hip` (v2 binary) | read/strided/gather/chase/fma/mfma/mixed; competitor windows realtime only. | Copied to `cubw/svc_bench_v3.hip`: `read_ilp`, `placement`, `idle`, monotonic stamps, lifecycle events. v2 binary unchanged. |
| `lane_counters.sh`, `analyze_counters.py` | Eager-mode counters, Triton excluded. | E5 runs graph-mode counters through `cubw.py` (one group per run, markers for decode ranges, Triton included in a pilot attempt). |
| `classify_kernels.py`, `latency_bounds.py` | Heuristic/conditional. | Not used for new claims. |
| `mi325x_full.sbatch` | Old campaign driver. | Not used for this assignment. |

## 4. New versioned adapters (`scripts/amd/cubw/`, cubw_v1)

- `svc_bench_v3.hip` → `$WORK/bin/svc_bench_gfx942_v3` (sha256 3dd152be…). Smoke-tested on an MI210 devel node
  (gfx90a build, 2026-10-26): placement counted 104/104 CUs unmasked and 52/52 under `0:0-51`; read_ilp, chase
  (host-replay check passed), competitor SIGTERM produced `started`/`stopped` events with rc=0, idle mode emits
  zero-launch windows. gfx942 behaviour is validated only by the pilot.
- `plan.py` — immutable launch plans with seeded randomized orders, gates, retry policy, timeouts and budget.
- `cubw.py` — lane executor and per-attempt journal; command completion separate from scientific status;
  at most two attempts and only for execution/tool failures.
- `analyze.py` — gates, E2 block-paired noninferiority, E3 T(n,z) and contrasts, duration-weighted competitor
  joins, E4 curves, E5 coverage, E8 tables; `gate` writes `gates.json` for bounded release.
- `test_analyze.py` — CPU fixtures: status/correctness separation, nested-trial bootstrap units, missing clocks,
  coverage gaps, pooled-vs-cell clock gate, overlapping windows, gate logic. Result: all checks pass
  (17/17, 2026-10-26), after one fixture repair (the first overlapping-window fixture coincidentally gave equal
  old/new means; the realtime-step check read clipped stamps — the reader now records the offset before clipping).
- `manifest.py`, `counter_probe.py`, `cubw.sbatch`, `submit.sh` (frozen release snapshot per commit; dry run
  enumerates every step command without a GPU or scheduler job).

## 5. Launch dispositions and budget (estimates from MI300X per-acquisition durations: Qwen32B ~140 s, Qwen14B ~105 s)

| Job | Content | Est. node-h | Est. units |
|---|---|---:|---:|
| MI300X pilot (mi3008x) | E1 placement on all 8 GPUs, mask-syntax revalidation, competitor lifecycle/alone references, leak probes, Qwen32B/Qwen14B/bridge E2 pilots, n=232/152 E3 pilots, timeline/counter pilots, E4 pilot, E8 pilots | ~0.9 | ~7 |
| MI300X production (mi3008x, after pilot review) | 8 randomized complete blocks (Qwen32B ×5, Qwen14B ×3) of E2 + released E3 cells, E4 ×3, E5, E8 | ~2.7 | ~22 |
| MI325X cohort (mi3258x, `afterany:481072`, held until the MI300X pilot validates the tools) | auto: reduced pilot stage → `gates.json` → production stage (Qwen32B ×3, Qwen14B ×3 blocks) | ~3.0 | ~39 |
| MI325X job 481072 (already queued) | old phases 2–4 | ~4–5 | ~55–65 |

Release rules are predeclared in every plan (`gates`): E3 full grid only if placement, lifecycle, outputs, leak
and the pooled clock gate pass; otherwise the minimal observational subset (Qwen32B, n=232/152,
absent/mem50/mem100/arith100) if the non-clock gates pass; E2 runs observationally if only clocks fail.
