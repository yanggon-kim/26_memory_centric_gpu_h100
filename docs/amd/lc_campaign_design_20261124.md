# Long-context and long-output chunked prefill on AMD — design and pre-registration

Version 1, 2026-11-24, committed before any data from the plans below. Handoff:
[long_context_chunked_prefill_20261124.md](long_context_chunked_prefill_20261124.md). Baseline and reused methods: the
November 18 campaign ([design](cp_campaign_design_20261118.md), [results](cp_campaign_results_20261118.md) at 99e80ef5).
Branch `amd/mi300x-phase0`. Code: `scripts/amd/lc/` (new) and `scripts/amd/cp/` (reused; backward-compatible additions
listed in section 10).

## 1. Platform, model and invariants

- MI300X (gfx942, 304 CUs), vLLM 0.23.0 V1 in the pinned Apptainer image, ROCm 7.14.1, one GPU per case, in-process
  engine core, synchronous scheduling (exec_s = model execution incl. sampling). No clock, power, driver or firmware
  change. Cohorts are never pooled. Failed acquisitions are kept; at most two attempts per step.
- Qwen2.5-32B-Instruct BF16 at the pinned revision (5ede1c97). Verified from the model's `config.json`:
  `max_position_embeddings` 32768, `use_sliding_window` false, no RoPE scaling. 64 layers, 8 KV heads, head dim 128:
  256 KiB of KV per token (1 GiB = 4096 tokens). The longest prompt is therefore **32000 tokens** ("32K"), leaving
  positions for generated tokens; replays at a 32000-token resident context use `max_model_len` 32768.
- Attention backend (recorded per run): ROCM_ATTN. Verified from the November 18 kernel traces: decode attention =
  `paged_attention_ll4mi_*` kernels; prompt/chunk attention = Triton `_fwd_kernel` (context attention over cached prefix
  + new tokens); KV write = `reshape_and_cache_kernel`; linear layers = hipBLASLt `Cijk_*` and skinny `wvSplitK*`.
  Smoke must confirm the same names at 32K and for a whole prompt; any new name is classified before main data.
- Graph policy unchanged (FULL_AND_PIECEWISE, capture ≤ 512 tokens); steps above 512 tokens run eagerly. Per-step graph
  path is derived from total tokens.
- Synthetic prompts: seeded uniform token ids; greedy, EOS ignored, fixed output lengths. Output digests recorded.
- Allocation: the plans below are about 4.3 (lc1) + 0.3 (smoke) units; lc2 and lc3 are estimated before submission.

## 2. Anchors (L0)

- **Step anchor**: experiment-1 cell S = 8192, N = 8, C ∈ {0, 256, 1024}, with the s1 settings (budget 4096,
  max_model_len 16384, seed offset 1, 10 reps). Pass if the paired-median T(8,0,8192) and the marginal
  T(8,C,8192) − T(8,0,8192) are each within 5% (or 1 ms) of `results_published/amd_20261118_mi300x_cp_s1`.
- **Serving anchor**: experiment-2 held-out dheavy, mixed B = 1024, Poisson 2 rps, seed 201, 150 requests. Pass if
  the pure-decode share is within 95.5–96.0% of steps (±0.5 points) and 87.9–89.5% of logged execution (±1.5 points).
- A failed anchor is reported and investigated before the dependent family is interpreted; it does not stop the plan.

## 3. L1 — cost of chunking one long prompt

- One request, no concurrent decoder, max_tokens 1, prefix caching off. Lengths L ∈ {1024, 8192, 16384, 24576, 32000}.
- Arms: whole (`enable_chunked_prefill` off, budget 32768, verified one step) and mixed budgets B ∈ {256, 1024, 4096}.
  All arms: max_model_len 32768, explicit KV budget 40 GiB, same token ids per (rep, L). One engine per arm.
- Clean timing: 8 reps + warmup. Per prompt: TTFT (harness), sum of exec_s, GPU event span, chunk count, tokens and
  start (earlier prefix) per chunk, first output token (cross-arm agreement reported).
- Kernel trace (2 reps): GPU time by class — linear, prefill attention, KV update, other — kernel-union busy time and
  host gap (exec − busy) per prompt.
- Counters (eager, 1 rep, L ∈ {8192, 32000}, arms whole / B 256 / B 1024; one counter per process): FETCH_SIZE,
  WRITE_SIZE, TCC_EA0_RDREQ_DRAM_sum, SQ_INSTS_VALU_MFMA_MOPS_BF16. Eager timing calibration (3 reps) without counters.
- Derived: chunked/whole ratio of TTFT and GPU busy time per L with paired bootstrap 95% CI (pairs = same prompt);
  logical revisit demand R = Σ_j start_j tokens × 256 KiB (for k equal chunks of c: c·k(k−1)/2 tokens) plus k weight
  reads (63.97 GB each) as logical demand; measured FETCH bytes beside it (labelled with the counter's observation
  point, section 6); linear-time ratio (smaller GEMMs) and host-gap ratio (launch/step overhead).
- Expectations (reported as confirmed or rejected): **H1a** B = 4096 stays within 1.10× of whole at every L.
  **H1b** B = 256 exceeds 1.5× whole at L ≥ 16384. **H1c** the excess of chunked over whole is attention-dominated at
  L ≥ 16384 (attention share of the excess ≥ 0.5) and linear/overhead-dominated at L = 1024.

## 4. L2 — decode attention vs prefill attention at long resident context

- Replay (`lc_replay.py`, extends cp_replay): N decoders at resident context S built outside measurement, then paired
  steps: decode-only reference T(N,0,S) and hybrid T(N,C,S,P) = N decodes + one chunk of C new tokens after a cached
  prefix of P tokens (verified per step: cached_at_admit = P, executed = C).
- Grid: S ∈ {1024, 8192, 16384, 24576, 32000}; N ∈ {1, 8, 32}, skipping cells above 300 000 resident tokens (N = 32 runs
  at S ≤ 8192); C ∈ {0, 256, 1024}; fixed P = 4096. Control: at S = 8192, N = 8, P ∈ {0, 4096, 16384, 28672}.
  Anchors N = 0 (chunk alone after the same P). 10 reps + warmup; budget 4096; max_model_len 32768.
- Kernel trace on every context, N ∈ {1, 8}, 2 reps. Classes: decode attention, prefill attention, KV update, linear,
  other; fused or unclassified time reported explicitly with coverage (classified kernel time / busy time).
- Outcomes: decode-attention ms per step and share of GPU time vs S and N·S; prefill-attention ms vs C and P;
  hybrid vs decode alone (marginal) and vs separate execution T(N,0,S) + T(0,C,P).
- Expectations: **H2a** decode-attention time is linear in N·S (R² ≥ 0.95 over the grid) and changes by < 10% when a
  chunk is added. **H2b** prefill-attention time depends on P and C, not on S (within 10% across S at fixed P, C).
  **H2c** hybrid < separate sum at every S, and the saving as a fraction of the separate sum falls as S grows.

## 5. L3 — where the pure-decode tail begins

**Part A (lc1).** Request A: 128-token prompt, O ∈ {512, 2048, 4096} output tokens; request B injected at the step
boundary after A's first token, I ∈ {0 (none), 128, 8192} fresh tokens, 256 output tokens. Policies (common
max_model_len 16384): whole/separate (budget 16384, no chunking), chunked/separate (B 2048), chunked/mixed B ∈ {256,
2048, 4096}. 3 reps + warmup. Kernel trace on O = 512 (GPU-active time per step type).
- Outcomes per (policy, O, I): pure-decode fraction of steps, of exec time and (profiled) of GPU-active time; the step at
  which fresh prefill work ends; A's inter-token latency around the injection; TTFT of B.
- **H3a**: for O ≥ 2048, pure decode is ≥ 90% of execution time under every policy, including I = 8192.

**Part B (lc2, after the pilot).** Matched open-loop traces (identical requests and timestamps across policies) for
lo512 / lo2048 / lo4096 (prompt 128–512, outputs 384–640 / 1536–2560 / 3072–5120), the balanced control and the L5
workloads; a fixed-rate slice across the three output lengths; overload marked separately (incomplete > 0). Outcomes:
pure-decode fraction by count and exec time, split into the arrival window and the post-arrival drain; effective P:D
from executed token sums; TTFT, ITL, completion, SLO goodput. Bursty repeat (CV 4) near the transition.
- **H3b**: when effective P:D ≤ 0.2, pure decode is ≥ 90% of execution time under every policy in the arrival window.

## 6. L4 — memory service in the remaining work

- **Observation point.** Prior audit (goal v1, G4): FETCH_SIZE (KiB) and TCC_EA0_RDREQ(_DRAM) count **pre-MALL,
  request-side** reads at the L2→fabric interface (a 128 MiB MALL-resident read still counts ≈ 1.0× useful bytes);
  WRITE_SIZE is streaming-consistent. No physical-HBM counter is validated. lc1 reruns the known-byte audit
  (`counter_audit.py`, same node and firmware); labels follow its result, never the counter name.
- **Counters on long-context cells** (eager, one counter per process): S ∈ {1024, 16384, 32000} × N ∈ {1, 8} × C ∈
  {0, 1024}, P = 4096. Per step: bytes at the observation point, logical demand (weights + KV + activations), MFMA work
  (useful FLOPs analytically: 2 × parameters × tokens + attention FLOPs), achieved request-side bandwidth = bytes /
  eager busy time, against the streaming reference (svc_bench read, 1 GiB, full device and 256 CUs; native steps).
- **Disjoint-CU competition** (placement-qualified protocol of the November 18 experiment 4 / G5): decoder at 256t,
  competitor on CUs 256–303: absent, arith100, mem50, mem100; two blocks on two GPUs, placement proofs first,
  competitor throughput recorded. Same cells, 6 reps. Outcome: slowdown ratio vs absent per step type and S.
- **H4a**: mem100 slows pure-decode steps by ≥ 1.10× at every S; arith100 by ≤ 1.05×. **H4b**: the mem100 slowdown
  of a pure-decode step at S = 32000, N = 8 (KV ≈ weights) is at least that at S = 1024. Memory sensitivity is evidence
  about the stock part only; no projection above stock bandwidth is drawn from it.

## 7. L5 — does tuned chunking help serving in each regime

- **SLO pilot (lc1)**, frozen rule of the November 18 design section 4: stock (mixed, B = 2048), full allocation, low
  rate, development seed 101; TTFT limit = 5 × p90 TTFT and ITL limit = 5 × median per-request p95 ITL, two significant
  digits. Capacity ladder on seed 102; λ* = highest stock rate with attainment ≥ 0.9 and no incomplete request.
  Workloads: lo512, lo2048, lo4096, lctx (prompt 8192–16384, output 512–1536), lprompt (prompt 16384–30720, output
  32–256). Balanced reuses its November 18 SLO and λ*. All serving runs use max_model_len 32768 and 64 sequences.
- **Development (lc2)**: rates GRID × λ* (GRID = 0.5, 0.75, 1, 1.25, 1.5, 2), seeds 101 and 102, Poisson, policies
  whole (separate, no chunking, budget 32768), separate B 2048, mixed B ∈ {256, 512, 1024, 2048, 4096, 8192}.
  Selection: best mixed and best whole/separate per workload by sustainable rate on seed 101 (attainment ≥ 0.9 and no
  incomplete at that rate and all lower), ties broken by goodput at the top rate, then smaller budget.
- **Held-out (lc3)**: seeds 201–203, Poisson; the selected mixed vs the selected whole/separate; bursty repeat at the
  transition rate. Decision per workload: **win** if the all-seed sustainable rate is ≥ 1 grid step higher, **loss**
  if ≥ 1 lower, otherwise **uncertain**; TTFT, ITL tails, per-request stalls and J/token reported as tradeoffs.
- **Replication**: one boundary (win, loss or uncertain) repeated with Qwen2.5-7B on MI300X, or on MI325X if available,
  labelled as a substitution.
- lc2 and lc3 parameters (λ*, SLOs, rates) are written into this document as an amendment before their data.

## 8. Ranking for the architecture model

Remaining bottlenecks are ranked by measured time share (L2, L3) × service sensitivity (L4 slowdown per TB/s of
competitor traffic), with every input's scope stated. Masks do not measure area; request-side counters do not measure
HBM bytes.

## 9. Execution

- Smoke (mi3001x, one GPU; plan `smoke`): the 8K step anchor (2 reps), one 32000-token prompt whole and B 1024, a
  mixed step at S = 32000 with P = 4096, kernel-trace replay and whole prompt, one counter run, a two-request tail, an
  lctx serving trace and the counter audit. Pass = every step completes and the kernel classes resolve.
- lc1 (one mi3008x node, plan `lc1`): 77 steps on eight single-GPU lanes, longest lane ≈ 4 h estimated.
- Per-case status, journals, analysis tables, checksums and reproduction commands are published to
  `results_published/` after each plan.

## 10. Code (backward-compatible additions to the November 18 code)

- `cp_common.ENGINE_KEYS` accepts `kv_cache_memory_bytes` (absent from earlier runs).
- `cp_trace.WORKLOADS` adds lo512, lo2048, lo4096, lctx, lprompt; earlier workloads' traces are byte-identical
  (checked by trace digest).
- `cp_run.py` accepts a step `pmc` counter list (rocprofv3 `--pmc`) and a `host` step kind (counter audit).
- New: `lc_prefill.py` (L1), `lc_replay.py` (L2, L4), `lc_tail.py` (L3 A), `lc_plan.py`, `lc_submit.sh`, and the
  analysis `lc_analyze.py` (written before lc1 data is read).

## 11. Amendment 1 (2026-11-24, after smoke, before lc1 data)

- Smoke 547179 (mi3001x) failed on a harness bug and was cancelled after two steps (kept): vLLM appends a suffix to
  scheduler-side request ids, so the whole-prompt composition check matched no chunk and the prefix-build wait could
  return early. Fixed in b7e44302. Smoke 547207 (mi3001x, release b7e44302): 10/10 steps completed; kernel classes
  resolve as in section 1 (a whole prompt also uses `_fwd_kernel`; "other" < 4% of busy time); cached prefix verified
  per step; FETCH_SIZE path works in eager.
- **Clock state differs by node.** The smoke node ran the 8K anchor at median SCLK 1446 MHz against 1131 MHz in the
  November 18 run (similar board power), and the anchor was 9% faster in decode and 30–35% faster in mixed steps.
  Software, backend and graph settings were identical. Clocks are not changed (no authorization); every comparison in
  this campaign is within one node and job, and per-run SCLK/MCLK medians are reported beside the L0 anchor. The L0
  anchor's pass rule (section 2) stays as written; a failure that tracks a clock difference is reported as such.
- No hypothesis, grid or decision rule changes.

## 12. Amendment 2 (2026-11-25, after lc1, before lc2 data)

lc1 (job 547842, mi3008x node k002-002, release c0f3e871): 76 of 77 steps completed. What lc1 changed in the plan:

- **L0 step anchor fails on clock state, as amendment 1 predicted.** The lc0 cell ran at median SCLK 1440 MHz (vs 1131
  in s1). t_ref was 27.1 ms against 30.9 (0.88×), and the mixed marginals were 32 / 131 ms against 57 / 210 at
  C = 256 / 1024. The serving anchor passes (decode share 96.0% of steps, 89.6% of exec). Clock medians also differ
  between GPUs of the same node under different work (1366–1606 MHz across lc1 lanes). Cross-lane comparisons
  therefore carry a stated clock caveat. A built-in control that runs identical work on two lanes (L1 at L = 1024:
  mix1024 and mix4096 run one chunk, the same as whole) gives a cross-lane ratio of 0.995.
- **Failed and kept: lc4_pmc_FETCH_SIZE.** Both attempts hit an illegal memory access in the S = 32000, N = 8 cell
  under counter collection. The other three counters completed that cell on retry. lc2 re-runs FETCH_SIZE as two new
  steps: S × N ≤ 131072 (`lc4_pmc_FETCH_SIZE_r2_main`) and the S = 32000, N = 8 cell alone (`_r2_S32000_N8`).
- **Failed and kept: lc4_counter_audit.** It exited 0, but all 208 runs reported "no ROCm-capable device". The host
  launcher set HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES both to the physical index. This works on a one-GPU node
  and fails on lane 7 of an 8-GPU node. Fixed in `cp_run.run_host` (ROCR = physical index, HIP = 0). A host step that
  writes an audit with no successful run is now marked failed. lc2 re-runs it as `lc4_counter_audit_r2`. Until then the
  counter classes are those of the G4 audit (section 6).
- **SLO and λ\*, frozen rule applied to the lc1 pilot** (`scripts/amd/lc/params/mi300x_lc2.json`):

  | workload | TTFT limit | ITL limit | λ* (rps) | note |
  |---|---|---|---|---|
  | lo512 | 0.46 s | 110 ms | 4.0 | censored: attainment 1.0 at every ladder rate |
  | lo2048 | 0.45 s | 110 ms | 1.0 | censored |
  | lo4096 | 0.70 s | 130 ms | 0.5 | censored |
  | lctx | 25 s | 110 ms | 0.1 | attainment 0.93 at 0.1, 0.33 at 0.15 |
  | lprompt | 41 s | 110 ms | 0.05* | stock never reaches 0.9 (0.87 at 0.03) |
  | balanced | 1.4 s | 110 ms | 1.0 | s1 values |

  \*The frozen rule gives no λ\* for lprompt. Pre-registered substitute: the smallest ladder rate whose goodput is
  ≥ 90% of the ladder peak (0.041 of 0.043 rps at 0.05). This is reported as a substitution. Stock B = 2048 already
  breaks the lprompt ITL target at low load. That failure is itself an L5 finding: long-prompt chunks inflate co-running
  decode steps. For the censored workloads, GRID × λ\* extends to 2× the ladder top, so the knee falls inside the lc2
  grid.
- **lc2 (development, plan `lc2`, 76 steps, ≈ 8.5 h estimated, pilot actuals ≈ 0.7× estimates):**
  - L5 tuning for lo2048, lctx and lprompt: 8 policies (whole/separate, separate B 2048, mixed B 256 / 512 / 1024 /
    2048 / 4096 / 8192) × seeds 101 and 102 × GRID × λ\*.
  - L3 part B for lo512, lo4096 and balanced: 5 policies (whole, sep2048, mix256 / 2048 / 4096), seed 101.
  - Fixed-rate slice at λ\*(lo4096) = 0.5 rps for lo512 and lo2048.
  - The two FETCH_SIZE re-runs and the audit re-run.
  - Requests per rate point: lo512 200, lo2048 120, lo4096 60, lctx 40, lprompt 30, balanced 200.
- **Selection for lc3 (held-out seeds 201–203):** per workload, the best mixed budget and the better of the two
  separate policies, by goodput at the highest grid rate where either reaches attainment ≥ 0.9, ties to the smaller
  budget. Decision rule as in section 7. lc3 also carries the bursty repeat and the 7B replication, written as
  amendment 3 before its data.
- No hypothesis changes.

## 13. Amendment 3 (2026-11-25, after lc2, before lc3 data)

**lc2 outcome (job 548920, 76/76 steps).** 8 serving steps hit `hipErrorLaunchFailure` on attempt 1 and completed on
attempt 2; the failed attempts are kept. Analysis: `results_published/amd_20261125_mi300x_lc_lc2/lc_tables.json`.

**Selection (amendment-2 rule).** The first analysis broke a tie between the two separate policies toward whole, which
contradicts the written rule (ties to the smaller budget). The code is corrected to the text (whole counts as budget
32768) before any lc3 data:

| Workload | Rate used | Selected mixed | Selected separate | Note |
|---|---|---|---|---|
| lo2048 | 0.75 rps | mix256 | sep2048 | mix256 / 512 / 4096 tie, smallest budget |
| lctx | 0.20 rps (2 × λ\*) | mix8192 | sep2048 | whole and sep2048 tie exactly (attainment 1.0) |
| lprompt | 0.10 rps (2 × λ\*) | mix8192 | sep2048 | whole and sep2048 tie exactly (attainment 1.0) |

**Changes for lc3 (all before data).**
1. **whole is added to every held-out policy set** (selected mixed, selected separate, mix2048, whole). Reason: it ties
   the selected separate policy on lctx and lprompt and is the no-chunking baseline the question asks about.
2. **Extended ladder for lctx and lprompt: GRID ∪ {3, 4} × λ\*.** Reason: in lc2 both separate policies held
   attainment 1.0 at the top grid rate (2 × λ\*), so their capacity is censored and a win/loss could not be resolved.
   lo2048 is not extended (every policy fell below 0.9 at 1 × λ\*).
3. **Secondary, stall-aware SLO.** The primary SLO bounds each request's p95 ITL, which by construction ignores up to
   5% of a request's gaps. In lc2 whole and separate prefill kept attainment 1.0 on lctx and lprompt while the p99 of
   per-request worst gaps reached 5–50 s. The secondary SLO is the primary SLO **plus** each request's maximum ITL ≤
   1000 ms (named threshold), with 250 / 500 / 2000 ms reported as sensitivity. The lc3 decision is made under both;
   the primary SLO stays primary, and a disagreement between the two is reported as a counterexample, not resolved.
   Applied to lc2 the stall view is exploratory (defined after seeing lc2).
   Exploratory lc2 result under the stall view (same selection rule): lctx picks mix2048 at 1000 ms (goodput summed over the two seeds 0.114 vs
   0.037 for whole / sep2048 at 0.05 rps) and mix256 at 250 / 500 ms; lo2048 is unchanged (mix256 / sep2048); for
   lprompt no policy reaches attainment 0.9 at any rate with stalls ≤ 1000 ms. The 1000 ms picks are already in the
   held-out sets below, so no policy is added for them.
4. Decision per workload, both SLOs: selected mixed vs each of selected separate and whole, on all held-out seeds;
   **win** if the all-seed sustainable rate (attainment ≥ 0.9, nothing incomplete, at that and every lower rate) is ≥ 1
   ladder step higher, **loss** if ≥ 1 lower, otherwise **uncertain**, with goodput at λ\* and 2 × λ\* as the tie
   view. TTFT, ITL tails, stalls and J/token are reported as tradeoffs.
5. **Bursty repeat**: same four policies, seed 201, gamma gaps (CV 4), same ladder.
6. **7B replication (Qwen2.5-7B, MI300X, same harness)**: L1 timing (whole, B 256 / 1024 / 4096), L2 timing (S 1K–32K,
   N 1 / 8 / 32), L3 part A tails (whole, mix2048), its own SLO pilot (stock, seed 101) and capacity ladder for lctx
   (pilot 0.1 rps, ladder 0.2–1.6) and lprompt (pilot 0.08 rps, ladder 0.12–1.0), and matched ladders of the four
   32B held-out policies on seed 201 (experiment lc5r). The 7B SLO comes from its own pilot by the frozen rule; the
   replication question is whether the 32B direction (win / loss / uncertain) holds, not the numbers. 7B KV is
   56 KiB per token (vs 256 KiB) and is used for any byte arithmetic.
7. Parameters: `scripts/amd/lc/params/mi300x_lc3.json`. Plan: 71 steps, longest lane ≈ 8.6 h, ≈ 8.9 units.
- No hypothesis changes.

## 14. Amendment 4 (2026-11-25, during lc3, before any re-run data)

- **Harness bug.** `lc5h_lctx_whole_s203` failed both attempts at the extended rate 0.3 rps with
  `AssertionError: hidden waiting queue received requests` (`cp_sched.py`). Under the separate policy, a decode step
  hides the waiting queue; when KV runs out, vLLM preempts a running request and prepends it to `waiting`
  (`Scheduler._preempt_request`), which the wrapper treated as impossible. lc1 and lc2 never preempted under the separate
  policy, so their results are unaffected (the code path is identical when nothing is preempted).
- **Fix.** Preempted requests are prepended to the restored waiting queue, exactly as vLLM does without the wrapper.
- **Re-run rule.** Every lc3 step whose final attempt fails with this assertion is re-run once after lc3, with the fixed
  release, the same arguments and seed, under the id `<step>_fx` (a new step, two attempts as usual). The failed
  attempts are kept; their completed lower-rate traces are reported as partial evidence only and are not used in the
  decision. If a re-run is not possible within the allocation, the affected policy × workload is reported with the
  ladder truncated at the last completed rate, and its sustained rate is marked as a lower bound.
- **lc3 outcome (job 555664, 63/71).** The 8 steps failing with this assertion are exactly the separate policies on lctx
  (whole and sep2048 × seeds 201–203, and both bursty repeats); none on lprompt or lo2048. They are re-run unsplit
  (8 steps on 8 GPUs, original per-step protocol) as `lc3fx`, params `scripts/amd/lc/params/mi300x_lc3fx.json`.
- **lc3fx outcome (job 558807, 1:15:29, 8/8 completed on attempt 1).** The re-runs replace the failed steps in the
  held-out and bursty decisions (`results_published/amd_20261125_mi300x_lc_lc3/lc_decisions.json`). Preemptions occur
  only at ≥ 0.3 rps (1–5 per run), where the old code aborted. No truncated ladder was needed.

## 15. Amendment 5 (2026-11-26, after the final lc report, before any ld data): disaggregation at long prompts

Motivation: section 5 of the results report shows that on one GPU, long prompts force a choice between two failure
modes:
- mixing chunks with decode costs 1.5–5× in sustained rate under the p95 SLO;
- whole/separate prefill stalls co-running decoders for 3–28 s, so it fails a 1 s worst-gap cap.

The report's implication that prefill/decode disaggregation resolves this was an inference. This amendment measures it.
Plan `ld` (`scripts/amd/lc/lc_plan.py ld`, params `scripts/amd/lc/params/mi300x_ld.json`); smoke plan `ld_smoke`
runs first.

**Setup.** Qwen2.5-32B on MI300X, two 4-GPU lanes per mi3008x node (GPUs 0–3, 4–7). Driver
`scripts/amd/cp/cp_disagg.py` (s4 harness): HTTP servers, one asyncio client, and open-loop traces from
`cp_trace.py`.
- Every engine: 64 seats, max_model_len 32768, no prefix caching.
- The first 10% of requests are warm-up and are not scored.
- Energy is measured on all four boards.

**Deployments (four GPUs each).**
- `colo4_whole`: four colocated engines, separate policy, whole prompts (no chunking, budget 32768). This is the
  primary-SLO winner on one GPU.
- `colo4_mix2048`: four colocated engines, mixed B 2048. This is the stall-SLO-feasible policy on one GPU.
- Colocated routing: least outstanding work.
- `dXpY`: X producers and Y consumers, P2pNcclConnector with PUT_ASYNC.
  - Producers run whole prompts (separate, no chunking, budget 32768).
  - Consumers run stock mixed scheduling (`--decode-policy mixed`).
  - The connector fix is on.
  - Consumer receive buffer: 2 × 10¹⁰ B (a 30 K-token prompt is 8.05 × 10⁹ B). Overflow goes to vLLM's 32 GB host
    pool. Consumers use gpu_memory_utilization 0.80, to leave room for the buffer.
  - Ratios: lctx 1P3D and 2P2D (decode-heavier); lprompt 2P2D and 3P1D (prefill-heavy).

**Traces.**
- Workloads and seeds: lctx and lprompt, held-out seeds 201 and 202, Poisson.
- Rates: 4 × the per-GPU lc3 ladder. lctx 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.6 rps; lprompt 0.1, 0.15, 0.2,
  0.25, 0.3, 0.4, 0.6, 0.8 rps.
- Requests per rate: lctx 100, lprompt 80. Drain: lctx 300 s, lprompt 240 s.
- 16 steps; estimated ≤ 10 node-hours.

**SLOs.**
- Primary: the lc SLOs. lctx TTFT 25 s, ITL 110 ms; lprompt TTFT 41 s, ITL 110 ms; ITL as per-request p95.
- Stall: the primary SLO plus each request's maximum ITL ≤ 1 000 ms (sensitivity grid 250 / 500 / 1 000 / 2 000).
- TTFT runs from arrival to the first decode token. For disaggregated deployments, it includes prefill and KV
  transfer.

**Decision.**
- Per workload and SLO, compute the all-seed sustained rate per deployment: the highest rate at which both seeds
  reach attainment ≥ 0.9 with nothing incomplete, at that rate and every lower one.
- Compare the best disaggregated ratio with the best colocated deployment. Every ratio is also compared with each
  colocated deployment.
- Win or loss if ≥ 1 ladder step apart; otherwise uncertain. If both are infeasible at the lowest rate, the result
  is uncertain.
- Choosing the best of two ratios on the same data is a stated optimism bias; the per-ratio comparisons are reported
  beside it.
- Analysis: `lc_analyze.py ld`.

**Hypotheses.**
- H1, stall SLO: the best disaggregated ratio wins over the best colocated deployment on lctx and on lprompt, because
  consumers never run a prompt.
- H2, primary SLO: on lctx, the best disaggregated ratio is not a loss against colo4_whole. On lprompt, no
  prediction: total prefill capacity is the same or lower.
- H3, harness check: colo4_whole's primary sustained rate is within one ladder step of 4 × the single-GPU held-out
  rate (lctx 1.2, lprompt 0.6).

**Secondary outcomes.**
- KV hand-off time (first decode token after the prefill response, p50 / p99);
- consumer and producer queue and KV peaks; preemptions;
- p99 of each request's maximum ITL;
- joules per output token over four boards.

**Failure handling.** At most two attempts per step; failures are kept.
- If the long-prompt connector path fails in the smoke, the blocker is recorded, and the disaggregated arms are
  replaced by a labelled configuration (for example a smaller consumer buffer with host-pool spill) before `ld`.
- Code changes for this amendment (additive; defaults unchanged):
  - `cp_disagg.py`: `--decode-policy`, `--decode-kv-buffer`, `--kv-mem-pool-gb`.
  - `cp_analyze.disagg_trace`: optional stall grid.
  - `lc_analyze.py ld`: tested on the s4 e6 pheavy data, where it reproduces the published 3P1D 4.5 vs colo4_tuned
    3.0 rps.

## 16. Amendment 6 (2026-11-27, after ld, before any re-run data): decode-engine crashes

**What happened in `ld`** (job 560897, 16 of 16 steps exited normally, all on attempt 1):
- In four disaggregated steps a consumer engine died with `hipErrorLaunchOutOfResources`
  (`HSA_STATUS_ERROR_OUT_OF_RESOURCES` on a kernel dispatch). Each death came at light load, 1–3 running requests.
  After it, every remaining request in that step failed. Affected steps:
  - `ld_lctx_d2p2_s201`: died during 0.2 rps;
  - `ld_lctx_d2p2_s202`: died during 0.4 rps;
  - `ld_lprompt_d3p1_s201`: died during 0.1 rps;
  - `ld_lprompt_d2p2_s202`: died during 0.8 rps.
- The same configurations ran clean on the other seed. The receive buffer had not overflowed when the lctx engine
  died.
- The colocated steps and `ld_lctx_d1p3` (both seeds) had no errors.
- Working hypothesis (not established): memory headroom for runtime allocations. Consumers reserve 0.80 of the GPU
  for vLLM, plus a GPU receive buffer of up to 2 × 10¹⁰ B, plus RCCL buffers.

**Why re-run.** Three of these steps determine a stall-SLO comparison:
- lprompt d3p1, seed 202 sustained 0.3 rps under the stall SLO; the colo4_whole stall rate is 0.2.
- lctx d2p2 passed 0.3 rps before its engine died; the colo4_mix2048 stall rate is 0.4.

`ld_lprompt_d2p2_s202` is not re-run: its engine died at the top rate, above both seeds' capacity (0.15), so
neither the capacity nor any decision changes.

**Re-run `ldfx`** (`lc_plan.py ldfx`, params `mi300x_ld.json` keys `ldfx_steps` and `ldfx_decode_gpu_mem`):
- Steps `ld_lctx_d2p2_s201_fx`, `ld_lctx_d2p2_s202_fx` and `ld_lprompt_d3p1_s201_fx`.
- Arguments are identical to the original steps except consumer gpu_memory_utilization 0.70 instead of 0.80 (about
  19 GB more headroom). This is a labelled configuration change; no code changes in the driver.
- One mi3008x job, two 4-GPU lanes; estimated 2.5 node-hours.
- Analysis: `lc_analyze.py ld --root <ld root> <ldfx root>`. Each `_fx` step replaces its original in the tables. The
  original crashed acquisitions are kept and reported. `engine_dead` flags any step whose server log has
  `EngineDeadError`.
- If an `_fx` step's engine dies again, the cell is reported as failed and the comparison it decides as uncertain. No
  further attempt is made.

**Already observed (not changed by the re-run).**
- Primary SLO: disaggregation loses to colo4_whole on both workloads at every ratio:
  - lctx: d1p3 0.2 vs colo4_whole 1.6 rps;
  - lprompt: d2p2 0.15 and d3p1 at most 0.3 vs colo4_whole 0.8 rps.
  - H2 fails. No re-run can recover 1.6 rps on lctx with two producers.
- Mechanism, measured from step logs: a single whole-prompt prefill step takes 1.6–2.0× longer on a producer than on a
  colocated engine. The connector's per-layer KV send runs inside the forward pass. Examples, ms per 1 K prompt
  tokens:
  - lprompt 24–33 K tokens: 399–405 on producers vs 205 colocated;
  - lctx 8–16 K tokens: 254 vs 163.
- KV hand-off after the prefill response: p99 0.08–0.12 s.
- Consumers' p99 of each request's maximum ITL: 48–280 ms. Colocated: 0.65–2.7 s for mix2048 and 0.9–69 s for
  whole.

**Predictions for the re-run (made before its data).**
- lprompt d3p1, seed 201 sustains ≥ 0.25 rps under the stall SLO (seed 202: 0.3). If so, d3p1 vs colo4_whole under
  the stall SLO is a win.
- lctx d2p2 sustains 0.3–0.4 rps under both SLOs. Vs colo4_mix2048 under the stall SLO (0.4): uncertain or loss.

## 17. Amendment 7 (2026-11-27, after ld/ldfx, before any ld2 data): scenarios that may favour disaggregation

The owner asked for scenarios that favour disaggregation, including other prefill:decode GPU ratios on one workload,
and ordered all of them executed.

**Basis from ld/ldfx (report section 10).**
- Disaggregation was prefill-bound: producers prefill 1.6–2.0× slower than colocated engines.
- Consumers never stall: per-request max ITL ≤ 0.3 s.
- Colocated whole fails every stall cap. At caps ≤ 500 ms no colocated deployment is feasible on lctx; disaggregation
  sustains 0.2–0.4 rps there. This comes from the pre-registered stall grid of amendment 5.

**Plans and params.** Plans `ld2a`, `ld2b` and `ld2c` in `scripts/amd/lc/lc_plan.py`; params
`scripts/amd/lc/params/mi300x_ld2.json`.

**Code (additive; defaults and existing traces unchanged; lctx trace digests verified identical).**
- `cp_disagg.py`: `--kv-channels` (P2pNccl nccl_num_channels, default 16) and `--kv-send-type` (default PUT_ASYNC).
- `cp_trace.py`: workloads `lclo` and `mixdoc` (below).
- `cp_analyze.disagg_trace`, opt-in per-workload SLO keys:
  - `long_prompt_min` / `ttft_long_s`: per-class TTFT;
  - `itl_tight_ms`: the tight-ITL SLO `itl50`;
  - per-class short-request breakdown.
- `lc_analyze.py ld`:
  - decisions for every present SLO (primary, stall 250 / 500 / 1 000 / 2 000, itl50);
  - bursty rows are kept separate (`<workload>_bursty`);
  - a new `ld2a` command for phase A.

**Workloads.**
- `lclo` (long context, long output): prompt 8 192–16 384, output 2 048–4 096, P:D ≈ 4. Each colocated request
  lives through many other requests' multi-second whole-prompt prefills.
- `mixdoc` (mixed tenants): each request is independently a document with probability 0.1 (prompt 16 384–30 720,
  output 128–512) or otherwise a chat turn (prompt 128–512, output 512–1 024).

**SLOs.**
- `lclo`: TTFT 25 s (as lctx, same prompt range), ITL p95 110 ms.
- `mixdoc`: chat TTFT 2 s, document TTFT 41 s (as lprompt), ITL p95 110 ms. The 2 s chat TTFT is chosen, not derived.
- All workloads: the stall grid 250 / 500 / 1 000 / 2 000 ms (primary 1 000), and `itl50` (p95 ITL ≤ 50 ms with the
  same TTFT).

**Phase A (connector micro-test, plan `ld2a`).**
- Setup: lprompt 3P1D at 0.1 rps, 30 requests, development seed 101; two 4-GPU lanes.
- One step per setting: nccl_num_channels 16 / 4 / 2 / 1 with PUT_ASYNC, and 16 with PUT. Plus colo4_whole at the
  same load as reference.
- Measure: median producer single-prompt prefill time per 1 K prompt tokens (prompts ≥ 16 K, step logs); TTFT;
  hand-off p99; completion; engine deaths.
- Selection rule (`lc_analyze.py ld2a`):
  - Eligible: all 30 requests complete, no engine death, hand-off p99 ≤ 0.5 s.
  - Pick the eligible setting with the lowest prefill time if it is ≥ 10% below the 16-channel PUT_ASYNC control.
    Otherwise keep the control.
  - The pick is written to `ld2_kv` and committed before ld2b is submitted.
- HA: ≤ 4 channels cuts producer prefill time ≥ 10% vs 16 channels. Prediction: within 1.3× of colocated.

**Phase B (ratio sweep, one workload, plan `ld2b`).**
- Setup: lclo, one 8-GPU deployment at a time, held-out seeds 201 and 202, Poisson. Rates 0.2, 0.4, 0.6, 0.8, 1.2,
  1.6, 2.4, 3.2 rps; 100 requests per rate; drain 400 s.
- Deployments:
  - colo8_whole, colo8_mix2048 and colo8_mix8192 (both budgets that won a single-GPU long-prompt SLO);
  - d2p6, d3p5, d4p4, d5p3 and d6p2, using the phase A connector setting.
- Consumers: gpu_mem 0.80, buffer 2 × 10¹⁰ B. 16 steps, about 18 node-hours, one resubmission to the same root
  (12 h limit).
- Hypotheses:
  - HB1: under the stall SLO, the best ratio wins over the best colocated deployment.
  - HB2: under the primary SLO, the best ratio is not a loss. Prediction: loss, unless phase A removes most of the
    producer slowdown. Colocated deployments use the KV capacity of all eight GPUs for decode.
  - HB3: the best ratio is between 3:5 and 5:3. Estimate: a producer handles ≈ 0.2–0.3 rps of 12 K prompts, and a
    consumer ≈ 0.2 rps of 3 K-token outputs (KV-limited).
  - HB4: under itl50, prediction is uncertain. Large-batch decode at ≈ 14 K context runs near 50 ms per step for
    either deployment.

**Phase C (plan `ld2c`).** Deployments for C1 and C2 come from phase B by the rule below. The rule is applied
mechanically, written to `ld2c` and committed before ld2c is submitted.

- **C1, bursty arrivals (gamma, CV 4), lclo,** same rates and seeds. Deployments:
  - the best colocated deployment under the primary SLO;
  - the best colocated deployment under the stall SLO; if that is the same deployment, colo8_mix2048 is added;
  - the best disaggregated ratio under the stall SLO (ties: higher primary capacity, then fewer producers).
  - HC1: stall win; primary uncertain.
- **C2, CU-restricted decode, lclo, Poisson.** The C1 disaggregated ratio with consumers restricted to 152 of 304 CUs
  (`--decode-cus 152 --decode-grid 152`, as in s5). It emulates a decode part with half the compute.
  - HC2: not a loss (within one ladder step) against the same ratio at 304 CUs in phase B, under both SLOs.
- **C3, mixed tenants (`mixdoc`),** rates 0.5, 1, 1.5, 2, 3, 4, 6, 8 rps; 200 requests; drain 300 s. Deployments
  fixed now: colo8_whole, colo8_mix2048, d4p4 and d6p2.
  - HC3: under the stall SLO, the best ratio wins.
  - HC3: chat requests' p99 max ITL on disaggregated deployments ≤ 0.3 s, against ≥ 1 s for colo8_whole.
  - HC3: primary uncertain.
- Size: about 16.5 node-hours, one resubmission.

**Decision and failure handling.**
- Decisions use the all-seed sustained rate, win or loss at ≥ 1 ladder step, as in amendment 5.
- At most two attempts per step. No re-runs are planned under this amendment.
- A consumer-engine death makes the crashed cell a lower bound. A comparison that is a loss only because of a
  crash-limited cell is reported as uncertain.
- Total estimate: about 36 node-hours (≈ 36 units of the ≈ 490 remaining).

**Phase A outcome (2026-11-27, job 562877, recorded before ld2b is submitted).**
- The rule selected **ch16 PUT_ASYNC (the control)**, so `ld2_kv` = [16, "PUT_ASYNC"].
- Producer single-prompt prefill p50 (ms per 1K tokens, prompts ≥ 16384), lprompt d3p1, 0.1 rps, 30 requests:

  | Setting | Prefill | Complete | Real engine crash |
  |---|---|---|---|
  | ch16 PUT_ASYNC | 338 | 18/30 | w3 |
  | ch16 PUT | 348 | 20/30 | w3 |
  | ch2 PUT_ASYNC | 342 | 20/30 | w3 |
  | ch4 PUT_ASYNC | 360 | 30/30 | none |
  | ch1 PUT_ASYNC | 361 | 30/30 | none |
  | colo4_whole | 195 | 30/30 | none |

- Hand-off p99 was 0.08–0.09 s for every setting.
- No eligible setting was ≤ 0.9 × the control. ch4 and ch1 were eligible but 6–7% slower than the control.
- **HA is refuted.** RCCL channel count and send type do not explain the producer slowdown, which stays at 1.7–1.9×
  colocated.
- The crashes (`hipErrorLaunchOutOfResources`, consumer w3) occurred at 16 and 2 channels but not at 4 or 1. With one
  run per setting this is not evidence of a channel effect, so the rule is applied as written.
- **Analysis correction.** vLLM 0.23 also logs `EngineDeadError` during a normal SIGTERM shutdown. The engine-death flag
  (`lc_analyze._engine_crashed`) now counts an `EngineDeadError` only before the first `[shutdown]` line, and counts any
  HIP/accelerator abort. This removed false flags on colo4_whole. The ld/ldfx engine-death list and all ld decisions are
  unchanged.

## 18. Amendment 8 (2026-11-28, after the first ld2b job, before any re-run or ld2c data): compile-cache startup race

**What happened.** Step `ld2b_lclo_colo8_mix8192_s202` failed on both attempts (job 563412) before serving any request.
- Attempt 1: workers w5 and w6 failed engine initialisation in torch-inductor autotuning ("@jit functions should be
  defined in a Python file"; "Cubin file saved by TritonBundler not found").
- Attempt 2: worker w4 failed with the TritonBundler error.
- All workers of a `cp_disagg` step share one compile cache (`$WORK/cache/proc/<ts>_<pid>`, set once per container
  by `vllm_in_container.sh`), so concurrent first compilations race on it. The same deployment on seed 201 and the
  other 10 completed steps did not hit it.
- This is an infrastructure failure, not a measurement. Both failed attempts are preserved, and the runner does not
  retry the step.

**Change.**
- `cp_disagg.py --per-worker-cache` (opt-in) gives each worker its own `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR` and
  `TORCHINDUCTOR_CACHE_DIR` under the step's cache.
- Only the cache location changes. Every step already starts from an empty cache, so compiled kernels and steady-state
  performance are unaffected.

**Re-run and phase C.**
- One re-run: `ld2b_lclo_colo8_mix8192_s202_fx` (`ld2b_fx_steps`), same configuration plus `--per-worker-cache`. It runs
  under its own plan `ld2bfx` and run root, because the ld2b root resumes its frozen plan. It replaces the original in
  the analysis (multi-root, as the `_fx` steps of amendment 6 did).
- At most two attempts. If it also fails, colo8_mix8192 has one seed and is excluded from all-seed decisions.
- All ld2c steps use `--per-worker-cache` (`ld2_per_worker_cache`).
- The remaining four ld2b steps were deferred by the job deadline and run unchanged.
- Added cost: about 1.1 node-hours.
