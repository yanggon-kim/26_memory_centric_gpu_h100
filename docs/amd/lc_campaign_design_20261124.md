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
