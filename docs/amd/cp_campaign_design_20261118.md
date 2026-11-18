# Chunked prefill and decode specialization on AMD — execution design (pre-registration)

Owner handoff: `docs/amd/chunked_prefill_campaign_20261118.md` (commit 1d8d7b5c). Owner execution prompt 2026-11-18:
run all five experiments within the existing AMD allocation, including the two-GPU disaggregated comparison. This
document is committed before any production data. Values marked *pilot* are computed by the frozen rules below from the
stage-1 pilot (stock configuration, development seeds) and recorded in a dated amendment before the stage that uses them;
the rules themselves do not change after data.

No clock, power-limit, driver or firmware change. Cohorts are never pooled. Failed and unmatched runs are preserved.
AMD acknowledgement for any publication: "This work was supported in part by Advanced Micro Devices, Inc. under the AMD
University Program's AI & HPC Cluster".

## 1. Stack, models and allocations

| Item | Value |
|---|---|
| Engine | vLLM 0.23.0 V1 in `vllm_rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0.sif` (sha in `$WORK/containers/sha256.txt`), TP1, BF16 weights and KV, greedy, EOS ignored, seed 20261013 |
| Engine mode | in-process engine core (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, uni executor, synchronous scheduling) for experiments 1–4; `vllm serve` OpenAI servers for experiment 5 |
| Graphs | `compilation_config.max_cudagraph_capture_size = 512` everywhere: vLLM's own dense capture sizes (1, 2, 4, every 8 to 256, every 16 to 512); FULL graphs for uniform decode batches, PIECEWISE for mixed batches ≤ 512 tokens, eager above. This is stock `vllm serve` behaviour at its default `max_num_seqs`; vLLM would otherwise tie the limit to `max_num_seqs`. The graph path of every step is derived from its token count and the recorded capture sizes. |
| Model A | Qwen2.5-32B-Instruct rev `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`: 64 layers, 40 query / 8 KV heads, head dim 128, BF16, full attention (sliding window disabled), KV 256 KiB per token |
| Model B | Qwen2.5-7B-Instruct rev `a09a35458c702b33eeacc393d103063234e8bc28`: 28 layers, 28 query / 4 KV heads, head dim 128, BF16, full attention, KV 56 KiB per token (4.6× less KV per token than A) |
| Prompts | seeded synthetic token ids (uniform over the vocabulary), as in the G2 runner |
| Allocations (MI300X, MI325X) | full 304 CUs (stock kernels); 192 and 128 CUs = `HSA_CU_MASK` on physical indices 0..n−1 (equal per XCD, placement-proven in earlier jobs and re-proven in experiment 4) with **stock** grids (`num_compute_units` = 304, identical settings) and **tuned** grids (`CUBW_CU_COUNT_OVERRIDE = n`, the sized wvSplitK of G1) |
| Allocations (MI350X) | full 256; 192 and 128, stock and tuned as above |

A CU mask restricts execution and request resources on a present die; it does not reclaim area. Nothing here measures
reclaimed silicon or extra HBM.

Engine limits used by the harnesses: `max_model_len` 16384 (8192 for whole-prompt policies), `max_num_seqs` 64 for
serving, `gpu_memory_utilization` 0.90 (0.85 for `vllm serve`).

## 2. Scheduler control (all experiments)

`scripts/amd/cp/cp_sched.py` subclasses the vLLM 0.23 scheduler (`scheduler_cls`):

- **mixed**: stock V1 scheduling (running requests first, i.e. decodes and unfinished prompt chunks, then waiting
  requests, under the token budget). Sarathi-style decode-first mixing.
- **separate**: each step is prefill-only or decode-only; prefill steps run whenever prompt work is pending (decodes are
  hidden from that step), otherwise a decode step runs with the waiting queue hidden. With chunked prefill on, prompts
  are split by the budget (**chunked, separate**); with chunked prefill off the engine admits only whole prompts that fit
  the budget (**whole, separate**: vLLM V0's default). An empty prefill step (KV full) is followed by a decode step.

Every executed step is logged: kind (decode / prefill / mixed), decode width and context, every prompt chunk (request,
start, tokens, prompt length, cached tokens at admission, recompute flag), queue lengths, preemptions, scheduling time and
model execution time. Dev runs (2026-11-18, Qwen2.5-7B): separate policy produced 1774 decode-only and 40 prefill-only
steps and no mixed step; mixed produced mixed steps.

## 3. Experiment 1 — replayed iterations and prefill headroom

The engine has no iteration-replay API. Fallback (handoff): controlled admission/barrier harness
`scripts/amd/cp/cp_replay.py`. For each resident context S ∈ {1024, 8192} and decode width N ∈ {1, 8, 32}: admit N
requests of S fresh tokens, step to the barrier (a decode-only step with exactly N decodes and an empty queue), then for
11 repetitions (first = warmup, discarded) × shuffled C ∈ {0, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048}: one
decode-only step (reference) followed by one step that adds a fresh C-token request with `max_tokens` 1 (N decodes +
one C-token prompt with prefix length 0, in one iteration; the request completes and frees its blocks). C = 0 is a
null pair. Budget 4096 so every C fits in one step. Composition of every timed step is verified from the scheduler log;
a mismatch is a correctness failure. Pure-prefill anchors (N = 0) for every C.

Fidelity, reported: decode positions advance one per step (drift ≤ 2 × 11 × 11 = 242 tokens per cell, recorded per
step); pairs are adjacent so the drift is shared within a pair; KV state is real (prefilled, not synthesized).

Allocations: Qwen2.5-32B at full, 192 stock, 192 tuned, 128 stock, 128 tuned; Qwen2.5-7B at full, 128 stock,
128 tuned. Timing = model execution time per step (exec_s); harness wall and scheduling time also kept.

**Headroom rule.** Per (model, a, S, N, C): pair ratio = T_mix / T_ref of the same pair; estimate = median of the 10
pair ratios, 95 % CI by bootstrap over pairs (10 000 resamples, seed 20261118). C is *within headroom* if the upper CI
≤ 1.05. Headroom H(N, S, a) = the largest tested C such that every tested C' ≤ C is within headroom. If the first tested
C > 0 already fails, H = 0 and the boundary is reported as the interval (0, first C). If no tested C fails, H ≥ 2048 (open
boundary). Non-monotonic points (a failure followed by a pass) are reported. The common absolute deadline of the table
is the full-allocation T(N, 0, S) × 1.05 for each (N, S). Comparisons: full vs reduced masks at identical (stock) grids,
then each mask with its tuned grid.

Instrumented attribution: rocprofv3 kernel + marker traces of S = 8192, N = 8, C ∈ {0, 256, 1024} at full and 128-tuned
(separate runs; their timings are not used for headroom). Kernels inside each step's marker range are grouped into
linear (GEMM / wvSplitK / skinny GEMM), attention and other.

## 4. Experiment 2 — scheduling policies under identical open-loop traffic

Traces (`scripts/amd/cp/cp_trace.py`): request ids, lengths and contents from the seed alone; timestamps from (seed,
arrival process, rate). Workloads: **dheavy** prompt U[128, 512], output U[256, 768] (P:D ≈ 0.6); **balanced** U[512,
2048] / U[128, 384] (P:D ≈ 5); **pheavy** U[2048, 6144] / U[32, 128] (P:D ≈ 51). Arrivals: Poisson and bursty (gamma
gaps, CV 4) at matched mean rate and identical request set. Development seeds 101, 102; held-out seeds 201, 202, 203.

Driver (`scripts/amd/cp/cp_serve.py`): open loop; TTFT from the scheduled arrival time; token times = end of the step
that produced them. Windows: the first 10 % of requests by arrival are warmup (served, not scored); the rest are
measured; drain deadline = last arrival + 60 s; requests unfinished at the deadline are aborted and count as SLO failures.

Policies: **whole-separate** (chunked prefill off, budget 8192), **chunked-separate** (budget B), **mixed** (budget B).
Pure-prefill and pure-decode anchors come from experiment 1.

**SLO rule (frozen; values from the pilot).** Pilot = stock configuration (full allocation, mixed, B = 2048), Poisson,
development seed 101, 40 requests at a low rate (32B: 0.2 req/s, 0.1 for pheavy; 7B: 0.5, 0.25 for pheavy). Per
(model, workload): L_TTFT = 5 × p90 TTFT of the pilot's measured requests, L_ITL = 5 × median per-request p95
inter-token latency, both rounded up to two significant digits. A request is **compliant** if it completes with
TTFT ≤ L_TTFT and its p95 inter-token latency ≤ L_ITL. The same limits apply to every policy, allocation and deployment
for that model and workload.

**Load search.** Capacity pilot: rate ladder at full and 128-tuned (stock mixed B = 2048, development seed 102). The
experiment-2 rate grid per (model, workload) is 6 rates = {0.4, 0.6, 0.8, 1.0, 1.2, 1.5} × λ*, where λ* = the highest
pilot rate at full allocation with SLO attainment ≥ 0.9, rounded to two significant digits.

**Metrics.** Per trace: SLO attainment (compliant / measured), goodput (compliant measured requests and their output
tokens per second of the measurement window), output tokens/s, p50/p99 TTFT and ITL, incomplete and aborted requests,
queue length over time, step-kind fractions by count and by time, effective P:D (executed prompt tokens incl.
recomputation / decode ticks; the first output token is counted with the prompt step that produced it), preemptions,
energy per output token (board counter; scope: one board over the whole trace incl. idle gaps). **Sustainable rate** =
the highest rate of the grid with attainment ≥ 0.9 in every seed and arrival process tested, with no incomplete
requests; reported with the attainment curve.

**Stages.** (i) Common budget: all three policies at B = 2048 (whole-separate necessarily 8192) on development seeds.
(ii) Tuning (development seeds 101, 102, Poisson): budgets {256, 512, 1024, 2048, 4096} for mixed and {512, 2048, 4096}
for chunked-separate, per allocation and workload; the tuned budget maximizes sustainable rate, ties broken by goodput at
the highest common rate, then by the smaller budget. (iii) Held-out judging: each policy with its tuned budget, and the
stock reference (mixed, 2048, full allocation), on held-out seeds 201–203, Poisson and bursty, full and 128-tuned (and
192-tuned for the common-budget stage). Uncertainty: per-seed values and a bootstrap over requests within trace for
tails; differences in sustainable rate are reported as grid steps.

## 5. Experiment 3 — resident history and fresh prefill

Step level (`cp_replay.py --prefix`): S = 1024 resident context, N ∈ {8, 32} decodes, fresh suffix F = 256, prefix P ∈
{0, 1024, 2048, 4096, 8192}; treatments per P: **warm** (prefix cached by an untimed request with the same P tokens:
executes F, attends P + F), **cold** (same P + F tokens, nothing cached: executes P + F), **fresh** (F tokens only).
Prefixes are new random ids every repetition (independent across requests). Cache hits verified per step from the
scheduler (cached tokens at admission) and executed tokens; warm-build time recorded as warmup cost. 8 repetitions;
budget 8704; 32B and 7B, full and 128-tuned.

Serving level (`cp_serve.py --workload prefix`): every request has its own P-token prefix + 256 fresh tokens, output
128; warm (all prefixes prefilled before the trace, outside timing, cost recorded) vs cold (same trace, cache reset);
P ∈ {0, 1024, 4096, 8192}; rate and request count fixed per model so the warmed prefixes fit in the KV pool (stage 2).

Physical HBM bytes: no counter has been validated on these cohorts (G4); reported as unavailable. Logical KV demand =
context tokens × KV bytes per token. A prefix-cache hit describes reused KV state, not SRAM residency.

## 6. Experiment 4 — disjoint-CU competition on mixed iterations

Decoder masks 0..n−1, n ∈ {256, 192, 128} (MI300X/MI325X; MI350X 224, 192, 128) with tuned grids; competitor on the
owned disjoint pool (256–303; MI350X 224–255) with the cubw treatments absent, inactive (resident idle), mem25, mem50,
mem100 (sequential read, 1 GiB footprint, duty 0.25/0.5/1.0) and arith100 (FMA). Iterations held fixed: N = 32 decodes at
S ∈ {1024, 8192}, C ∈ {0 (decode-only), 256 (light mix), 1024 (heavy mix)}, 8 repetitions, same replay harness. Four
blocks (one per GPU), treatment order randomized per block. Placement proofs (svc_bench v3) of every decoder mask and the
pool in every block. Isolation of this decoder / pool arrangement was established by leak test v6 (511191, 511192)
for the decode kernels; prefill kernels run under the same hardware queue CU mask. Achieved competitor traffic is read
from the competitor's own log. Response = decoder step time / competitor-absent in the same block, within a mask first;
across masks at matched achieved traffic where it overlaps. Above-stock bandwidth effects are analytical only.

## 7. Experiment 5 — colocated replicas vs disaggregated workers (two GPUs)

`scripts/amd/cp/cp_disagg.py`, two GPUs of one node (lane pairs 0-1, 2-3, 4-5, 6-7; peer topology recorded with
`amd-smi topology`). Deployments at equal GPU count, same model, precision, outputs, traces and SLOs:

- **colocated-stock**: two servers (mixed, B = 2048, full), router = least outstanding work;
- **colocated-tuned**: two servers with the best tuned policy/budget/allocation from experiment 2 for that workload;
- **disagg-full**: prefill worker + decode worker, KV transferred by vLLM's P2pNcclConnector (RCCL, PUT_ASYNC); both
  workers see both GPUs, own GPU first (with only its own GPU visible, RCCL communicator setup failed in the dev run);
- **disagg-restricted**: the same with the decode worker on 128 CUs, tuned grid.
- single-GPU reference (one server; the second board metered idle).

Platform adaptations for the real KV path, found in the dev allocation 529777 (2026-11-18) and required for any
end-to-end disaggregated run on this stack (Qwen2.5-7B dev trace: 30/30 requests completed through the KV transfer):
(1) both workers see both GPUs, own GPU first; (2) the connector forces `NCCL_CUMEM_ENABLE=1`, but RCCL's cuMem peer
path needs Linux ≥ 6.8 (nodes run 5.14), so `cp_sched.py` sets it to 0 inside the same context (`CP_P2P_CUMEM=0`), RCCL
then connects over P2P/IPC; (3) vLLM 0.23 appends a random suffix to request ids per engine, so the two workers keyed
the KV tensors differently and decode waited forever; `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` on both servers (the
client makes every id unique); (4) servers run with `--no-async-scheduling` to match experiments 1–4; (5) per-lane
ports. Peer-copy and RCCL KV-sized transfers reach 41–50 GB/s for ≥ 1024-token KV of Qwen2.5-7B between two GPUs.

Workloads dheavy, balanced, pheavy; open-loop Poisson load grid {0.4 … 1.5} × 2λ* (experiment-2 λ*), held-out seed 201.
Measured: both queues (server metrics every 0.25 s), prefill-worker response time, decode first-token time after the
prefill response (transfer + decode queue), logical KV bytes per request, sustainable goodput, tails, and energy per
output token for both boards over the whole trace (idle capacity of either worker included). KV link: peer-copy and RCCL
send/recv of KV-sized buffers (one buffer and per-layer messages) on the same GPU pairs (`cp_link.py`).

Fallback if a two-device route fails after diagnosis and one retry on another supported route: real KV-size transfers on
the link + both worker service curves (experiment 2 serving at full and restricted allocations) in a labelled modelled
deployment comparison; the real end-to-end case stays open with its blocker and resume command.

## 8. Cross-cohort and second-model reproduction

MI325X (mi3258x) and MI350X (mi3508x) when allocated: experiment 1 at full and 128-tuned (both models) and experiment 2
common-budget points below, near and above the MI300X transitions (balanced and pheavy; rates from that cohort's own
capacity pilot). Qwen2.5-7B: experiments 1, 3 (step level), the pilot, and experiment-2 common budget at full and
128-tuned on MI300X.

## 9. Execution, resumption and allocation

Runner `scripts/amd/cp/cp_run.py` under `scripts/amd/cp/cp.sbatch` from a frozen release snapshot; persistent run roots
under `$WORK/results_amd/cp_<cohort>_<plan>`; a resubmitted job resumes from the journals. At most two attempts per step
(second only after an execution failure). Plans: `scripts/amd/cp/cp_plan.py`. All eight GPUs of a node are used (8
single-GPU lanes or 4 two-GPU lanes). Estimated allocation: stage 1 ≈ 2.5 units (MI300X), stages 2–3 ≈ 6–10 units,
cross-cohort ≈ 4–6 units.

## 10. Delivery

Per plan: published folder `results_published/amd_<date>_<cohort>_cp_<plan>/` with compact tables (JSON), the case
manifest, plan, journals and SHA-256 lists; raw roots on `$WORK` with checksums. Analysis `scripts/amd/cp/cp_analyze.py`;
report `docs/amd/cp_campaign_results_20261118.md` with, per experiment, completed / partial / blocked, evidence and
reproduction commands.

## 11. Amendment 2026-11-18 (after the stage-1 pilot, before any policy-comparison data)

Pilot values (stock: full allocation, mixed, budget 2048; job 529791; frozen SLO rule of section 4):

| Model / workload | L_TTFT (s) | L_ITL (ms, per-request p95) | pilot p90 TTFT (s) | pilot median per-request p95 ITL (ms) |
|---|---|---|---|---|
| Qwen2.5-32B dheavy | 0.48 | 120 | 0.095 | 22.4 |
| Qwen2.5-32B balanced | 1.4 | 110 | 0.263 | 20.1 |
| Qwen2.5-32B pheavy | 4.4 | 89 | 0.871 | 17.7 |
| Qwen2.5-7B dheavy | 0.12 | 28 | 0.022 | 5.45 |
| Qwen2.5-7B balanced | 0.30 | 25 | 0.058 | 4.97 |
| Qwen2.5-7B pheavy | 1.2 | 25 | 0.220 | 4.86 |

**Rate-grid anchor changed (grid placement only).** The pre-registered anchor λ* (highest stock-pilot rate with SLO
attainment ≥ 0.9) is undefined for Qwen2.5-32B pheavy (attainment 0.79 already at 0.25 req/s: with budget 2048 each
prompt chunk stalls co-running decodes beyond the strict ITL limit), is non-monotone for dheavy (0.84 at 2.5, 0.96 at 3.0,
0.93 at 4.0) and gives 0.5 req/s for balanced, which would place the whole grid below saturation. The grid must span
saturation for the policies to be compared, so it is now anchored at **λ_q = the highest ladder rate such that this and
every lower ladder rate at full allocation has no incomplete request and p99 TTFT ≤ L_TTFT**, and the grid is
**{0.5, 0.75, 1, 1.25, 1.5, 2} × λ_q**. λ_q: 32B dheavy 2.0, balanced 1.0, pheavy 0.75; 7B dheavy 10, balanced 10,
pheavy 3 req/s (`scripts/amd/cp/params/mi300x_s2.json`). SLO limits, compliance, sustainable rate, tuning and held-out
rules are unchanged. Experiment-3 serving rate: 0.25 req/s (cold 8448-token prompts keep the worker about one-third busy;
40 warmed 8K prefixes fit in the 450K-token KV pool of Qwen2.5-32B).

**Stage-1 observation that bears on the design (not a judged result).** Fresh prompt tokens cost almost linearly from the
first token at every decode width tested (Qwen2.5-32B full, S = 8192: N = 32 decode step 48 ms; +16 tokens 51 ms; +256
tokens 80 ms; +2048 tokens 352 ms; the same 2048 tokens alone take 418 ms). The 5 % headroom is therefore 0 or 16
tokens everywhere; experiment 1 reports the curves and the attribution.

## 12. Tuning-selection details (2026-11-18, before any stage-2 data)

`cp_analyze.py select`: per (workload, allocation ∈ {full, 128-tuned}, family ∈ {whole-separate, chunked-separate,
mixed}) the tuned budget is chosen on development seed 101, Poisson (the only development trace every candidate has:
the B = 2048 configurations of the common-budget stage ran seed 101 only); sustainable rate = the highest grid rate such
that every trace at that and lower rates has attainment ≥ 0.9 and no incomplete request; ties → mean goodput at the top
grid rate over the available development seeds → smaller budget. The best colocated configuration for experiment 5 is
the family/budget/allocation with the highest sustainable rate per GPU on the same development data (ties → full
allocation). Held-out judging uses seeds 201–203 only.

## 13. Amendment 2026-11-18 — MI350X graph-capture fallback

MI350X job 529874 (x1): every Qwen2.5-32B engine under the 128-CU mask (stock or tuned grid) hung while capturing the
PIECEWISE (mixed) graphs at capture 41 of 51 (size 64), in the 8-GPU job and again on a single-GPU MI350X slice
(diagnostic 530036, `scripts/amd/cp/diag_mi350x_capture.sh`); Qwen2.5-7B at 128 and 192 CUs and Qwen2.5-32B at full
allocation captured normally (530032). Limits 256 and 512 hang; limit 32 completes. Fallback (documented platform
limitation, no driver/firmware change): the MI350X reproduction reruns with `max_cudagraph_capture_size = 32` for every
step at every allocation (plan x1_cap32), so mixed iterations above 32 tokens run eagerly on MI350X; the completed
full-allocation steps of 529874 (limit 512) are kept as an additional comparison. MI350X timings are therefore not
graph-path matched to MI300X/MI325X; cohorts are never pooled.

Update (same day, x1_cap32 530065): with capture limit 32, Qwen2.5-32B at 128 CUs hung again at the first **eager**
mixed iteration of 64 tokens (N = 32 decodes + C = 32), after completing the N = 1 cells; the cap-32 diagnostic had
passed only because its iterations were 72 and 264 tokens. The hang is therefore a 64-token iteration of this model under
the 128-CU mask on MI350X, graph or eager (Qwen2.5-7B at 128 CUs and Qwen2.5-32B at 192 CUs run normally, 530032 and
530036). **Qwen2.5-32B at 128 CUs on MI350X is blocked** (exact blocker above; resume: `cp_submit.sh x1 mi350x 02:30:00
--params scripts/amd/cp/params/mi300x_s2.json` after a ROCm/vLLM change). The MI350X reproduction uses **192 CUs** as its
reduced allocation (stock 512 capture limit; plan x1_low192). Completed steps of 529874 and 530065 are kept and labelled
by their capture limit.
