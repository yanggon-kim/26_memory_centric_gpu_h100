# Chunked prefill and decode specialization on AMD — results

Branch `amd/mi300x-phase0`. Handoff: [chunked_prefill_campaign_20261118.md](chunked_prefill_campaign_20261118.md).
Pre-registered design and amendments: [cp_campaign_design_20261118.md](cp_campaign_design_20261118.md) (sections 11–13
were committed before the data they govern). Code: `scripts/amd/cp/`. Compact results: `results_published/amd_20261118_*_cp_*`.
Raw roots are on `$WORK/results_amd/` and are listed in each folder's `raw_root.txt` with `raw_SHA256SUMS`.

Stack: vLLM 0.23.0 (V1 engine), ROCm 7.14.1, PyTorch 2.11 in a pinned Apptainer image. Models: Qwen2.5-32B-Instruct
(BF16; 64 layers, 8 KV heads, head dim 128, 256 KiB KV per token) and Qwen2.5-7B-Instruct (BF16; 28 layers, 4 KV heads,
56 KiB KV per token); no sliding window. Allocations are HSA_CU_MASK 0..n−1. "s" means the stock grid and "t" means a
grid tuned to n CUs. No clocks, power limits, drivers or firmware were changed.

## Summary for the owner

| # | Experiment | Status | Cohorts |
|---|---|---|---|
| 1 | Replayed mixed steps and prefill headroom | **completed** | MI300X (both models; 304, 192s/t, 128s/t); MI325X (304, 128s/t); MI350X (256, 192s/t; 128 blocked for 32B) |
| 2 | Three scheduling policies under identical open-loop traffic | **completed** (development, tuning, held-out, stock reference, cross-cohort) | MI300X (dev: 32B and 7B; held-out: 32B); MI325X and MI350X common-budget reproduction |
| 3 | Resident history vs fresh prefill | **completed** (step and serving level; physical HBM bytes unavailable) | MI300X, both models (step), 32B (serving) |
| 4 | Disjoint-CU competition on mixed steps | **completed** | MI300X 32B, masks 128/192/256 |
| 5 | Colocated replicas vs disaggregated workers | **completed** (all four pre-registered deployments and the single-GPU reference on three workloads; two disaggregated cases completed in a rerun after a connector fix; replicated on seeds 202–203, and extended to four-GPU worker ratios and decode KV limits in section 6) | MI300X 32B, GPU pairs and quads |

Main answers (details and qualifications below):

- **Prefill headroom in a decode step is essentially zero on these GPUs.** For every model, allocation, decode width and
  context, a step that adds even 16 fresh tokens takes more than 5% longer than the decode-only step. The 32B model is
  the only exception: 16 tokens fit at N = 32. Prefill cost then grows almost linearly in the added tokens:
  - MI300X full: ≈150–220 µs per token;
  - MI300X at 128 CUs: ≈250–300 µs per token;
  - MI325X full: ≈110 µs per token;
  - MI350X full: ≈62 µs per token.

  Decode steps on these GPUs are not compute-starved enough to hide prefill work for free.
- **Chunked prefill with mixing helps decode-heavy serving and hurts prefill-heavy serving.** Held-out results (seeds
201–203, Poisson):
- decode-heavy: mixed (B = 1024) sustains 3.0 rps against 2.5 for whole or separate scheduling;
- balanced: a tuned small budget (B = 256) ties separate scheduling at 1.5 rps, where stock B = 2048 manages 0.75, and
  holds 0.91 attainment at 2 rps;
- prefill-heavy: separate or whole scheduling sustains 0.56 rps, and mixed stays below 0.375.

  Mixing always gives the smallest stall tails (0.10–0.16 s against up to 6 s). Bursty arrivals defeat every policy at
  these SLOs. The ordering reproduces on MI325X and MI350X.
- **Fewer CUs do not preserve serving capacity for the 32B model once prefill is in the mix, even after tuning.**
  - Decode-only steps lose 0–15% at 128 CUs on MI300X (12–15% at S = 8K, N = 32), and up to 24% on MI325X.
  - Prefill costs ≈ 1.6–1.9× per token.
  - Held-out best sustainable rate at 128t ÷ best at full = 0.67 for all three workloads (±1 grid step), after tuning
    budget and policy.
  - 192t on the common budget keeps 50–83%.
  - The 7B model loses half its rate at 128 CUs.
  - Exception: as a disaggregated **decode** worker, 128 CUs keep the full rate for balanced and prefill-heavy traffic
    (section 5).
- **Follow-up tests (section 6)** settle three open questions:
  - With the right worker ratio, disaggregation wins: 3P1D beats four colocated servers on prefill-heavy traffic (4.5 vs
    3.0–3.75 rps on both seeds), with 34 ms p99 ITL.
  - The experiment 5 conclusions hold on seeds 202 and 203.
  - Below about 16–32 GiB the decode worker becomes KV-limited, and a 128-CU decode worker needs about 11–50% more KV for
    the same rate, because each request stays resident longer.
- **Decode-only CU sweep (section 7).** A decode worker loses ≤ 1.21× per-token speed down to 224 of 304 CUs, 1.3–1.8×
  at 128–192 and 2–3× below 128. With 256 seats and unconstrained KV, its sustainable rate holds down to 96 CUs
  (decode-heavy) and 128 CUs (balanced). The earlier decode-heavy loss at 128 CUs was a seat limit. With KV fixed at 16
  GiB, the rate falls with every CU step below 256.
- **Prefix reuse saves the compute but not the attention service.** For 32B with a warm 8K prefix, the mixed step costs
  104 ms, against 55 ms for the same fresh tokens without history and 1339 ms when cold. The warm-history surcharge is
  about 6 ms per 1K prefix tokens per request (≈1.5 GB/s-equivalent of logical KV per 1K tokens; see section 3).
- **Memory competition on disjoint CUs costs mixed steps about as much as it costs decode steps.**
  - A 2 TB/s streaming competitor slows the decoder 1.13–1.29×.
  - An arithmetic competitor slows it 1.04–1.13×.
  - Light and heavy mixes respond like decode-only steps, within ±0.07.
- **A specialized decode worker improves decode latency and energy, but not capacity, once both GPUs and the KV transfer
are charged.** All results are 32B, one held-out seed.
- **Latency:** disaggregation cuts p99 ITL to 27–47 ms (colocated-tuned: 79–121 ms for decode-heavy and balanced
  traffic; colocated-stock: up to 472 ms). The handoff
  adds ≈ 40–60 ms to TTFT.
- **Energy:** joules per token over both boards drop by 9–31% for decode-heavy and balanced traffic.
- **Capacity against colocated-tuned:**
  - balanced: a tie at the top of the grid (≥ 4 rps);
  - prefill-heavy: a loss (1.12 vs 1.88 rps), with the single prefill GPU saturating;
  - decode-heavy: a loss (3 vs ≥ 8 rps) with the shared 64-seat limit; a tie (≥ 8) with 128 decode seats (exploratory).
- **Decode worker on 128 CUs:** keeps the full-allocation rate for balanced and prefill-heavy traffic, and costs one grid
  step for decode-heavy traffic.

## Deviations and platform adaptations (consequential)

1. **Scheduler control.** vLLM 0.23 has no switch to separate prefill from decode. `cp_sched.PolicyScheduler`
   (passed as `scheduler_cls`) implements it:
   - `separate`: every step is prefill-only or decode-only. Prefill steps hide the running decodes; decode steps hide
     the waiting queue.
   - `--no-chunk`: prompts are scheduled whole.
   - `mixed`: stock V1.

   Each step is logged with its composition, which verifies the policy:
   - `separate` and `whole`: 0 mixed steps in every trace;
   - `mixed`: mixed steps present wherever prefill arrived during decoding.

   Separate mode alternates once per step and adds no priority beyond that.
2. **Replay fidelity (experiment 1).** No engine-state snapshot or restore exists, so the documented fallback was used: a
   barrier admission harness.
   - N requests are prefilled to context S outside the timed window.
   - The harness then alternates paired reference (decode-only) and treatment (decode + C fresh tokens) steps.
   - The scheduler's per-step record checks every timed step's composition. Composition mismatches: 0 in all cohorts.
   - Decode positions drift by ≤ 2×reps tokens within a cell (`ctx_drift_tokens`).
3. **Graph policy.** vLLM's stock serve setting is kept: capture limit 512 tokens.
   - Decode-only steps run FULL graphs.
   - Mixed steps up to 512 tokens run PIECEWISE graphs.
   - Larger steps run eagerly.

   As a result, part of the small-C step increase at N = 1 is the change from FULL to PIECEWISE graph. MI325X, 32B, S = 1K,
   N = 1: +43% at C = 16 versus +44% at C = 32. The `graph_path` and `padded_tokens` columns record this for every cell.
   A larger custom capture list was slower (43.8 vs 35.7 ms at C = 1024), so it was not adopted.
4. **SLO and rate-grid amendment** (design section 11, committed before any policy data):
   - The pre-registered λ* was not defined for every workload, so the grid uses λ_q (the highest monotone ladder rate at
     full allocation with p99 TTFT within the limit).
   - Grid = {0.5, 0.75, 1, 1.25, 1.5, 2}×λ_q.
   - The SLO values are in `scripts/amd/cp/params/mi300x_slo.json` and apply unchanged to every policy, allocation,
     deployment and cohort:

     | Workload | 32B TTFT / ITL | 7B TTFT / ITL |
     |---|---|---|
     | dheavy | 0.48 s / 120 ms | 0.12 s / 28 ms |
     | balanced | 1.4 s / 110 ms | 0.3 s / 25 ms |
     | pheavy | 4.4 s / 89 ms | 1.2 s / 25 ms |
5. **Disaggregated KV path** (design section 7). Five adaptations were needed for vLLM's P2pNcclConnector on RCCL:
   - both GPUs are visible to each worker, own GPU first;
   - `NCCL_CUMEM_ENABLE=0` inside the connector, because RCCL's cuMem path needs kernel ≥ 6.8 and the nodes run 5.14;
   - `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`, because per-engine random id suffixes broke KV tensor keys;
   - `--no-async-scheduling`;
   - per-lane ports.
6. **MI350X at 128 CUs (32B) is blocked.** The engine hangs on a 64-token iteration under graph capture and also
   eagerly. Diagnosed in jobs 530032/530036 on mi3501x and confirmed by the cap-32 retry (530065).
   - Evidence: `results_published/amd_20261118_mi350x_cp_x1_cap32/capture_hang_diagnostics.json`.
   - Fallback (design section 13): MI350X is reproduced at 192 CUs (job 530192, `..._mi350x_cp_x1_low192`).
7. **RAS event during s2** (job 529873, node k002-002, 14:24):
   - All eight engines failed with "unspecified launch failure".
   - GPU3 reported 73 deferred ECC errors and GPU2 reported 2 (`ras_ecc_after_1424.txt`).
   - All eight affected steps were rerun as their second attempt and completed.
   - The failed first attempts are kept in the journal.
8. **Physical HBM bytes** remain unavailable: no validated counter exists on these cohorts (carried over from G4). Only
   logical KV bytes are reported.
9. **Separate scheduling and the per-request p95-ITL rule.**
   - Separate scheduling turns many moderate slowdowns into a few long stalls, and a per-request p95 inter-token limit
     forgives those stalls when a request generates many tokens.
   - Every serving table therefore also reports two exploratory stall metrics, which were not used for selection:
     the p99 over requests of the longest inter-token gap (`req_itl_max_p99_ms`) and the mean ITL.
10. **Held-out stock reference added late.** The s3 plan omitted the pre-registered held-out stock arm (mixed, B = 2048,
    full). It ran afterwards as plan s3r (job 530904), with the same seeds, traces and SLOs, and with
    `family = stock` in the tables. No other held-out arm was rerun or changed.
11. **P2pNcclConnector crash on chunked prompts** (vLLM 0.23 bug).
    - When a continuation chunk needs no new KV block, the scheduler passes `None` in
      `scheduled_cached_reqs.new_block_ids`. `P2pNcclConnector.build_connector_meta` then fails with
      `assert new_block_ids is not None`, killing the prefill worker.
    - In s3 this happened for balanced traffic at ≥ 2 rps and prefill-heavy traffic at ≥ 1.5 rps. Every later request
      failed, and those traces are kept.
    - Fix: `cp_sched._p2p_none_blocks` (enabled by `CP_P2P_NONE_BLOCKS=1`, via `cp_disagg.py --p2p-none-blocks-fix`).
      It substitutes an empty block list for `None` for the duration of that one call and restores the original. KV
      contents and transfer are unchanged.
    - It was unit-tested on the real class, then the four affected cases were rerun in full under new step ids (s3d, job
      530958).
12. **Inductor/Triton cache race.** The colocated-tuned balanced step first failed ("Cubin file saved by TritonBundler
    not found") because two servers compiled into a shared cache at once. The automatic second attempt completed, and
    the failure is kept in the journal.
13. **Decode-seat limit in experiment 5.** All deployments used `max_num_seqs = 64` per engine, as pre-registered. That
    cap, not compute or the KV link, limited the decode-heavy disaggregated runs. The 128-seat runs (disagg
    `--decode-max-seqs 128`, colocated `--max-seqs 128`) are exploratory, added after the data, and are labelled as
    such. The s4 follow-up (design section 14) pre-registers and replicates them.
14. **Decode-worker abort with three prefill peers** (e6 balanced 3P1D). `HSA_STATUS_ERROR_OUT_OF_RESOURCES` hit the
    single decode worker at the top rate in both attempts, the second with more device-memory headroom (s4r). KV usage
    was ≤ 38% and no other e6, e5s or e7 step was affected. The row is reported as a lower bound. The likely cause is
    per-peer RCCL resources (three communicators × 16 channels) combined with deep decode batches, but this is not
    isolated further.
15. **Decode-worker abort in the CU sweep** (e8 balanced, ≤ 192 CUs). The same
    `HSA_STATUS_ERROR_OUT_OF_RESOURCES` abort as deviation 14 ("Available Free mem: 0 MB", in an RCCL kernel) hit the
    single decode worker in 10 of 16 balanced runs at 64–192 CUs. It came at 4–6 rps, always above that run's
    sustainable rate (attainment had already dropped below 0.9), with 74–80 requests running and ≤ 27% KV in use.
    Later traces in those steps record 0 completions. No dheavy run and no run at 224–304 CUs aborted. No second
    attempt was made, because no sustainable rate or knee depends on the affected traces.

## 1. Prefill headroom (replayed steps)

Data:
- `results_published/amd_20261118_mi300x_cp_s1/s1_tables.json` (`e1`, `e1p` profiles, graph paths, CIs);
- `..._mi325x_cp_x1/s1_tables.json`;
- `..._mi350x_cp_x1_low192/s1_tables.json`.

Grid: N ∈ {1, 8, 32}, S ∈ {1K, 8K}, C ∈ {0, 16, 32, 64, 128, 256, 512, 1024, 2048}. Ten pairs per cell.

Headroom rule: the median ratio T_mix/T_ref must have an upper bootstrap 95% CI bound ≤ 1.05. H is the largest C for
which every smaller C is also within the rule.

**Headroom H** (fresh tokens) and decode-only step time at S = 8K, N = 32:

| Cohort / model | allocation | H at N = 1, 8, 32 (S = 1K and 8K) | decode-only step (S 8K, N 32) | marginal prefill (µs per token, C ≥ 256) |
|---|---|---|---|---|
| MI300X 32B | full 304 | 0 / 0 / 0 | 48 ms | 140–218 |
| MI300X 32B | 192s / 192t | 0 / 0 / 16 | 52 ms | 186–201 |
| MI300X 32B | 128s / 128t | 0 / 0 / 16 | 54 / 55 ms | 250–265 / 264–299 |
| MI300X 7B | full / 128s / 128t | 0 everywhere, except 16 at 128t (S 1K, N 8) | 12 / 14 / 14 ms | 30–32 / 51–56 / 51–55 |
| MI325X 32B | full / 128s / 128t | 0 (16 at S 8K, N 32) / 0 or 16 / 0 or 16 | 41 / 51 / 51 ms | 106–116 / 264–278 / 263–278 |
| MI325X 7B | full / 128t | 0 / 32 at N = 8 | 10 / 13 ms | 25–26 / 52–57 |
| MI350X 32B | full 256 / 192s / 192t | 0 everywhere | 29 / 35 / 34 ms | 62–64 / 80–85 / 80–87 |
| MI350X 7B | full / 192t | 0 everywhere | 8 / 10 ms | 15–18 / 19–21 |

Example curve (MI300X 32B full, S = 8K, N = 32):

| Fresh tokens added | Step time |
|---|---|
| 0 (decode only) | 48 ms |
| 16 | 51 ms |
| 256 | 80 ms |
| 2048 | 352 ms |

For comparison, a 2048-token prefill alone takes 418 ms. Per-cell CIs, graph paths and padded token counts are in the
tables.

**Profiles** (rocprofv3, MI300X 32B, S = 8K, N = 8; `e1p`). Linear (GEMM) time per step:

| Allocation | Decode only | +256 tokens | +1024 tokens |
|---|---|---|---|
| full | 18 ms | 44 ms | 128 ms |
| 128t | 23.6 ms | 79 ms | 242 ms |

Attention is a small share at these contexts. Prefill linear work therefore scales with CUs, while decode linear work
does not (it is bandwidth-bound).

**Interpretation.**
- The decode step's GEMMs are weight-streaming. Adding rows does not ride along for free: the first 16 fresh tokens
  already move the kernels onto a different tile and graph path.
- From about 64 tokens onward, every extra token adds the full prefill arithmetic.
- The SARATHI premise ("piggyback prefill onto memory-bound decode at near-zero cost") does not hold at the 5% level on
  MI300X, MI325X or MI350X with this stack. Mixing still amortizes the weight read once per step, so it is cheaper than
  running a separate prefill step of the same size. The +256 mixed step at 80 ms compares with roughly 48 + 47 ms for
  two separate steps.

**Tuned vs stock grid at reduced CUs:**
- Decode: identical within 2%.
- Prefill: within ±10% (MI300X 128t is slightly slower than 128s at S = 1K; MI325X is identical).
- Tuning the grid does not recover the prefill cost of removing CUs.

## 2. Scheduling policies under identical open-loop traffic

Data:
- `results_published/amd_20261118_mi300x_cp_s2/serve_tables.json` (development; per trace: attainment, goodput, p50/p99
  TTFT and ITL, stall metrics, step fractions by count and time, effective P:D, preemptions, energy per output token,
  trace SHA-256);
- `..._mi300x_cp_s3/serve_tables.json` (held-out);
- `..._mi325x_cp_x1` and `..._mi350x_cp_x1_low192` (reproduction).

Traces:
- Request contents are identical across treatments (`content_sha256`). Arrival timestamps are identical per (seed,
  arrival process, rate) (`trace_sha256`).
- 150 requests per trace (32B) or 300 (7B). 10% warmup, then measurement. The drain deadline is the last arrival + 60 s,
  and requests unfinished by then count as failures.

### 2a. Common budget (B = 2048), development seed 101, MI300X 32B

Sustainable rate (requests/s; Poisson; attainment ≥ 0.9 and 0 incomplete at that rate and every lower rate):

| workload (λ_q) | allocation | whole-separate | chunked-separate | mixed |
|---|---|---|---|---|
| dheavy (2.0) | full | 3.0 | 3.0 | 3.0 |
|  | 192t | 1.5 | 1.5 | **2.0** |
|  | 128t | 1.5 | 1.5 | 1.5 |
| balanced (1.0) | full | **1.5** | **1.5** | 1.25 |
|  | 192t | 1.25 | 1.25 | 0.75 |
|  | 128t | 1.0 | 1.0 | 0.75 |
| pheavy (0.75) | full | 0.56 | **0.75** | < 0.375 |
|  | 192t | 0.375 | 0.375 | < 0.375 |
|  | 128t | **0.56** | 0.375 | < 0.375 |

**Bursty arrivals** (gamma gaps, CV 4, same request set and mean rate): no policy or allocation reaches 0.9 attainment
at any grid rate, so the bursty sustainable rate is below 0.5 λ_q everywhere.

The ordering at full allocation is preserved:
- dheavy: mixed ≥ separate (0.92 vs 0.92 at 2.0 rps);
- balanced: whole ≈ separate > mixed (0.65–0.79 vs 0.35–0.64);
- pheavy: separate and whole far above mixed (0.21–0.67 vs 0.04–0.16).

Tail stalls (p99 over requests of the longest inter-token gap, at the highest grid rate, Poisson):
- mixed: 133–699 ms;
- separate or whole: 234–2538 ms for dheavy and balanced, and 15–17 s for pheavy.

Mixing keeps decodes moving through prefill bursts. Separating them keeps TTFT low and the typical ITL clean, but some
requests sit through multi-second prefill phases.

Other measures:
- **Energy per output token** at the highest rate is within ±3% across policies, with mixed slightly higher for
  prefill-heavy traffic (7.4 vs 6.8–6.9 J/token).
- **Effective P:D** (executed prompt tokens including recomputation ÷ decode ticks) matches the offered P:D: dheavy ≈ 0.6,
  balanced ≈ 5, pheavy ≈ 50.
- **Preemptions and recomputation:** 0 in every 32B trace.

**Qwen2.5-7B** (full, common budget):

| Workload | whole / chunked-separate | mixed |
|---|---|---|
| dheavy | 10 rps | 10 rps |
| balanced | 5 rps | < 5 rps (0.84 attainment) |
| pheavy | 3 rps | < 1.5 rps |

At 128t: dheavy 5 rps for every policy; balanced below 5 rps for every policy; pheavy 1.5 rps for separate or whole and
below 1.5 rps for mixed.

### 2b. Tuning (development seeds 101 and 102, Poisson)

Selection rule (design section 12): maximize sustainable rate on seed 101, then goodput at the top rate, then the
smaller budget. Selection files:
- `scripts/amd/cp/params/mi300x_s3.json` (`selection` lists every option);
- `scripts/amd/cp/cp_analyze.py select`.

| workload | allocation | whole | chunked-separate | mixed | best colocated (used in experiment 5) |
|---|---|---|---|---|---|
| dheavy | full | 3.0 | sep2048: 3.0 | mix1024: 3.0 | mixed, B = 1024, full |
|  | 128t | 1.5 | sep512: 1.5 | mix512: 1.5 | |
| balanced | full | 1.5 | sep512: 1.5 | **mix256: 1.5** (attainment 0.84 at 2 rps vs 0.23–0.42) | mixed, B = 256, full |
|  | 128t | 1.0 | sep512: 1.25 | mix256: 0.75 | |
| pheavy | full | 0.56 | **sep4096: 0.75** | mix256: < 0.375 | separate, B = 4096, full |
|  | 128t | 0.56 | sep512: 0.375 | mix256: < 0.375 | |

Effects of tuning:
- **Balanced traffic:** a small mixed budget (B = 256) closes the gap with separate scheduling. It also keeps the best
  attainment past saturation, with the smallest stall tail: 114 ms against 1.5–2.2 s for separate.
- **Prefill-heavy traffic:** no mixed budget reaches 0.9 attainment even at 0.5 λ_q. The ITL limit (89 ms, from a
  low-load pilot) is exceeded by every request whose decode overlaps chunk processing.

### 2c. Held-out judging (seeds 201–203, Poisson and bursty)

Each policy family runs with its tuned budget from 2b, at full and 128t, on held-out seeds 201, 202 and 203.
Arrivals are Poisson and bursty, with 150 requests per trace (648 traces, all complete). Data:
`results_published/amd_20261118_mi300x_cp_s3/serve_tables.json` (`experiment = e2h`). The stock reference (mixed,
B = 2048, full) is in `..._mi300x_cp_s3r` (job 530904; deviation 10).

Sustainable rate (rps) requires attainment ≥ 0.9 and no incomplete requests in **every** held-out seed at that rate and
every lower rate (Poisson). Per-seed values are in brackets. "@λ_q" gives goodput (rps), p99 TTFT (s) and p99 of the
per-request maximum stall (s) at λ_q.

| workload (λ_q) | alloc | whole | chunked-separate (tuned) | mixed (tuned) |
|---|---|---|---|---|
| dheavy (2.0) | full | 2.5 [2.5, 3, 3]; @λ_q 2.09 / 0.14 / 0.23 | sep4096 2.5 [2.5, 3, 3]; 2.09 / 0.14 / 0.23 | **mix1024 3.0** [3, 3, 3]; 2.09 / 0.16 / 0.12 |
|  | 128t | 1.5 [2, 1.5, 1.5]; 1.41 / 0.27 / 0.45 | sep2048 1.5 [2, 1.5, 1.5]; 1.30 / 0.27 / 0.48 | **mix256 2.0** [2, 2, 2]; 2.09 / 0.41 / 0.12 |
| balanced (1.0) | full | 1.25 [1.5, 1.5, 1.25]; 1.04 / 0.50 / 1.06 | sep4096 1.5 [1.5, 1.5, 1.5]; 1.05 / 0.42 / 0.84 | **mix256 1.5** [2, 1.5, 1.5]; 1.05 / 1.10 / 0.10; attainment 0.91 at 2.0 vs 0.37–0.47 |
|  | 128t | **1.0** [1, 1.5, 1]; 1.01 / 1.32 / 2.2 | sep512 **1.0** [1, 1.5, 1]; 1.01 / 1.41 / 2.7 | mix256 0.5 [0.75, 1, 0.5]; 0.86 / 2.27 / 0.18 |
| pheavy (0.75) | full | **0.56** [0.56, 0.75, 0.94]; 0.70 / 2.8 / 4.1 | sep4096 **0.56** [0.56, 0.75, 0.75]; 0.69 / 3.4 / 6.1 | mix256 < 0.375 (attainment 0.85 at 0.375); 0.30 / 10.0 / 0.14 |
|  | 128t | **0.375** [0.56, 0.375, 0.56] | sep512 < 0.375 [0.375, 0.375, 0] | mix256 < 0.375 (0.24) |

**Held-out stock reference** (mixed, B = 2048, full; `..._cp_s3r`, 108 traces, all complete). Sustainable rate (Poisson,
every seed):
- dheavy: 3.0 [3, 3, 3], the same as tuned mixed B = 1024;
- balanced: **0.75** [1.0, 0.75, 1.25], against 1.5 for tuned mixed B = 256 (attainment at 1.5 rps: 0.60 / 0.33 / 0.83);
- pheavy: < 0.375 (0.73–0.84 at 0.375), the same as tuned mixed and below separate (0.56).

Tuning therefore matters most for balanced traffic, where it doubles the mixed rate. For prefill-heavy traffic the gain
comes from switching to separate scheduling (B = 4096). For decode-heavy traffic the stock setting is already as good as
the tuned one. Bursty stock attainment is 0.0–0.99, which is no better than the tuned arms.

**Bursty arrivals** (same request sets and mean rates): no configuration sustains 0.9 attainment in every seed at any
grid rate. The best attainment at 0.5 λ_q:
- dheavy full: mixed 0.89, separate 0.87;
- balanced full: separate 0.82, whole 0.79, mixed 0.50;
- pheavy full: whole 0.59, separate 0.55, mixed 0.16.

Burst-induced queueing breaks the TTFT limits, which were set from a low-load Poisson pilot, for every policy. Mixing
keeps the smallest stall tail under bursts too: 0.11–0.16 s, against 0.9–22 s for the separate policies.

**Energy per output token at λ_q** is within ±3% across policies for each workload and allocation:
- dheavy: 0.85 J (full) and 0.88–0.89 J (128t);
- balanced: 3.0 J;
- pheavy: 10.9–11.5 J (full) and 11.9–15.0 J (128t).

At equal rate and policy, 128t costs 0–4% more energy per token for dheavy and balanced. For pheavy it costs 10–30%
more, because prefill dominates there and each prefill token takes ≈1.8× longer.

**Held-out vs development.** The held-out ranking agrees with development, with two changes:
- At full allocation, decode-heavy mixed (B = 1024) now beats separate by one grid step (3.0 vs 2.5): separate dropped on
  seed 201.
- At 128t, decode-heavy mixed (B = 256) beats separate by one step (2.0 vs 1.5).

**Reduced CUs after tuning** (best family at 128t ÷ best at full):
- dheavy: 2.0 / 3.0 = 0.67;
- balanced: 1.0 / 1.5 = 0.67;
- pheavy: 0.375 / 0.56 = 0.67.

Sustainable rates are grid-quantized, so these ratios are uncertain by ±1 grid step (about ±0.17).

### 2d. Cross-cohort reproduction (common budget, MI300X SLOs and rates)

Sustainable rate (Poisson, seed 101):

| cohort | workload | allocation | whole | chunked-separate | mixed |
|---|---|---|---|---|---|
| MI325X | balanced | full | 2.0 | 2.0 | 1.5 |
|  |  | 128t | 1.25 | 1.25 | 0.75 |
|  | pheavy | full | 0.94 | 0.94 | < 0.375 |
|  |  | 128t | 0.56 | 0.375 | < 0.375 |
| MI350X | balanced | full | 2.0 (≥ top rate) | 2.0 (≥ top) | 2.0 (≥ top) |
|  |  | 192t | 2.0 | 2.0 | 1.5 |
|  | pheavy | full | 1.5 (≥ top) | 0.94 | < 0.375 |
|  |  | 192t | 1.5 (≥ top) | 0.94 | < 0.375 |

On MI350X:
- Prefill-heavy traffic: separate and whole kept attainment at 0.87–0.99 up to 1.5 rps, while mixed fell to 0.25–0.48.
- Balanced traffic stayed at 0.85–1.0 for every policy up to 2 rps.
- At the MI300X rates, 192 CUs on MI350X lose almost nothing (pheavy whole: 1.5 rps at both allocations).

The policy ordering reproduces on all three generations. Each newer GPU raises absolute capacity.

## 3. Resident history and fresh prefill

Data: `..._mi300x_cp_s1/s1_tables.json` (`e3s`, step level) and `..._mi300x_cp_s2/serve_tables.json` (`e3v`, serving).
Every warm step's prefix hit was verified from the scheduler record: cached = P, executed = 256.

**Step level.** Mixed-step time (ms) for N decodes at S = 1K plus 256 fresh tokens (F = 256). The suffix attends to a
prefix of P tokens that is either cached (warm) or recomputed (cold). Columns:
- decode: decode-only step;
- fresh: decode + 256 fresh tokens with no prefix;
- warm and cold: at P = 1K / 4K / 8K;
- build: warm-build cost at P = 8K.

| model / allocation | N | decode | fresh | warm 1K / 4K / 8K | cold 1K / 4K / 8K | build 8K |
|---|---|---|---|---|---|---|
| 32B full | 8 | 23 | 55 | 62 / 81 / 104 | 178 / 635 / 1339 | 1262 |
| 32B full | 32 | 27 | 58 | 65 / 83 / 107 | 182 / 639 / 1337 | 1257 |
| 32B 128t | 8 | 30 | 108 | 117 / 136 / 162 | 397 / 1380 / 2967 | 2803 |
| 32B 128t | 32 | 38 | 115 | 122 / 143 / 166 | 412 / 1395 / 2946 | 2803 |
| 7B full | 8 | 6 | 15 | 18 / 26 / 36 | 45 / 149 / 323 | 311 |
| 7B 128t | 8 | 9 | 24 | 27 / 34 / 44 | 74 / 269 / 596 | 564 |

**Serving level** (32B, 40 requests at 0.25 rps, own prefix + 256 fresh tokens, output 128, mixed B = 2048; TTFT
p50 / p99 in seconds, attainment against the balanced SLOs):

| P | full: warm | full: cold | 128t: warm | 128t: cold |
|---|---|---|---|---|
| 0 | 0.05 / 0.09 (1.00) | — | 0.08 / 0.14 (1.00) | — |
| 1K | 0.06 / 0.11 (1.00) | 0.18 / 0.30 (1.00) | 0.09 / 0.16 (1.00) | 0.33 / 0.59 (1.00) |
| 4K | 0.08 / 0.14 (1.00) | 0.74 / 1.73 (0.89) | 0.11 / 0.21 (1.00) | 1.72 / 4.07 (0.22) |
| 8K | 0.11 / 0.20 (1.00) | 2.17 / 6.06 (0.00) | 0.14 / 0.27 (1.00) | 11.7 / 34.0 (0.00) |

Energy per output token:

| Allocation | Warm, P = 0 → 8K | Cold, 8K |
|---|---|---|
| full | 11.9 → 12.8 J | 16.3 J |
| 128t | 12.1 → 12.9 J | 19.0 J |

**Interpretation.**
- Prefix reuse removes the recompute: a warm step is 8–13× cheaper than a cold one at 8K.
- Warm attention over the resident history still costs a measurable, linear surcharge of ≈6 ms per 1K prefix tokens (32B
  full) or ≈2.6 ms per 1K (7B). Logical KV read: 2 GiB per 8K 32B prefix.
- The warm surcharge is almost allocation-independent: +49 ms at full vs +54 ms at 128t for 32B at 8K. The fresh
  compute part, by contrast, doubles at 128t.
- So with warm prefixes, resident-history service, not arithmetic, dominates the difference from fresh-only steps. That
  service is a memory-side demand. A software prefix hit says nothing about SRAM residency, and physical HBM bytes are
  unavailable.

## 4. Disjoint-CU competition on mixed steps

Data: `..._mi300x_cp_s1/s1_tables.json` (`e4`). Decoder masks 0..n−1 (n = 128, 192, 256, tuned grid); the competitor
runs on CUs 256–303.
- Four blocks (one per GPU) with randomized treatment order and eight repetitions.
- Placement proofs for every mask and pool in every block; leak qualification v6.
- Competitor traffic is read from the competitor's own log: ≈2.05 TB/s at mem100.

Decoder step time ÷ competitor-absent in the same block (median over blocks), shown as C = 0 / 256 / 1024 fresh tokens
at N = 32:

| mask | S | inactive | mem25 | mem50 | mem100 | arith100 |
|---|---|---|---|---|---|---|
| 128 | 1K | 1.00/0.99/1.00 | 1.04/1.05/1.07 | 1.10/1.10/1.13 | 1.22/1.23/1.29 | 1.05/1.09/1.09 |
| 128 | 8K | 1.00/1.00/1.00 | 1.05/1.05/1.07 | 1.10/1.10/1.13 | 1.22/1.22/1.29 | 1.04/1.07/1.09 |
| 192 | 1K | 1.00/1.00/1.00 | 1.05/1.02/1.06 | 1.10/1.06/1.12 | 1.23/1.19/1.26 | 1.04/1.08/1.12 |
| 192 | 8K | 1.00/1.00/1.00 | 1.04/1.04/1.05 | 1.09/1.08/1.11 | 1.22/1.21/1.25 | 1.05/1.05/1.13 |
| 256 | 1K | 1.00/1.00/0.99 | 1.05/1.01/1.02 | 1.10/1.04/1.06 | 1.26/1.13/1.18 | 1.06/1.07/1.07 |
| 256 | 8K | 1.00/1.00/1.00 | 1.05/1.03/1.01 | 1.12/1.07/1.06 | 1.22/1.16/1.18 | 1.06/1.04/1.09 |

**Interpretation.**
- The response is roughly linear in achieved competitor traffic: ≈1.05 at 0.5 TB/s, 1.10 at 1 TB/s, 1.22 at 2 TB/s.
- The response is almost the same for decode-only and mixed steps.
- At 128 CUs, heavy mixes respond slightly more (1.29), because the longer prefill phase overlaps more competitor traffic
  while running on fewer CUs.
- At 256 CUs, mixes respond less (1.13–1.18), because compute dominates there.
- The arithmetic competitor's effect (≤ 1.13) reflects shared power and clock as well as fabric. Clocks were recorded,
  not controlled.
- Shared L2, Infinity Fabric, power and HBM all contribute, and this experiment cannot apportion them.

## 5. Colocated replicas vs disaggregated workers (two GPUs)

Data:
- `results_published/amd_20261118_mi300x_cp_s3/e5_tables.json` (pre-registered run, job 530467);
- `..._mi300x_cp_s3d/e5_tables.json` (job 530958): reruns of the four disaggregated cases that crashed, plus exploratory
  cases.

Setup: MI300X, Qwen2.5-32B, held-out seed 201, Poisson arrivals, 150 requests per trace (135 measured), SLOs as in
experiment 2.
- Two-GPU deployments are offered the aggregate grid {0.5 … 2}×2λ_q.
- The single-GPU reference is offered {0.5 … 2}×λ_q. Its second board is metered idle.
- Energy is joules per output token over **both** boards for the whole trace, so idle capacity on either worker is
  charged.
- KV transfer: P2pNcclConnector over RCCL (PUT_ASYNC) between the two GPUs of a lane pair.

**Sustainable rate** (requests/s, whole two-GPU deployment; attainment ≥ 0.9 and 0 incomplete at that rate and every
lower rate; "≥" means the top of the grid):

| deployment | dheavy (grid 2–8) | balanced (grid 1–4) | pheavy (grid 0.75–3) |
|---|---|---|---|
| single GPU (×2 for two GPUs) | 2.5 (5.0) | 1.25 (2.5) | < 0.375 (attainment 0.73) |
| colocated-stock (mixed, B = 2048) | **≥ 8** | 2.5 | 0.75 |
| colocated-tuned (from 2b) | **≥ 8** (0.97 at 8) | **≥ 4** (0.90 at 4) | **1.88** |
| disagg-full | 3.0 † | **≥ 4** ‡ | 1.12 ‡ |
| disagg-restricted (decode worker on 128t) | 2.0 † | **≥ 4** ‡ | 1.12 ‡ |
| *exploratory:* disagg-full, decode worker 128 seats | ≥ 8 (0.96 at 8) | — | — |
| *exploratory:* disagg-restricted, decode worker 128 seats | 6.0 (0.85 at 8) | — | — |
| *exploratory:* colocated-stock, 128 seats | ≥ 8 (1.00 at 8) | — | — |

† The decode worker hits `max_num_seqs = 64` (64 running) at ≥ 4 rps (full) or ≥ 3 rps (128t). The KV handoff then
queues: the median time from prefill response to first decode token rises from 0.05 s to 1.2–12 s. The setting is
identical in every deployment, but colocated replicas split the load over two engines and 128 seats. The exploratory
128-seat rows remove this limit. They were added after the data and are not pre-registered.

‡ From the s3d rerun with the connector fix (deviation 11). In the pre-registered run, the prefill worker crashed at
balanced ≥ 2 rps and pheavy ≥ 1.5 rps. Those traces are kept in `..._cp_s3` (status complete, requests incomplete). At
the rates valid in both runs, the rerun reproduces s3 to within 0.01 attainment and ±0.03 s p50 TTFT.

**Latency and energy at a common rate** (p50 / p99 TTFT in s; p99 ITL in ms; J per output token over two boards):

| workload, rate | colocated-stock | colocated-tuned | disagg-full | disagg-restricted |
|---|---|---|---|---|
| dheavy, 3 rps | 0.08 / 0.16; 79; 1.20 | 0.10 / 0.15; 90; 1.21 | 0.12 / 0.22; 35; **0.83** | 0.73 / 4.05; 49; 0.93 (seat-capped) |
| dheavy, 8 rps (128 seats where noted) | 0.09 / 0.16; 91; 0.71 | 0.11 / 1.19; 121; 0.76 | 128 seats: 0.15 / 0.63; 47; **0.58** | 128 seats: 0.18 / 7.07; 65; 0.72 |
| balanced, 2 rps | 0.22 / 0.49; 210; 3.41 | 0.41 / 0.95; 82; 3.35 | 0.28 / 0.65; 33; **2.58** | 0.30 / 0.65; 43; 2.59 |
| balanced, 4 rps | 0.29 / 0.73; 280; 1.62 | 0.80 / 1.66; 110; 1.61 | 0.57 / 1.29; 41; **1.47** | 0.63 / 1.35; 62; 1.54 |
| pheavy, 1.12 rps | 0.72 / 1.66; 314; 14.76 | 0.66 / 1.60; 27; **14.50** | 1.34 / 3.66; 28; 14.79 | 1.38 / 3.81; 33; 15.01 |
| pheavy, 1.88 rps | 1.16 / 3.64; 472; 9.86 | 1.09 / 3.16; 964; **9.93** | 16.9 / 25.6; 29; 12.93 | 17.4 / 26.4; 36; 13.04 |

**KV handoff** (disaggregated runs below saturation):

| workload | logical KV per request | prefill-worker response p50 | handoff p50, full / 128t decode |
|---|---|---|---|
| dheavy | 84 MB | 0.07–0.08 s | 0.046–0.063 / 0.060–0.089 s |
| balanced | 345 MB | 0.22–0.52 s | 0.040–0.061 / 0.051–0.089 s |
| pheavy | 1057 MB | 1.0–1.6 s | 0.037–0.043 / 0.044–0.052 s |

The handoff time is the prefill response to the first decode token, so it includes transfer, decode admission and the
first decode step. It barely depends on KV size (84 MB to 1.06 GB): the PUT_ASYNC transfer overlaps the prefill
response, and the remainder is decode-side admission and one step. The 1.06 GB pheavy handoff at ≈ 40 ms is consistent
with the 41–50 GB/s RCCL rate for KV-sized buffers (design section 7) only because the transfer overlaps the prefill
response.

**Interpretation.**
- **Decode quality.** The specialized decode worker gives 2–3× lower p99 ITL than colocated-tuned for decode-heavy and
  balanced traffic (27–47 ms against 79–121 ms), and 6–15× lower than colocated-stock for balanced and prefill-heavy
  traffic (210–472 ms). For prefill-heavy traffic, colocated-tuned (separate scheduling) matches it at low rate (27 ms)
  but its p99 ITL reaches 0.5–1.8 s from 1.5 rps. No prefill ever runs on the decode worker's GPU. It also lowers energy per token by 9–31% for decode-heavy and balanced
  traffic, mostly because the prefill GPU idles cheaply while the decode GPU batches deeper.
- **Capacity after charging both GPUs:**
  - Disaggregation never beats colocated-tuned.
  - dheavy: both reach the top of the grid only with 128 decode seats (exploratory). With the pre-registered 64 seats,
    disaggregation sustains 3 rps against ≥ 8.
  - balanced: both reach the top of the grid (≥ 4 rps), and disaggregation keeps much lower ITL.
  - pheavy: disaggregation loses (1.12 vs 1.88 rps). A single prefill GPU saturates (prefill response grows from 1.3 s
    to 17 s at 1.88 rps) while the decode GPU runs 5 requests. That is a 1:1 pool imbalance, and a P:D worker ratio
    greater than 1 would be needed.
- **Disaggregation beats colocated-stock** for balanced (≥ 4 vs 2.5) and pheavy (1.12 vs 0.75), and beats the
  single-GPU reference ×2 except for decode-heavy traffic.
- **TTFT.** Disaggregation adds ≈ 40–60 ms of handoff to TTFT. That is harmless against the 32B limits for balanced
  and prefill-heavy traffic, and costs part of the 0.48 s decode-heavy margin.
- **Decode worker on 128 CUs.** It costs nothing in sustainable rate for balanced or prefill-heavy traffic. ITL p99
  rises by 5–21 ms and energy by 0–2%. For decode-heavy traffic, it costs one grid step (2 vs 3 with 64 seats; 6 vs ≥ 8
  with 128 seats), because the per-step time rises with batch width at 128 CUs (ITL p50 59 vs 41 ms at 8 rps).

## 6. Follow-up tests (pre-registered in design section 14)

Data:
- `results_published/amd_20261119_mi300x_cp_s4/e5_tables.json` (job 531381; 70 steps, 420 traces). Rows are
  distinguished by `experiment` = e6, e5s or e7.
- `..._cp_s4r` (job 531813): the second attempt of the 3P1D balanced row.
- `..._cp_s4_smoke` (job 531379): the four-GPU path check.

Settings: MI300X, Qwen2.5-32B, Poisson arrivals. SLOs and the sustainable-rate rule are unchanged. Every disaggregated
worker runs with the connector fix. Energy counts every board of the deployment over the whole trace.

Because the arrival window fixes the trace span, joules per token at equal rate mostly reflect average board power. Two
deployments that both keep up can land within 1% of each other.

### 6a. Prefill:decode worker ratio on four GPUs (e6)

Sustainable rate (requests/s, whole four-GPU deployment; seed 201 / seed 202; "≥" means the top of the grid):

| workload (grid) | colo4_tuned | 1P3D | 2P2D | 3P1D | pre-registered decision |
|---|---|---|---|---|---|
| dheavy (4–16) | ≥ 16 / ≥ 16 | ≥ 16 / ≥ 16 | ≥ 16 / ≥ 16 | — | **uncertain** (tie at top of grid) |
| balanced (2–8) | ≥ 8 / ≥ 8 | 4 / 4 | ≥ 8 / ≥ 8 | 6 / 6 † (≥ 6 lower bound) | **uncertain** (2P2D ties at top of grid) |
| pheavy (1.5–6) | 3.75 / 3.0 | — | 2.25 / 2.25 | **4.5 / 4.5** | **win** for 3P1D (+1 and +2 grid steps) |

Secondary outcomes at the highest rate where both deployments are sustainable:

| workload, rate | colo4_tuned: p99 TTFT s; p99 ITL ms; J/token | best ratio: same |
|---|---|---|
| dheavy, 16 rps | 0.14; 88; 0.69–0.71 | 2P2D: 1.4–1.75; 46–47; **0.57–0.58** (1P3D: 0.50–0.62 s; 38–39 ms; 0.65–0.66 J) |
| balanced, 8 rps | 1.15–1.17; 69–80; 1.70–1.84 | 2P2D: 0.82–1.02; 37–38; **1.52–1.64** |
| pheavy, 3 rps | 1.63–1.81; 553–571; 11.3–11.7 | 3P1D: 2.0–2.1; 34; **9.8–10.0** |

- **The right ratio turns the two-GPU loss for prefill-heavy traffic into a win.**
  - At 1P1D, disaggregation lost (1.12 vs 1.88 rps per two GPUs).
  - At 3P1D on four GPUs it sustains 4.5 rps on both seeds, against 3.75 and 3.0 for four tuned colocated servers.
    That is a pre-registered win of one and two grid steps.
  - Its p99 ITL is 34 ms against 0.55–1.2 s for colocated separate scheduling, which stalls decodes behind 4K-token
    prefills.
  - 2P2D loses (2.25): prefill capacity, not decode, is the limit for this workload.
- **A wrong ratio loses badly.** For balanced traffic, 1P3D sustains half of what colo4 or 2P2D sustain (4 vs ≥ 8),
  because one prefill GPU saturates. For prefill-heavy traffic, 2P2D trails colo4.
- **Decode-heavy and balanced capacity tie at the top of the grid.** At equal rate, disaggregation still halves the p99
  ITL and uses 6–18% less energy per token. For decode-heavy traffic at 16 rps it raises the p99 TTFT above the 0.48 s
  limit (0.5–1.75 s, attainment still 0.95–0.98): the handoff queues behind the decode workers' admission.
- † **3P1D balanced: the decode worker aborts at the top rate** (deviation 14). The worker receives from three
  prefill peers with about 75–99 requests running and 25–38% KV in use. It aborts with
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES` ("too many resources requested for launch").
  - First attempt (s4, `gpu_memory_utilization` 0.85): the abort came at 8 rps on seed 201 and at 6 rps on seed 202.
  - Second attempt (s4r, decode worker at 0.80): both seeds sustain 6 rps, and the abort came at 8 rps.
  - Both attempts are used and kept. The 3P1D row is therefore a lower bound (≥ 6). It does not change the balanced
    decision, because 2P2D already ties colo4 at the top of the grid.

### 6b. Experiment 5 on more seeds (e5s)

All-seed sustainable rate, the minimum over seeds 201, 202 and 203 (per-seed values in brackets). Seed 201 comes from s3
or s3d; seeds 202 and 203 come from s4.

| deployment (two GPUs) | dheavy | balanced | pheavy |
|---|---|---|---|
| colo_stock | ≥ 8 [8, 8, 8] | 2.5 [2.5, 3, 3] | < 0.75 [0.75, 0.75, < 0.75 (0.89)] |
| colo_tuned | ≥ 8 [8, 8, 8] | ≥ 4 [4, 4, 4] | 1.88 [1.88, 1.88, 1.88] |
| disagg_full (64 seats) | 3 [3, 3, 3] | ≥ 4 [4, 4, 4] | 1.12 [1.12, 1.12, 1.12] |
| disagg_r128 (64 seats) | 2 [2, 2, 3] | ≥ 4 [4, 4, 4] | 1.12 [1.12, 1.12, 1.12] |
| disagg_full, 128 decode seats | ≥ 8 [8, 8, 8] | — | — |
| disagg_r128, 128 decode seats | 6 [6, 6, 6] | — | — |
| colo_stock, 128 seats | ≥ 8 [8, 8, 8] | — | — |

Every seed-201 ranking against colo_tuned holds on seeds 202 and 203, so the section 5 conclusions are **confirmed**:
- the tie for balanced traffic;
- the loss at 1P1D for prefill-heavy traffic;
- the seat-capped loss and the 128-seat tie for decode-heavy traffic;
- the one-step cost of a 128t decode worker for decode-heavy traffic only.

The 128-seat decode-heavy rows are now pre-registered and replicated. The latency and energy pattern also repeats on the
new seeds:
- disaggregated p99 ITL is 26–64 ms, against 56–89 ms for colo_tuned;
- energy at equal rate, against colo_tuned, is 10–31% lower for disagg_full on decode-heavy and balanced traffic, and
  3–30% lower with a 128t decode worker.

### 6c. Decode KV capacity (e7; 1P1D, 256 decode seats, seed 201)

Sustainable rate (two-GPU deployment) by decode-worker KV budget. 1 GiB holds 4096 tokens of Qwen2.5-32B KV.

| workload (grid) | decode allocation | 4 GiB | 8 GiB | 16 GiB | 32 GiB | default (≈ 100 GiB) | knee |
|---|---|---|---|---|---|---|---|
| dheavy (2–8) | full | < 2 | 3 | 6 | ≥ 8 | ≥ 8 | 16 GiB |
| dheavy (2–8) | 128t | < 2 | 2 | 4 | ≥ 8 | ≥ 8 | 32 GiB |
| balanced (1–4) | full | 1.5 | 2 | ≥ 4 | ≥ 4 | ≥ 4 | 16 GiB |
| balanced (1–4) | 128t | 1.0 | 2 | 3 | ≥ 4 | ≥ 4 | 16 GiB |

Peak decode-worker KV in use when not limited (default or 32 GiB budget, top rate):

| workload, rate | full: running / KV GiB | 128t: running / KV GiB | p50 ITL full / 128t |
|---|---|---|---|
| dheavy, 8 rps | 129 / ≈ 18 | 146 / ≈ 20 | 39 / 58 ms |
| balanced, 4 rps | 42 / ≈ 15 | 62–63 / ≈ 22 | 33 / 50–52 ms |

- **Below the knee, KV capacity sets the rate.** At 4–8 GiB, KV usage reaches 100% and the decode worker preempts:
  - 49–995 preemptions per step;
  - none in any unconstrained row;
  - handoff time grows from 0.05 s to 2–27 s.
- **Reduced decode compute raises the KV needed for the same rate.** This rejects the pre-registered hypothesis that
  capacity-limited rates would be equal at full and 128t.
  - At 128t each decode step is 1.5× slower (p50 ITL 50–58 vs 33–39 ms). Each request stays resident longer, so more
    requests are running at once (Little's law). Peak KV at the same rate rises by about 11% (dheavy) and about 50%
    (balanced).
  - Consequences: at 8 and 16 GiB the 128t worker sustains one to two grid steps less than full. Its knee moves from 16
    to 32 GiB for decode-heavy traffic.
- **Model input.** Decode KV demand ≈ rate × decode residency × mean resident tokens per request. Residency includes the
  per-token step time, which depends on decode CUs.
  - Measured: ≈ 18 GiB at 8 rps (dheavy) and ≈ 15 GiB at 4 rps (balanced) on full allocation.
  - These rise to ≈ 20 GiB and ≈ 22 GiB at 128t.
  - A decode-specialized part that gives up compute must therefore budget more KV bytes per unit of rate, not fewer.

## 7. Decode-only CU sweep (e8; pre-registered in design section 15)

Plan `s5`, job 534996 (2:22 on one mi3008x node; 40/40 steps completed). 1P1D on GPU pairs: the prefill worker at full
allocation, the decode worker at n CUs with the matching grid (nt), 256 decode seats, 150 requests, held-out seeds 201
and 202 (main arm) or 201 (kv16 arm). Published: `results_published/amd_20261120_mi300x_cp_s5` (`e5_tables.json`, 320
rows). SLOs as before: dheavy TTFT 0.48 s and per-request ITL p95 120 ms, balanced 1.4 s and 110 ms.

**Sustainable rate (two-GPU deployment, all-seed minimum) by decode CUs**

| arm (grid) | 64 | 96 | 128 | 160 | 192 | 224 | 256 | 304 (full) |
|---|---|---|---|---|---|---|---|---|
| dheavy, default KV (2–12) | 3 | ≥ 12 | ≥ 12 | ≥ 12 | ≥ 12 | ≥ 12 | ≥ 12 | ≥ 12 |
| balanced, default KV (1–6) | 2.5 | 3 | 4 | 4 | 4 | 4 | 4 (5 on s202) | 4 |
| dheavy, KV fixed 16 GiB (2–12, s201) | 2 | 3 | 4 | 4 | 4 | 5 | 6 | 6 |

**Decode ITL p50 in ms (mean of seeds) at fixed rates, and ratio to full allocation**

| arm, rate | 64 | 96 | 128 | 160 | 192 | 224 | 256 | 304 |
|---|---|---|---|---|---|---|---|---|
| dheavy, 3 rps | 94 (3.2×) | 64 (2.2×) | 51 (1.8×) | 44 (1.5×) | 46 (1.6×) | 35 (1.21×) | 32 (1.10×) | 29 |
| dheavy, 12 rps | 118 (2.7×) | 84 (2.0×) | 64 (1.5×) | 62 (1.4×) | 62 (1.4×) | 52 (1.21×) | 45 (1.05×) | 43 |
| balanced, 2.5 rps | 67 (2.2×) | 49 (1.6×) | 40 (1.3×) | 39 (1.26×) | 37 (1.19×) | 35 (1.13×) | 33 (1.06×) | 31 |
| balanced, 4 rps | — | — | 53 (1.6×) | 46 (1.4×) | 41 (1.24×) | 39 (1.18×) | 36 (1.09×) | 33 |

**Knee (pre-registered definition: within one grid step of the full-allocation rate and ITL p50 ≤ 1.25× full at the
highest rate every level sustains)**

| arm | rate-only knee | ITL criterion (rate used) | knee |
|---|---|---|---|
| dheavy, default KV | 96 (top of grid; lower bound on rate) | 224 (3 rps) | **224** |
| balanced, default KV | 96 | 192 (2.5 rps) | **192** |
| dheavy, KV 16 GiB | 224 | 192 (2 rps) | **224** |

**Hypotheses**
- **H1 rejected for dheavy, confirmed for balanced.** With 256 decode seats and unconstrained KV, the dheavy sustainable
  rate does not fall until 64 CUs. It stays at the top of the extended grid (12 rps, 1.5× the old ceiling) at 96 CUs.
  The one-step dheavy loss at 128t in experiment 5 and e5s was therefore a seat limit (64 or 128 seats), not a compute
  limit: slower decode keeps more requests running. Balanced holds the full-allocation rate down to 128 CUs and loses
  one step at 96. At 4–5 rps the balanced runs are limited by the single prefill worker at every CU level (TTFT p99
  1.8–3.7 s), so the top of that curve is not decode-limited.
- **H2 rejected for dheavy, confirmed for balanced.** ITL p50 rises as CUs fall at every rate (160 and 192 CUs are
  within ±2 ms of each other, the only non-monotone pair). 304 → 192 costs 1.4–1.6× for dheavy (above the 1.25× bound)
  and 1.09–1.24× for balanced. The cost is small down to 224–256 CUs (≤ 1.21×), moderate at 128–192 (1.3–1.8×) and
  steep below 128 (2–3×).
- **H3 rejected on the stated criterion, but the effect is large.** The kv16 knee (224) equals the main-arm knee (224)
  because the ITL criterion already places the main knee at 224. On rate alone, the two arms diverge sharply. With
  unconstrained KV, dheavy holds ≥ 12 rps down to 96 CUs. With 16 GiB, the rate falls almost monotonically with CUs: 6
  at 256–304, 5 at 224, 4 at 128–192, 3 at 96 and 2 at 64. Preemptions start at 8 rps on full, 6 at 224–256, 5 at
  160–192, 4 at 96–128 and 3 at 64, which matches the rise in residency (Little's law). This comparison is a reading
  added after the data, not the pre-registered test.

**Other outcomes**
- **Energy.** Joules per output token over both boards rise as decode CUs fall, because the run is longer and masked CUs
  do not stop the board's power draw. At dheavy 12 rps: 0.50 J at 304 CUs, 0.53 at 256, 0.58 at 224, 0.61 at 192, 0.65 at
  128, 0.75 at 96 and 1.46 at 64. Balanced varies by at most 5% from 128 to 304 CUs at sustained rates. Masks do not
  stand for the power or area of a part designed with fewer CUs.
- **KV and running requests.** Unconstrained, the decode worker never exceeded 22% of its default KV (≈ 21–22 GiB), with
  0 preemptions. At dheavy ≥ 8 rps, peak running reaches 150, i.e. every request in the trace is resident at once. The
  top dheavy rates therefore test absorption of a 150-request burst, not steady state.
- **SLO dependence.** The dheavy rate holds down to 96 CUs because 84 ms ITL p50 still passes the 120 ms SLO. With a
  tighter ITL target, the rate knee would move toward the ITL knee. The ITL curve is the SLO-independent result.
- **Crash.** In 10 of the 16 balanced runs at ≤ 192 CUs, the decode worker aborted with the deviation 14 error once the
  rate was already past its sustainable point (deviation 15). No dheavy run and no run at 224–304 CUs aborted.

**What this means for the decode-specialized part.**
- On MI300X with this model, a decode worker keeps ≥ 0.8× of its per-token speed down to about 224 CUs (74%), and it
  keeps its sustainable rate at the stated SLOs down to 96–128 CUs if it has enough seats and KV.
- Between 224 and 128 CUs the cost is per-token latency (1.3–1.8×), which converts into more resident requests and more
  KV. Below 96 CUs both latency and rate collapse.
- With KV fixed, every CU removed below about 256 costs rate. A design that removes compute must add capacity in step.

## What wins, loses or stays uncertain

**Wins (design better than its baseline, beyond grid quantization):**
- Mixed chunked prefill with a tuned budget for **decode-heavy** traffic: +1 grid step held-out at full (3.0 vs 2.5 rps)
  and at 128t (2.0 vs 1.5), with the smallest stall tails.
- Tuning the mixed budget for **balanced** traffic: B = 256 doubles the stock B = 2048 held-out rate (1.5 vs 0.75 rps),
  ties separate scheduling, and degrades most gracefully past saturation (0.91 vs 0.37–0.47 attainment at 2 rps).
- **Separate or whole scheduling for prefill-heavy traffic** at every allocation and on all three GPU generations (mixed
  never reaches 0.9 attainment).
- **Disaggregated decode** for decode latency and energy: p99 ITL 2–3× lower than colocated-tuned and up to 15× lower than colocated-stock, and 9–31% fewer joules per token for
  dheavy and balanced at equal rate, with both boards charged.
- **Decode worker at 128 CUs** for balanced and prefill-heavy traffic: same sustainable rate as full. With 256 decode
  seats and unconstrained KV, decode-heavy traffic also keeps its rate, down to 96 CUs (section 7).

- **Disaggregation with a prefill-weighted ratio for prefill-heavy traffic** (section 6a): 3P1D sustains 4.5 rps on four
  GPUs, against 3.0–3.75 for colo4_tuned, on both seeds.

**Loses:**
- Free prefill piggybacking: headroom is 0–16 tokens in every cohort, model and allocation.
- Mixed scheduling for prefill-heavy traffic: < 0.375 rps against 0.56 held-out (MI300X), and against 0.94–1.5 on
  MI325X and MI350X.
- Fewer CUs in colocated serving: 128t keeps only two thirds of the held-out rate in every workload, even after tuning.
- Disaggregation for prefill-heavy traffic at a 1:1 worker ratio: 1.12 vs 1.88 rps for colocated-tuned, confirmed on
  three seeds. Mismatched ratios (1P3D for balanced, 2P2D for pheavy) also lose.
- A reduced-CU decode worker with a tight KV budget: it is 1–2 grid steps below full at 8–16 GiB (section 6c).
- Disaggregation with the shared 64-seat limit for decode-heavy traffic: 3 vs ≥ 8 rps.
- Every policy under bursty arrivals (CV 4) at these SLOs.

**Uncertain:**
- Disagg vs colocated capacity for **balanced** and **decode-heavy** traffic: both are at or above the top of the grid
  on two and four GPUs, on every seed.
- 3P1D balanced beyond 6 rps: the decode worker aborts (deviation 14).
- Ratios of reduced-CU to full capacity: ±1 grid step, about ±0.17.
- Worker ratios beyond four GPUs and across nodes: not tested.
- Physical HBM traffic: unavailable. Energy figures come from board counters, with no external calibration.
- The four-GPU ratio test used two seeds, and the KV-capacity test used one.
- The decode CU knee depends on the ITL SLO and was measured for one model; at dheavy ≥ 8 rps the 150-request traces
  are fully resident, so the top rates measure burst absorption (section 7).

## Implications for the area model

These are model inputs, not architectural results. CU masks do not reclaim area, and HBM stacks are not compute-die area
(see the fixed-area model V0).
- **Decode vs prefill sensitivity to compute.**
  - Decode-only steps lose 12–15% at 128 of 304 CUs on MI300X (S 8K, N 32) and up to 24% on MI325X (41 → 51 ms).
  - Marginal prefill cost per token rises ≈ 1.6–1.9× (MI300X 140–218 → 250–299 µs per token; MI325X 106–116 →
    263–278).
  - So a decode-only (D) part tolerates a large compute cut, but a prefill-plus-decode (PD) part pays almost
    proportionally on the prefill share.
- **Step-time model for PD.**
  - Use T_mix(C) ≈ T_dec + c_p(CUs)·C for C ≥ 64 with H ≈ 0. The c_p values are in the experiment 1 tables.
  - Do not assume free piggybacking. Mixing does save the second weight read (one 80 ms step against 48 + 47 ms for two
    separate steps).
- **Serving-level calibration.** Best held-out capacity at 128t ÷ full = 0.67 for every workload (±0.17). With 192t on
  the common budget it is 0.50–0.83. The serving loss exceeds the decode-step loss, because prefill sets the queueing.
- **Disaggregated D pool.**
  - A 128-CU decode worker loses no sustainable rate for balanced or prefill-heavy traffic, and one grid step for
    decode-heavy.
  - A specialized decode die with reduced compute is therefore viable when the prefill pool keeps full compute and the
    decode pool has enough KV seats.
  - KV capacity (seats), not compute, limited the decode worker first.
- **Decode KV term** (section 6c). KV demand ≈ rate × decode residency × resident tokens per request.
  - Measured at full allocation: ≈ 18 GiB at 8 rps (dheavy) and ≈ 15 GiB at 4 rps (balanced). At 128t: ≈ 20 and 22 GiB.
  - Below that, the rate collapses with preemptions.
  - A decode part that gives up compute must budget more KV bytes per unit of rate. The memory-side term therefore grows
    when the compute area shrinks.
- **Decode compute curve** (section 7). Per-token decode time at 224 / 192 / 128 / 96 / 64 CUs ≈ 1.13–1.21 / 1.2–1.6 /
  1.3–1.8 / 1.6–2.2 / 2.2–3.2 × full (MI300X, 32B, 256 seats). Use this curve, not a linear interpolation between 128 and
  304, for the decode-only part. Rate is preserved to 96–128 CUs only when seats and KV grow with the residency the slower
  step creates.
- **Worker ratio** (section 6a). The prefill pool must match the offered prompt work: 3P1D for P:D ≈ 50, 2P2D for
  P:D ≈ 5, and 1P3D–2P2D for P:D ≈ 0.6. With the right ratio, a full-compute prefill pool and a decode pool beat
  colocated serving on prefill-heavy traffic and tie elsewhere.
- **Transfer term.**
  - KV handoff costs ≈ 40–90 ms per request for 84 MB to 1.06 GB of logical KV over the intra-node link (41–50 GB/s).
  - It overlaps the prefill response, so it adds latency, not a throughput limit, at ≤ 4 rps.
  - Charge it as TTFT, plus link bandwidth of KV bytes per request × request rate (≤ 2 GB/s here).
- **Shared-service term.**
  - Mixed steps respond to memory competition like decode steps: ≈ +0.11 slowdown per TB/s of competitor traffic.
  - Resident-history attention costs ≈ 6 ms per 1K tokens (32B), independent of CU count.
  - Both are memory-side demands that do not shrink with compute area.
- **Energy.** At equal rate, a 128t allocation costs 0–4% more energy per token for decode-heavy and balanced traffic, and
  10–30% more for prefill-heavy traffic. Disaggregation saves 9–31% at equal rate for decode-heavy and balanced traffic.

## Reproduction

All commands run from the repository root on the login node; jobs go to Slurm. Plans are generated inside a frozen
release snapshot (`$WORK/cp_release/<sha>`), and resubmitting with `--root <root>` resumes a run root.

```bash
# stage 1: experiments 1, 3 (step), 4, pilot, KV link — MI300X (job 529791)
scripts/amd/cp/cp_submit.sh s1 mi300x 03:00:00
# stage 2: experiment 2 common budget + tuning, experiment 3 serving (job 529873)
scripts/amd/cp/cp_submit.sh s2 mi300x 10:00:00 --params scripts/amd/cp/params/mi300x_s2.json
# stage 3: experiment 2 held-out + experiment 5 (job 530467)
scripts/amd/cp/cp_submit.sh s3 mi300x 09:30:00 --params scripts/amd/cp/params/mi300x_s3.json
# held-out stock reference (job 530904) and experiment-5 reruns with the connector fix + exploratory seats (job 530958)
scripts/amd/cp/cp_submit.sh s3r mi300x 02:30:00 --params scripts/amd/cp/params/mi300x_s3.json
scripts/amd/cp/cp_submit.sh s3d mi300x 01:45:00 --params scripts/amd/cp/params/mi300x_s3.json
# follow-up (design section 14): smoke (531379), main (531381), 3P1D second attempt (531813)
scripts/amd/cp/cp_submit.sh s4 mi300x 01:00:00 --smoke --params scripts/amd/cp/params/mi300x_s3.json
scripts/amd/cp/cp_submit.sh s4 mi300x 10:00:00 --params scripts/amd/cp/params/mi300x_s3.json
scripts/amd/cp/cp_submit.sh s4r mi300x 01:30:00 --params scripts/amd/cp/params/mi300x_s3.json
# decode-only CU sweep (design section 15; job 534996)
scripts/amd/cp/cp_submit.sh s5 mi300x 06:00:00 --params scripts/amd/cp/params/mi300x_s3.json
# cross-cohort: MI325X (529875), MI350X at 192 CUs (530192)
scripts/amd/cp/cp_submit.sh x1 mi325x 02:30:00 --params scripts/amd/cp/params/mi300x_s2.json
scripts/amd/cp/cp_submit.sh x1 mi350x 02:30:00 --params scripts/amd/cp/params/mi300x_s2.json --low 192
# MI350X 128-CU attempts kept for the record: 529874 (stock capture) and 530065 (--max-capture 32)
# analysis (CPU; devel partition)
srun -p devel -N 1 -n 1 -c 4 -t 00:30:00 python3.12 scripts/amd/cp/cp_analyze.py s1 --root <root>
srun -p devel -N 1 -n 1 -c 4 -t 00:30:00 python3.12 scripts/amd/cp/cp_analyze.py serve --root <root> --slo scripts/amd/cp/params/mi300x_slo.json
srun -p devel -N 1 -n 1 -c 4 -t 00:30:00 python3.12 scripts/amd/cp/cp_analyze.py disagg --root <root> --slo scripts/amd/cp/params/mi300x_slo.json
python3.12 scripts/amd/cp/cp_analyze.py select --serve <s2 root>/analysis/serve_tables.json --params scripts/amd/cp/params/mi300x_s2.json --out scripts/amd/cp/params/mi300x_s3.json
scripts/amd/cp/cp_publish.sh <root>
```

MI350X 128-CU blocker, reproduced with: `scripts/amd/cp/diag_mi350x_capture.sh` (mi3501x).

## Acknowledgement

This work was supported in part by Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster.
