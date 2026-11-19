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
| 2 | Three scheduling policies under identical open-loop traffic | __E2_STATUS__ | MI300X (dev: 32B and 7B; held-out: 32B); MI325X and MI350X common-budget reproduction |
| 3 | Resident history vs fresh prefill | **completed** (step and serving level; physical HBM bytes unavailable) | MI300X, both models (step), 32B (serving) |
| 4 | Disjoint-CU competition on mixed steps | **completed** | MI300X 32B, masks 128/192/256 |
| 5 | Colocated replicas vs disaggregated workers | __E5_STATUS__ | MI300X 32B, four GPU pairs |

Main answers (details and qualifications below):

- **Prefill headroom in a decode step is essentially zero on these GPUs.** For every model, allocation, decode width and
  context, a step that adds even 16 fresh tokens takes more than 5% longer than the decode-only step. The 32B model is
  the only exception: 16 tokens fit at N = 32. Prefill cost then grows almost linearly in the added tokens:
  - MI300X full: ≈150–220 µs per token;
  - MI300X at 128 CUs: ≈250–300 µs per token;
  - MI325X full: ≈110 µs per token;
  - MI350X full: ≈62 µs per token.

  Decode steps on these GPUs are not compute-starved enough to hide prefill work for free.
- __E2_SUMMARY__
- **Fewer CUs do not preserve serving capacity for the 32B model once prefill is in the mix, even after tuning.**
  - Decode-only steps lose 7–15% at 128 CUs.
  - Prefill costs ≈1.8× per token.
  - Sustainable rate at 128 CUs is half to two thirds of full allocation for decode-heavy and balanced traffic, and half
    or less for prefill-heavy traffic.
  - Tuning the budget and policy does not recover this.
  - 192 CUs keeps 67–83% of the full-allocation rate.
  - The 7B model loses more: half its rate at 128 CUs in every workload.
- **Prefix reuse saves the compute but not the attention service.** For 32B with a warm 8K prefix, the mixed step costs
  104 ms, against 55 ms for the same fresh tokens without history and 1339 ms when cold. The warm-history surcharge is
  about 6 ms per 1K prefix tokens per request (≈1.5 GB/s-equivalent of logical KV per 1K tokens; see section 3).
- **Memory competition on disjoint CUs costs mixed steps about as much as it costs decode steps.**
  - A 2 TB/s streaming competitor slows the decoder 1.13–1.29×.
  - An arithmetic competitor slows it 1.04–1.13×.
  - Light and heavy mixes respond like decode-only steps, within ±0.07.
- __E5_SUMMARY__

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

__E2H__

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

__E5__

## What wins, loses or stays uncertain

__WLU__

## Implications for the area model

__AREA__

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
