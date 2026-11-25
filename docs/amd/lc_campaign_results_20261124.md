# Long-context and long-output chunked prefill on MI300X — results

Status: **DRAFT (lc1 and lc2 in; lc3 running)**. Design and pre-registration: `docs/amd/lc_campaign_design_20261124.md`
(amendments 1–3). Handoff: `docs/amd/long_context_chunked_prefill_20261124.md`. Baseline: the November 18 campaign.

All numbers: Qwen2.5-32B, BF16, one MI300X per lane (TP1), vLLM 0.23.0 V1 in process, ROCM_ATTN, CUDA graphs
FULL_AND_PIECEWISE ≤ 512 tokens unless "eager". 256 KiB of KV per token, 64 GB of weights. Clocks are not controlled.
Medians, with paired bootstrap 95% CIs where stated. Sources: `results_published/amd_20261124_mi300x_lc_lc1/lc_tables.json`
(job 547842) and `results_published/amd_20261125_mi300x_lc_lc2/lc_tables.json` (job 548920); lc3 to follow.

## 0. Answers in one paragraph each (lc1 scope; serving answers pending lc2/lc3)

1. **Chunking cost vs a whole prompt.** At 32 000 tokens, chunking costs 1.08× (B 4096), 1.54× (B 1024) and 2.44×
   (B 256) the whole-prompt time. The penalty grows with length at every budget, and at 1 024 tokens one chunk is free
   (0.996×). The cost is not extra arithmetic: MFMA instruction counts are within 2–3% of whole. It is not added reads
   either. The c·k(k−1)/2 prefix revisit is 133 GB at B 1024 (≈ 33 ms at the measured 4.1 TB/s), against a 3.8 s penalty,
   and the measured pre-MALL request-side reads are *lower* at B 1024 (5.0 TB) than whole (10.5 TB). The penalty is
   kernel efficiency. The prefill-attention kernel over a paged prefix takes 1.93× (B 1024) and 3.55× (B 256) the
   whole-prompt attention time, and the smaller-M GEMMs take 1.16× and 1.52× the linear time.
2. **Decode attention vs resident context.** Decode attention per step is linear in resident tokens N·S:
   0.95 ms + 82.5 µs per 1 K resident tokens (R² = 0.9997, 30 cells, N = 1 / 8 / 32, S = 1 K–32 K). That is 3.1 TB/s of
   KV, 75% of the measured streaming reference. Its share of a pure-decode step rises from 7% (N 8, S 1 K) to 51%
   (N 8, S 32 K). In a mixed step, the chunk's prefill attention does not depend on the decoders' S (60–62 ms at
   P 4096 + C 1024 for N = 8 at every S). The linear-layer time is shared and set by total tokens.
3. **Pure-decode fraction as outputs lengthen.** With a 128-token prompt, ≥ 99.6% of iterations and ≥ 99.6% of
   execution time are pure decode at O = 512, rising with O. A single 8 192-token injection drops the execution share to
   78–88% at O = 512 and 93–98% at O = 4 096. The spread depends on policy, and B 256 is lowest because its chunks run
   longest. Under stock serving, the pure-decode time share falls from ≈ 100% at P:D < 0.3 to 34–82% at P:D ≈ 12 and
   1–30% at P:D ≈ 170 (long prompts), depending on load.
4. **Memory-service sensitivity of the remaining steps.** A disjoint-CU streaming competitor drawing 1.88 TB/s slows
   pure-decode steps by 16–20% and mixed steps by 18–23%. At 0.95 TB/s the slowdown is 5–9%, and a compute-only
   competitor gives 1–5%. The slowdown does not grow with S. On this platform the memory competitor also lowers the
   decoder's clocks (median SCLK −12…−16%, MCLK −9%, board power ≈ 740 W). The slowdown is therefore an upper bound on
   the bandwidth-sharing effect. Bytes: FETCH_SIZE / WRITE_SIZE (KiB) and TCC_EA0_RDREQ_DRAM (128-byte requests) count
   L2→fabric requests, upstream of the 256 MB MALL. No counter on this stack measures HBM bytes (section 4.3).
5. **Where tuned chunking helps or hurts serving.** Pending lc2 (development) and lc3 (held-out). The lc1 pilot
   already shows one hurt case. For long prompts (16–30 K tokens, short outputs), stock B 2048 misses the 110 ms ITL
   target for 13% of requests even at 0.03 rps, because long-prefix chunks inflate co-running decode steps.

## 1. Platform, controls and failed cases

| item | value |
|---|---|
| node | mi3008x k002-002 (lc1, job 547842, release c0f3e871), 8 lanes × 1 GPU |
| software | vLLM 0.23.0, ROCm 7.14.1 container, torch 2.11, rocprofv3 1.0.0 (ROCm 7.1.0), CP firmware 177 |
| clocks | not changed; per-run medians recorded. lc1 lanes 1366–1606 MHz SCLK during timing |
| anchor (L0) | step anchor **fails** the 5% rule: t_ref 27.1 ms vs 30.9 ms in s1, at SCLK 1440 vs 1131 MHz. Serving anchor passes (96.0% of steps / 89.6% of exec pure decode, rule 95.5–96 / 87.9–89.5) |
| cross-lane control | identical work on two lanes (L1, L = 1024, one chunk): ratio 0.996 [0.992, 1.001] |

Failed acquisitions (kept, raw under the run root):
- `lc1_pmc_mix1024_SQ_INSTS_VALU_MFMA_MOPS_BF16` attempt 1: rocprofv3 queue error (malformed AQL packet). Attempt 2
  completed.
- `lc4_pmc_{WRITE_SIZE, TCC_EA0_RDREQ_DRAM_sum}` attempt 1: execution failures. Attempt 2 of each completed.
- `lc4_pmc_FETCH_SIZE`: illegal memory access in the S = 32 000, N = 8 cell on both attempts. Re-run in lc2 (amendment 2).
- `lc4_counter_audit`: exit 0, but 0 / 208 runs saw a device (launcher bug, fixed). Re-run in lc2.
- lc2: 8 serving steps (`lc5d_*`) hit `hipErrorLaunchFailure` on attempt 1 and completed on attempt 2; kept. Both
  re-runs (FETCH_SIZE S 32 000 N 8 split out; counter audit) completed.
- Smoke 547179 (harness id-suffix bug), kept. The smoke anchor ran on a faster-clocked node (amendment 1).

## 2. L1 — chunked vs whole prefill

![L1](figs/lc/l1_chunked_over_whole.png)

TTFT ratio chunked / whole (same prompt token ids, 8 paired reps, first sampled token identical in 120 / 120 pairs):

| L | B 4096 | B 1024 | B 256 | whole TTFT |
|---|---|---|---|---|
| 1 024 | 0.996 | 0.996 | 1.406 | 0.146 s |
| 8 192 | 1.031 | 1.221 | 1.899 | 1.23 s |
| 16 384 | 1.053 | 1.371 | 2.159 | 2.87 s |
| 24 576 | 1.068 | 1.438 | 2.319 | 4.91 s |
| 32 000 | 1.084 | 1.535 | 2.441 | 7.04 s |

Kernel time at L = 32 000 (rocprofv3 kernel trace, ms):

| arm | linear | prefill attn | other | busy | host gaps |
|---|---|---|---|---|---|
| whole | 3 503 | 3 166 | 196 | 6 892 | 5 |
| B 4096 | 3 690 | 3 686 | 195 | 7 605 | 39 |
| B 1024 | 4 071 | 6 112 | 234 | 10 475 | 164 |
| B 256 | 5 333 | 11 238 | 350 | 17 060 | 503 |

Counters at L = 32 000 (separate eager passes, one counter each):

| arm | MFMA PFLOP | FETCH_SIZE | TCC_EA0_RDREQ_DRAM × 128 B | WRITE_SIZE |
|---|---|---|---|---|
| whole | 2.67 | 10.5 TB | 10.5 TB | 0.60 TB |
| B 1024 | 2.74 | 5.0 TB | 5.0 TB | 0.63 TB |
| B 256 | 2.76 | 12.1 TB | 12.1 TB | 0.68 TB |

Supported:
- Chunking costs time, not FLOPs. The extra time grows with L and with smaller B.
- The c·k(k−1)/2 revisit model predicts the *shape* (attention time grows ≈ quadratically in the number of chunks), but
  through attention kernel efficiency, not bytes. Its byte term (0.13 TB at B 1024, 0.52 TB at B 256) is under 5% of the
  measured request-side reads.

Strongest counterexample: request-side reads are not monotone in B. They are lowest at B 1024 and highest at B 256 and
whole, so "chunking adds memory traffic" is false at the L2→fabric observation point. The whole prompt's 32 000-row
GEMMs re-fetch weight tiles per M-tile.

Gap: whether HBM bytes (after MALL) follow the same order is unmeasured. Other attention kernels (CK / FA) may change
the 1.9× attention penalty.

## 3. L2 — decode attention vs prefill attention vs shared linear

![L2](figs/lc/l2_decode_attention.png)

Pure-decode steps (graphs, kernel trace):

| N | S | step busy | decode attn | share | linear |
|---|---|---|---|---|---|
| 1 | 1 K → 32 K | 17.0 → 20.2 ms | 1.1 → 3.7 ms | 6% → 18% | 14.2 → 14.7 ms |
| 8 | 1 K → 32 K | 21.9 → 43.9 ms | 1.6 → 22.4 ms | 7% → 51% | 18.4 → 19.3 ms |
| 32 | 1 K, 8 K | 26.0, 49.2 ms (timed) | — | — | — |

Fit over 30 profiled cells: decode attention (ms) = 0.949 + 8.25 × 10⁻⁵ × N·S, R² = 0.9997 (H2a supported).

Mixed steps, N decoders + one C-token chunk after a cached P-token prefix:
- **H2b supported.** The chunk's prefill attention is independent of the decoders' S: 27.6–28.7 ms (C 256) and
  59.7–62.5 ms (C 1024) for N = 8 over S = 1 K–32 K. It grows with the prefix P (anchors: C 1024 alone takes 161 / 224 /
  359 / 506 ms at P = 0 / 4 K / 16 K / 28 K).
- **Counterexample (implementation).** With N = 1 the same chunk's prefill attention is 83–88 ms, against 60 ms alone
  or with 8 decoders, at every S. The kernel's launch depends on the batch composition, not on S. This explains the
  smoke "85 vs 61 ms" observation.
- **H2c partly supported.** The hybrid saving (separate − mixed) / separate is 12–15% at C 1024, N = 8, and 22–27% at
  C 256, flat in S (no shrink). It shrinks with prefix P: 3.6% at P = 16 K and 28 K. One shared weight pass is worth
  ≈ 20–30 ms against a long-prefix chunk of 360–500 ms.

Gap: N = 32 at S ≥ 16 K exceeds the 300 K-token per-step limit and the KV pool. Wider batches at long S need TP > 1.

## 4. L3 part A — the pure-decode tail; L4 — memory service

### 4.1 Two-request tails (A: 128-token prompt, O outputs; B: I-token prompt, 256 outputs, injected after A's first token)

| O | I | decode share of steps | of exec | A max ITL after inject | B TTFT |
|---|---|---|---|---|---|
| 512 | 0 | 99.8% | 99.6% | — | — |
| 4 096 | 0 | 99.98% | 99.95% | — | — |
| 512 | 8 192 | 93.4–99.6% | 77.9 (B 256) – 88.1% (whole) | 99 (B 256) – 1 358 ms (sep 2048) | 1.23 (whole) – 2.46 s (B 256) |
| 4 096 | 8 192 | 99.2–99.95% | 96.7–98.3% | same | same |

![L3](figs/lc/l3_decode_share_vs_pd.png)

### 4.2 Competition (decoder on CUs 0–255, owned competitor on 256–303, two blocks, eager)

| treatment | competitor | pure decode (ref) | mixed C 1024 |
|---|---|---|---|
| arith100 | compute only | 1.01–1.05 | 0.99–1.08 |
| mem50 | 0.95 TB/s | 1.05–1.09 | 1.01–1.09 |
| mem100 | 1.88 TB/s | 1.16–1.20 | 1.18–1.23 |

The slowdown is flat in S (1 K, 16 K, 32 K) and about the same for pure-decode and mixed steps (H4a not supported as
stated). Telemetry during mem100: decoder-run median SCLK 1305 / 1510 MHz vs 1547 / 1716 absent; "memory clock"
1141 / 1160 vs 1251 / 1281 MHz; power 739–747 W vs 618–639 W. The competitor's effect therefore includes a
power-managed clock reduction that a bandwidth-only model does not have.

### 4.3 Where each counter measures bytes

| counter | unit | observation point | evidence |
|---|---|---|---|
| FETCH_SIZE | KiB | TCC (L2) → EA read requests, before MALL / Infinity Cache | lc2 audit: 1.000× useful bytes for 1 GiB and 4 GiB streaming reads, **0.999× for a 128 MiB MALL-resident read** (not filtered by MALL), 0.004× for an 8 MiB L2-resident read. Long-context re-run, pure decode: 1.00× (N 1, S 1 K) to 1.05× (N 8, S 32 K) of logical bytes |
| WRITE_SIZE | KiB | L2 → EA write requests | lc2 audit: 1.000× for a 4 GiB streaming write |
| TCC_EA0_RDREQ(_DRAM), TCC_MISS | 128-byte requests | L2 misses / EA reads, before MALL | lc2 audit: 1.000× streaming, 0.999× MALL-resident |
| TCC_EA0_WRREQ(_64B, _DRAM) | 64-byte requests | L2 → EA writes | lc2 audit: 1.000× for a 4 GiB write (0.5 per 32 B) |
| TCC_EA0_RDREQ_DRAM_sum | 128-byte requests (lc1 fit) | L2 → EA reads tagged DRAM-bound, before MALL | × 128 B = 1.00 / 1.04 / 1.05 × logical bytes for pure decode at N·S = 1 K / 131 K / 256 K |
| SQ_INSTS_VALU_MFMA_MOPS_BF16 | × 512 FLOP | issued MFMA | L1 whole matches analytic linear + attention FLOPs within 4% |
| HBM bytes | — | **not observable** on this stack (no DF / HBM counters exposed by rocprofv3 here) | — |

Every read counter that passes the streaming test also counts the MALL-resident read at full size, so the audit finds
**no post-MALL (HBM-side) counter**; 11 candidates tested, 2 reps each. A mixed step with a 1 024-token chunk fetches
2.3× its logical bytes (C 1024, N 1 / 8, all S: e.g. 153.7 vs 65.6 GB, 191.9 vs 99.7 GB), while the same decoders
alone fetch 1.00–1.05×. The excess is chunk activation traffic at the request side, and whether it reaches HBM is not
observable.

## 5. L3 part B and L5 — serving under matched arrivals

SLO and λ\*: amendment 2 table. Primary SLO: TTFT ≤ 5 × pilot p90 and per-request p95 ITL ≤ 5 × pilot median.
Secondary (amendment 3) adds each request's **maximum** ITL ≤ 1 000 ms. Sustained rate = highest rate with attainment
≥ 0.9 and nothing incomplete at it and every lower rate.

### 5.1 Development (lc2, seeds 101 / 102, Poisson): sustained rate under the primary SLO

| workload (P:D) | whole | sep2048 | mix256 | mix512 | mix1024 | mix2048 | mix4096 | mix8192 |
|---|---|---|---|---|---|---|---|---|
| lo2048 (0.15) | 0.75 / 0.5 | 0.75 / 0.75 | 0.75 / 0.75 | 0.75 / 0.75 | 0.75 / 0.75 | 0.75 / 0.75 | 0.75 / 0.75 | 0.75 / 0.75 |
| lctx (12) | ≥ 0.2 / ≥ 0.2 | ≥ 0.2 / ≥ 0.2 | 0.05 / — | — / — | 0.05 / 0.05 | 0.1 / 0.075 | 0.15 / 0.125 | 0.15 / ≥ 0.2 |
| lprompt (161) | ≥ 0.1 / ≥ 0.1 | ≥ 0.1 / ≥ 0.1 | — | — | — | — | 0.0375 / 0.025 | 0.0375 / 0.025 |

(rps, seed 101 / seed 102; "≥" = censored at the top grid rate; "—" = below 0.9 at the lowest rate.)

L3 part B (seed 101): **lo512 (P:D 0.6)** every mixed budget sustains 3.0 rps vs 2.0 for whole / sep2048 (mixed
wins one step); **balanced (P:D 5)** whole, sep2048 and mix256 sustain 1.5 vs 1.25 for mix2048 / 4096; **lo4096
(P:D 0.08)** all five policies are censored at 1.0 rps (2 × λ\*) with identical goodput.

### 5.2 The SLO definition decides the long-prompt answer

Under the primary SLO, separate / whole prefill **beats every chunked budget on long prompts** (lctx, lprompt), and
the smaller the budget the worse. A chunk co-scheduled with decoders stretches every decode step it shares, so small
budgets spread a 16–30 K prefill over hundreds of slow steps and push requests' p95 ITL over the limit (mix256 lctx
p95 ITL 54–230 ms vs 23–38 ms whole). Whole prefill instead stalls decoders once per arriving prompt, which the p95
metric ignores when the stall is < 5% of a request's gaps. The cost appears in the worst gap: p99 of per-request
maximum ITL is 3.6–15 s (lctx) and 4.9–28 s (lprompt) for whole, against 0.2–0.4 s for mix256.

With the 1 s stall cap (exploratory on lc2): lctx goodput at 0.05 rps is 0.057 (mix2048) / 0.056 (mix1024) vs 0.019
for whole and sep2048 (mean of two seeds), and at 0.1 rps 0.088 (mix2048) vs 0.008 (whole, sep2048), so **chunking wins about 3–11×**; lprompt reaches attainment 0.9 at no rate under any policy. The
short-prompt workloads (lo2048, lo512, lo4096) do not change between the two SLOs. The held-out decision (lc3) is made
under both and reported side by side.

![goodput lc5d](figs/lc/goodput_lc5d.png)

## 6. Ranking (pending)

## 7. Implications for the memory-centric GPU design (draft)

- Long context makes decode attention the dominant KV-streaming term. At N·S = 256 K resident tokens it is half of a
  pure-decode step and runs at 75% of streaming bandwidth. That is the regime where more bandwidth per CU pays off, and
  it grows linearly with resident tokens.
- Chunked prefill on MI300X does not create long-prompt traffic. It creates compute-efficiency loss in prefill
  attention over a paged prefix. A decode-specialised part with fewer CUs is not penalised by chunking itself, but
  long-prompt chunks co-scheduled with decode hurt decode ITL (lc1 pilot, lprompt). That supports keeping long prefills
  off the decode device.
- Memory-competition sensitivity is ≈ 0.1 per TB/s of competitor traffic, but on MI300X it is confounded with power
  management. A bandwidth model must carry this as an upper bound.

## 8. Remaining uncertainties (draft)

- No physical HBM byte counter. All byte statements are request-side (pre-MALL).
- Clocks are uncontrolled and differ by node and by lane under load. Cross-lane serving comparisons carry this caveat.
- One attention backend (ROCM_ATTN). The N = 1 mixed-step anomaly shows backend sensitivity.
- TP1 only. 32 K tokens is the model's position limit.
