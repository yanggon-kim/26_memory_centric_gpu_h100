# MI300X shoreline-aware decode study: opportunity report

Allocation 468885 (`mi3008x`, node k002-003, 8× AMD Instinct MI300X), 2026-10-18 12:12–(see log), plus MI325X job 468949. Governing scope: `docs/shoreline_aware_research_plan_2.0.md` and the common measurement contract `docs/shoreline_hopper_measurements.md`, adapted in `docs/AMD_PORT_PLAN.md`. Execution details, failures and tool workarounds are in `docs/amd/experiment_log.md`; artifact locations in `docs/amd/artifacts.md`.

Labels: **measured** = real MI300X/MI325X; **analytical** = model computed from measured inputs; **profiled** = under rocprofv3/rocprof-compute (timings perturbed, never baselines). Nothing here establishes a physical architecture benefit. Area, PHY, shoreline, package and power models are later stages.

## Bottom line

1. **Decode at batch 1 streams every weight and KV byte from memory once per step, and L2 holds nothing across steps.** Counter-measured L2→memory read bytes per step equal the logical demand within 0.2–0.5% (1.5B, 7B, Mistral-7B, 14B; eager and graph) and within 3–4% at batch 8. *(measured, profiled)*
2. **Extra on-chip capacity barely helps today's decode.** Evicting L2 and the 256 MB Infinity Cache between tokens slows a step by only 0.2–1.4%. The Infinity Cache serves streaming reads at about 5.65 TB/s, only about 1.25× HBM's 4.55 TB/s. *(measured)*
3. **Decode is not bandwidth-saturated on MI300X for small and medium models.** The time to stream a step's bytes at the measured 4.55 TB/s is 32% (1.5B) to 82% (32B) of the measured step time. Arithmetic at the measured BF16 rate is under 2%. The remainder (1.5–3.5 ms per step) is latency, kernel and launch inefficiency, and per-step fixed cost. *(analytical from measured inputs)*
4. **Compute-unit trimming (software proxy) costs decode little but prefill a lot.** At 32% of the compute units, batch-1 decode slows 9–24% (least for the largest models), batch 8–64 decode slows 40–70%, and prefill slows 2.1–2.3×. Arithmetic throughput falls to 36–47%, but HBM streaming keeps 87% of its full-GPU rate. L2-resident reads and random gathers fall roughly in proportion to the compute units: compute units are also the request generators. *(measured)*
5. **Energy: decode runs at the 750 W power cap from 7B upward.** Masking compute units does not lower power (clocks rise instead) and raises J/token by 10–19%. Energy savings from compute trimming need physically removed or power-gated silicon, which a software proxy cannot show. *(measured)*
6. **More delivered bandwidth at equal compute helps in proportion to the bandwidth share.** See the TP section (compute-matched TP2: Qwen 32B 1.48×, Qwen 7B 1.17×, 1.5B 0.92×). MI325X vs MI300X (+7–10% measured HBM bandwidth, but also +14% compute) gives 1.04–1.15× decode, again growing with model size. *(measured; cross-configuration confounds stated)*

**Implication for plan 2.0:** on this hardware the evidence favours **compute → delivered-bandwidth** (plus request-generation capacity) over **compute → SRAM capacity** for dense small-batch decode. The case is strongest for large models at small batch. Two caveats: (a) roughly 20–70% of step time is neither bandwidth nor arithmetic, so better kernels and lower per-step overhead may beat either hardware change; (b) compute cannot be cut freely, because compute units issue the memory requests, prefill needs them, and larger batches need them. The c/r/f analysis below quantifies this conditionally.

## 1. What was run

| Family | Cases | Status |
|---|---|---|
| J0 inventory, identity, tool feasibility | 8× MI300X; counters permitted; CU masks validated | complete |
| M1 baselines (vLLM 0.23 graph + eager, 3 acq × 10 trials) | bridge B1/C1024, B8/C4096; Qwen7B, Mistral7B B1/B8 C4096; Qwen14B, Qwen32B B1/C4096 | complete (17/17 × 3) |
| M1 representative text | bridge B1/B4 C128 | complete |
| M1 cross-GPU reproducibility | 8 graph cases on a second GPU | complete/partial (see manifest) |
| HF StaticCache gate + compiled timing | bridge | passed |
| M2/M3 counters (rocprofv3, per dispatch) | 12 cases × 4 counter groups | 44+ groups; some groups failed 4 attempts (recorded) |
| M3 rocprof-compute | HF Tensile kernels, single-pass sets; service benchmarks | partial (tool crashes recorded) |
| M4/A–D service controls | 34 conditions × 4 CU levels × 3 acq + counter acq | complete, 0 failures |
| M5 cache-history pressure | bridge B1 C1024/2048/4096, B8 C4096; Qwen7B B1 C4096 | complete (3 acq each) |
| M6 CU-restricted decode | 8 focused cases + Qwen7B batch 1–64 × 4 CU levels | complete (a few cells 1–2 acq) |
| M7 service under CU masks | all M4 conditions | complete |
| M8 energy | 7 graph cases; energy under CU masks (7B, 32B); prefill vs CUs (4 cases) | complete/partial (see manifest) |
| E/F + J9 | latency bounds and c/r/f grid from measured inputs | analytical, complete |
| TP extension (owner-authorized) | TP2/TP4/TP8, full and compute-matched | see TP section |
| MI325X comparison | M4 full-CU sweep + 8 graph baselines | complete (job 468949) |

## 2. Baselines (M1, graph mode, mean of 3 acquisitions, 95% CI)

| Case | ms/step | CI | tok/s | J/token (graph, qualified) |
|---|---:|---|---:|---:|
| Qwen2.5-1.5B B1 C1024 | 2.187 | 2.182–2.194 | 457 | 1.246 |
| Qwen2.5-1.5B B8 C4096 | 3.155 | 3.145–3.169 | 2536 | 0.249 |
| Qwen2.5-7B B1 C4096 | 4.682 | 4.667–4.704 | 214 | 3.594 |
| Qwen2.5-7B B8 C4096 | 6.654 | 6.601–6.696 | 1202 | 0.659 |
| Mistral-7B B1 C4096 | 5.111 | 5.072–5.176 | 196 | 3.916 |
| Mistral-7B B8 C4096 | 7.601 | 7.566–7.643 | 1053 | — |
| Qwen2.5-14B B1 C4096 | 8.638 | (3 acq) | 116 | 6.742 |
| Qwen2.5-32B B1 C4096 | 17.532 | (3 acq) | 57 | 14.001 |

Eager vLLM is host-bound at batch 1 (337 launches/step; GPU busy ≈15% of step wall in the profiled trace) and is a separate program, not a baseline. The contract's pooled ≤5% clock envelope failed in most conditions because MI300X's power-managed clocks move with load and the memory clock ramps at trial start. Rows are reported with that flag (`clock_qualified_v1`) rather than dropped. Energy qualification passed (counter vs integrated power within 0.2–1.4%, windows 7–77 s). It is internal consistency only, not absolute calibration, and has socket scope.

## 3. Traffic: logical vs physical (M2)

Per decode step (tick 5), L2→memory-side bytes from FETCH_SIZE/WRITE_SIZE. These are bytes leaving the per-XCD L2s, before the Infinity Cache; there is no MALL counter on this stack, so they are an upper bound on HBM bytes.

| Case | Read GB | Logical U GB | D/U | Written MB | L2 hit (requests) |
|---|---:|---:|---:|---:|---:|
| bridge B1 C1024 (eager / graph) | 3.133 / 3.131 | 3.117 | 1.005 | 7 | 0.13 |
| bridge B8 C4096 eager | 4.174 | 4.028 | 1.036 | 164 | 0.21 |
| Qwen7B B1 C4096 (eager / graph) | 14.41 | 14.38 | 1.002 | 31 | 0.06 |
| Qwen7B B8 C4096 eager | 16.47 | 16.02 | 1.028 | 550 | 0.09 |
| Mistral7B B1 C4096 eager | 14.80 | 14.77 | 1.002 | 12 | 0.05 |
| Qwen14B B1 C4096 eager | 28.85 | 28.79 | 1.002 | 95 | 0.05 |

The aggregate D/U near 1 means "about once per step", not a per-address proof. Issued BF16 MFMA work (SQ_INSTS_VALU_MFMA_MOPS_BF16 × 512) is about 4–5× the useful FLOPs at batch 1 (GEMV padded to 16-row MFMA tiles) and about 2× at batch 8. The ×512 unit is the rocprof-compute convention, not independently verified here.

## 4. Service controls (M4, A–D; M7 under CU masks)

Full-CU values (3 acquisitions, 10 trials each):

| Control | MI300X | MI325X |
|---|---:|---:|
| Read, 32 MiB (L2-resident) | 24.2 TB/s | 24.4 TB/s |
| Read, 128–256 MiB (Infinity Cache) | 5.65 TB/s | 5.65 TB/s |
| Read, 4–16 GiB (HBM) | 4.48–4.55 TB/s | 4.80–5.00 TB/s |
| Copy / write 4 GiB | 3.82 / 4.85 TB/s | 4.08 / 5.31 TB/s |
| Dependent-load latency L2 / MALL / HBM | 103 / 245 / 268 ns | same |
| Chains needed to approach HBM bandwidth | ~32k (1.2 MB in flight) | similar |
| BF16 MFMA (register-resident) | 1071 TFLOP/s | 1230 TFLOP/s |
| FP32 FMA | 95 TFLOP/s | 108 TFLOP/s |

M7 (ratio to full CUs at 232 / 152 / 96 CUs): MFMA 0.74 / 0.53 / 0.36; FMA 0.81 / 0.64 / 0.47; HBM read 0.95 / 0.88 / 0.87; copy 1.00 / 0.95 / 0.90; L2-resident read 0.87 / 0.63 / 0.46; random gather 0.96 / 0.72 / 0.50; dependent latency unchanged.

## 5. Retention (M5)

Pressure/normal step-time ratio (1 GiB read between steps, outside timed intervals; outputs identical): bridge B1 C1024 1.014, C2048 1.009, C4096 1.005; B8 C4096 1.003; Qwen7B B1 C4096 1.002. This holds even where the whole KV (≈115 MB at bridge B1 C4096) fits the Infinity Cache. HF StaticCache compiled path, not vLLM.

## 6. Compute right-sizing (M6, prefill, energy)

Slowdown vs 304 CUs (same GPU; median GFX clock rises as CUs are masked because the board is power-capped):

| Case | 232 CU | 152 CU | 96 CU |
|---|---:|---:|---:|
| bridge B1 C1024 decode | 1.02 | 1.13 | 1.21 |
| Qwen7B B1 C4096 decode | 1.06 | 1.13 | 1.24 |
| Mistral7B B1 C4096 decode | 1.05 | 1.11 | 1.17 |
| Qwen14B B1 C4096 decode | 1.05 | 1.10 | 1.11 |
| Qwen32B B1 C4096 decode | 1.03 | 1.08 | 1.09 |
| Qwen7B B8 C4096 decode | 1.19 | 1.43 | 1.64 |
| Qwen7B B16–B64 C1024 decode | 1.18–1.22 | 1.38–1.48 | 1.51–1.70 |
| Prefill (4 cases, C4096) | 1.22–1.31 | 1.56–1.73 | 2.08–2.25 |

Noninferiority (≤5% decode regression, upper CI bound) holds at 232 CUs only for the largest models at batch 1 (32B; 14B and Mistral are borderline). Energy under masks (7B, 32B, B1 C4096): board power stays at about 722–751 W and J/token rises (7B 3.55 → 3.94 → 4.23; 32B 13.5 → 14.5 → 14.6). Masked CUs are not removed silicon, so this does not test physical leakage savings.

## 7. Latency bounds and c/r/f sensitivity (E/F, J9)

`results/.../analysis/shoreline_latency_bounds.json`. C = F/1071 TFLOP/s, M = D/4.55 TB/s (D from counters where available, else logical U), T = measured graph step:

| Case | T ms | C ms | M ms | M/T | O feasible ms |
|---|---:|---:|---:|---:|---|
| bridge B1 C1024 | 2.187 | 0.003 | 0.690 | 0.32 | 1.49–1.50 |
| bridge B8 C4096 | 3.155 | 0.028 | 0.886 | 0.28 | 2.24–2.27 |
| Qwen7B B1 C4096 | 4.682 | 0.015 | 3.161 | 0.68 | 1.51–1.52 |
| Qwen7B B8 C4096 | 6.653 | 0.118 | 3.523 | 0.53 | 3.01–3.13 |
| Mistral7B B1 C4096 | 5.111 | 0.015 | 3.246 | 0.64 | 1.85–1.86 |
| Qwen14B B1 C4096 | 8.638 | 0.030 | 6.329 | 0.73 | 2.28–2.31 |
| Qwen32B B1 C4096 | 17.532 | 0.065 | 14.301 | 0.82 | 3.17–3.23 |

All intervals are nonempty (consistency, not validation). The c/r/f grid (c ∈ {1, .75, .5, .25}, r ∈ {.7, 1, 1.2, 1.4}, f ∈ {0, .05, .1, .2}) is in the same file, under same-O/X and envelope assumptions. **M6 falsifies reading c as FLOP capacity:** C/c predicts under 1% slowdown at c = 0.32, but 9–24% was measured, because compute units also generate memory requests and drive L2 bandwidth (M7). Any architecture model must couple "compute" removal to request-generation and on-chip delivery capacity.

## 8. Multi-GPU extension (TP) — owner-authorized 2026-10-18

(Filled from `results/.../tp/summary.json` at close-out.)

## 9. Limitations and failures

- HBM bytes are pre-Infinity-Cache upper bounds; MALL behaviour is inferred from timing controls.
- Counter collection on real LLM workloads crashed or hung intermittently (runtime code-object loads; AQL malformed; illegal address). Runtime Triton kernels were excluded from collection, so a few small per-step kernels are uncovered. Several counter groups and rocprof-compute sets failed all attempts; each is recorded.
- Sporadic `unspecified launch failure` in about 1 in 20 vLLM runs while 8 lanes shared the node (no ECC errors); failed runs are kept and repeated.
- CU masking is a software proxy: masked CUs keep their area, leakage and clock domain, and freed power raises clocks. It is not area-matched compute removal.
- TP and MI325X comparisons change several things at once (communication, power caps, HBM type) and are reported as such.
- The common contract's ≤5% clock envelope failed for most conditions (power-managed clocks); flagged, not dropped.
- No native instruction/address traces (no NVBit on AMD); reuse is from counters and pressure controls, not traces. Simulator calibration is not attempted.
