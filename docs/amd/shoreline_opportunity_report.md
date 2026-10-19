# MI300X shoreline-aware decode study: opportunity report

Allocation 468885 (`mi3008x`, node k002-003, 8× AMD Instinct MI300X), 2026-10-18 12:12–18:57 (cancelled when all planned work was done), plus MI325X jobs 468949 and 469084 (both complete). Governing scope: `docs/shoreline_aware_research_plan_2.0.md` and the common measurement contract `docs/shoreline_hopper_measurements.md`, adapted in `docs/AMD_PORT_PLAN.md`. Execution details, failures and tool workarounds are in `docs/amd/experiment_log.md`; artifact locations in `docs/amd/artifacts.md`.

Labels: **measured** = real MI300X/MI325X; **analytical** = model computed from measured inputs; **profiled** = under rocprofv3/rocprof-compute (timings perturbed, never baselines). Nothing here establishes a physical architecture benefit. Area, PHY, shoreline, package and power models are later stages.

## Bottom line

1. **Decode at batch 1 streams every weight and KV byte from memory once per step, and L2 holds nothing across steps.** Counter-measured L2→memory read bytes per step equal the logical demand within 0.2–0.5% (1.5B, 7B, Mistral-7B, 14B; eager and graph) and within 3–4% at batch 8. *(measured, profiled)*
2. **Extra on-chip capacity barely helps today's decode.** Evicting L2 and the 256 MB Infinity Cache between tokens slows a step by only 0.2–1.4%. The Infinity Cache serves streaming reads at about 5.65 TB/s, only about 1.25× HBM's 4.55 TB/s. *(measured)*
3. **Decode is not bandwidth-saturated on MI300X for small and medium models.** The time to stream a step's bytes at the measured 4.55 TB/s is 32% (1.5B) to 82% (32B) of the measured step time. Arithmetic at the measured BF16 rate is under 2%. The remainder (1.5–3.5 ms per step) is latency, kernel and launch inefficiency, and per-step fixed cost. *(analytical from measured inputs)*
4. **Compute-unit trimming (software proxy) costs decode little but prefill a lot.** At 32% of the compute units, batch-1 decode slows 9–24% (least for the largest models), batch 8–64 decode slows 40–70%, and prefill slows 2.1–2.3×. Arithmetic throughput falls to 36–47%, but HBM streaming keeps 87% of its full-GPU rate. L2-resident reads and random gathers fall roughly in proportion to the compute units: compute units are also the request generators. *(measured)*
5. **Energy: decode runs at the 750 W power cap from 7B upward.** Masking compute units does not lower power (clocks rise instead) and raises J/token by 10–19%. Energy savings from compute trimming need physically removed or power-gated silicon, which a software proxy cannot show. *(measured)*
6. **Delivered bandwidth at fixed compute moves decode time, in proportion to the bandwidth share.** Three independent routes agree:
   - Compute-matched TP: about one GPU's CUs spread over 2/4 GPUs gives Qwen 32B 1.48×/1.86×, Qwen 7B 1.16×/1.33×, while the 1.5B loses to communication.
   - A same-GPU competitor removing up to 1.1–1.9 TB/s slows decode by up to 15–26% at fixed decoder CUs.
   - MI325X's +7–10% bandwidth (with +14% compute) gives 1.04–1.15×. A compute-matched MI325X attempt failed to isolate bandwidth, because CU masking also cut its bandwidth by 9–11%.

   Gains saturate: TP8 matched < TP4 matched. They also need request issuers: at about 160 total CUs, 4 GPUs × 40 CUs < 2 × 80. *(measured; confounds stated per experiment)*

**Implication for plan 2.0:** on this hardware the evidence favours **compute → delivered-bandwidth** (plus request-generation capacity) over **compute → SRAM capacity** for dense small-batch decode. The case is strongest for large models at small batch. Two caveats: (a) roughly 20–70% of step time is neither bandwidth nor arithmetic, so better kernels and lower per-step overhead may beat either hardware change; (b) compute cannot be cut freely, because compute units issue the memory requests, prefill needs them, and larger batches need them. The c/r/f analysis below quantifies this conditionally.

## 1. What was run

| Family | Cases | Status |
|---|---|---|
| J0 inventory, identity, tool feasibility | 8× MI300X; counters permitted; CU masks validated | complete |
| M1 baselines (vLLM 0.23 graph + eager, 3 acq × 10 trials) | bridge B1/C1024, B8/C4096; Qwen7B, Mistral7B B1/B8 C4096; Qwen14B, Qwen32B B1/C4096 | complete (17/17 × 3) |
| M1 representative text | bridge B1/B4 C128 | complete |
| M1 cross-GPU reproducibility | 8 graph cases on a second GPU | GPU 6 within 0.1–0.8% of GPU 0 (2–3 acq) |
| HF StaticCache gate + compiled timing | bridge | passed |
| M2/M3 counters (rocprofv3, per dispatch) | 12 cases × 4 counter groups | complete (all groups, after retries; failed attempts kept) |
| M3 rocprof-compute | HF Tensile kernels, single-pass sets; service benchmarks | partial (tool crashes recorded) |
| M4/A–D service controls | 34 conditions × 4 CU levels × 3 acq + counter acq | complete, 0 failures |
| M5 cache-history pressure | bridge B1 C1024/2048/4096, B8 C4096; Qwen7B B1 C4096 | complete (3 acq each) |
| M6 CU-restricted decode | 8 focused cases + Qwen7B batch 1–64 × 4 CU levels | complete (3 acq each) |
| M7 service under CU masks | all M4 conditions | complete |
| M8 energy | 8 graph cases (3 acq; Mistral B8 2 acq; all trials qualified); energy under CU masks (7B, 32B); prefill vs CUs (4 cases) | complete |
| E/F + J9 | latency bounds and c/r/f grid from measured inputs | analytical, complete |
| TP extension (owner-authorized) | TP2/TP4/TP8, full and compute-matched at ≈304 and ≈160 total CUs | complete (3 acq; one TP2 cell 2 acq) |
| J6 same-GPU competitor (added) | 4 cases × {solo, read 25/50/75/100%, FMA} | complete (3 acq) |
| MI325X comparison | M4 full-CU sweep + 8 graph baselines (468949); 264-CU vs 304-CU paired runs (469084) | complete; the compute-matched pairing did not hold bandwidth fixed |

## 2. Baselines (M1, graph mode, mean of 3 acquisitions, 95% CI)

| Case | ms/step | CI | tok/s | J/token (graph, qualified) |
|---|---:|---|---:|---:|
| Qwen2.5-1.5B B1 C1024 | 2.187 | 2.182–2.194 | 457 | 1.246 |
| Qwen2.5-1.5B B8 C4096 | 3.155 | 3.145–3.169 | 2536 | 0.249 |
| Qwen2.5-7B B1 C4096 | 4.682 | 4.667–4.704 | 214 | 3.594 |
| Qwen2.5-7B B8 C4096 | 6.654 | 6.601–6.696 | 1202 | 0.659 |
| Mistral-7B B1 C4096 | 5.111 | 5.072–5.176 | 196 | 3.916 |
| Mistral-7B B8 C4096 | 7.601 | 7.566–7.643 | 1053 | 0.782 (2 acq; third cut by allocation end) |
| Qwen2.5-14B B1 C4096 | 8.638 | (3 acq) | 116 | 6.742 |
| Qwen2.5-32B B1 C4096 | 17.532 | (3 acq) | 57 | 14.001 |

The contract's pooled ≤5% clock envelope passed for only 2 of 17 conditions. A post-hoc diagnostic restricted to the middle 80% of each trial window gives the same 2 of 17. MI300X GFX clocks genuinely vary by more than 5% between trials under power management at the 750 W cap, so this gate cannot be met on this hardware without clock locking, which the contract forbids. Comparisons are reported with the flag; CIs come from three independent acquisitions.

Eager vLLM is host-bound at batch 1 (337 launches/step; GPU busy ≈15% of step wall in the profiled trace) and is a separate program, not a baseline. Energy qualification passed (counter vs integrated power within 0.2–1.4%, windows 7–77 s). It is internal consistency only, not absolute calibration, and has socket scope.

## 3. Traffic: logical vs physical (M2)

Per decode step (tick 5), L2→memory-side bytes from FETCH_SIZE/WRITE_SIZE. These are bytes leaving the per-XCD L2s, before the Infinity Cache; there is no MALL counter on this stack, so they are an upper bound on HBM bytes.

| Case | Read GB | Logical U GB | D/U | Written MB | L2 hit (requests) | Issued MFMA / useful FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| bridge B1 C1024 (eager / graph) | 3.133 / 3.131 | 3.117 | 1.005 | 7 | 0.13 | 4.9 |
| bridge B8 C4096 (eager / graph) | 4.174 / 4.175 | 4.028 | 1.036 | 161–164 | 0.21 | 2.2 |
| Qwen7B B1 C4096 (eager / graph) | 14.41 | 14.38 | 1.002 | 31 | 0.06 | 4.2 |
| Qwen7B B8 C4096 eager | 16.47 | 16.02 | 1.028 | 550 | 0.09 | 2.1 |
| Mistral7B B1 C4096 (eager / graph) | 14.80 | 14.77 | 1.002 | 11–12 | 0.05 | 3.6 |
| Mistral7B B8 C4096 eager | 18.78 | 18.53 | 1.014 | 171 | 0.17 | 1.9 |
| Qwen14B B1 C4096 eager | 28.85 | 28.79 | 1.002 | 95 | 0.05 | 4.0 |
| Qwen32B B1 C4096 eager | 65.14 | 65.05 | 1.001 | 130 | 0.04 | 4.0 |

The aggregate D/U near 1 means "about once per step", not a per-address proof. The issued/useful column uses SQ_INSTS_VALU_MFMA_MOPS_BF16 × 512. That unit is the rocprof-compute convention, not independently verified here. At batch 1 the GEMVs are padded to 16-row MFMA tiles.

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
| bridge B8 C4096 | 3.155 | 0.028 | 0.954 | 0.30 | 2.17–2.20 |
| Qwen7B B1 C4096 | 4.682 | 0.015 | 3.175 | 0.68 | 1.49–1.51 |
| Qwen7B B8 C4096 | 6.653 | 0.118 | 3.742 | 0.56 | 2.79–2.91 |
| Mistral7B B1 C4096 | 5.111 | 0.015 | 3.256 | 0.64 | 1.84–1.86 |
| Mistral7B B8 C4096 | 7.601 | 0.122 | 4.167 | 0.55 | 3.31–3.43 |
| Qwen14B B1 C4096 | 8.638 | 0.030 | 6.363 | 0.74 | 2.25–2.28 |
| Qwen32B B1 C4096 | 17.532 | 0.065 | 14.349 | 0.82 | 3.12–3.18 |

D comes from counters (FETCH_SIZE + WRITE_SIZE, eager program, ticks 4–6) for every case.

All intervals are nonempty (consistency, not validation). The c/r/f grid (c ∈ {1, .75, .5, .25}, r ∈ {.7, 1, 1.2, 1.4}, f ∈ {0, .05, .1, .2}) is in the same file, under same-O/X and envelope assumptions. **M6 falsifies reading c as FLOP capacity:** C/c predicts under 1% slowdown at c = 0.32, but 9–24% was measured, because compute units also generate memory requests and drive L2 bandwidth (M7). Any architecture model must couple "compute" removal to request-generation and on-chip delivery capacity.

## 8. Bandwidth at fixed compute: multi-GPU (TP), owner-authorized 2026-10-18

Speedup over TP1 (graph, synchronized-wall ms/step for TP; HIP-event ms/step for TP1, so ratios carry that scope difference), 3 acquisitions each unless noted. **Matched** restricts each GPU to ⌈304/N⌉₈ CUs, so the total active CUs stay at about one GPU (304–320) while HBM stacks, bandwidth and Infinity Cache scale by N. **Matched152** holds about 160 total CUs and compares with TP1 restricted to 152 CUs (M6).

| Case (B1 C4096 unless noted) | TP2 full | TP4 full | TP8 full | TP2 matched | TP4 matched | TP8 matched | TP2 m152 | TP4 m152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bridge B1 C1024 | 1.01 | 1.12 | n/a¹ | 0.92 | 0.89 | n/a¹ | — | — |
| Qwen7B | 1.34 | 1.71 | n/a¹ | 1.16 | 1.33 | n/a¹ | 1.12 | 1.05 |
| Qwen7B B8 | 1.38 | 1.87 | n/a¹ | 1.11 | 1.21 | n/a¹ | — | — |
| Mistral7B | 1.31 | 1.64 | 1.90 | 1.19 | 1.29 | 1.07 | — | — |
| Qwen14B | 1.44 | 1.93 | 2.28 | 1.28 | 1.45 | 1.23 | 1.21 | 1.11 |
| Qwen32B | 1.63² | 2.43 | 3.16 | 1.48 | 1.86 | 1.74 | 1.39 | 1.33 |

¹ Head count not divisible by 8. ² 2 acquisitions (one attempt failed with a launch failure).

- **Batch-1 output IDs match TP1 exactly in every case. At batch 8 they differ** (changed reduction order); the divergence is recorded, not hidden.
- **At fixed total compute, more delivered bandwidth helps in order of the bandwidth share:** 32B > 14B > 7B ≈ Mistral > 1.5B (which loses).
- **The gain saturates and then reverses:** TP8 matched is below TP4 matched for every model, because all-reduce and per-GPU fixed costs grow with N.
- **With fewer CUs per GPU the extra bandwidth is worth less:** at about 160 total CUs, TP4 (40 CUs per GPU) is below TP2 (80 per GPU). This is the same request-generation coupling as M7: bandwidth must be paired with enough request issuers per HBM stack.
- Energy was not measured for TP: 64-token windows are below the 5 s qualification minimum.

## 8b. Bandwidth at fixed compute: same-GPU competitor (contract J6)

The decoder is fixed on CUs 0–231 of one GPU. An owned competitor on CUs 232–303 streams HBM at a requested duty, or runs register FMA as a power/clock control. Its achieved bandwidth during the decoder's trials is logged. Ratio to the solo decoder on the same 232 CUs (3 acquisitions):

| Case | read25 | read50 | read75 | read100 | fma100 |
|---|---:|---:|---:|---:|---:|
| bridge B1 C1024 | 1.042 (0.48 TB/s taken) | 1.065 (0.95) | 1.101 (1.36) | 1.147 (1.93) | 1.009 |
| Qwen7B B1 C4096 | 1.051 (0.38) | 1.105 (0.72) | 1.158 (1.08) | 1.236 (1.49) | 1.044 |
| Qwen7B B8 C4096 | 1.048 (0.41) | 1.115 (0.79) | 1.166 (1.16) | 1.259 (1.62) | 1.032 |
| Qwen32B B1 C4096 | 1.058 (0.29) | 1.119 (0.55) | 1.178 (0.81) | 1.246 (1.09) | 1.068 |

Decode slows nearly linearly with the bandwidth taken, at constant decoder compute. Power/clock interference alone (the FMA control) accounts for 1–7%. The competitor obtains less bandwidth against the heavier decoders, so the two genuinely contend. This is shared-service sensitivity (L2, Infinity Cache and fabric are shared too), not a physical change of memory interfaces.

## 8c. MI325X vs MI300X

Full CUs (job 468949): HBM stream +7–10%, write +10%, Infinity Cache and L2 equal, latency equal, BF16 MFMA +15%, FMA +14% (higher power limit). Decode is 1.04× (1.5B) to 1.15× (32B) faster. This is not a same-compute pair.

**The compute-matched attempt (job 469084, 2026-10-18 19:46–20:41) did not isolate bandwidth.** MI325X was restricted to 264 CUs to match MI300X's MFMA rate; it measured 997 vs 1071 TFLOP/s (FMA 98 vs 95). The restriction also cut MI325X's HBM streaming bandwidth from 4.99 to 4.43 TB/s (4 GiB) and from 4.63 to 4.26 TB/s (16 GiB), below MI300X's full-CU 4.55/4.48 TB/s. Decode ms/step, MI325X@264 vs MI300X@304: 7B B1 4.95 vs 4.68, 7B B8 8.15 vs 6.65, Mistral 5.27 vs 5.11, 14B 8.73 vs 8.64, 32B 17.45 vs 17.53. With r ≈ 0.97 and c ≈ 0.93, this pair adds no r > 1 evidence. It independently confirms M7: removing CUs removes request-generation capacity and hence deliverable bandwidth, on MI325X as on MI300X.

## 9. Limitations and failures

- HBM bytes are pre-Infinity-Cache upper bounds; MALL behaviour is inferred from timing controls.
- Counter collection on real LLM workloads crashed or hung intermittently (runtime code-object loads; AQL malformed; illegal address). Runtime Triton kernels were excluded from collection, so a few small per-step kernels are uncovered. Several counter groups and rocprof-compute sets failed all attempts; each is recorded.
- Sporadic `unspecified launch failure` in about 1 in 20 vLLM runs while 8 lanes shared the node (no ECC errors); failed runs are kept and repeated.
- CU masking is a software proxy: masked CUs keep their area, leakage and clock domain, and freed power raises clocks. It is not area-matched compute removal.
- TP and MI325X comparisons change several things at once (communication, power caps, HBM type) and are reported as such.
- The common contract's ≤5% clock envelope failed for most conditions (power-managed clocks); flagged, not dropped.
- No native instruction/address traces (no NVBit on AMD); reuse is from counters and pressure controls, not traces. Simulator calibration is not attempted.

## 10. Recommended next steps

1. **Architecture modelling (plan 2.0 Phase 2–4).** Use these measured inputs: decode C/M/O per case, M7 service-vs-CU curves (request generation), the competitor and compute-matched TP bandwidth sensitivities, prefill and large-batch penalties, and power-cap behaviour. Couple "compute removal" to request-issue and on-chip delivery capacity, not FLOPs alone.
2. **Reduce O before adding hardware.** 1.5–3.4 ms per step is neither bandwidth nor arithmetic. Profile graph-mode per-kernel efficiency (wvSplitK at about 2.5 TB/s vs 4.5 TB/s stream) and launch/sync structure. A software fix here is the strongest simpler alternative (plan 2.0 Challenge F).
3. **Energy.** A proxy for removed or power-gated silicon is needed; CU masking cannot show leakage savings. Collect component power where the platform exposes it.
4. **A clean r > 1 at fixed c is still missing on real hardware.** CU masking couples compute to bandwidth. Future same-compute comparisons should hold CUs fixed and vary memory configuration through the platform (memory partition modes or another SKU at equal CU count), if an admin can provide them.
5. **Traces.** Address-level reuse needs a tracer that works on gfx942 (rocprofv3 ATT or a binary-instrumentation tool). It was not attempted here.
