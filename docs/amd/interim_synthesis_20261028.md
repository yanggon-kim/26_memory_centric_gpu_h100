# Interim synthesis: MI300X pilot + MI300X/MI325X campaigns (28 October 2026)

Cross-run reading of evidence already published on this branch. It adds no new data and draws no
controlled conclusions; it does not replace the versioned follow-up results report, which comes after
jobs 483498 (MI300X) and 483499 (MI325X).

Sources:
- MI300X campaign 468885: [shoreline_opportunity_report.md](shoreline_opportunity_report.md).
- MI325X campaign 470176 + 481072: [mi325x_full_campaign_results_20261028.md](mi325x_full_campaign_results_20261028.md).
- MI300X follow-up pilot 481167: [cubw_pilot_review_20261028.md](cubw_pilot_review_20261028.md). Single
  acquisitions, labelled pilot.

**Every number here is observational.** The pooled ≤5% clock gate fails on both SKUs: gfx clocks range about
1.3–2.0 GHz, and the clock check passes for only 1/16 (MI325X) and 2/16 (MI300X) baseline conditions.
Competitor cells also raise the decoder's clock (~1.92 vs ~1.27 GHz on MI300X), which biases their measured
slowdowns *downward*.

## 1. CU headroom (H1)

Qwen32B B1/C4096, R(n) = T(n)/T(304):

| n | MI300X campaign | MI300X pilot (2 GPUs) | MI325X campaign |
|---:|---:|---:|---:|
| 232 | 1.034 | 1.028 / 1.034 | 1.031 |
| 200 | – | 1.015 / 1.023 | – |
| 152 | 1.084 | 1.084 / 1.087 | 1.115 |
| 96 | 1.092 | 1.095 / 1.102 | 1.124 |

- Removing about a quarter to a third of the CUs costs ~2–3%; the step is between 200 and 152.
- At batch 8, cutting to 96 CUs costs 1.64× (MI300X) and 1.90× (MI325X) on Qwen7B, so the headroom shrinks
  quickly with batch size.
- MI325X is more CU-sensitive at 152. A plausible reading: with more bandwidth, compute becomes a larger
  share of each step.

## 2. Mechanism: fewer requests in flight (MI300X pilot E4, ratio to 304 CUs)

| control | 232 | 200 | 152 | 96 |
|---|---:|---:|---:|---:|
| 4 GiB streaming read | 0.95 | 0.96 | 0.88 | 0.87 |
| read, 9728-block grid | 1.01 | 1.02 | 1.01 | 0.98 |
| read, ILP 1 | 0.84 | 0.88 | 0.72 | 0.61 |
| 8 MiB reuse (cache-resident) | 0.73 | 0.66 | 0.54 | 0.37 |
| FMA | 0.81 | 0.74 | 0.64 | 0.46 |
| MFMA | 0.75 | 0.66 | 0.53 | 0.35 |

- With enough parallelism, 96 CUs still nearly saturate HBM.
- Low-ILP and cache-resident access lose much more.
- Small-batch decode kernels have limited memory-level parallelism, so a CU cut mainly reduces outstanding
  requests. It does not reduce peak HBM bandwidth.

## 3. Shared memory service (H2)

Decoder slowdown with a competitor on reserved CUs 232–303:

| competitor | MI300X pilot n=232 | MI300X pilot n=152 | MI300X campaign | MI325X campaign |
|---|---:|---:|---:|---:|
| memory 100% | 1.269 | 1.306 | 1.247 | 1.146 |
| memory 50% | 1.122 | 1.127 | – | – |
| arithmetic 100% | 1.062 | 1.062 | 1.069 | 1.015 |
| inactive process | 0.998 | – | – | – |

- Removing memory service costs 15–30%; removing arithmetic costs 2–7%.
- MI325X absorbs a memory competitor better.
- Shared-service balance (pilot): T(152, absent) / T(232, mem100) = 0.83. This demonstrates a balance
  between the two; it is not a stock-GPU win.
- Open item: the arithmetic competitor itself ran at ~0.6 of its solo rate beside the decoder. CU leakage
  vs inter-process contention is tested by `leak_ok_v11` in 483498/483499.

## 4. SKU comparison

| | MI325X | MI300X | ratio |
|---|---:|---:|---:|
| Qwen32B B1 decode ms/step | 15.76 | 17.53 | 0.90 |
| 4 GiB streaming read, GB/s | 5021 | 4549 | 1.10 (inverse 0.91) |
| MFMA TFLOP/s | 1215 | 1071 | 1.13 |
| Qwen32B J/token (mean W) | 15.43 (967) | 14.00 (751) | 1.10 |

The decode-time ratio tracks the inverse streaming-bandwidth ratio. It does not track the MFMA ratio.
MI325X is ~10% faster, but its J/token is ~10% worse.

## 5. Same total CUs across more HBM (TP, owner-authorized)

Qwen32B TP8 holding total CUs at 304 (38 per GPU): 1.74× faster (MI300X) and 1.40× (MI325X) than one full
GPU. Full-CU TP8 gives 3.16× and 2.90×.

This is consistent with bandwidth per CU mattering. It is not an isolation: the eight GPUs also bring 8×
L2/MALL capacity and power budget, plus communication effects.

## 6. What remains unavailable or pending

- Fixed-clock inference: unavailable without changing clocks or power, which the assignment does not
  authorize.
- Graph-mode hardware counters: rocprofv3 `--pmc` crashes or hangs; rocprof-compute on MI325X completed
  5/13 sets.
- Pending: the E2 noninferiority boundary with ≥3 blocks, released E3 cells, E4 ×3, E5 timelines and E8,
  from 483498 and 483499.
