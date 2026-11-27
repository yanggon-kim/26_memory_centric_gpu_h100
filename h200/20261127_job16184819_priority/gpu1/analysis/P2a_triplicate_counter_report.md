# Full-GPU optimized32B physical DRAM capture comparison

Qwen2.5-32B BF16/TP1, batch1/context4096, fixed outputs, decode ticks4–6 (three generated tokens). Three independent instrumented producers on `GPU-7838118b-b9d4-2865-898f-f2dc79e741e7`; exact replay outputs and counter-reader gates pass. All three use the same frozen numerical sources.

| Acquisition | Physical DRAM read GB/token | Physical DRAM write MB/token | Instrumented three-tick ms |
|---|---:|---:|---:|
| 0 | 65.081213 | 49.650176 | 74.078400 |
| 1 | 65.081645 | 50.067541 | 74.112160 |
| 2 | 65.081699 | 49.260544 | 73.760288 |

GB/MB are decimal. These are aggregate physical counter observations under NCU application-range replay. The duration column describes instrumented execution, not ordinary decode latency.

The clean reference failed its predeclared5percent SM-clock envelope (1260–1980MHz); every trial remains saved. No clean achieved-GB/s or latency ratio is derived. L2 traffic, kernel attribution and matched restricted-arm traffic remain unmeasured, so this is a partial P2a comparison. No inference about address-level exactly-once access, SRAM capacity or architectural area follows.

Evidence and raw SHA256s: `../results/P2a_triplicate_counter_review.json`; original acquisitions: `../results/P2a_graph_b1_c4096_a0.json`, `a1.json`, `a2.json`. The original raw profiler reports remain in `../profiles/`.
