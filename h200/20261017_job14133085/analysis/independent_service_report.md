# Independent native service characterization

All selected J4 captures are saved: three native and three extension ordinary acquisitions,39 native plus6 extension physical-counter captures, and an instrumented same-pass overlap timeline. The15 pattern joins pass their source, isolation-record and pooled-clock checks. These are native-pattern estimates, not decoder service rates or physical architecture predictions.

Physical read+write bytes come from separate kernel-replay counter captures with cache/clock control disabled. GPU-event time comes from ordinary acquisitions, normalized by recorded repetitions. Counter replay and cache-history perturbations remain limitations; zero HBM bytes identify no positive HBM service rate. Confidence intervals resample independent acquisitions and their timed trials.

| Native pattern | Physical service TB/s (95% CI) |
|---|---:|
| stream_1mib | NA: zero physical demand |
| stream_28mib | 3.175 [3.135, 3.212] |
| stream_32mib | 3.265 [3.242, 3.295] |
| stream_512mib | 3.962 [3.952, 3.969] |
| stride17_512mib | 3.499 [3.492, 3.505] |
| gather_512mib | 3.381 [3.376, 3.384] |
| paged128_512mib | 3.986 [3.962, 4.001] |
| pointer_0.03125mib_chains1 | 1.293e-05 [1.055e-05, 5.3e-05] |
| pointer_96mib_chains1 | 0.0002303 [0.0002302, 0.0002303] |
| pointer_96mib_chains32 | 0.003365 [0.003361, 0.003366] |
| fma_blocks528 | 0.003727 [0.002724, 0.005004] |
| bf16_gemm_4096 | 0.7012 [0.6882, 0.7129] |
| softmax_width8192 | 0.04795 [0.04707, 0.05032] |
| read_only_512mib | 3.104 [3.04, 3.122] |
| read_stride17_512mib | 4.446 [4.439, 4.448] |

The fixed70/30 native overlap-family held-out model is rejected: one condition has10.54% relative error against the predeclared10% limit, and its point estimates give an empty coupled residual interval. No threshold/split/attempt selection was changed to obtain a pass. Sampling uncertainty and analytical assumption envelopes remain distinct.

The raw footprint/pointer figure and illustrative joint-sensitivity figure are under `analysis/figures/`. The latter uses declared hypothetical inputs and does not predict H200 speedup. Authoritative source-bound joins are in `results/shoreline_independent_service_qualification.json`; completed coverage and its source hashes are in `results/J4_final_capture_review_v2.json`.

Next: finish focused current-GPU decoder controls; run the bounded J11 semantic pilot, J5 actual resource-allocation measurements, J8 layer/event controls and J9 conditional analysis, then selected J7/J10 and supported J6/J11 closeout. No physical SRAM/PHY/package change is enacted.
