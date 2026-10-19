# Current research interpretation — 2026-10-19

These are measured motivation and conditional research directions, not demonstrated architecture benefits. No new GPU measurements were collected for this review. All H200 allocations/cohorts remain separate.

## New native-service evidence

The existing reader qualifies all twelve J5 pattern/fraction joins using three ordinary acquisitions with ten timed trials each and three matched physical counter captures. Source hashes, correctness, process records and pooled clocks pass. Cross-fraction endpoints are identical and every per-pattern pooled clock ratio is1.0; see results/J5_native_service_review_v1.json for exact sources, values and95% confidence intervals.

Physical read+write service estimates (TB/s, decimal):

| Actual SMs | Stream512MiB | Gather512MiB | Dependent pointer96MiB, one chain |
|---|---:|---:|---:|
|132|3.9713|3.3848|0.00022856|
|104|3.9673|3.3830|0.00022794|
|72|4.0149|3.3842|0.00022801|
|40|3.7102|3.3507|0.00022805|

This supports a pattern-dependent service plateau and a stream-service decline at the lowest tested allocation. It motivates testing the SM/request-delivery resources needed to preserve useful memory service, with72SMs a candidate region for further decoder validation. It establishes neither decoder behavior at72SMs nor reclaimed die area. The one-chain pointer condition is dependency limited and does not represent parallel decoder service. Separate counter replay can perturb cache history; the listed intervals are per-condition estimates, not pairwise difference intervals.

## Other established observations and limits

The separate14150903 bridge B1/C1024 eager cohort tolerates132→40SMs with a latency ratio1.003745 and95%CI[0.993853,1.011830]. This is a software restriction result. The separate14133085 optimized-graph/eager comparison gives graph/eager wall latency0.292817 (95%CI[0.290147,0.294904]); execution strategy materially changes latency and can change apparent hardware headroom. Restricted graph placement is unsupported in the reviewed route. These cohorts are not pooled.

The14131987 full-window traffic accounting reports Qwen14B graph B1/C4096 read demand near1.00054× summed logical per-tick demand, versus about3.0016× window-unique demand across three ticks. This is consistent with autoregressive recurrence, not evidence of threefold avoidable traffic within a tick. Aggregate ratios do not establish exactly-once access or rule out targeted retention. Modest SRAM benefits need J7 retention/pressure evidence; KV capacity estimates alone do not prove residency.

J8 B1 individual controls pass; pooled layer timing/intrusion analysis remains pending. B8 timing controls are clock rejected. J7/J6 and qualified energy remain unfinished. The earlier native held-out overlap model failed its declared threshold (10.54% versus10%); its coefficients cannot predict decoder speedups. No energy gain, manufacturable extra HBM interface, package feasibility or H200-calibrated architecture simulation is established.

## Research direction

Test whether a smaller compute allocation preserves weight/KV request delivery under optimized decoding. Identify which working sets benefit from targeted SRAM retention, then evaluate conditional bandwidth/SRAM/NoC designs with explicit physical shoreline, package, energy and prefill constraints. A validated resource-service knee plus workload-specific retention opportunities would support the architecture motivation. A common overlap model, physical implementation feasibility and energy savings still require independent validation.
