# Allocation 16228671 closeout

Slurm state: **PREEMPTED**. Allocated GPU-hours: **2.0150**, including staging and idle.

CPU reconciliation only. Original producers, partial runs and failures are retained. Different physical devices are not pooled.

| GPU UUID | Engine | Batch/context | Acquisition set captured | Qualified curve | Disposition |
|---|---|---|---|---|---|
| GPU-d6a286c1-f8de-4287-55f5-2ba29b153b9c | eager | 1/4096 | True | False | captured_quality_unresolved |
| GPU-28fa5fe0-0a71-78dc-ff99-a1fb75e9d459 | graph | 1/4096 | False | False | incomplete |
| GPU-28fa5fe0-0a71-78dc-ff99-a1fb75e9d459 | graph | 1/16384 | False | False | incomplete |

See `../results/allocation_closeout.json` for exact accounting, source identities, thresholds, interrupted producers and physical capture records.
A completed acquisition set with failed clock/source gates remains unqualified. Do not resample solely for favorable clocks. An interrupted proof does not establish that SM restriction is unsupported.
Physical counters remain instrumented evidence; this review does not establish a clean GB/s join or a serving result.
Before another authorized two-H200 submission, select the highest-priority unfinished eligible comparison and validate its launcher. Reuse completed results only within their recorded scope.
