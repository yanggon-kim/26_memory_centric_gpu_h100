# Recovery after network interruption — 2026-10-17

The local and fetched remote `codex/h200-phase01` heads both matched `c709bd6fd723d40494b0d0dde79cc83bbac3193b` before this recovery. Completed acquisitions had already been pushed. This recovery publishes the final log tail, refreshed CPU-only SM statistics and corrected status; it starts no GPU measurements.

## Actual execution state

Scheduler accounting reports allocation14133085 PREEMPTED, end13:15:22 EDT; batch cancelled with signal15. No research queue/process is live. The recovery shell belongs to existing allocation14134786 on the same host, but the old allocation-local Python/runtime path is absent. The original session configuration remains historical. A new versioned configuration, current UUID/isolation/deadline/storage inventory and runtime restoration are required before restart. No Slurm job or system configuration change was made. Absence of a final queue exit record is not successful completion.

## Saved work and qualification

| Job | Saved work | Remaining qualification/work |
|---|---|---|
| J0 | Prior allocation inventory/runtime/contracts passed | Fresh inventory/runtime for the recovery allocation |
| J1 | Focused graph/eager controls: both shapes 3 acquisitions x10; all3 model strict smoke comparisons exact | B1 engine timing qualified; B8 timing rejected by clocks; broader timeline/scientific gates remain partial |
| J2 | Prior physical-traffic captures and provenance retained | Unresolved whole-model/clock/source joins; do not pool GPU cohorts |
| J3 | Prior54 selected operation/resource captures retained | Dynamic productive/issued work and decoder-wide interpretation incomplete |
| J4 | Selected ordinary acquisitions,45 physical captures and15 qualified native pattern joins complete | Held-out common-model error10.5376% exceeds10% threshold; decoder transfer rejected/unqualified |
| J5 | Eager B1C1024 f100 acq00-02 and f75 acq00-01 complete: exact outputs,10 trials each, individual clock gates passed. B1/B8 eager f75 placement pilots passed | f75 acq02 absent; zero qualified comparisons. Remaining supported eager fractions/shapes plus physical/native-service sweeps pending. Graph restriction unsupported after bounded reviewed pilots |
| J8 | Adapter/analysis paths prepared | Current complete-layer timing and perturbation collection pending |
| J9 | 320 conditional analytical scenarios saved | Refresh with qualified evidence; physical area/shoreline/package feasibility and decoder coefficients unvalidated |
| J7 | Logical KV geometry and independent-reference route prepared | Normal/pressure acquisition pairs and physical attribution pending |
| J10 | Energy methodology retained | Selected long-window energy and negative-regime captures pending; primary energy claims remain gated |
| J6 | Controlled-competition implementation path prepared | Placement/concurrency pilot and collection pending, conditional on supported disjoint resources |
| J11 | Native micro capture/postprocess/local archive and positive parser replay;3x10 same-harness eager calibration;24 exact-output native model kernel traces saved | Eight kernels trigger unsafe wait-pruning precheck; model postprocess/replay and full-model trace blocked. H200 timing/power calibration incomplete. Private archive upload/readback HTTP404 |

J1 and J4 completion reviews pin immutable copies under `results/J1_completed_v1/` and `results/J4_completed_v1/`; use these instead of assuming regenerated analysis paths are unchanged. J11 details are in `analysis/trace_semantic_review.md`.

## Integrity and publication

`results/recovery_20261017_network_v2/audit.json` independently verifies all96 raw inventory entries:96 SHA256 matches, zero missing or changed artifacts. Scheduler evidence and original status/manifest/queue snapshots are preserved beside it. Large raw profiler/trace files and the full measurement aggregate remain on persistent shared storage, outside Git; compact evidence and bounded lossless measurement shards are committed. Private independent archive readback is still unavailable (HTTP404), so Git publication must not be described as backup of every raw trace/database.

CPU-only SM analysis was refreshed: no supported family yet has complete matched3x10 arms plus all required source/context/isolation/pooled-clock gates. No performance benefit or noninferiority is claimed from the incomplete75%-SM arm.

## Resume order

1. Reconcile the actual allocated GPU, deadline, source/runtime/model identity and budgets into a new session; preserve all old cohorts separately.
2. Finish supported eager J5 families and their physical/native-service controls after bounded pilots. Reuse old functional/semantic evidence where applicable; quantitative cross-GPU pooling is forbidden. Do not repeat unsupported graph pilots without a new justified adapter/review.
3. Collect J8 complete-layer timing and refresh J9 bounded/conditional analysis.
4. Collect selected J7 retention/pressure controls, then J10 energy/negative controls.
5. Attempt J6 only after disjoint placement/concurrency proof and sufficient closeout time. J11 full-model capture remains blocked until async/wait semantics pass; resolve private archive access for independent transfer verification.
6. Publish acquisition/job checkpoints, final qualified/failed/pending ledger and immutable evidence handoff. Do not restart the exhaustive old Phase1 portfolio.
