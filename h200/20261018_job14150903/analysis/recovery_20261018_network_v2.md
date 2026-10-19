# GPU-first campaign recovery — October18 evening

## Git and execution state

Before recovery, local HEAD and remote `codex/h200-phase01` both matched `99c7ee0740d30ff52021f99e4d3e3295bdc70642`. The last B8 full-SM physical capture completed and was already pushed. This recovery publishes its remaining log tail, an integrity audit, qualified SM statistics and corrected queue/status records.

No research queue is live. Scheduler accounting reports allocation14150903 PREEMPTED, end12:18:04EDT October18; its extern step ended11:18:15EDT. The saved queue still said J5/running and has been marked interrupted, preserving original snapshots under `results/recovery_20261018_network_v2/`. All saved numerical producer attempts have terminal records; the parent/queue exit code is unknown. The detached launcher did not continue beyond the old execution environment. Starting a new terminal session does not preserve a process against scheduler/cgroup cleanup.

The recovery shell is on `atl1-1-03-019-2-0.pace.gatech.edu`, existing allocation14162783. This is a different node; no new GPU measurement or runtime restoration was started during recovery. Historical UUID/runtime paths/deadlines must not be reused silently.

## Newly saved J5 coverage

- **Ordinary timing:24/24 acquisitions**, each10 timed trials after3 warmups, covering B1/C1024 and B8/C4096 at actual132/104/72/40SM. All exact-output checks passed;18 acquisitions passed individual clock gates and6 B8 acquisitions were clock-rejected (132SM and104SM arms).
- **Placement:8 successful selected-decode NSYS pilots**, covering both shapes and all four returned SM allocations. Restricted graph execution remains unavailable under the reviewed adapter; these new sweeps use eager execution.
- **Physical HBM traffic:15/24 selected captures complete**, with replay/output/metric-reader checks: B1 at all four allocations (12 captures), and B8 at132SM (3 captures). These are instrumented measurements; ordinary timing/clock joins remain separate.
- **Native-service allocation sweeps:0 new acquisitions.** The campaign stopped before this stage.

## Qualified B1 timing result

Limited CPU-only recovery qualification checked complete3x10 arms, exact outputs, immutable run/source hashes, completed isolated producers, measured kernel placement and pooled SM/memory clocks. The pooled clock ratios are1.0. Three eager B1/C1024 comparisons pass the predeclared upper95% latency-ratio bound<=1.05:

| Returned SMs vs132SM | Latency ratio | 95% interval | Upper regression |
|---|---:|---:|---:|
|104|1.00645|[0.99986,1.01266]|1.27%|
|72|1.00314|[0.99236,1.01403]|1.40%|
|40|1.00374|[0.99385,1.01183]|1.18%|

Scope: same in-process eager/stream harness, synchronized wall per pure-decode step over63 ticks, fixed64-token greedy output. This supports the declared timing noninferiority envelope for this software/hardware case. SM restriction also changes issue/LSU/MLP resources; it creates no SRAM, adds no bandwidth interfaces, and is not an iso-area compute-only or architecture-benefit result. B8 comparisons remain unqualified because the baseline clock gate failed.

Full machine-readable statistics are in `results/shoreline_sm_statistics.json`; immutable completion copy and hash are pinned by `results/J5_recovery_summary_v1.json` under `results/J5_partial_completed_v1/`.

## Done versus remaining

| Job | Saved work | Remaining |
|---|---|---|
|J0|Latest allocation14150903 inventory/runtime/contracts passed; H200132SM/queried60MiBL2|Fresh allocation/runtime inventory before new measurements|
|J1|Latest bridge strict HF/graph exact smoke pair; earlier focused baselines and model smokes retained|Prior B8 timing remains clock-rejected; larger-model checks should occur only before their own captures|
|J2/J3|Prior traffic/metadata and54 operation/resource captures retained|Existing scientific attribution/productive-work/clock joins remain incomplete; no cohort pooling|
|J4|Earlier selected collection complete:45 physical captures,15 native pattern joins|Held-out common model rejected (10.5376% error exceeds10%); decoder transfer unqualified|
|J5|All24 ordinary acquisitions,8 placement pilots,15 physical captures; three B1 timing comparisons qualify|Nine B8 restricted physical captures (104/72/40SM x3), all allocation-specific native ordinary/physical service sweeps, final service/traffic qualification|
|J8|Implementation prepared|All selected complete-layer/perturbation captures pending|
|J7|Implementation and logical KV geometry prepared|Matched normal/pressure ordinary and physical captures pending|
|J10|Sensor preflight/energy/prefill path prepared|Sensor diagnostic, eligible energy captures and selected prefill/larger-batch controls pending|
|J6|Owned disjoint-context route prepared|Disjoint placement/concurrency pilot first, expansion only if supported|
|J11|Prior micro replay/calibration/local archive and24 model kernel traces retained|Full trace blocked by unsafe wait pruning/async semantics; no new full-model trace. Private independent archive readback previously HTTP404|
|J9/final CPU|Prior conditional scenarios retained; limited J5 recovery qualification now saved|Broad CPU qualification, joint scenarios, reports, figures and archive transfer follow remaining eligible GPU work|

## Integrity and publication

The recovery independently rehashed127 raw artifacts:31 from this allocation plus96 in the previous retained inventory. All127 match; no missing or changed artifacts. All47 compact J5 profiles (8 placement pilots,24 ordinary captures,15 physical readers) also exactly match their already-pushed Git blobs. The full audit and scheduler evidence are under `results/recovery_20261018_network_v2/`.

Large raw databases/traces remain on persistent shared storage with SHA256 inventories, outside Git. Compact evidence and recovery analysis are committed. Git synchronization does not establish independent private backup of all raw artifacts.

## Resume order

1. Verify the current allocated GPU/UUID, deadline, isolation and budgets; restore pinned runtime/model paths into a new versioned session. Preserve this completed cohort rather than rerunning its ordinary matrix.
2. Prioritize remaining J5 restricted-B8 physical traffic and independent native service sweeps. New-GPU quantitative timing/service joins require complete matched new cohorts; do not append new-GPU data to old captures or silently pool devices. Keep fresh controls narrowly scoped to the comparisons requiring them.
3. GPU J8, then J7, J10, conditional J6 and any eligible J11 GPU work. Establish required larger-model smokes immediately before use. Preserve known J11 semantic blocker instead of repeating unsafe captures.
4. After GPU work ends, run J9/full qualification, reports/figures and private archive readback. Publish every acquisition and job disposition with verified Git push, retaining failed/unsupported/pending scientific gates.
