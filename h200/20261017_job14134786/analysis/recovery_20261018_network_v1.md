# Network recovery — October18,2026

## Branch and live state

Before recovery, local HEAD and fetched origin `codex/h200-phase01` both matched `26c38f385b7459b33c87a3fd5f03791e74ffd8be`. All three completed run JSONs in allocation14134786 exactly match their pushed Git blobs. This recovery publishes the remaining log tail, canonical Mistral HF qualification and unfinished graph-attempt records, plus corrected status and this handoff.

No research queue is live. Allocation14134786 is now PREEMPTED: scheduler end18:45:55EDT October17. The last monitor observation was17:46:06.517803EDT, shortly after the Mistral graph process started; its log is empty and no completed run JSON exists. No exit code was saved. Do not infer that this attempt completed, timed out, or stopped specifically because of the later preemption. Original process/status/manifest records are snapshotted under `results/recovery_20261018_network_v1/` before marking the attempt interrupted with exit code null.

The recovery shell is on `atl1-1-03-020-18-0.pace.gatech.edu`, existing allocation14150903, whose current scheduler end is18:09:52EDT October18. This is a different node; the old node-local runtime/model/tool paths are not reusable without restoration. No new Slurm job or GPU measurement was started during this recovery.

## Most recent progress (allocation14134786)

- Runtime restored at the pinned package versions; dependency checks and expanded CPU contracts passed.
- J0 inventory passed: assigned H200,132SM, queried62914560-byte/60MiB L2, CUDA13 runtime, counter catalog and CPU contracts.
- Bridge strict StaticCache HF and graph C128/B1/64-token smoke runs passed, with exact cross-engine IDs over all3 smoke trials. Their timings are functional smoke evidence, not3x10 quantitative controls.
- Pinned bridge and Mistral model materialization and Mistral tokenizer proof completed.
- Mistral strict HF smoke passed and was pushed in26c38f3. Its canonical qualification is included by recovery.
- Mistral graph smoke is interrupted/unqualified; Qwen14B startup smoke did not run.
- **The remaining-job queue never started. There are no new J5/J8/J7/J10/J6 measurements from this last attempt.**

The prepared implementation is committed: supported eager-only SM continuation, unrestricted physical-counter controls, stop-before-expansion guards for new physical/native routes, sensor preflight and clock gates for energy, and per-job CPU analysis/figure/push boundaries.

## Accumulated completed coverage and remaining gates

| Job | Saved coverage from earlier sessions | Left / limitation |
|---|---|---|
| J0 | Prior inventories/runtime/contracts plus latest14134786 inventory passed | Fresh inventory/runtime/configuration for14150903 |
| J1 | Focused graph/eager B1C1024 and B8C4096 acquisitions; strict model smokes; latest bridge exact pair and Mistral HF smoke | Prior B1 engine comparison qualified; prior B8 timing clock-rejected. Latest Mistral graph smoke unfinished. Preserve stronger graph baseline separately from eager restrictions |
| J2 | Prior physical traffic and metadata captures retained with hashes | Remaining source/clock/whole-model attribution gates; no cross-session quantitative pooling |
| J3 | Prior54 selected operation/resource captures complete | Productive/issued work and decoder-wide interpretation remain incomplete |
| J4 | Selected native/extension collection complete:45 physical captures and15 qualified native pattern joins | Common-model held-out error10.5376% exceeds10% threshold; decoder coefficient transfer remains unqualified |
| J5 | Earlier eager B1C1024 f100 acq00-02 and f75 acq00-01 saved,10 trials each; eager placement pilots on B1/B8 | Third f75 acquisition absent; zero fully qualified matched SM comparisons. Remaining fractions/shapes, physical traffic and independent service sweeps pending. Restricted graph route unsupported |
| J8 | Adapter and analysis paths prepared | Complete-layer timing and perturbation captures pending |
| J9 | Prior320 conditional analytical scenarios saved | Refresh with qualified new evidence; physical feasibility and decoder-model transfer unvalidated |
| J7 | Logical KV geometry and independent-reference path prepared | Matched retention/pressure timing and physical attribution pending |
| J10 | Sensor preflight/qualification path prepared | Selected sensor-gated energy and prefill/larger-batch controls pending; primary J/token remains gated |
| J6 | Owned disjoint-context route prepared | Resource/concurrency pilot first; expand only if proven |
| J11 | Prior native micro positive replay/local archive,3x10 same-program calibration,24 exact-output model kernel traces | Eight kernels require unsafe wait pruning in pinned postprocessor: preserve waits, block full-model expansion. H200 timing/power calibration incomplete. Independent private archive readback previously HTTP404 |

## Data integrity and publication

All96 previously inventoried raw artifacts were independently rehashed during this recovery:96 matches, zero mismatches/missing files. All3 completed latest-session run JSONs match their already-pushed blobs. The machine-readable audit is `results/recovery_20261018_network_v1/audit.json`; scheduler snapshots are adjacent. Original records are preserved before recovery edits.

Large profiler/trace files remain on persistent shared storage with SHA256 inventories, outside Git. Compact runs, logs, manifests, source/provenance records and bounded measurement shards are published. Git synchronization does not establish independent private backup of every raw artifact.

## Remaining order

1. Fresh J0 allocation/UUID/isolation/deadline/budget inventory and pinned runtime restoration for14150903. Establish required exact bridge controls, then prioritize core work rather than repeating the exhaustive old Phase1 portfolio. Gate larger models before their own new captures.
2. J5 supported eager allocation/ordinary/physical/native controls with bounded pilots, complete3x10 cohorts and strict pooled clocks. Never append across GPUs or silently pool old partial cohorts. Retain reviewed unsupported graph disposition.
3. J8 complete-layer timing, then J9 conditional/bounded analysis and figures.
4. J7 matched retention/pressure controls, then J10 sensor-gated energy and negative regimes.
5. Conditional J6 after disjoint concurrency proof; J11 full-model work remains blocked until safe async/wait semantics pass. Retry authorized private archive/readback only with working access.
6. Commit/push every acquisition and job analysis with remoteHEAD verification; close out before the actual scheduler deadline, leaving blocked/pending scientific gates explicit.
