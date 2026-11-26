# Authorized two-H200 execution

Owner authorization: one eight-hour Embers allocation, two H200s on one node, account gts-dmahajan7. No automatic resubmission, system changes, or tensor parallelism.

Repository fast-forward pull completed before launch; required commit7843c845 is present. Existing local f634c0fc and all historical/untracked evidence are preserved. Login-node staging ended137; staging moves into the allocated compute step.

The launcher records two Slurm-visible UUIDs before spawning separate single-GPU workers. Every producer uses allocation/deadline checks, an idle guard, a cooperating physical-device lock, source snapshots, and ongoing NVML process/clock monitoring. Source code is frozen per worker. Exact token, prefill-barrier, three independent acquisition/ten timed-trial, and <=5% clock gates are unchanged. No cross-board timing pooling.

GPU0: smoke graph/eager and independent HF references, target B1/C4096 reference, actual decoder graph-placement proofs, broad 100/85/75/60/45/30% SM curve, interleaved full references, and explicit stock/full-green controls. Primary thresholds compare against stock unrestricted latency; same-green-path ratios are also retained. Requested fractions use actual CUDA-supported rounding/counts.

GPU1: inventory, known-payload NCU permission/control smoke, independent native streaming/gather/arithmetic SM controls. These are useful payload/service controls; they are never relabeled as decoder physical traffic or architectural bandwidth gains.

After the primary curve, prioritize P2a actual traffic and eligible scheduling/long-context cases. Optimized restriction failures receive immediate limitations entries, a bounded diagnosis/repair, and a separately labeled eager fallback. Missing serving/transfer adapters require versioned implementation and a bounded pilot; pending rows are not evidence. Four-device P5b is deferred.

Persist logs, manifests and limitations throughout. Publish small complete comparisons and retain partial/failed captures. Stop new GPU launches 30minutes before the actual Slurm deadline and reserve closeout for evidence, checksums and remote-SHA verification.

## Reviewed continuation within the same allocation

After both initial workers finish, the coordinator keeps the same allocation available until the 30-minute closeout boundary. The active run's `control/*.task.json` accepts explicit single-device script tasks with `id`, `gpu` (0 or1), repository script basename, `arguments` (a JSON string list), and `timeout_s`. The coordinator fixes the UUID, snapshots a fresh immutable numerical source tree, records a new session identity, and uses the existing allocation/isolation supervisor. It never evaluates shell text or submits another Slurm job. Scientific acceptance remains separate from task exit status. Creating `control/finalize` requests closeout at the next completed task boundary.

The early paired peer-copy pilot is a synthetic known-value transfer with sampled process-ancestry monitoring. It is explicitly not real decoder KV handoff or evidence of disaggregated serving. The spare GPU also attempts its own full-GPU32B physical traffic captures after native controls, preserving separate physical-device results.

The new SM targets use nearest supported co-scheduled groups (ties upward), bounded by the queried minimum partition; full allocation requests the actual device count. Requested fractions, selected counts, and actual driver-granted counts are retained. Historical helper calls continue to use upward rounding by default; no earlier evidence is relabeled.
