# Authorized two-H200 execution

Owner authorization: one eight-hour Embers allocation, two H200s on one node, account gts-dmahajan7. No automatic resubmission, system changes, or tensor parallelism.

Repository fast-forward pull completed before launch; required commit7843c845 is present. Existing local f634c0fc and all historical/untracked evidence are preserved. Login-node staging ended137; staging moves into the allocated compute step.

The launcher records two Slurm-visible UUIDs before spawning separate single-GPU workers. Every producer uses allocation/deadline checks, an idle guard, a cooperating physical-device lock, source snapshots, and ongoing NVML process/clock monitoring. Source code is frozen per worker. Exact token, prefill-barrier, three independent acquisition/ten timed-trial, and <=5% clock gates are unchanged. No cross-board timing pooling.

GPU0: smoke graph/eager and independent HF references, target B1/C4096 reference, actual decoder graph-placement proofs, broad 100/85/75/60/45/30% SM curve, interleaved full references, and explicit stock/full-green controls. Primary thresholds compare against stock unrestricted latency; same-green-path ratios are also retained. Requested fractions use actual CUDA-supported rounding/counts.

GPU1: inventory, known-payload NCU permission/control smoke, independent native streaming/gather/arithmetic SM controls. These are useful payload/service controls; they are never relabeled as decoder physical traffic or architectural bandwidth gains.

After the primary curve, prioritize P2a actual traffic and eligible scheduling/long-context cases. Optimized restriction failures receive immediate limitations entries, a bounded diagnosis/repair, and a separately labeled eager fallback. Missing serving/transfer adapters require versioned implementation and a bounded pilot; pending rows are not evidence. Four-device P5b is deferred.

Persist logs, manifests and limitations throughout. Publish small complete comparisons and retain partial/failed captures. Stop new GPU launches 30minutes before the actual Slurm deadline and reserve closeout for evidence, checksums and remote-SHA verification.
