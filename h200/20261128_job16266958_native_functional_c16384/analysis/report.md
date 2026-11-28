# Native model KV functional prerequisites

No deployment timing or serving qualification. Preserve known-data component, metadata-only model controls and intrusive payload proof separately. At initial setup, no GPU measurements had been made.

- Limitation slow_runtime_preparation: Initial runtime check took380.2288seconds but completed correctly; reduced available measurement/continuation time, root cause unresolved. Evidence: results/runtime_preparation_limitations.json; results/runtime_check.json; logs/runtime_check.log; results/runtime_preparation_process_snapshot.json. Tried: Read-only owned CPU process snapshots, no restart/signals/system changes. Later filesystem waits are evidence of preparation I/O, not proof of a CUDA hang; runtime check passed before its600second bound.. Resume: Proceed with original actual-time admission and strict case gates; do not count preparation as model timing. Revisit startup/cache behavior only on recurrence; preserve all original logs..

## Handoff

functional_prerequisites_passed. Only the component/model functional stages explicitly passed by their CPU readers count; no producer-penalty, full-versus-restricted live worker or goodput comparison is completed here

Next: /storage/scratch1/6/jkim4112/envs/h200_priority_20261125/bin/python -m json.tool /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261128_job16266958_native_functional_c16384/results/worker_outcome.json.

- Limitation post_functional_producer_budget: insufficient protected budget; preserve completed functional case. Evidence: results/planned_producer_followup.json. Tried: Original functional case completed, published and archived first; require32minutes for a complete separately supervised producer pilot. Resume: Reuse the qualified functional software evidence within scope; prepare the next complete producer comparison after this allocation closes. Never weaken source/clock/output/isolation gates or start an incomplete contrast..
