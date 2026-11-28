# Native model KV functional prerequisites

No deployment timing or serving qualification. Preserve known-data component, metadata-only model controls and intrusive payload proof separately. At initial setup, no GPU measurements had been made.

- Limitation slow_runtime_preparation: Initial runtime check took380.2288seconds but completed correctly; reduced available measurement/continuation time, root cause unresolved. Evidence: results/runtime_preparation_limitations.json; results/runtime_check.json; logs/runtime_check.log; results/runtime_preparation_process_snapshot.json. Tried: Read-only owned CPU process snapshots, no restart/signals/system changes. Later filesystem waits are evidence of preparation I/O, not proof of a CUDA hang; runtime check passed before its600second bound.. Resume: Proceed with original actual-time admission and strict case gates; do not count preparation as model timing. Revisit startup/cache behavior only on recurrence; preserve all original logs..
