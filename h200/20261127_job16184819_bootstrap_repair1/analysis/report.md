# Allocated staging and launch

No model measurement inferred from installation.

- Limitation runtime_import_io_wait: Retry progress stops at importing torch; allocated process snapshot at00:42:40 Eastern shows runtime PID1484434 in cl_sync_io_wait. This observes a filesystem wait, not GPU execution.. Evidence: logs/runtime_check.log; environment/runtime_process_snapshot.txt. Tried: Preserved process state; allowing the single 600s bounded startup retry without changing packages or device settings. Resume: Successful pinned imports and exact two-device validation; if timeout persists investigate node-local runtime staging as a separate recorded repair.

- Recovery update: Bounded startup retry resolved: pinned version checks and exact two-H200 validation passed after 368.8s. Both actual devices report132SM and150156607488B total HBM; no package or GPU setting changed. Evidence: h200/20261127_job16184819_bootstrap_repair1/logs/runtime_check.log and results/runtime_check.json. All subsequent model/config/CPU contracts passed; coordinator started. Initial180s timeout and filesystem-wait snapshots retained. This is startup evidence only, not model performance.
