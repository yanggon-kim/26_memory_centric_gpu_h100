# Allocated staging and launch

No model measurement inferred from installation.

- Limitation runtime_import_io_wait: Retry progress stops at importing torch; allocated process snapshot at00:42:40 Eastern shows runtime PID1484434 in cl_sync_io_wait. This observes a filesystem wait, not GPU execution.. Evidence: logs/runtime_check.log; environment/runtime_process_snapshot.txt. Tried: Preserved process state; allowing the single 600s bounded startup retry without changing packages or device settings. Resume: Successful pinned imports and exact two-device validation; if timeout persists investigate node-local runtime staging as a separate recorded repair.
