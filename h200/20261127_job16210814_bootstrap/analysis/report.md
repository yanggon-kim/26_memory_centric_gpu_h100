# Allocated staging and launch

No model measurement inferred from installation.

- Limitation startup_import_latency: Pinned runtime validation remains at importing torch for several minutes on this fresh node; this is startup wall time, not decoder latency.. Evidence: logs/runtime_check.log; environment/startup_process_state.txt. Tried: Prepared600s budget and unbuffered progress from the initial attempt; retained owned-process snapshot. Resume: Require successful pinned versions and exact two-device validation before model work; actual Slurm cutoff still governs.
