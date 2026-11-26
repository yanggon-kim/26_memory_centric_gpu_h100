# Two-H200 priority campaign

No decoder-curve measurements yet. Primary: Qwen2.5-32B BF16/TP1 B1/C4096 SM curve. Secondary UUID: independent native controls.

- Limitation P0_inventory_startup_IO: Fresh device-query process required several minutes; process snapshot showed cl_sync_io_wait, then query completed and paired pilot launched without repair. Evidence: h200/20261126_job16163464_bootstrap_repair1/environment/coordinator_start_processes.txt; h200/20261126_job16163464_bootstrap_repair1/environment/coordinator_query_io.txt; environment/allocated_devices.json. Tried: Observed the owned process and retained both snapshots; no process termination or system changes. Resume: Budget from actual measured producer pilots; startup waiting is not decoder latency and is not a GPU measurement.

## P0 synthetic peer-copy control

The allocated pair completed3 warmups and10 instrumented128MiB BF16 known-value copies. Correctness passed, peer access was available in both directions, and the sampled ownership monitor recorded no failures. The pilot copies in one direction; bidirectional peer capability is not a bidirectional bandwidth measurement. Nsight Systems produced `profiles/P0_p2p.nsys-rep`; raw results and scope are in `results/P0_p2p.json`. Timing is an instrumented single-acquisition pilot, with no primary transfer-latency, physical-link-byte or serving claim. Real all-layer KV handoff is a separate pending pilot.
