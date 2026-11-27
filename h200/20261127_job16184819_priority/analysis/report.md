# Two-H200 priority campaign

No measurements yet. Primary: Qwen2.5-32B BF16/TP1 B1/C4096 SM curve. Secondary UUID: independent native controls.

- Limitation P0_new_pair_KV: Fresh paired HF handoff deferred until after first complete SM sweep; prior adapter validation applies only to its recorded physical pair. Evidence: results/P0_prior_KV_software_validation.json. Tried: Validated unchanged adapter/model source hashes; fresh allocated-pair peer-copy smoke remains separate. Resume: At a safe completed-curve boundary, run the existing real-KV pilot on the new allocated pair; never relabel prior UUIDs.

- Limitation model_startup_io_wait: Both first model producers spend minutes before decode; owned process snapshot shows cl_sync_io_wait and package-bytecode descriptors. No decoder latency is inferred from startup wall time.. Evidence: environment/early_model_process_snapshot.txt; environment/early_model_io_detail.txt; gpu0/logs/graph_b1_c4096_stock_f100_a0_proof.log; gpu1/logs/graph_b1_c4096_stock_f100_a0_traffic_reference.log. Tried: Retained live process/I/O evidence and versioned1200s producer budgets; pinned sources/versions unchanged. Resume: Require complete run publications, exact outputs and measured placement; use actual observed costs and allocation cutoff for curve admission.

- Prepared bounded import-order correction: results/capture_binding_import_preflight_preparation.json. Nine CPU contracts passed; no GPU validation, active frozen worker unchanged. Any use requires a fresh identified cohort and fresh placement proofs.

- GPU1 completed a three-acquisition full-GPU physical DRAM capture comparison: [report](../gpu1/analysis/P2a_triplicate_counter_report.md). Replay/source/reader gates pass; clean-clock/L2/kernel/restricted-arm joins remain incomplete. No qualified clean bandwidth or latency inference.
