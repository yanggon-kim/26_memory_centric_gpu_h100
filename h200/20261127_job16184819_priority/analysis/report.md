# Two-H200 priority campaign

No measurements yet. Primary: Qwen2.5-32B BF16/TP1 B1/C4096 SM curve. Secondary UUID: independent native controls.

- Limitation P0_new_pair_KV: Fresh paired HF handoff deferred until after first complete SM sweep; prior adapter validation applies only to its recorded physical pair. Evidence: results/P0_prior_KV_software_validation.json. Tried: Validated unchanged adapter/model source hashes; fresh allocated-pair peer-copy smoke remains separate. Resume: At a safe completed-curve boundary, run the existing real-KV pilot on the new allocated pair; never relabel prior UUIDs.
