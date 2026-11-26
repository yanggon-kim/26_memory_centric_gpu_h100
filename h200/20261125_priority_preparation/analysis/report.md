# Preparation checkpoint

No GPU experiments have started.

- Limitation preparation: Login-node runtime installation and model download both terminated with exit137; cause not established. Evidence: h200/20261125_priority_preparation/logs/runtime_install.log and model_download.log. Tried: Separate pinned runtime and bounded immutable model download; partial files retained. Resume: Perform staging inside the authorized Slurm allocation with requested host memory.

Submitted job16163464: two H200s/16CPUs/392GiB/eight hours, pending Priority at receipt. No GPU allocation or measurements yet. Preparation checkpoint ef59ed37 pushed, including preserved historical LFS ancestor f634c0fc.

- Limitation P0_HF_input_contract: Preflight code/config mismatch: HF dynamic generate infers padding from model pad token151643, while synthetic vLLM prompt IDs are all active. No new GPU token mismatch has been observed.. Evidence: scripts/hf_full_reference.py and immutable Qwen32B generation_config.json at revision5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd. Tried: Inspected actual staged metadata and reference input path; explicit all-ones attention mask is required before new comparisons. Resume: Run the corrected source and unchanged exact-token gate on allocated GPUs; earlier evidence remains unchanged.
