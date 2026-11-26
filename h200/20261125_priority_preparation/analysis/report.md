# Preparation checkpoint

No GPU experiments have started.

- Limitation preparation: Login-node runtime installation and model download both terminated with exit137; cause not established. Evidence: h200/20261125_priority_preparation/logs/runtime_install.log and model_download.log. Tried: Separate pinned runtime and bounded immutable model download; partial files retained. Resume: Perform staging inside the authorized Slurm allocation with requested host memory.

Submitted job16163464: two H200s/16CPUs/392GiB/eight hours, pending Priority at receipt. No GPU allocation or measurements yet. Preparation checkpoint ef59ed37 pushed, including preserved historical LFS ancestor f634c0fc.
