# H200 measurement limitations and recovery log

Use with the [priority runbook](h200_priority_experiments_20261125.md). Add an entry as soon as a limitation changes what a case can establish. Link the entry from the relevant per-run Markdown report. Record observed results separately from inherited risks. At handoff the optimized Qwen2.5-32B restricted path awaits validation on the new rental.

| Date, run and case | Limitation and evidence (command/config/log) | Scientific consequence | Attempts and fallback evidence | Status and next action |
|---|---|---|---|---|
| 2026-11-26T03:35:19.426038+00:00 h200/20261125_priority_preparation preparation | Login-node runtime installation and model download both terminated with exit137; cause not established; evidence: h200/20261125_priority_preparation/logs/runtime_install.log and model_download.log | No claim beyond the recorded qualified scope. | Separate pinned runtime and bounded immutable model download; partial files retained | open; resume: Perform staging inside the authorized Slurm allocation with requested host memory |
