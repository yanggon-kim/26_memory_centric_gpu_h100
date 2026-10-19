# Allocation 14171804 recovery

Slurm granted the H200 at 2026-10-19 05:47:08 EDT and preempted it at 06:56:20 EDT (69 minutes 12 seconds). The batch started automatically, restored the runtime, passed inventory and bridge exact functional smokes, and pushed acquisitions through commit 3798db3. This was scheduler preemption, not a setup failure or completed campaign.

J5 native-service capture coverage finished: 12 ordinary acquisitions and 36 physical counter captures across full/104/72/40 SMs. Scientific service/clock qualification remains separate. Previously completed decoder timing and physical traffic were retained without repeating them.

J8 finished nine bridge B1 controls with exact outputs and individual clock gates passing. Eight bridge B8 controls retained exact outputs but failed the clock envelope; no qualified B8 timing comparison is claimed. The ninth B8 acquisition was interrupted before a child exit record or final result. Larger-model J8 captures did not start. Pooled timing/event-intrusion qualification remains pending.

All 36 inventoried excluded raw artifacts passed independent size/SHA256 verification. Pre-recovery queue/bootstrap/current-process/job/process records are preserved under results/preemption_20261019_v1. Scheduler accounting and batch stdout are archived under environment/. Missing child exit status is recorded as null, not invented.

Remaining order: unfinished J8, then J7, J10, conditional J6 and J11 GPU disposition; broad CPU analysis/figures/archive closeout follow. The existing J11 semantic blocker still prevents full-model tracing. No GPU job was submitted during recovery. A subsequent batch derives the completed J5 command disposition and skips its GPU sweep.
