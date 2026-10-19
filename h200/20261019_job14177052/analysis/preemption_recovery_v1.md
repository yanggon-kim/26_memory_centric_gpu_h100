# Allocation 14177052 recovery

Slurm started the automatic H200 batch at 2026-10-19 07:48:27 EDT and preempted it at09:39:11 EDT (1h50m44s). Startup/runtime, inventory and exact bridge controls succeeded. Completed acquisitions were pushed through ce06e147 before interruption. No new GPU job was submitted during recovery.

J5 reused the prior command-finished disposition without a GPU recapture. J8 finished45 selected captures:9 individual checks passed and36 were clock rejected; this is capture completion, not qualified full-model layer timing/overlap evidence. J7 finished18 timing captures (12 individually complete,6 clock rejected) and18 physical-counter reader results (all failed). J7 physical traffic remains unqualified and requires a versioned counter/replay/range diagnosis. Command exit0 does not close that scientific gate.

J10 completed bridge energy/prefill raw acquisitions and the first Mistral B1 energy acquisition; Mistral B1 energy acquisition01 was interrupted before a recovered exit record. Numeric energy telemetry alone is not full clock/output/family qualification. J10 family qualification and remaining larger-model controls are unfinished. J6 and J11 GPU disposition did not start; broad CPU analysis/figures/archive closeout did not start.

All18 inventoried raw artifacts passed size/SHA256 verification. Original queue/bootstrap/current-process/job/unfinished-process records are saved under results/preemption_20261019_v1/. Scheduler accounting and batch stdout are retained under environment/. Missing child exit status is null.

A new automatic batch will retain command-finished J5/J8/J7 dispositions and continue J10→J6→J11_gpu, followed by CPU analysis/archives. This skip policy does not repair or pass J7's failed counter gate, J8's clock rejections or the existing J11 trace-semantic blocker. Those remain explicit scientific limitations.
