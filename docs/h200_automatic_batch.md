# Automatic H200 campaign

Current default: the targeted [version2 gap-remediation plan](h200_gap_remediation.md).
The completed broad continuation is retained in `configs/h200_batch_v1.json` as historical policy.
Both the standard entry point below and `batch/h200_gap_v2.sbatch` now use
`configs/h200_batch_gap_v2.json`: corrected J6, J7 counters, then CPU closeout.
The original broad-continuation behavior described below applies to version1.

Submit the research batch script instead of the old allocate-and-sleep script:

```bash
cd ~/scratch/gpullm/02_mem_centric/26_memory_centric_gpu_h100
git switch codex/h200-phase01
git pull --no-rebase --ff-only origin codex/h200-phase01
sbatch batch/h200_shoreline.sbatch
```

The explicit `--no-rebase` overrides a local `pull.rebase=true` setting. Generated
job logs can be modified locally; preserve them in a checkpoint rather than
discarding or stashing research evidence just to update the branch.

If the preceding research allocation is still scheduled, submit with
`sbatch --dependency=afterany:PREVIOUS_JOB_ID batch/h200_shoreline.sbatch`.
This lets Slurm wait for that allocation to end before granting the new one,
including when its interactive step has ended but preemption cleanup is pending.
It avoids the bootstrap's deliberate refusal to overlap with a scheduled predecessor.

No Codex session, SQLite directory, interactive shell or follow-up message is needed. This file requests the same account, embers QoS, one H200, 196 GiB and eight-hour limit as the existing allocation script, with eight CPUs. The implementation has not submitted a job or changed the existing file at `/storage/home/hcoda1/6/jkim4112/batch/h200_embers_yg`.

The batch holds a cooperating repository lock, verifies `codex/h200-phase01`, fetches and fast-forwards from origin, and starts a foreground GPU-bound `srun` step. It refuses staged changes, modified research source, a divergent branch, or another automatic campaign. The branch must remain a runnable, reviewed campaign while a job waits in the queue. Git credentials and existing model access must work without prompts on the compute node.

On allocation, `bootstrap_h200_batch.py` discovers the actual scheduler deadline and the single Slurm-visible H200 UUID. It creates a fresh shared-storage result namespace and node-local Python 3.10/CUDA 13 runtime from `configs/runtime.lock.txt` and `configs/tracer_requirements.lock.txt`, verifies historical output references/raw checksums, freezes current source, and runs the existing guarded campaign. Models download at first use at their pinned revisions. Runtime installation/download time consumes allocation time. Dependencies are installed in user space; no system, driver, clock, power or MIG changes occur.

`configs/h200_batch_v1.json` is the versioned continuation policy. It selects the newest earlier campaign ledger, refuses a predecessor still scheduled, and preserves its source/evidence namespaces. An interrupted J5 gets a fresh native-service cohort at full/104/72/40 SMs, including new full-SM controls. Those repeats are necessary for a same-GPU comparison; prior timing and the twelve completed B8 traffic captures are retained separately. The bootstrap checks that all twelve B8 traffic reader results exist and passed before using the native-only scope. No old exhaustive timing matrix is resumed.

The batch executes this implemented policy; it does not interpret new Markdown research directions or repair unsupported adapters. Future direction changes require a reviewed policy/code update before submission.

After inventory and strict bridge functional checks, the GPU order is J5 → J8 → J7 → J10 → J6 → J11. A prior queue step is skipped only if its ledger records `command_complete` and exit zero, with that ledger's hash retained. This preserves both successful and scientifically partial dispositions; it does not turn unsupported/blocked gates into passes. An interrupted job starts its unfinished family in a fresh allocation namespace; acquisitions from different GPUs are never combined into a quantitative cohort. Unsupported adapters stop at their existing bounded pilots and require a versioned fix if diagnosis is needed. The retained J11 semantic blocker prevents unsupported full tracing and avoids rebuilding trace tools automatically.

Broad CPU analysis, figures, archive attempts and final publication follow GPU collection. A 30-minute closeout reserve stops new GPU launches. Completed acquisitions/job dispositions are committed and pushed by the existing checkpoint helper. Private remote archive failures remain explicit; shared raw data and hashes do not imply a verified independent backup.

Watch `h200-shoreline-JOBID.out` in your submission directory. Detailed logs, `STATUS.json`, `results/batch_bootstrap.json` and `results/campaign_queue.json` live in `h200/YYYYMMDD_jobJOBID/`. A failed setup or queue stop makes the batch fail, retaining its reason and best-effort final checkpoint. Terminal loss does not stop the batch. Preemption, node loss or SIGKILL can stop it immediately; finalization cannot be guaranteed then. Submit the same script for a later allocation to derive a new continuation from durable ledgers. The script does not requeue or submit another job itself.

Paths can be overridden before submission using `H200_REPO`, `H200_BOOTSTRAP_PYTHON`, and `H200_BATCH_PLAN`. The defaults target this user's shared repository, cluster Python 3.10.13 and the checked-in policy. The pinned CUDA installation must exist on the assigned node. Storage free-space checks cover the filesystems; user-specific inode quotas still need enough headroom for the compact result files.

Validation on 2026-10-18: Bash syntax, ShellCheck, six batch CPU contracts and the existing shoreline contracts passed. Slurm `sbatch --test-only` accepted the resource request without submitting a job. The actual predecessor's output-reference hashes, 31 inventoried raw artifacts and all twelve completed B8 traffic readers were verified. Batch contracts cover final-push failure reporting, allocation/deadline rejection, evidence tampering, interrupted versus command-finished reuse and queue skipping. Actual batch startup, dependency restoration and GPU measurements require an allocation and have not been claimed as tested by these checks.
