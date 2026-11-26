# Allocated staging and launch

No model measurement inferred from installation.

- Limitation P0_visibility: Allocated job has two H200 GRES (IDX1-2), but main step exports CUDA_VISIBLE_DEVICES=1 and SLURM_STEP_GPUS=2; two-device CUDA validation is not established. Evidence: environment/main_step_visibility.txt; environment/visible_device_mapping.txt; environment/slurm.txt. Tried: Preserved main-process environment and read-only NVML inventory; checking driver minor numbers against the authorized allocation before any visibility repair. Resume: Require explicit mapping of both allocated GPU UUIDs and successful exact two-device runtime validation; no measurements accepted yet.

## Allocation inventory and visibility gate

Started 2026-11-26 at10:31:18 Eastern; scheduled end18:31:24. Driver-visible allocated H200 pair: `GPU-951a11c2-40ee-3fcb-f71f-49849e744cb1` and `GPU-b51c6186-ac3b-5149-8b8a-f33943072c47`, each143771MiB total/143201MiB initially free, driver615.71.09, topologyNV18. `RmProfilingAdminOnly=0`; actual counter smoke remains pending. Exact evidence is in `environment/allocation_receipt.json` and its raw inventory paths.

The inherited CUDA mask exposes only one device despite a two-GPU Slurm allocation. Driver minor numbers2/5 differ from Slurm GRES indices1/2, so ordinal equality is not assumed. The bounded existing-allocation retry script `batch/h200_priority_visibility_repair_16163464.sh` verifies the node, RUNNING allocation and exact driver-visible pair, then selects those UUIDs. It has passed shell syntax validation but has not run yet. Original staging logs remain separate. No new model measurement exists.

- Limitation bootstrap: Allocated staging/launch failed: runtime_check failed with exit 1. Evidence: h200/20261126_job16163464_bootstrap. Tried: Preserved each stage log and pinned requirements; no system changes. Resume: Inspect failed stage and use existing allocation if time remains.
