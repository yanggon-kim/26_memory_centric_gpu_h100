# Native model KV functional prerequisites

No deployment timing or serving qualification. Preserve known-data component, metadata-only model controls and intrusive payload proof separately. At initial setup, no GPU measurements had been made.

## Handoff

functional_prerequisites_passed. Only the component/model functional stages explicitly passed by their CPU readers count; no producer-penalty, full-versus-restricted live worker or goodput comparison is completed here

Next: /storage/scratch1/6/jkim4112/envs/h200_priority_20261125/bin/python -m json.tool /storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100/h200/20261128_job16281263_native_functional_c16384/results/worker_outcome.json.

- Limitation C16384_current_pair_native_host_staging: All12current-pair functional handoffs recorded3,355,443,200host-store tensor bytes and equal host-load bytes out of4,294,967,296requested model KV bytes per request. This is not a pure device-to-device transfer or isolated network measurement.. Evidence: results/native_host_pool_summary.json; model/results/native_model_CPU_review.json. Tried: Retained native defaults and actual store/load/free API evidence; no source, transport or system settings changed. Pre-cleanup retention is not a post-exit leak. Original archive remains byte-identical; this derived note postdates it.. Resume: Separate attributed host copy/blocking and communication intervals from clean matched producer host time. Do not infer timed-cohort spill per request from this instrumented functional proof. Physical HBM/link counters and end-to-end serving require separate measurements..

## Allocation closeout

Job16281263 ended FAILED at08:38:47Eastern after2009seconds (1.116111GPU-hours). Native functional gates passed. Separate producer cohort stopped on its165second stock-after process timeout, not preemption. Original functional archive SHA256: bb0170afd0046ab801b1d50dac1ead43fca5b88ffe62ce50ff2ccaff0e53e7eb. This closeout and host-staging summary postdate that archive.
