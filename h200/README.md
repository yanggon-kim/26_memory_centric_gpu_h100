# H200 measurements

Fresh H200 experiments are separated from the historical H100 and RTX evidence. The current session is [20261016_job14073758](20261016_job14073758/analysis/intermediate_results.md). Its [status](20261016_job14073758/STATUS.json), configurations, raw run JSON, measurement table, and analysis are scoped to that allocation.

BF16 weights/KV, one physical GPU, immutable model revisions, seed 20261013, three warmups/trials, and exact output gates remain required. H200 measurements use the queried 60 MiB L2 and actual memory system. Historical H100 timings, counter denials, and 50 MiB L2 are not substituted.

Large profiler databases and trace archives are ignored by Git. Their locations and checksums remain in the session manifests. Runtime environments, model downloads, and build trees live in an owned job-specific node-local workspace; package locks and source identities are retained here.

The scripts `h200_session.py`, `run_h200_phase01.py`, and `summarize_h200.py` preserve commands, per-case results, and unfinished work. The session wrapper sets the explicit CUDA toolkit, GPU UUID, cache directories, thread limits, and launch deadline. It never changes device clocks, power limits, drivers, or scheduler settings.
