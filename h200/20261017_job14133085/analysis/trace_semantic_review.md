# Current native trace review

The native micro is nonempty real sm_90 and replays4,121cycles/19,456active-thread instructions/one kernel under the stock H100 parser-smoke configuration; this is not H200 timing/power calibration. All three standalone same-program eager acquisitions pass exact-prefix and pooled-clock gates.

The two-step layer-zero capture reaches the verified prefill barrier, matches the standalone greedy prefix, and yields24 nonempty binary-version90 kernel traces. Raw inventory includes HGMMA, TMA, arrival and TRYWAIT records. Eight kernels trigger the pinned postprocessor's wait-pruning rule. The model trace is therefore preserved as unprocessed/semantically unqualified; no waits are deleted, no model replay success is claimed, and full-model expansion is blocked. Raw/active-lane record contributions are not dynamic issued work.

The initial micro setup failure and its versioned directory correction remain preserved. The current native-micro private upload returns404; persistent local archive/readback and independent private transfer remain separate gates. Continue J5-J10 while retaining these limits.
