# Correctness audit v4

CPU analysis/consolidation completed after the import fix. Original failed batch and raw runs remain preserved.

B8 first differs at generated token 6 in request 0: HF selects token 4; vLLM eager and graph select token 15. The three routes share the prompt digest and are internally deterministic. Exact cross-engine correctness remains rejected.

The prior eager/StaticCache diagnostic used a Boolean 4D mask with an implementation that adds the mask to scores. It did not mask unused KV slots correctly; its agreement with vLLM cannot be used. SDPA accepts Boolean masks, so this does not establish the cause of the production SDPA mismatch. A corrected, forced-common-prefix GPU diagnostic is required.

No physical compute/memory/overlap decomposition or new calibration success is claimed.
