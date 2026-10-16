# H200 on chip residency opportunity

The current Phase 1 measurements distinguish software execution paths and sample the memory/resource behavior of actual decode kernels. No added-SRAM traffic reduction has been measured.

The next discriminating evidence is a calibrated full-model access stream across successive decode steps, charged against existing 60 MiB L2 behavior. Weight/KV pinning and ordinary replacement must preserve the intervening model traffic.

Traffic reduction, higher achieved bandwidth, and changed kernel work are different hypotheses. A controlled decode improvement alone cannot establish serving p99, area parity, or architectural energy benefit. SRAM banking, data-delivery costs, retained compute/control, and leakage remain outside the measured board-energy results.
