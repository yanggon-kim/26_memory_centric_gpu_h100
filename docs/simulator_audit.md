# Pinned simulator evidence and remaining gates

Framework v2.0.0 `64653015f85fb5664c84a10f48527e8897d289d0`; backend `91880c53383d5a6a6742bfb1be2c5f34e39c7871`; pybind11 `6254086d96062249c5153c17eab6b4437dd1589f`; NVBit 1.8. Pins are in `configs/simulator_pin.json`. The parent's CPU build, Python-extension import and postprocessor build passed. The CMake project VERSION still says 1.2: use immutable source pins/release tag, not that stale field as evidence of deployed functionality.

A public genuine V100 LUD fixture replayed 10 kernels, 127,309 cycles and 418,884 active-thread instructions. The same compatible legacy fixture under H100 configuration produced 118,888 cycles and the same instructions. These are CPU build/config regression smokes, not native Hopper traces, H100 timing calibration or LLM results.

A small native RTX sm_120 trace produced 20,224 reported warp instructions and nonempty compressed traces after fixing PATH and folder setup. Its simulation is correctly rejected: the trace frontend has no sm_120 dispatch and can report unsupported architecture while exiting zero. Never relabel its header as sm_90. Native H100 capture, modern Hopper opcode replay and workload calibration remain pending.

The source audit found these consequential limitations:

- H100 `gpgpusim.config` lines 187–188 disables power until a real energy model exists. An older power XML is not H100 calibration. Physical SRAM/tags/ECC/ports/controllers/links, leakage and timing remain independent PPA requirements.
- `gpgpu_num_mem_units` is explicitly unused. Changing it cannot demonstrate LSU throughput scaling.
- Hopper GMMA goes through specialized unit 5; the generic tensor-core knob is not automatically the WGMMA throughput knob. Tile-dependent timing overrides require opcode-path inspection.
- TMA shader-side handling represents global-load work; prefetch behavior and complete end-to-end control/data-movement semantics are not all implemented. Inspect actual relevant source paths and do not assume release claims cover every traced instruction or workload.
- Address traces do not execute a newly designed AGU/page-walk/descriptor engine. A local interconnect without physical flit/wire accounting cannot prove hop-byte energy. New memory-serving SM roles/topology are not supplied merely by changing a count.
- Shipped H100 simple-DRAM configuration requires achieved-bandwidth and memory-timing calibration for the exact SKU. Spinlock replay/watchdog behavior must be separately checked.
- `gpu_tot_sim_insn` is active-thread instruction count (`shader.cc` accumulation by active count); tracer statistics are warp records. Their denominators differ.

Upstream trace-gating bug is reproduced at tracer_tool.cu 655–658: ACTIVE_FROM_START=0 overwrites active_region at each launch even after the ProfilerStart handler enables it. Native PASS/exit0 can therefore produce empty trace lists. Our scaffold uses ACTIVE_FROM_START=1, starts instrumentation disabled, attaches the official runtime-enable/disable hooks only after prefill, and insists on actual positive trace instructions and grouped files. This hook workaround still requires native H100 validation.

The shipped stock H100 configuration is132SM/SXM-like. A PCIe/NVL or another H100 allocation requires an explicitly derived, hashed SKU configuration; compare observed SM/HBM/L2/clocks/bandwidth and preserve calibration differences. The rental replay smoke intentionally uses stock config only for parser/completion integrity, never as automatic device accuracy.

Only after matching eager H100 kernel/runtime/counter measurements should fixed-trace timing sensitivity be treated as calibrated for those accesses. Fixed traces do not establish changed scheduling correctness, energy or iso-area feasibility. Path A is a smaller possible later strawman; Path B remains conditional on measured access-path and physical costs.
