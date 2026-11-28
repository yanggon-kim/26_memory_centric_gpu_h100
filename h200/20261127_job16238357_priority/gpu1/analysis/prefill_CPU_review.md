# Resident prefill comparison: CPU review

Instrumented pretokenized engine submission and prefill service only. Policy/observer/event costs remain included; ordinary TTFT, offered-load SLOs, physical bytes and energy remain separate gates. All rejected observations are retained.

Complete acquisition set: **True**. Instrumented service qualified: **False**. Instrumented submitted first token qualified: **False**. Ordinary TTFT remains unqualified.

| Condition | Acquisitions | Raw median service wall ms | Raw median submitted first token ms |
|---|---|---|---|
| whole | [0, 1, 2] | [110.27206294238567, 109.64460764080286, 110.08384567685425] | [110.90245237573981, 110.26143748313189, 110.731550026685] |
| chunk256 | [0, 1, 2] | [136.78126712329686, 137.65834621153772, 136.96756679564714] | [137.40848447196186, 138.33633111789823, 137.59630266577005] |
| chunk1024 | [0, 1, 2] | [109.85563066788018, 109.5527047291398, 109.86514459364116] | [110.53657648153603, 110.24307645857334, 110.54445337504148] |
| chunk4096 | [0, 1, 2] | [111.06189643032849, 110.46489421278238, 110.36184197291732] | [111.74481711350381, 111.11901188269258, 111.04848003014922] |
| whole_end | [0, 1, 2] | [109.9262700881809, 110.39458797313273, 110.95797503367066] | [110.62654294073582, 111.09780543483794, 111.66728055104613] |

All source/monitor hashes, phase-specific clock windows, partial/error records and any qualified intervals are in `../results/prefill_CPU_review.json`.
