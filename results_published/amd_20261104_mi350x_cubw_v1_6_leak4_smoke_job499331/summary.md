# cubw_v1 pilot — mi350x

## Gates

- placement_ok: **True**
- lifecycle_ok: **True**
- outputs_ok: **True**
- leak_ok: **False**
- leak_ok_v11: **False**
- leak_ok_v2: **False**
- leak_ok_v3: **False**
- leak_ok_v4: **True**
- clock_ok: **False**
- release_e2: **True**
- release_e3_minimal: **False**
- release_e3_minimal_v2: **False**
- release_e3_minimal_v3: **False**
- release_e3_minimal_v4: **True**
- release_e3_full: **False**
- release_e5e8: **True**
- outputs: {'runs': 4, 'complete': 4, 'distinct_digests': 1}
- leak ratios (probe during decode / probe alone): {}
- leak v1.1 (relative to placement-proven native co-runners): {}
- pooled clocks (Qwen32B E2): {'pooled_ok': False, 'gfx_min': None, 'gfx_max': None, 'gfx_ratio': None, 'mem_min': None, 'mem_max': None, 'samples': 0, 'coverage_min': None}

## Placement

| probe | distinct CUs | expected | per-XCD | balanced | rc |
|---|---:|---:|---|---|---:|
| L0_n256 | 256 | 256 | {'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32} | True | 0 |
| L0_n200 | 200 | 200 | {'0': 25, '1': 25, '2': 25, '3': 25, '4': 25, '5': 25, '6': 25, '7': 25} | True | 0 |
| L0_n192 | 192 | 192 | {'0': 24, '1': 24, '2': 24, '3': 24, '4': 24, '5': 24, '6': 24, '7': 24} | True | 0 |
| L0_n136 | 136 | 136 | {'0': 17, '1': 17, '2': 17, '3': 17, '4': 17, '5': 17, '6': 17, '7': 17} | True | 0 |
| L0_n128 | 128 | 128 | {'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16} | True | 0 |
| L0_competitor_192-255 | 64 | 64 | {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8} | True | 0 |

## E2 latency ratio R(n) = T(n)/T(304)

## E3 T(n, z)

## E4 service (mean over acquisitions; ratio to 304)


## E5 / timeline


## Leak test v4 (per-CU start-delay probe)

- qwen32b_graph_b1_c4096__n192__gapprobe_L0: {'expected_cus': 64, 'alone_cus': 64, 'flag_excess': 0.1, 'alone_late_max': 0.0, 'launches': [96150, 97183, 97083, 94219, 22558, 23355], 'native_flags': [0, 0], 'native_max_excess': [0.00031898583085519073, 0.00046352090479280614], 'positive_flags': [8, 8], 'positive_top8_excess': [[0.7349897579044566, 0.7407635402625797, 0.7430030036404547, 0.7475562253897834, 0.7489678302677804, 0.7538182319914243, 0.7539031405555143, 0.7542321612413632], [0.4534533203298165, 0.45779767709903363, 0.4599255253125277, 0.4651121553329196, 0.46661938115081125, 0.4667523716641546, 0.47388952921358274, 0.48825250465466796]], 'decoder_flags': [0], 'decoder_max_excess': [0.0025262256476129307], 'info_positive_flags': {'mem100': 8, 'arith100': 8}, 'sensitive': True, 'ok': True}
- qwen32b_graph_b1_c4096__n128__gapprobe_L0: {'expected_cus': 128, 'alone_cus': 128, 'flag_excess': 0.1, 'alone_late_max': 0.0, 'launches': [96476, 96411, 95949, 93421, 21084, 22125], 'native_flags': [0, 0], 'native_max_excess': [0.00026967877109458464, 0.0002188662727073758], 'positive_flags': [8, 8], 'positive_top8_excess': [[0.6287772556491581, 0.6378009227047452, 0.6445874054013552, 0.6447051519465645, 0.6496398026139732, 0.6629772749167746, 0.6677834748075915, 0.6683507990708727], [0.48795295010434453, 0.48837981407702524, 0.48956554733447166, 0.4977233921457029, 0.4986245494213622, 0.5053595143236578, 0.5103395940049327, 0.5205369000189717]], 'decoder_flags': [0], 'decoder_max_excess': [0.0018983050847457628], 'info_positive_flags': {'mem100': 8, 'arith100': 8}, 'sensitive': True, 'ok': True}

## E8 guardrails

