# Operation resource and loaded-SASS evidence

Kernel replay samples retain their exact metric names/units and source/producer identities. Static SASS counts and logical operator FLOPs are separate from dynamic issued work.

| Case | Status | Static SASS rows | Logical operator FLOPs |
|---|---|---:|---:|
| J3_bridge_b1_c1024_attention_combine_acq00_filter_v2 | complete | 920 | NA |
| J3_bridge_b1_c1024_attention_combine_acq01_filter_v2 | complete | 920 | NA |
| J3_bridge_b1_c1024_attention_combine_acq02_filter_v2 | complete | 920 | NA |
| J3_bridge_b1_c1024_attention_forward_acq00 | failed | 0 | NA |
| J3_bridge_b1_c1024_attention_forward_acq00_filter_v2 | complete | 5424 | NA |
| J3_bridge_b1_c1024_attention_forward_acq01 | failed | 0 | NA |
| J3_bridge_b1_c1024_attention_forward_acq01_filter_v2 | complete | 5424 | NA |
| J3_bridge_b1_c1024_attention_forward_acq02 | failed | 0 | NA |
| J3_bridge_b1_c1024_attention_forward_acq02_filter_v2 | complete | 5424 | NA |
| J3_bridge_b1_c1024_down_acq00 | complete | 840 | 27525120 |
| J3_bridge_b1_c1024_down_acq01 | complete | 840 | 27525120 |
| J3_bridge_b1_c1024_down_acq02 | complete | 840 | 27525120 |
| J3_bridge_b1_c1024_gate_up_acq00 | complete | 936 | 55050240 |
| J3_bridge_b1_c1024_gate_up_acq01 | complete | 936 | 55050240 |
| J3_bridge_b1_c1024_gate_up_acq02 | complete | 936 | 55050240 |
| J3_bridge_b1_c1024_qkv_acq00 | complete | 880 | 6291456 |
| J3_bridge_b1_c1024_qkv_acq01 | complete | 880 | 6291456 |
| J3_bridge_b1_c1024_qkv_acq02 | complete | 880 | 6291456 |

No productive/empty-worker counts, padded dynamic tensor work, whole-step traffic or exact DRAM-wait time are inferred from these samples.
