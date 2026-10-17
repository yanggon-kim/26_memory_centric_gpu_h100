# Operation resource and loaded-SASS evidence

Kernel replay samples retain their exact metric names/units and source/producer identities. Static SASS counts and logical operator FLOPs are separate from dynamic issued work.

| Case | Status | Static SASS rows | Logical operator FLOPs |
|---|---|---:|---:|
| J3_bridge_b1_c1024_qkv_acq00 | complete | 880 | 6291456 |
| J3_bridge_b1_c1024_qkv_acq01 | complete | 880 | 6291456 |
| J3_bridge_b1_c1024_qkv_acq02 | complete | 880 | 6291456 |

No productive/empty-worker counts, padded dynamic tensor work, whole-step traffic or exact DRAM-wait time are inferred from these samples.
