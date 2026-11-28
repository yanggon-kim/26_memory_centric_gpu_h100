# Native known-data transport component

Synthetic BF16 blocks only; not model KV or serving.

## Native component CPU review

known_data_transport_functional_review_passed. Six known-data transfer trials per role:3warm/3diagnostic, one process per device. Independently recomputed BF16 payload hashes, paired completion and source/isolation checks only. No model KV injection, serving, clean latency or calibrated bandwidth.
