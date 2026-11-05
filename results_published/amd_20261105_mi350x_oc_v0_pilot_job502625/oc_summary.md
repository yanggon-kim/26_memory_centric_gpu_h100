# OC-v0 pilot — mi350x

## Probe (OC0)

- pc_host_trap: rc 0, 2 files
- pc_stochastic: rc 0, 2 files
- att_gemv: rc 0, 45 files
- att_native_split: rc 0, 61 files
- att_native_fused: rc 0, 45 files
- att_native_mfma4: rc 0, 30 files
- att_native_read: rc 0, 45 files

## Counter validation

- SQ_INSTS_VMEM_RD: validated (ratio 1.0)
- SQ_INSTS_MFMA_w1: validated (ratio 1.0)
- SQ_INSTS_VALU_MFMA_MOPS_BF16_w1: measured-unvalidated (ratio None)
- SQ_VALU_MFMA_BUSY_CYCLES_w1: measured-unvalidated (ratio None)
- SQ_INSTS_VALU_w1: invalid (ratio 0.500518798828125)
  - ISA-amended expectation: validated (ratio 1.00103759765625)
- SQ_INSTS_MFMA_w4: validated (ratio 1.0)
- SQ_INSTS_VALU_MFMA_MOPS_BF16_w4: measured-unvalidated (ratio None)
- SQ_VALU_MFMA_BUSY_CYCLES_w4: measured-unvalidated (ratio None)
- SQ_INSTS_VALU_w4: invalid (ratio 0.500518798828125)
  - ISA-amended expectation: validated (ratio 1.00103759765625)
- FETCH_SIZE: invalid (ratio 0.5000134706497192)
- TCC_EA0_RDREQ_32B: unavailable (ratio None)
- TCC_EA0_RDREQ_64B: measured-unvalidated (ratio {32: 6.854534149169922e-06, 64: 1.3709068298339844e-05, 128: 2.7418136596679688e-05})
- TCC_EA0_RDREQ_128B: validated (ratio {32: 0.25, 64: 0.5, 128: 1.0})
- TCC_EA0_RDREQ_DRAM: validated (ratio {32: 0.25000685453414917, 64: 0.5000137090682983, 128: 1.0000274181365967})
- TCC_EA0_RDREQ: validated (ratio {32: 0.25000667572021484, 64: 0.5000133514404297, 128: 1.0000267028808594})
- TCC_EA0_RDREQ_LEVEL: validated (ratio None)
- SQ_ACTIVE_INST_VALU: instruction count (not a time-domain counter); w4/w1 [2.3625185316830914, 1.4004089829872717]; ilp8/ilp1 None
- SQ_ACTIVE_INST_VMEM: unidentified; w4/w1 None; ilp8/ilp1 None
- SQ_ACTIVE_INST_ANY: ambiguous; w4/w1 [2.315730775089132, 1.4382096613967934]; ilp8/ilp1 None

## OC4 candidate A: **unavailable**

```
{
 "i_valu_any_active": false,
 "i_vmem_ilp8_over_ilp1": null,
 "i_vmem_any_active": false,
 "ii_mfma4_vmem_frac": null,
 "ii_read_valu_frac": 0.044989968806886826,
 "ii": false,
 "iii_split_lower": null,
 "iv_fused_lower": null,
 "iii": false,
 "iv": false,
 "fractions_in_unit_interval": true
}
```

## OC4 candidate B (ATT): **bounded (per-CU scope)**

```
{
 "ii_mfma4_A_M_over_T": 0.0,
 "ii_read_A_C_over_T": 0.0,
 "iii_split_X_over_T": 0.0,
 "iv_fused_X_over_T": 0.471738638932851,
 "iii_split_traced_cu_ran_both_types": false,
 "gemv_decoded": true,
 "ii": true,
 "iii": true,
 "iv": true
}
```

Caveat: split check is degenerate: the traced CU ran one workgroup type only (X = 0 trivially); a split trace with both types on the traced CU is added to the development job

| route | dispatch | CU | T cycles | A_C/T | A_M/T | X/T | loads w/o stall |
|---|---|---|---:|---:|---:|---:|---:|
| gemv | 37 | se0_cu1 | 23436 | 0.131 | 0.907 | 0.131 | 0.0 |
| fused | 2 | se0_cu1 | 70768 | 0.472 | 0.948 | 0.472 | 0.0 |
| mfma4 | 1 | se0_cu1 | 286188 | 0.952 | 0.0 | 0.0 | None |
| read | 2 | se0_cu1 | 78164 | 0.0 | 0.953 | 0.0 | 0.0 |
| split | 2 | se0_cu1 | 2265192 | 0.998 | 0.0 | 0.0 | None |

## References

| cell | mode | median ms | GB/s | TFLOP/s | cv |
|---|---|---:|---:|---:|---:|
| fused_n192_g192 | fused | 2.044859 | 5250.933 | 84.015 | 0.013 |
| fused_n256_g192 | fused | 2.03512 | 5276.061 | 84.417 | 0.0156 |
| fused_n256_g256 | fused | 1.727056 | 6217.18 | 99.475 | 0.018 |
| ilp1_n192_g192 | read_ilp | 4.288764 | 5007.232 | None | 0.0042 |
| ilp1_n256_g192 | read_ilp | 4.251755 | 5050.817 | None | 0.0046 |
| ilp1_n256_g256 | read_ilp | 3.460972 | 6204.857 | None | 0.0048 |
| ilp2_n192_g192 | read_ilp | 3.37663 | 6359.843 | None | 0.0027 |
| ilp2_n256_g192 | read_ilp | 3.383146 | 6347.594 | None | 0.0028 |
| ilp2_n256_g256 | read_ilp | 3.291427 | 6524.476 | None | 0.0064 |
| ilp4_n192_g192 | read_ilp | 3.885391 | 5527.072 | None | 0.0016 |
| ilp4_n256_g192 | read_ilp | 3.887071 | 5524.684 | None | 0.0017 |
| ilp4_n256_g256 | read_ilp | 3.648952 | 5885.207 | None | 0.0025 |
| ilp8_n192_g192 | read_ilp | 3.593829 | 5975.475 | None | 0.0016 |
| ilp8_n256_g192 | read_ilp | 3.585226 | 5989.814 | None | 0.002 |
| ilp8_n256_g256 | read_ilp | 3.611432 | 5946.349 | None | 0.0012 |
| mfma4_n192_g192 | mfma4 | 20.892584 | None | 394.702 | 0.0006 |
| mfma4_n256_g192 | mfma4 | 20.961758 | None | 393.399 | 0.0008 |
| mfma4_n256_g256 | mfma4 | 21.03326 | None | 522.749 | 0.0008 |
| mfma4lds_n192_g192 | mfma4_lds | 13.994624 | None | 294.625 | 0.0008 |
| mfma4lds_n256_g192 | mfma4_lds | 14.001935 | None | 294.471 | 0.0007 |
| mfma4lds_n256_g256 | mfma4_lds | 15.317509 | None | 358.907 | 0.0066 |
| read_mall_n192_g192 | read | 2.835464 | 7573.659 | None | 0.001 |
| read_mall_n256_g192 | read | 2.830654 | 7586.528 | None | 0.0013 |
| read_mall_n256_g256 | read | 2.862025 | 7503.371 | None | 0.0008 |
| read_n192_g192 | read | 3.942933 | 5446.412 | None | 0.0027 |
| read_n192_g8x | read | 3.261737 | 6583.865 | None | 0.0035 |
| read_n256_g192 | read | 3.945506 | 5442.86 | None | 0.0028 |
| read_n256_g256 | read | 3.32375 | 6461.026 | None | 0.0026 |
| read_n256_g8x | read | 3.299146 | 6509.211 | None | 0.0038 |
| split_n192_g192 | split | 42.707417 | 251.418 | 193.089 | 0.0007 |
| split_n256_g192 | split | 43.03056 | 249.53 | 191.639 | 0.0013 |
| split_n256_g256 | split | 42.894066 | 250.324 | 256.332 | 0.002 |

## Kernel controls (median µs)

| cell | model | shape | route | history | grid | µs |
|---|---|---|---|---|---:|---:|
| kernels_n192_g192 | None | tiny | wvSplitK | same | 256 | 5.82 |
| kernels_n192_g192 | None | tiny | wvSplitK | same | 192 | 5.58 |
| kernels_n192_g192 | qwen32b | qkv | wvSplitK | stream | 256 | 17.25 |
| kernels_n192_g192 | qwen32b | qkv | wvSplitK | same | 256 | 15.44 |
| kernels_n192_g192 | qwen32b | qkv | wvSplitK | stream | 192 | 13.25 |
| kernels_n192_g192 | qwen32b | qkv | wvSplitK | same | 192 | 12.51 |
| kernels_n192_g192 | qwen32b | o_proj | wvSplitK | stream | 256 | 13.29 |
| kernels_n192_g192 | qwen32b | o_proj | wvSplitK | same | 256 | 12.52 |
| kernels_n192_g192 | qwen32b | o_proj | wvSplitK | stream | 192 | 10.25 |
| kernels_n192_g192 | qwen32b | o_proj | wvSplitK | same | 192 | 9.67 |
| kernels_n192_g192 | qwen32b | gate_up | wvSplitK | stream | 256 | 109.78 |
| kernels_n192_g192 | qwen32b | gate_up | wvSplitK | same | 256 | 103.17 |
| kernels_n192_g192 | qwen32b | gate_up | wvSplitK | stream | 192 | 84.23 |
| kernels_n192_g192 | qwen32b | gate_up | wvSplitK | same | 192 | 83.75 |
| kernels_n192_g192 | qwen32b | down | wvSplitK | stream | 256 | 61.05 |
| kernels_n192_g192 | qwen32b | down | wvSplitK | same | 256 | 60.31 |
| kernels_n192_g192 | qwen32b | down | wvSplitK | stream | 192 | 48.9 |
| kernels_n192_g192 | qwen32b | down | wvSplitK | same | 192 | 43.0 |
| kernels_n192_g192 | qwen32b | lm_head | wvSplitK | stream | 256 | 283.44 |
| kernels_n192_g192 | qwen32b | lm_head | wvSplitK | same | 256 | 283.01 |
| kernels_n192_g192 | qwen32b | lm_head | wvSplitK | stream | 192 | 229.59 |
| kernels_n192_g192 | qwen32b | lm_head | wvSplitK | same | 192 | 229.03 |
| kernels_n192_g192 | qwen1p5b | qkv | wvSplitK | stream | 256 | 5.46 |
| kernels_n192_g192 | qwen1p5b | qkv | wvSplitK | same | 256 | 5.56 |
| kernels_n192_g192 | qwen1p5b | qkv | wvSplitK | stream | 192 | 5.21 |
| kernels_n192_g192 | qwen1p5b | qkv | wvSplitK | same | 192 | 5.07 |
| kernels_n192_g192 | qwen1p5b | o_proj | wvSplitK | stream | 256 | 5.5 |
| kernels_n192_g192 | qwen1p5b | o_proj | wvSplitK | same | 256 | 4.94 |
| kernels_n192_g192 | qwen1p5b | o_proj | wvSplitK | stream | 192 | 4.84 |
| kernels_n192_g192 | qwen1p5b | o_proj | wvSplitK | same | 192 | 6.16 |
| kernels_n192_g192 | qwen1p5b | gate_up | wvSplitK | stream | 256 | 15.48 |
| kernels_n192_g192 | qwen1p5b | gate_up | wvSplitK | same | 256 | 14.42 |
| kernels_n192_g192 | qwen1p5b | gate_up | wvSplitK | stream | 192 | 12.17 |
| kernels_n192_g192 | qwen1p5b | gate_up | wvSplitK | same | 192 | 11.76 |
| kernels_n192_g192 | qwen1p5b | down | wvSplitK | stream | 256 | 10.64 |
| kernels_n192_g192 | qwen1p5b | down | wvSplitK | same | 256 | 10.14 |
| kernels_n192_g192 | qwen1p5b | down | wvSplitK | stream | 192 | 10.68 |
| kernels_n192_g192 | qwen1p5b | down | wvSplitK | same | 192 | 10.16 |
| kernels_n192_g192 | qwen1p5b | lm_head | wvSplitK | stream | 256 | 95.71 |
| kernels_n192_g192 | qwen1p5b | lm_head | wvSplitK | same | 256 | 94.65 |
| kernels_n192_g192 | qwen1p5b | lm_head | wvSplitK | stream | 192 | 78.21 |
| kernels_n192_g192 | qwen1p5b | lm_head | wvSplitK | same | 192 | 77.32 |
| kernels_n256_g192 | None | tiny | wvSplitK | same | 256 | 5.32 |
| kernels_n256_g192 | None | tiny | wvSplitK | same | 192 | 5.24 |
| kernels_n256_g192 | qwen32b | qkv | wvSplitK | stream | 256 | 13.62 |
| kernels_n256_g192 | qwen32b | qkv | wvSplitK | same | 256 | 11.33 |
| kernels_n256_g192 | qwen32b | qkv | wvSplitK | stream | 192 | 13.24 |
| kernels_n256_g192 | qwen32b | qkv | wvSplitK | same | 192 | 12.53 |
| kernels_n256_g192 | qwen32b | o_proj | wvSplitK | stream | 256 | 10.35 |
| kernels_n256_g192 | qwen32b | o_proj | wvSplitK | same | 256 | 9.57 |
| kernels_n256_g192 | qwen32b | o_proj | wvSplitK | stream | 192 | 10.35 |
| kernels_n256_g192 | qwen32b | o_proj | wvSplitK | same | 192 | 9.74 |
| kernels_n256_g192 | qwen32b | gate_up | wvSplitK | stream | 256 | 89.74 |
| kernels_n256_g192 | qwen32b | gate_up | wvSplitK | same | 256 | 83.45 |
| kernels_n256_g192 | qwen32b | gate_up | wvSplitK | stream | 192 | 84.32 |
| kernels_n256_g192 | qwen32b | gate_up | wvSplitK | same | 192 | 83.75 |
| kernels_n256_g192 | qwen32b | down | wvSplitK | stream | 256 | 46.58 |
| kernels_n256_g192 | qwen32b | down | wvSplitK | same | 256 | 45.58 |
| kernels_n256_g192 | qwen32b | down | wvSplitK | stream | 192 | 48.94 |
| kernels_n256_g192 | qwen32b | down | wvSplitK | same | 192 | 42.9 |
| kernels_n256_g192 | qwen32b | lm_head | wvSplitK | stream | 256 | 229.65 |
| kernels_n256_g192 | qwen32b | lm_head | wvSplitK | same | 256 | 227.33 |
| kernels_n256_g192 | qwen32b | lm_head | wvSplitK | stream | 192 | 229.93 |
| kernels_n256_g192 | qwen32b | lm_head | wvSplitK | same | 192 | 229.17 |
| kernels_n256_g192 | qwen1p5b | qkv | wvSplitK | stream | 256 | 5.68 |
| kernels_n256_g192 | qwen1p5b | qkv | wvSplitK | same | 256 | 4.8 |
| kernels_n256_g192 | qwen1p5b | qkv | wvSplitK | stream | 192 | 4.78 |
| kernels_n256_g192 | qwen1p5b | qkv | wvSplitK | same | 192 | 4.81 |
| kernels_n256_g192 | qwen1p5b | o_proj | wvSplitK | stream | 256 | 5.13 |
| kernels_n256_g192 | qwen1p5b | o_proj | wvSplitK | same | 256 | 5.22 |
| kernels_n256_g192 | qwen1p5b | o_proj | wvSplitK | stream | 192 | 4.76 |
| kernels_n256_g192 | qwen1p5b | o_proj | wvSplitK | same | 192 | 5.48 |
| kernels_n256_g192 | qwen1p5b | gate_up | wvSplitK | stream | 256 | 12.08 |
| kernels_n256_g192 | qwen1p5b | gate_up | wvSplitK | same | 256 | 10.97 |
| kernels_n256_g192 | qwen1p5b | gate_up | wvSplitK | stream | 192 | 12.18 |
| kernels_n256_g192 | qwen1p5b | gate_up | wvSplitK | same | 192 | 11.78 |
| kernels_n256_g192 | qwen1p5b | down | wvSplitK | stream | 256 | 10.67 |
| kernels_n256_g192 | qwen1p5b | down | wvSplitK | same | 256 | 10.19 |
| kernels_n256_g192 | qwen1p5b | down | wvSplitK | stream | 192 | 10.69 |
| kernels_n256_g192 | qwen1p5b | down | wvSplitK | same | 192 | 10.14 |
| kernels_n256_g192 | qwen1p5b | lm_head | wvSplitK | stream | 256 | 74.61 |
| kernels_n256_g192 | qwen1p5b | lm_head | wvSplitK | same | 256 | 72.8 |
| kernels_n256_g192 | qwen1p5b | lm_head | wvSplitK | stream | 192 | 78.57 |
| kernels_n256_g192 | qwen1p5b | lm_head | wvSplitK | same | 192 | 77.39 |
| kernels_n256_g256 | None | tiny | wvSplitK | same | 256 | 5.74 |
| kernels_n256_g256 | qwen32b | qkv | wvSplitK | stream | 256 | 13.5 |
| kernels_n256_g256 | qwen32b | qkv | wvSplitK | same | 256 | 11.26 |
| kernels_n256_g256 | qwen32b | qkv | linear | stream | None | 313.72 |
| kernels_n256_g256 | qwen32b | o_proj | wvSplitK | stream | 256 | 10.28 |
| kernels_n256_g256 | qwen32b | o_proj | wvSplitK | same | 256 | 9.52 |
| kernels_n256_g256 | qwen32b | o_proj | linear | stream | None | 200.19 |
| kernels_n256_g256 | qwen32b | gate_up | wvSplitK | stream | 256 | 89.68 |
| kernels_n256_g256 | qwen32b | gate_up | wvSplitK | same | 256 | 83.46 |
| kernels_n256_g256 | qwen32b | gate_up | linear | stream | None | 2136.56 |
| kernels_n256_g256 | qwen32b | down | wvSplitK | stream | 256 | 46.64 |
| kernels_n256_g256 | qwen32b | down | wvSplitK | same | 256 | 45.7 |
| kernels_n256_g256 | qwen32b | down | linear | stream | None | 987.54 |
| kernels_n256_g256 | qwen32b | lm_head | wvSplitK | stream | 256 | 228.53 |
| kernels_n256_g256 | qwen32b | lm_head | wvSplitK | same | 256 | 227.13 |
| kernels_n256_g256 | qwen32b | lm_head | linear | stream | None | 5963.4 |
| kernels_n256_g256 | qwen1p5b | qkv | wvSplitK | stream | 256 | 5.96 |
| kernels_n256_g256 | qwen1p5b | qkv | wvSplitK | same | 256 | 5.81 |
| kernels_n256_g256 | qwen1p5b | qkv | linear | stream | None | 34.23 |
| kernels_n256_g256 | qwen1p5b | o_proj | wvSplitK | stream | 256 | 6.04 |
| kernels_n256_g256 | qwen1p5b | o_proj | wvSplitK | same | 256 | 5.59 |
| kernels_n256_g256 | qwen1p5b | o_proj | linear | stream | None | 28.97 |
| kernels_n256_g256 | qwen1p5b | gate_up | wvSplitK | stream | 256 | 11.92 |
| kernels_n256_g256 | qwen1p5b | gate_up | wvSplitK | same | 256 | 11.02 |
| kernels_n256_g256 | qwen1p5b | gate_up | linear | stream | None | 249.95 |
| kernels_n256_g256 | qwen1p5b | down | wvSplitK | stream | 256 | 10.71 |
| kernels_n256_g256 | qwen1p5b | down | wvSplitK | same | 256 | 10.23 |
| kernels_n256_g256 | qwen1p5b | down | linear | stream | None | 138.0 |
| kernels_n256_g256 | qwen1p5b | lm_head | wvSplitK | stream | 256 | 74.55 |
| kernels_n256_g256 | qwen1p5b | lm_head | wvSplitK | same | 256 | 72.84 |
| kernels_n256_g256 | qwen1p5b | lm_head | linear | stream | None | 2098.93 |
