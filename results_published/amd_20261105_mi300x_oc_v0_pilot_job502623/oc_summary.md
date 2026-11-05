# OC-v0 pilot — mi300x

## Probe (OC0)

- pc_host_trap: rc 0, 2 files
- pc_stochastic: rc 0, 2 files
- att_gemv: rc 0, 44 files
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
- FETCH_SIZE: validated (ratio 1.0000134706497192)
- TCC_EA0_RDREQ_32B: unavailable (ratio None)
- TCC_EA0_RDREQ_64B: unavailable (ratio None)
- TCC_EA0_RDREQ_128B: unavailable (ratio None)
- TCC_EA0_RDREQ_DRAM: validated (ratio {32: 0.25000688433647156, 64: 0.5000137686729431, 128: 1.0000275373458862})
- TCC_EA0_RDREQ: validated (ratio {32: 0.25000691413879395, 64: 0.5000138282775879, 128: 1.0000276565551758})
- TCC_EA0_RDREQ_LEVEL: validated (ratio None)
- SQ_ACTIVE_INST_VALU: instruction count (not a time-domain counter); w4/w1 [2.366008057787504, 1.3996494430365585]; ilp8/ilp1 None
- SQ_ACTIVE_INST_VMEM: unidentified; w4/w1 None; ilp8/ilp1 None
- SQ_ACTIVE_INST_ANY: ambiguous; w4/w1 [2.3438293802982457, 1.4315847413311311]; ilp8/ilp1 None

## OC4 candidate A: **unavailable**

```
{
 "i_valu_any_active": false,
 "i_vmem_ilp8_over_ilp1": null,
 "i_vmem_any_active": false,
 "ii_mfma4_vmem_frac": null,
 "ii_read_valu_frac": 0.021211595770018854,
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
 "iv_fused_X_over_T": 0.23410464873879508,
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
| gemv | 36 | se0_cu1 | 47084 | 0.043 | 0.954 | 0.043 | 0.0 |
| fused | 2 | se0_cu1 | 134316 | 0.234 | 0.97 | 0.234 | 0.0 |
| mfma4 | 1 | se0_cu1 | 285976 | 0.944 | 0.0 | 0.0 | None |
| read | 2 | se0_cu1 | 130168 | 0.0 | 0.971 | 0.0 | 0.0 |
| split | 2 | se0_cu1 | 2307896 | 0.983 | 0.0 | 0.0 | None |

## References

| cell | mode | median ms | GB/s | TFLOP/s | cv |
|---|---|---:|---:|---:|---:|
| fused_n128_g128 | fused | 2.863651 | 3749.555 | 59.993 | 0.0096 |
| fused_n304_g128 | fused | 2.909925 | 3689.93 | 59.039 | 0.0093 |
| fused_n304_g304 | fused | 2.641913 | 4064.259 | 65.028 | 0.0345 |
| ilp1_n128_g128 | read_ilp | 5.818735 | 3690.637 | None | 0.0047 |
| ilp1_n304_g128 | read_ilp | 5.877033 | 3654.027 | None | 0.0043 |
| ilp1_n304_g304 | read_ilp | 4.757852 | 4513.557 | None | 0.0014 |
| ilp2_n128_g128 | read_ilp | 5.672407 | 3785.842 | None | 0.0059 |
| ilp2_n304_g128 | read_ilp | 5.691547 | 3773.111 | None | 0.0075 |
| ilp2_n304_g304 | read_ilp | 4.964198 | 4325.943 | None | 0.0038 |
| ilp4_n128_g128 | read_ilp | 5.944807 | 3612.369 | None | 0.0049 |
| ilp4_n304_g128 | read_ilp | 5.984964 | 3588.131 | None | 0.0052 |
| ilp4_n304_g304 | read_ilp | 5.151204 | 4168.896 | None | 0.0011 |
| ilp8_n128_g128 | read_ilp | 5.696052 | 3770.126 | None | 0.006 |
| ilp8_n304_g128 | read_ilp | 5.703738 | 3765.046 | None | 0.0061 |
| ilp8_n304_g304 | read_ilp | 5.09888 | 4211.677 | None | 0.0016 |
| mfma4_n128_g128 | mfma4 | 22.092133 | None | 248.847 | 0.0003 |
| mfma4_n304_g128 | mfma4 | 22.069836 | None | 249.098 | 0.0003 |
| mfma4_n304_g304 | mfma4 | 28.718491 | None | 454.644 | 0.0026 |
| mfma4lds_n128_g128 | mfma4_lds | 15.684334 | None | 175.256 | 0.003 |
| mfma4lds_n304_g128 | mfma4_lds | 15.779474 | None | 174.2 | 0.0034 |
| mfma4lds_n304_g304 | mfma4_lds | 21.983337 | None | 296.968 | 0.0032 |
| read_mall_n128_g128 | read | 3.497862 | 6139.418 | None | 0.0008 |
| read_mall_n304_g128 | read | 3.498143 | 6138.925 | None | 0.0013 |
| read_mall_n304_g304 | read | 3.687406 | 5823.833 | None | 0.0044 |
| read_n128_g128 | read | 5.251833 | 4089.017 | None | 0.0052 |
| read_n128_g8x | read | 6.519418 | 3293.981 | None | 0.0007 |
| read_n304_g128 | read | 5.342853 | 4019.357 | None | 0.0035 |
| read_n304_g304 | read | 4.800365 | 4473.584 | None | 0.0014 |
| read_n304_g8x | read | 4.83224 | 4444.075 | None | 0.0054 |
| split_n128_g128 | split | 43.743168 | 245.465 | 125.678 | 0.0001 |
| split_n304_g128 | split | 23.474979 | 457.398 | 234.188 | 0.0023 |
| split_n304_g304 | split | 47.603916 | 225.557 | 274.278 | 0.0023 |

## Kernel controls (median µs)

| cell | model | shape | route | history | grid | µs |
|---|---|---|---|---|---:|---:|
| kernels_n304_g304 | None | tiny | wvSplitK | same | 304 | 6.24 |
| kernels_n304_g304 | qwen32b | qkv | wvSplitK | stream | 304 | 18.13 |
| kernels_n304_g304 | qwen32b | qkv | wvSplitK | same | 304 | 14.21 |
| kernels_n304_g304 | qwen32b | qkv | linear | stream | None | 535.01 |
| kernels_n304_g304 | qwen32b | o_proj | wvSplitK | stream | 304 | 14.86 |
| kernels_n304_g304 | qwen32b | o_proj | wvSplitK | same | 304 | 13.67 |
| kernels_n304_g304 | qwen32b | o_proj | linear | stream | None | 396.04 |
| kernels_n304_g304 | qwen32b | gate_up | wvSplitK | stream | 304 | 137.3 |
| kernels_n304_g304 | qwen32b | gate_up | wvSplitK | same | 304 | 135.57 |
| kernels_n304_g304 | qwen32b | gate_up | linear | stream | None | 4172.48 |
| kernels_n304_g304 | qwen32b | down | wvSplitK | stream | 304 | 72.8 |
| kernels_n304_g304 | qwen32b | down | wvSplitK | same | 304 | 74.0 |
| kernels_n304_g304 | qwen32b | down | linear | stream | None | 2024.17 |
| kernels_n304_g304 | qwen32b | lm_head | wvSplitK | stream | 304 | 380.24 |
| kernels_n304_g304 | qwen32b | lm_head | wvSplitK | same | 304 | 374.36 |
| kernels_n304_g304 | qwen32b | lm_head | linear | stream | None | 11953.81 |
| kernels_n304_g304 | qwen1p5b | qkv | wvSplitK | stream | 304 | 6.63 |
| kernels_n304_g304 | qwen1p5b | qkv | wvSplitK | same | 304 | 6.63 |
| kernels_n304_g304 | qwen1p5b | qkv | linear | stream | None | 64.83 |
| kernels_n304_g304 | qwen1p5b | o_proj | wvSplitK | stream | 304 | 6.92 |
| kernels_n304_g304 | qwen1p5b | o_proj | wvSplitK | same | 304 | 6.79 |
| kernels_n304_g304 | qwen1p5b | o_proj | linear | stream | None | 57.66 |
| kernels_n304_g304 | qwen1p5b | gate_up | wvSplitK | stream | 304 | 16.02 |
| kernels_n304_g304 | qwen1p5b | gate_up | wvSplitK | same | 304 | 14.53 |
| kernels_n304_g304 | qwen1p5b | gate_up | linear | stream | None | 457.21 |
| kernels_n304_g304 | qwen1p5b | down | wvSplitK | stream | 304 | 11.11 |
| kernels_n304_g304 | qwen1p5b | down | wvSplitK | same | 304 | 10.62 |
| kernels_n304_g304 | qwen1p5b | down | linear | stream | None | 221.4 |
| kernels_n304_g304 | qwen1p5b | lm_head | wvSplitK | stream | 304 | 122.69 |
| kernels_n304_g304 | qwen1p5b | lm_head | wvSplitK | same | 304 | 120.66 |
| kernels_n304_g304 | qwen1p5b | lm_head | linear | stream | None | 3593.27 |
| kernels_n128_g128 | None | tiny | wvSplitK | same | 304 | 5.72 |
| kernels_n128_g128 | None | tiny | wvSplitK | same | 128 | 5.67 |
| kernels_n128_g128 | qwen32b | qkv | wvSplitK | stream | 304 | 22.45 |
| kernels_n128_g128 | qwen32b | qkv | wvSplitK | same | 304 | 17.86 |
| kernels_n128_g128 | qwen32b | qkv | wvSplitK | stream | 128 | 17.98 |
| kernels_n128_g128 | qwen32b | qkv | wvSplitK | same | 128 | 16.86 |
| kernels_n128_g128 | qwen32b | o_proj | wvSplitK | stream | 304 | 17.86 |
| kernels_n128_g128 | qwen32b | o_proj | wvSplitK | same | 304 | 16.71 |
| kernels_n128_g128 | qwen32b | o_proj | wvSplitK | stream | 128 | 15.73 |
| kernels_n128_g128 | qwen32b | o_proj | wvSplitK | same | 128 | 15.03 |
| kernels_n128_g128 | qwen32b | gate_up | wvSplitK | stream | 304 | 144.89 |
| kernels_n128_g128 | qwen32b | gate_up | wvSplitK | same | 304 | 139.15 |
| kernels_n128_g128 | qwen32b | gate_up | wvSplitK | stream | 128 | 136.24 |
| kernels_n128_g128 | qwen32b | gate_up | wvSplitK | same | 128 | 134.91 |
| kernels_n128_g128 | qwen32b | down | wvSplitK | stream | 304 | 82.4 |
| kernels_n128_g128 | qwen32b | down | wvSplitK | same | 304 | 81.82 |
| kernels_n128_g128 | qwen32b | down | wvSplitK | stream | 128 | 66.28 |
| kernels_n128_g128 | qwen32b | down | wvSplitK | same | 128 | 67.93 |
| kernels_n128_g128 | qwen32b | lm_head | wvSplitK | stream | 304 | 388.01 |
| kernels_n128_g128 | qwen32b | lm_head | wvSplitK | same | 304 | 382.08 |
| kernels_n128_g128 | qwen32b | lm_head | wvSplitK | stream | 128 | 371.15 |
| kernels_n128_g128 | qwen32b | lm_head | wvSplitK | same | 128 | 369.97 |
| kernels_n128_g128 | qwen1p5b | qkv | wvSplitK | stream | 304 | 5.98 |
| kernels_n128_g128 | qwen1p5b | qkv | wvSplitK | same | 304 | 5.89 |
| kernels_n128_g128 | qwen1p5b | qkv | wvSplitK | stream | 128 | 5.94 |
| kernels_n128_g128 | qwen1p5b | qkv | wvSplitK | same | 128 | 5.83 |
| kernels_n128_g128 | qwen1p5b | o_proj | wvSplitK | stream | 304 | 6.11 |
| kernels_n128_g128 | qwen1p5b | o_proj | wvSplitK | same | 304 | 5.97 |
| kernels_n128_g128 | qwen1p5b | o_proj | wvSplitK | stream | 128 | 6.0 |
| kernels_n128_g128 | qwen1p5b | o_proj | wvSplitK | same | 128 | 5.9 |
| kernels_n128_g128 | qwen1p5b | gate_up | wvSplitK | stream | 304 | 19.6 |
| kernels_n128_g128 | qwen1p5b | gate_up | wvSplitK | same | 304 | 18.02 |
| kernels_n128_g128 | qwen1p5b | gate_up | wvSplitK | stream | 128 | 17.96 |
| kernels_n128_g128 | qwen1p5b | gate_up | wvSplitK | same | 128 | 16.86 |
| kernels_n128_g128 | qwen1p5b | down | wvSplitK | stream | 304 | 13.91 |
| kernels_n128_g128 | qwen1p5b | down | wvSplitK | same | 304 | 13.09 |
| kernels_n128_g128 | qwen1p5b | down | wvSplitK | stream | 128 | 10.82 |
| kernels_n128_g128 | qwen1p5b | down | wvSplitK | same | 128 | 10.46 |
| kernels_n128_g128 | qwen1p5b | lm_head | wvSplitK | stream | 304 | 134.02 |
| kernels_n128_g128 | qwen1p5b | lm_head | wvSplitK | same | 304 | 132.26 |
| kernels_n128_g128 | qwen1p5b | lm_head | wvSplitK | stream | 128 | 125.29 |
| kernels_n128_g128 | qwen1p5b | lm_head | wvSplitK | same | 128 | 117.57 |
| kernels_n304_g128 | None | tiny | wvSplitK | same | 304 | 6.3 |
| kernels_n304_g128 | None | tiny | wvSplitK | same | 128 | 6.61 |
| kernels_n304_g128 | qwen32b | qkv | wvSplitK | stream | 304 | 18.09 |
| kernels_n304_g128 | qwen32b | qkv | wvSplitK | same | 304 | 13.88 |
| kernels_n304_g128 | qwen32b | qkv | wvSplitK | stream | 128 | 18.57 |
| kernels_n304_g128 | qwen32b | qkv | wvSplitK | same | 128 | 16.92 |
| kernels_n304_g128 | qwen32b | o_proj | wvSplitK | stream | 304 | 15.11 |
| kernels_n304_g128 | qwen32b | o_proj | wvSplitK | same | 304 | 13.48 |
| kernels_n304_g128 | qwen32b | o_proj | wvSplitK | stream | 128 | 15.87 |
| kernels_n304_g128 | qwen32b | o_proj | wvSplitK | same | 128 | 15.0 |
| kernels_n304_g128 | qwen32b | gate_up | wvSplitK | stream | 304 | 139.07 |
| kernels_n304_g128 | qwen32b | gate_up | wvSplitK | same | 304 | 135.67 |
| kernels_n304_g128 | qwen32b | gate_up | wvSplitK | stream | 128 | 136.42 |
| kernels_n304_g128 | qwen32b | gate_up | wvSplitK | same | 128 | 135.07 |
| kernels_n304_g128 | qwen32b | down | wvSplitK | stream | 304 | 70.96 |
| kernels_n304_g128 | qwen32b | down | wvSplitK | same | 304 | 73.01 |
| kernels_n304_g128 | qwen32b | down | wvSplitK | stream | 128 | 69.55 |
| kernels_n304_g128 | qwen32b | down | wvSplitK | same | 128 | 68.31 |
| kernels_n304_g128 | qwen32b | lm_head | wvSplitK | stream | 304 | 375.78 |
| kernels_n304_g128 | qwen32b | lm_head | wvSplitK | same | 304 | 370.11 |
| kernels_n304_g128 | qwen32b | lm_head | wvSplitK | stream | 128 | 372.03 |
| kernels_n304_g128 | qwen32b | lm_head | wvSplitK | same | 128 | 370.43 |
| kernels_n304_g128 | qwen1p5b | qkv | wvSplitK | stream | 304 | 6.71 |
| kernels_n304_g128 | qwen1p5b | qkv | wvSplitK | same | 304 | 6.73 |
| kernels_n304_g128 | qwen1p5b | qkv | wvSplitK | stream | 128 | 6.88 |
| kernels_n304_g128 | qwen1p5b | qkv | wvSplitK | same | 128 | 6.91 |
| kernels_n304_g128 | qwen1p5b | o_proj | wvSplitK | stream | 304 | 6.96 |
| kernels_n304_g128 | qwen1p5b | o_proj | wvSplitK | same | 304 | 6.94 |
| kernels_n304_g128 | qwen1p5b | o_proj | wvSplitK | stream | 128 | 7.06 |
| kernels_n304_g128 | qwen1p5b | o_proj | wvSplitK | same | 128 | 7.02 |
| kernels_n304_g128 | qwen1p5b | gate_up | wvSplitK | stream | 304 | 17.08 |
| kernels_n304_g128 | qwen1p5b | gate_up | wvSplitK | same | 304 | 14.17 |
| kernels_n304_g128 | qwen1p5b | gate_up | wvSplitK | stream | 128 | 18.33 |
| kernels_n304_g128 | qwen1p5b | gate_up | wvSplitK | same | 128 | 16.89 |
| kernels_n304_g128 | qwen1p5b | down | wvSplitK | stream | 304 | 11.06 |
| kernels_n304_g128 | qwen1p5b | down | wvSplitK | same | 304 | 10.6 |
| kernels_n304_g128 | qwen1p5b | down | wvSplitK | stream | 128 | 10.89 |
| kernels_n304_g128 | qwen1p5b | down | wvSplitK | same | 128 | 10.5 |
| kernels_n304_g128 | qwen1p5b | lm_head | wvSplitK | stream | 304 | 122.51 |
| kernels_n304_g128 | qwen1p5b | lm_head | wvSplitK | same | 304 | 120.21 |
| kernels_n304_g128 | qwen1p5b | lm_head | wvSplitK | stream | 128 | 126.95 |
| kernels_n304_g128 | qwen1p5b | lm_head | wvSplitK | same | 128 | 125.55 |
