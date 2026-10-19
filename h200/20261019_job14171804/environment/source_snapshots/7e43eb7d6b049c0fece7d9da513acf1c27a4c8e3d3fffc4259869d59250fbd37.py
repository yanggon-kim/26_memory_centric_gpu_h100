"""Classify observed FA3 kernels by implementation identity, never a 'flash' substring."""
from pathlib import Path

SOURCE_AT_IMPORT=Path(__file__).read_bytes()


def attention_role(symbol):
    if 'reshape_and_cache' in symbol:
        return 'kv_update'
    if 'prepare_varlen_num_blocks_kernel' in symbol:
        return 'attention_scheduler'
    if 'FlashAttnFwdCombine<' in symbol:
        return 'attention_combine'
    if 'FlashAttnFwdSm90<' in symbol:
        return 'attention_forward'
    return None
