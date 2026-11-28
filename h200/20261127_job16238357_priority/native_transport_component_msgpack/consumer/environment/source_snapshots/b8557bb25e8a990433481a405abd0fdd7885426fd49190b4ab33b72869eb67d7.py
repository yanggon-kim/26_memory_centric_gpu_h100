"""Decoder for the pinned ATen FA3 scheduler layout; padding is never interpreted."""
import re


SOURCE_PIN='bce29425653ec0fbc579d329883030e832d15ada'


def decode(raw,batch,causal=True):
    # hopper/flash_api.cpp: varlen_sort_batches=false. For the supported
    # causal paged head-dim128 path: {packed_q, optional splits, head swizzle,
    # semaphore}. This is NOT flash_api_stable.cpp's different ABI/layout.
    if not causal:raise ValueError('only the verified causal head-swizzle layout is supported')
    rounded=(batch+3)//4*4
    if len(raw)==3*rounded+1:
        splits=raw[rounded:rounded+batch]
        head_offset=2*rounded
    elif len(raw)==2*rounded+1:
        splits=[1]*batch
        head_offset=rounded
    else:raise ValueError('scheduler layout/length differs from verified ATen ABI')
    if not all(isinstance(x,int) and 1<=x<=128 for x in splits):
        raise ValueError('invalid active split metadata; padding must not be decoded')
    return {'packed_query_lengths':raw[:batch],'per_request_dynamic_splits':splits,
            'num_nheads_in_l2':raw[head_offset:head_offset+batch],
            'tile_count_semaphore':raw[-1],'unused_padding_ignored':True,
            'source_pin':SOURCE_PIN,'source_file':'hopper/flash_api.cpp',
            'scope':'actual active scheduler fields decoded from GPU metadata; not productive CTA counts'}


def tile(name):
    integer=r'(?:\(int\))?\s*(\d+)'
    shape=r'cute::tuple<cute::C<'+integer+r'>,\s*cute::C<'+integer+r'>,\s*cute::C<'+integer+r'>>'
    match=re.search(r'CollectiveMainloopFwdSm90<'+integer+r',\s*'+shape+r',\s*'+shape,name)
    if not match:return None
    values=[int(x) for x in match.groups()]
    result={'stages':values[0],'cluster_shape':values[1:4],
            'BM':values[4],'BN':values[5],'BK':values[6],
            'source_pin':SOURCE_PIN,'source_file':'hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp',
            'scope':'observed kernel template specialization; not a measured useful/empty-work count'}
    body=name.split('CollectiveMainloopFwdSm90<',1)[1].split('flash::CollectiveEpilogueFwd',1)[0]
    tail=re.split(r'(?:cutlass::)?arch::Sm90,',body,maxsplit=1)
    if len(tail)==2:
        flags=re.findall(r'(?:\(bool\))?\s*([01])\s*,',tail[1].split('cutlass::bfloat16_t')[0])
        keys=['causal','local','softcap','varlen','paged_kv_non_tma','append_kv',
              'has_qv','mma_pv_register_source','intra_warpgroup_overlap','pack_gqa','split','v_colmajor']
        if len(flags)>=len(keys):result['flags']={key:bool(int(value)) for key,value in zip(keys,flags)}
    return result
