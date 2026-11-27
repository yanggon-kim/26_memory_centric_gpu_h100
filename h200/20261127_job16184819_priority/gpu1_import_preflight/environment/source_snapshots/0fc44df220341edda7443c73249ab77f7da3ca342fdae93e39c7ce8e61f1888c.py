"""Synthetic read-service controls; payload and physical traffic remain distinct."""
import triton
import triton.language as tl


@triton.jit
def read_reference(X,Y,N:tl.constexpr,PAGED:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0);i=pid*BLOCK+tl.arange(0,BLOCK)
    index=((i//64*17)%(N//64))*64+i%64 if PAGED else i
    value=tl.load(X+index,i<N,other=0,cache_modifier='.cg').to(tl.float32)
    tl.store(Y+pid,tl.sum(value,axis=0))
