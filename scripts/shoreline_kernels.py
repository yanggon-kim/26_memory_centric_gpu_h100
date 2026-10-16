"""Versioned scalar memory, dependent-load and arithmetic controls for Hopper."""
import triton
import triton.language as tl


@triton.jit
def condition(A,B,N:tl.constexpr,STEPS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(A+i,i<N,other=0)
    for _ in range(STEPS):x=tl.fma(x,0.9999999,0.0000001)
    tl.store(B+i,x,i<N)


@triton.jit
def transfer(A, B, SMIDS, N:tl.constexpr, PATTERN:tl.constexpr, BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    lane=tl.arange(0,BLOCK)
    i=pid*BLOCK+lane
    if PATTERN == 0:
        source=i
    elif PATTERN == 1:
        source=(i*17)%N
    elif PATTERN == 2:
        source=((i.to(tl.int64)*1103515245+12345)%N).to(tl.int32)
    else:
        source=((i//128*17)%(N//128))*128+i%128
    value=tl.load(A+source, i<N, other=0, cache_modifier='.cg')
    tl.store(B+i,value,i<N)
    smid=tl.inline_asm_elementwise('mov.u32 $0, %smid;',constraints='=r',args=[],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(SMIDS+pid,smid)


@triton.jit
def dependent(NEXT, STARTS, OUT, CYCLES, SMIDS, N:tl.constexpr, CHAINS:tl.constexpr, STEPS:tl.constexpr):
    lane=tl.arange(0,triton.next_power_of_2(CHAINS))
    index=tl.load(STARTS+lane,lane<CHAINS,other=0)
    start=tl.inline_asm_elementwise('mov.u64 $0, %clock64;',constraints='=l',args=[],dtype=tl.int64,is_pure=False,pack=1)
    for _ in range(STEPS):
        index=tl.load(NEXT+index,lane<CHAINS,other=0,cache_modifier='.ca',volatile=True)
    end=tl.inline_asm_elementwise('mov.u64 $0, %clock64;',constraints='=l',args=[],dtype=tl.int64,is_pure=False,pack=1)
    tl.store(OUT+lane,index,lane<CHAINS)
    tl.store(CYCLES+lane,end-start,lane<CHAINS)
    smid=tl.inline_asm_elementwise('mov.u32 $0, %smid;',constraints='=r',args=[],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(SMIDS,smid)


@triton.jit
def arithmetic(A,B,SMIDS,N:tl.constexpr,STEPS:tl.constexpr,INTEGER:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    i=pid*BLOCK+tl.arange(0,BLOCK)
    if INTEGER:
        x=tl.load(A+i,i<N,other=0).to(tl.uint32)
        for _ in range(STEPS):
            x=x*1664525+1013904223
    else:
        x=tl.load(A+i,i<N,other=0).to(tl.float32)
        for _ in range(STEPS):
            x=tl.fma(x,1.0001,0.0001)
    tl.store(B+i,x,i<N)
    smid=tl.inline_asm_elementwise('mov.u32 $0, %smid;',constraints='=r',args=[],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(SMIDS+pid,smid)
