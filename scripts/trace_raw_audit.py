"""Audit raw warp masks/opcodes and refuse the pinned postprocessor's wait pruning."""
from collections import Counter
import lzma
import re


def instruction(line,lineinfo=0):
    parts=line.split()[11+lineinfo:]
    pc=int(parts[0],16);mask=int(parts[1],16);destinations=int(parts[2])
    assert 0<=mask<2**32
    oi=3+destinations;opcode=parts[oi];sources=int(parts[oi+1]);mi=oi+2+sources
    width=int(parts[mi]);kind=None;address=None
    wait='PHASECHK' in opcode and 'TRYWAIT' in opcode
    tma=opcode.startswith('UTMALDG')
    arrive=opcode.startswith(('SYNCS.ARRIVE','ARRIVES.LDGSTSBAR'))
    if wait or arrive or tma:
        if width:
            if tma:address=int(parts[mi+1],16)&0xffffffff
            else:
                mode=int(parts[mi+1]);assert mode in (0,1,2),'unsupported mbarrier address encoding'
                address=int(parts[mi+2],16)&0xffffffff
            kind='wait' if wait else 'arrive'
        else:raise ValueError('async barrier instruction lacks required memory address payload')
    return opcode,mask,kind,address


def audit(root):
    paths=sorted(p for p in root.rglob('*') if p.name.endswith(('.trace','.trace.xz')))
    assert paths,'raw trace files missing'
    output={'status':'passed_no_wait_pruning','files':[],
        'semantic_scope':'raw syntax/opcode/mask inventory and pinned wait-pruning precheck only; async descriptors, productive workers and timing remain uncalibrated',
        'instruction_count_note':'raw records with multi-memory operands can represent the same instruction more than once; active-lane contributions are not issued instruction counts'}
    for path in paths:
        counts=Counter();arrived=set();waits=Counter();lanes=0;records=0;lineinfo=0;binary=None
        opener=lzma.open if path.name.endswith('.xz') else open
        with opener(path,'rt') as f:
            for number,line in enumerate(f,1):
                if line.startswith('-enable lineinfo'):lineinfo=int(line.rsplit('=',1)[1])
                if line.startswith('-binary version'):binary=int(line.rsplit('=',1)[1])
                if not line[:1].isdigit():continue
                try:opcode,mask,kind,address=instruction(line,lineinfo)
                except (ValueError,IndexError,AssertionError) as error:
                    raise ValueError(f'{path.name}:{number}: {error}') from error
                counts[opcode]+=1;records+=1;lanes+=mask.bit_count()
                if kind=='arrive' and address:arrived.add(address)
                if kind=='wait' and address:waits[address]+=1
        assert binary==90 and records>0,'raw capture must be nonempty actual sm_90'
        orphaned={hex(a):n for a,n in waits.items() if a not in arrived}
        output['files'].append({'path':str(path.relative_to(root)),'raw_records':records,
            'raw_active_lane_record_contributions':lanes,'opcode_counts':dict(counts),
            'would_prune_trywaits':orphaned,'binary_version':binary})
        if orphaned:output['status']='rejected_wait_pruning_required'
    return output
