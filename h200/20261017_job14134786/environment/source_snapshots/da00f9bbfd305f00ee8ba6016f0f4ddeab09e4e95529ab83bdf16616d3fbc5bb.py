#!/usr/bin/env python3
"""Give each Nsight application replay its own preserved native output directory."""
import os
from pathlib import Path
import sys
from h200_session import PACKAGE,PYTHON


def main():
    arguments=sys.argv[1:]
    index=arguments.index('--output')+1
    base=Path(arguments[index])
    for replay in range(100):
        output=base.with_name(base.name+f'_pass{replay:03d}')
        if not output.exists():break
    else:raise RuntimeError('unexpected application replay count')
    arguments[index]=str(output)
    os.execv(str(PYTHON),[str(PYTHON),str(PACKAGE/'scripts/controlled_vllm_experiments.py'),*arguments])


if __name__=='__main__':main()
