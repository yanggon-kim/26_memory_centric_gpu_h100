#!/usr/bin/env python3
"""Publish every application-replay process before it starts native execution."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    args=sys.argv[1:]
    i=args.index('--output')+1
    base=Path(args[i])
    base.mkdir(parents=True,exist_ok=True)
    for index in range(100):
        output=base/f'pass{index:03d}'
        try:output.mkdir(exist_ok=False)
        except FileExistsError:continue
        break
    else:raise RuntimeError('application replay pass limit exceeded')
    # The target requires a fresh directory; its reservation/publication lives beside it.
    output.rmdir()
    publication=base/f'pass{index:03d}.json'
    from software_decode_experiments import dump
    dump(publication,{'status':'running','pass':index,'started_unix_ns':time.time_ns(),
                      'output':str(output),'argv':args})
    args[i]=str(output)
    import shoreline_decode
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',default='timing');parser.add_argument('--external-profiler',action='store_true')
    parser.add_argument('--collect-worker-metadata',action='store_true')
    parser.add_argument('--batch',type=int);parser.add_argument('--context',type=int)
    parser.add_argument('--generated-tokens',type=int,default=64)
    parser.add_argument('--fraction',type=float)
    parser.add_argument('--pressure',choices=['normal','pressure'])
    parsed=parser.parse_args(args);parsed.output.mkdir(parents=True,exist_ok=False)
    try:
        if parsed.fraction is not None:
            import shoreline_restricted_decode
            shoreline_restricted_decode.main(parsed)
        else:shoreline_decode.main(parsed)
        dump(publication,{'status':'complete','pass':index,'output':str(output),
                          'finished_unix_ns':time.time_ns()})
    except BaseException as error:
        dump(publication,{'status':'failed','pass':index,'output':str(output),
                          'error_type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':main()
