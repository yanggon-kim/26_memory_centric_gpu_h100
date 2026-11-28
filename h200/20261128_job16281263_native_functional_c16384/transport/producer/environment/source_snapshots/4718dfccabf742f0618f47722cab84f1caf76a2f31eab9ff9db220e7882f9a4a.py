#!/usr/bin/env python3
"""Instrumented native model functional pilot with explicit connector spans.

Separate entrypoint, output and pair descriptor required. This reuses exact
functional gates and preserves its serialized handoff. It is not a clean
producer comparison or a serving benchmark. Metadata-only and payload-hashing
arms remain distinct; payload-arm API spans include proof overhead.
"""
import argparse
from pathlib import Path
import json


def main(args):
    pair=json.loads(args.pair_config.read_text())
    assert pair.get('connector_attribution_protocol')=='host_API_NVTX_functional_v1'
    assert pair['protocol']=='native_model_KV_long_context_functional_v2'
    args.output.mkdir(parents=True,exist_ok=False)
    import h200_native_model_pilot as functional
    from h200_connector_spans import ConnectorSpans
    from software_decode_experiments import dump
    original=functional.NativeLedger

    class ObservedLedger(original):
        def __init__(self,connector):
            super().__init__(connector)
            import torch
            self.connector_spans=ConnectorSpans(connector,torch.cuda.nvtx.range_push,torch.cuda.nvtx.range_pop)

        def close(self):
            # Preserve spans even on failed or partial functional requests.
            # Do not restore active native calls while a background operation
            # is in progress; cleanup remains bounded by the role supervisor.
            spans=self.connector_spans
            document={'classification':spans.classification,'events':spans.events,
                'protocol':pair['connector_attribution_protocol'],
                'clean_timing_qualified':False,'GPU_compute_attributed':False,
                'per_request_mode':'request_id contains metadata_control or payload_proof; never pool arms',
                'scope':'save_kv_layer includes KV extraction and enqueue, not transformer layer compute. send_sync includes protocol/communication blocking; native send/recv include their existing stream synchronization. Nested/concurrent intervals are not additive. Payload-arm spans include hashing proof overhead. Serialized functional handoff is retained and cannot be called deployment latency.'}
            try:
                document['summary']=spans.summary()
                spans.close()
            except AssertionError as error:
                # An interrupted native call remains process-owned. Preserve
                # its partial span and let the existing engine cleanup and
                # outer supervisor finish; never strand cleanup on this check.
                document['incomplete_span_or_restore_error']=str(error)
            else:
                super().close()
            dump(args.output/'connector_host_spans.json',document)

    functional.NativeLedger=ObservedLedger
    try:
        return functional.main(args)
    finally:
        functional.NativeLedger=original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-config',type=Path,required=True)
    parser.add_argument('--role',choices=('producer','consumer'),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
