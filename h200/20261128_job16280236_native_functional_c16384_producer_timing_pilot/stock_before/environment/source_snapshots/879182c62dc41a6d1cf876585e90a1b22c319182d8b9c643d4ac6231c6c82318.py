"""CPU protocol guards for a prospective native P2pNccl 1P1D pilot.

These checks do not implement or qualify a GPU transfer. Measured layer
records must come from the actual native send/receive/load paths.
"""
import ipaddress
import math
import re


def request_id(prefix, producer_host, producer_port, consumer_host, consumer_port):
    assert re.fullmatch(r'[A-Za-z0-9_]+', prefix)
    assert '___prefill_addr_' not in prefix and '___decode_addr_' not in prefix
    for host in (producer_host, consumer_host):
        assert ipaddress.ip_address(host).version == 4
    assert all(isinstance(p, int) and 0 < p < 65536 for p in (producer_port, consumer_port))
    assert (producer_host, producer_port) != (consumer_host, consumer_port)
    return (f'{prefix}___prefill_addr_{producer_host}:{producer_port}'
            f'___decode_addr_{consumer_host}:{consumer_port}')


def phase_contract(prompt_ids, first_token, generated_tokens=64, block_size=16):
    assert prompt_ids and all(isinstance(x, int) and x >= 0 for x in prompt_ids)
    assert isinstance(first_token, int) and first_token >= 0
    assert isinstance(generated_tokens, int) and generated_tokens >= 2
    assert isinstance(block_size, int) and block_size > 0
    context = len(prompt_ids)
    producer_blocks = math.ceil(context/block_size)
    consumer_blocks = math.ceil((context+1)/block_size)
    # Pinned Qwen2.5-32B BF16:64layers,8KVheads,128head dimension, K+V.
    bytes_per_cached_token = 64*8*128*2*2
    return {'classification': 'analytical prospective protocol check, not a GPU measurement',
            'original_context_tokens': context, 'producer_generated_tokens': 1,
            'consumer_prompt_ids': list(prompt_ids)+[first_token],
            'consumer_generated_tokens': generated_tokens-1,
            'canonical_generated_tokens': generated_tokens,
            'consumer_external_matched_tokens_expected': context,
            'consumer_original_prompt_queries_expected': 0,
            'consumer_first_query_original_position': context,
            'consumer_prefill_classified_queries_expected': 1,
            'consumer_post_prompt_queries_expected': generated_tokens-2,
            'canonical_decode_queries_expected': generated_tokens-1,
            'block_size': block_size, 'producer_blocks': producer_blocks,
            'consumer_blocks': consumer_blocks,
            'packed_payload_tokens_including_padding': producer_blocks*block_size,
            'padding_tokens': producer_blocks*block_size-context,
            'analytical_requested_KV_tensor_bytes': producer_blocks*block_size*bytes_per_cached_token,
            'logical_meaningful_KV_bytes': context*bytes_per_cached_token,
            'physical_HBM_or_NVLink_bytes': None}


def validate_layers(sent, loaded, rid, expected_tensor_bytes):
    """Require64packed BF16 layers, matching bytes/hashes and completed sends."""
    assert len(sent) == len(loaded) == 64
    assert expected_tensor_bytes > 0 and expected_tensor_bytes % 64 == 0
    def index(rows, producer):
        result = {}
        for row in rows:
            layer = row['layer_index']
            assert isinstance(layer, int) and layer in range(64) and layer not in result
            assert row['tensor_id'].startswith(rid+'#') and len(row['tensor_id']) > len(rid)+1
            assert row['dtype'] == 'torch.bfloat16' and row['contiguous'] is True
            assert row['completed'] is True, 'queued transfer is not completion'
            shape = row['shape']
            assert shape and all(isinstance(n, int) and n > 0 for n in shape)
            assert len(shape) == 5 and 2 in shape[:2] and shape[-1] == 128
            assert 8 in shape[2:4], 'pinned dense-model KV-head layout required'
            assert math.prod(shape)*2 == row['nbytes'] == expected_tensor_bytes//64
            assert re.fullmatch(r'[a-f0-9]{64}', row['sha256'])
            if not producer:
                assert row['injected'] is True, 'received bytes do not prove KV injection'
            result[layer] = row
        assert len({r['tensor_id'] for r in rows}) == 64
        return result
    source, target = index(sent, True), index(loaded, False)
    for layer in range(64):
        for key in ('tensor_id', 'shape', 'dtype', 'nbytes', 'sha256'):
            assert source[layer][key] == target[layer][key], f'KV mismatch: layer{layer} {key}'
    return {'status': 'paired_layer_ledger_passed', 'packed_layer_tensors': 64,
            'semantic_K_and_V_roles': 128, 'requested_tensor_bytes': expected_tensor_bytes,
            'physical_link_or_HBM_bytes': None,
            'scope': 'source/received/injected ledger agreement only; exact full outputs, actual transport, pair isolation, phase/query counts and independent references still required'}
