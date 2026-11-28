import copy
import hashlib
import struct
import unittest
from h200_review_native_transport import check_trials


class KnownPayload(unittest.TestCase):
    def fixture(self):
        source, target = [], []
        for trial in range(6):
            layers = []
            for layer in range(64):
                raw = struct.pack('<f',float((trial*64+layer)%128))[2:]*(2*8*16*8*128)
                layers.append({'layer_index':layer,'tensor_id':f'r{trial}#model.layers.{layer}.self_attn.attn',
                    'shape':[2,8,16,8,128],'dtype':'torch.bfloat16','nbytes':len(raw),
                    'sha256':hashlib.sha256(raw).hexdigest()})
            row = {'trial':trial,'warmup':trial<3,'request_id':f'r{trial}',
                'status':'known_data_transport_passed','all_native_send_completions_observed':True,'layers':layers}
            source.append(row); received = copy.deepcopy(row)
            for item in received['layers']:
                item.update(received_sha256=item['sha256'],known_values_verified=True)
            target.append(received)
        return source,target

    def test_wrong_payload_or_unfinished_last_send_is_rejected(self):
        source,target = self.fixture(); check_trials(source,target)
        for corruption in ('payload','completion','missing_layer','warmup'):
            a,b = copy.deepcopy(source),copy.deepcopy(target)
            if corruption == 'payload': b[5]['layers'][63]['received_sha256'] = '0'*64
            elif corruption == 'completion': a[5]['all_native_send_completions_observed'] = False
            elif corruption == 'missing_layer': b[5]['layers'].pop()
            else: b[5]['warmup'] = True
            with self.assertRaises(AssertionError): check_trials(a,b)


if __name__ == '__main__': unittest.main()
