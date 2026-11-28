import unittest
from h200_catalog_decoder_grids import attribute


class Attribution(unittest.TestCase):
    def test_unique_host_thread_and_interval_assigns_real_operation(self):
        api={'globalTid':9,'start':20,'end':30}
        ranges=[{'globalTid':9,'start':0,'end':100,'text':'tick_4'},
                {'globalTid':9,'start':10,'end':40,'text':'operation_layer_3_self_attn_o_proj'},
                {'globalTid':10,'start':10,'end':40,'text':'operation_layer_8_mlp_down_proj'}]
        self.assertEqual(attribute(api,ranges),'tick4/layer3/self_attn_o_proj')

    def test_no_guess_from_unrelated_or_ambiguous_ranges(self):
        api={'globalTid':9,'start':20,'end':30}
        ranges=[{'globalTid':9,'start':0,'end':100,'text':'tick_4'},
                {'globalTid':9,'start':10,'end':40,'text':'cutlass_attention_name_is_not_an_operation_range'}]
        self.assertIn('unattributed',attribute(api,ranges))
        self.assertIn('missing_unique',attribute(None,ranges))


if __name__=='__main__':unittest.main()
