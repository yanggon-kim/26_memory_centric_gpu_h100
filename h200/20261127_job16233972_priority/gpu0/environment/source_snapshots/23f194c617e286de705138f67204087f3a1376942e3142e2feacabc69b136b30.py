import unittest
from h200_postcore_pilots import validate_idle


class Idle(unittest.TestCase):
    def test_completed_core_and_idle_gpu_are_required(self):
        core={'complete_acquisition_set':True};outcome={'status':'resident_curve_disposition'}
        validate_idle(core,{'status':'idle'},outcome,[])
        for c,current,o,pids in (({}, {'status':'idle'},outcome,[]),
            (core,{'status':'running'},outcome,[]),(core,{'status':'idle'},outcome,[123]),
            (core,{'status':'idle'},{'status':'failed'},[])):
            with self.assertRaises(AssertionError):validate_idle(c,current,o,pids)

    def test_active_conditioned_continuation_blocks_idle_transition(self):
        with self.assertRaises(AssertionError):
            validate_idle({'complete_acquisition_set':True},{'status':'idle'},
                {'status':'resident_curve_disposition','conditioned_followup':{'status':'admitted'}},[])


if __name__=='__main__':unittest.main()
