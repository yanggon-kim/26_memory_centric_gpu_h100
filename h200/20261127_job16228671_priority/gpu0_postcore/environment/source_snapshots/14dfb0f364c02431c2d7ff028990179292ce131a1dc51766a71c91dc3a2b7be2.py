import types
import unittest
from h200_serving_policy import schedule_phase


def request(name,prefill=False):
    return types.SimpleNamespace(request_id=name,num_computed_tokens=0 if prefill else 4096,num_prompt_tokens=4096)


class Mask(unittest.TestCase):
    def test_held_decode_keeps_seat_and_order_during_prefill(self):
        d=request('decode');p=request('partial',True);w=request('waiting',True)
        waiting=[w];skipped=[]
        scheduler=types.SimpleNamespace(running=[d,p],waiting=waiting,skipped_waiting=skipped,max_num_running_reqs=3)
        def original():
            self.assertEqual(scheduler.running,[p]);self.assertEqual(scheduler.max_num_running_reqs,2)
            scheduler.running.append(scheduler.waiting.pop(0));return 'output'
        self.assertEqual(schedule_phase(scheduler,original,'prefill',list),'output')
        self.assertEqual(scheduler.running,[d,p,w]);self.assertEqual(scheduler.max_num_running_reqs,3)
        self.assertIs(scheduler.waiting,waiting);self.assertEqual(waiting,[])

    def test_decode_masks_waiting_and_restores_it_on_failure(self):
        d=request('decode');p=request('partial',True);waiting=[request('new',True)];skipped=[request('skip',True)]
        scheduler=types.SimpleNamespace(running=[p,d],waiting=waiting,skipped_waiting=skipped,max_num_running_reqs=2)
        def original():
            self.assertEqual(scheduler.running,[d]);self.assertFalse(scheduler.waiting or scheduler.skipped_waiting)
            self.assertEqual(scheduler.max_num_running_reqs,1)
            raise RuntimeError('producer failure must propagate')
        with self.assertRaisesRegex(RuntimeError,'must propagate'):schedule_phase(scheduler,original,'decode',list)
        self.assertEqual(scheduler.running,[p,d]);self.assertIs(scheduler.waiting,waiting)
        self.assertIs(scheduler.skipped_waiting,skipped);self.assertEqual(scheduler.max_num_running_reqs,2)

    def test_eligible_preemption_does_not_drop_held_request(self):
        d=request('decode');p=request('partial',True)
        scheduler=types.SimpleNamespace(running=[d,p],waiting=[],skipped_waiting=[],max_num_running_reqs=2)
        def original():scheduler.waiting.append(scheduler.running.pop());return 'empty'
        schedule_phase(scheduler,original,'prefill',list)
        self.assertEqual(scheduler.running,[d]);self.assertEqual(scheduler.waiting,[p])


if __name__=='__main__':unittest.main()
