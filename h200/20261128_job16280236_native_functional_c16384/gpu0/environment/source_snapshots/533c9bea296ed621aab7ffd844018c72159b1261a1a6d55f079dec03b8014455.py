#!/usr/bin/env python3
import copy
import os
from pathlib import Path
import unittest
from h200_spare_linker_followup import validate_transition


class Contracts(unittest.TestCase):
    def test_only_owned_idle_matching_worker_can_transition(self):
        root=Path('/tmp/CPU_fixture/gpu1')
        command=['python','scripts/h200_priority_campaign.py','--worker','1','--root',str(root),'--session',str(root/'session.json')]
        args=[{'gpu_uuid':'GPU-fixture'},root,{'status':'idle'},command,os.getuid(),[],[]]
        validate_transition(*args)
        for index,value in ((2,{'status':'running'}),(4,os.getuid()+1),(5,['active-checkpoint']),(6,[123])):
            bad=copy.deepcopy(args);bad[index]=value
            with self.assertRaises(AssertionError):validate_transition(*bad)
        for index,value in ((3,'0'),(5,'/tmp/other/gpu1'),(7,'/tmp/other/session.json')):
            bad=copy.deepcopy(args);bad[3][index]=value
            with self.assertRaises(AssertionError):validate_transition(*bad)


if __name__=='__main__':unittest.main()
