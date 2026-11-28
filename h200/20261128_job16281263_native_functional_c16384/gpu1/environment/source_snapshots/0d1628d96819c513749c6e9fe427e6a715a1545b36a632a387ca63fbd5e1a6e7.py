"""CPU entrypoint admission/output contracts; no CUDA initialization."""
import json
from pathlib import Path
from types import ModuleType,SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

from h200_native_connector_attribution import main


class Attribution(unittest.TestCase):
    def args(self,root,protocol='host_API_NVTX_functional_v1'):
        pair=root/'pair.json'
        pair.write_text(json.dumps({'connector_attribution_protocol':protocol,
            'protocol':'native_model_KV_long_context_functional_v2'}))
        return SimpleNamespace(pair_config=pair,output=root/'new-output',role='producer',gpu_uuid='GPU-toy')

    def module(self):
        module=ModuleType('h200_native_model_pilot');module.NativeLedger=type('OriginalLedger',(),{})
        return module

    def test_output_directory_exists_before_functional_ledger_write(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.args(Path(folder));module=self.module();original=module.NativeLedger
            def delegated(actual):
                self.assertIs(actual,args)
                (actual.output/'run.json').write_text('{"CPU_fixture":true}')
                return 'delegated'
            module.main=delegated
            with patch.dict('sys.modules',{'h200_native_model_pilot':module}):
                self.assertEqual(main(args),'delegated')
            self.assertTrue((args.output/'run.json').exists())
            self.assertIs(module.NativeLedger,original)

    def test_existing_output_is_preserved_and_never_delegated(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.args(Path(folder));args.output.mkdir();sentinel=args.output/'raw.json';sentinel.write_text('preserved')
            module=self.module();module.main=Mock()
            with patch.dict('sys.modules',{'h200_native_model_pilot':module}):
                with self.assertRaises(FileExistsError):main(args)
            module.main.assert_not_called();self.assertEqual(sentinel.read_text(),'preserved')

    def test_unknown_protocol_never_creates_output_or_delegates(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.args(Path(folder),'unreviewed');module=self.module();module.main=Mock()
            with patch.dict('sys.modules',{'h200_native_model_pilot':module}):
                with self.assertRaises(AssertionError):main(args)
            module.main.assert_not_called();self.assertFalse(args.output.exists())

    def test_delegate_failure_restores_original_class_and_preserves_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.args(Path(folder));module=self.module();original=module.NativeLedger
            def failed(actual):
                (actual.output/'partial.json').write_text('retained')
                raise RuntimeError('CPU fixture stopped')
            module.main=failed
            with patch.dict('sys.modules',{'h200_native_model_pilot':module}):
                with self.assertRaisesRegex(RuntimeError,'CPU fixture stopped'):main(args)
            self.assertIs(module.NativeLedger,original)
            self.assertEqual((args.output/'partial.json').read_text(),'retained')


if __name__=='__main__':unittest.main()
