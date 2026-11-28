import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from software_decode_experiments import dump


class AtomicDump(unittest.TestCase):
    def test_exact_json_bytes_and_replace_failure_preserve_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run.json';data={'status':'running','trials':[1,2]}
            dump(path,data);expected=json.dumps(data,indent=2)+'\n'
            self.assertEqual(path.read_text(),expected)
            with patch.object(Path,'replace',side_effect=OSError('interrupted publication')):
                with self.assertRaises(OSError):dump(path,{'status':'complete'})
            self.assertEqual(path.read_text(),expected)
            self.assertEqual(list(Path(directory).iterdir()),[path])


if __name__=='__main__':unittest.main()
