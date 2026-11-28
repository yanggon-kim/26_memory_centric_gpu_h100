import tempfile
from pathlib import Path
import unittest
from h200_runtime_prewarm import warm,candidates


class Bounds(unittest.TestCase):
    def test_byte_budget_never_reads_beyond_declared_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'binary.so';path.write_bytes(b'x'*100)
            result=warm([path],17,1)
            self.assertEqual(result['bytes_read'],17);self.assertEqual(result['stop_reason'],'byte_budget')
            self.assertEqual(path.read_bytes(),b'x'*100)

    def test_external_symlinks_and_unrelated_files_are_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'runtime';site=root/'lib/python3.10/site-packages/torch/lib';site.mkdir(parents=True)
            outside=Path(folder)/'outside.so';outside.write_bytes(b'x')
            (site/'foreign.so').symlink_to(outside)
            kept=site/'owned.so';kept.write_bytes(b'y')
            (site/'unrelated.txt').write_text('not a shared library')
            self.assertEqual(candidates(root),[kept.resolve()])


if __name__=='__main__':unittest.main()
