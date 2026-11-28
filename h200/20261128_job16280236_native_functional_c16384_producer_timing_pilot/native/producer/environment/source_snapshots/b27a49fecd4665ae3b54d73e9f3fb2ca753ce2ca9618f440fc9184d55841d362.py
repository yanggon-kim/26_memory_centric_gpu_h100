#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
import archive_h200_priority as archiver


class Contracts(unittest.TestCase):
    def test_verified_idempotent_copy_does_not_follow_external_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'run';root.mkdir();(root/'data.txt').write_text('measured raw bytes\n')
            outside=base/'outside';outside.write_text('not part of evidence');(root/'link').symlink_to(outside)
            result=archiver.archive(root,base/'archive')
            self.assertTrue(result['every_archive_member_verified'])
            self.assertEqual(result['archive_sha256'],archiver.digest(Path(result['archive'])))
            manifest=json.loads(Path(result['source_manifest']).read_text())
            self.assertEqual([r['path'] for r in manifest['files']],['data.txt'])
            self.assertEqual(len(manifest['excluded_symlinks']),1)
            self.assertEqual(archiver.archive(root,base/'archive'),result)
            Path(result['archive']).write_bytes(b'corrupted destination')
            with self.assertRaises(AssertionError):archiver.archive(root,base/'archive')
            self.assertEqual((root/'data.txt').read_text(),'measured raw bytes\n')

    def test_source_budget_rejects_before_archive_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'run';root.mkdir();(root/'data').write_bytes(b'1234')
            with self.assertRaises(AssertionError):archiver.archive(root,base/'archive',3)
            self.assertFalse((base/'archive').exists())


if __name__=='__main__':unittest.main()
