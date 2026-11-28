"""CPU archive binding failures cannot supply future runtime estimates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h200_producer_runtime_observations_v2 as reader


class Archive(unittest.TestCase):
    def fixture(self,folder,condition='valid'):
        package=Path(folder);root=package/'h200/closed';(root/'results').mkdir(parents=True)
        member=root/'run.json';member.write_text('{"fixture":"closed"}')
        rows=[dict(path='run.json',sha256=hashlib.sha256(member.read_bytes()).hexdigest())]
        if condition=='duplicate':rows+=rows
        manifest=root/'results/manifest.json';manifest.write_text(json.dumps({'files':rows}))
        receipt=dict(status='archive_transfer_verified',every_archive_member_verified=True,
            archive_sha256='fixture-pin',destination_sha256='fixture-pin',source_root=str(root),
            source_manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        if condition=='unverified':receipt['every_archive_member_verified']=False
        if condition=='wrong_root':receipt['source_root']=str(package/'h200/other')
        if condition=='wrong_archive':receipt['destination_sha256']='other'
        (root/'results/archive_transfer_fixture.json').write_text(json.dumps(receipt))
        return package,root,member

    def test_original_member_passes_but_later_byte_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            package,root,member=self.fixture(folder)
            with patch.object(reader,'PACKAGE',package):
                checked,_=reader.archived_reader(root);self.assertEqual(checked(member),member.resolve())
                member.write_text('{"fixture":"changed"}')
                with self.assertRaisesRegex(AssertionError,'differs'):checked(member)

    def test_archive_verification_root_and_unique_members_are_required(self):
        for condition in ('unverified','wrong_root','wrong_archive','duplicate'):
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as folder:
                package,root,_=self.fixture(folder,condition)
                with patch.object(reader,'PACKAGE',package),self.assertRaises(AssertionError):reader.archived_reader(root)

    def test_outside_or_unarchived_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            package,root,_=self.fixture(folder)
            with patch.object(reader,'PACKAGE',package):
                checked,_=reader.archived_reader(root)
                with self.assertRaises(ValueError):checked(package/'outside.json')
                extra=root/'later.json';extra.write_text('{}')
                with self.assertRaises(KeyError):checked(extra)


if __name__=='__main__':unittest.main()
