import subprocess
import tempfile
import unittest
from pathlib import Path
from checkpoint_h200 import scan_staged


def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL)


class IndexScanner(unittest.TestCase):
    def test_index_secret_is_rejected_even_after_working_copy_is_cleaned(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);git(root,'init','-q')
            p=root/'evidence with space.json'
            p.write_bytes(b'{"synthetic_test": "'+b'hf_'+b'A'*30+b'"}')
            git(root,'add','--',p.name);p.write_bytes(b'clean working copy')
            with self.assertRaisesRegex(RuntimeError,'publication refused'):
                scan_staged(root)

    def test_deletion_rename_binary_and_unusual_paths_keep_index_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);git(root,'init','-q')
            git(root,'config','user.name','CPU contract');git(root,'config','user.email','cpu@example.invalid')
            (root/'removed').write_text('old');(root/'old name').write_text('retained')
            git(root,'add','.');git(root,'commit','-qm','CPU fixture')
            git(root,'rm','-q','removed');git(root,'mv','old name','new\nname')
            (root/'binary').write_bytes(b'\0clean\xff');git(root,'add','binary')
            # Unstaged content cannot change what the index gate actually scans.
            (root/'binary').write_bytes(b'ghp_'+b'B'*40)
            scan_staged(root)
            git(root,'add','binary')
            with self.assertRaisesRegex(RuntimeError,'publication refused'):
                scan_staged(root)

    def test_empty_index_has_no_reader_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);git(root,'init','-q');scan_staged(root)


if __name__=='__main__':unittest.main()
