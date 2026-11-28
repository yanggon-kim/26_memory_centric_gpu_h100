"""CPU copy-event schema/ambiguity/unit contracts; no new GPU evidence."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import test_connector_timeline as kernel_fixtures
from h200_connector_copy_timeline import read_copies


class Copies(unittest.TestCase):
    def fixture(self,path,copy_count=1):
        db,actor,other=kernel_fixtures.Timeline().fixture(path)
        db.executescript('''
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,bytes INTEGER,copyKind INTEGER,copyCount INTEGER);
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,name TEXT,label TEXT);
        ''')
        db.executemany('INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (?,?,?)',[(1,'CUDA_MEMCPY_KIND_HTOD','Host-to-Device'),(2,'CUDA_MEMCPY_KIND_DTOH','Device-to-Host')])
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)',(31,34,actor+100,8,2))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?)',(500,600,actor,8,100,1,copy_count))
        db.commit();return db,actor,other

    def test_async_copy_matches_launch_scope_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,_,_=self.fixture(path);db.close()
            before=hashlib.sha256(path.read_bytes()).hexdigest();result=read_copies(path,100,'capture')
            self.assertEqual(result['selected_launch_correlated_copy_records'],1)
            self.assertEqual(result['copy_record_bytes_field_sum'],100)
            self.assertEqual(result['selected_copies'][0]['innermost_connector_NVTX'],'h200_connector/Test.send/0')
            self.assertEqual(result['by_copy_kind']['CUDA_MEMCPY_KIND_HTOD']['bytes_field_sum'],100)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
            self.assertIsNone(result['physical_HBM_bytes']);self.assertIsNone(result['physical_link_bytes'])
            self.assertFalse(result['scientific_qualified'])

    def test_copy_count_is_preserved_without_multiplying_byte_field(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,_,_=self.fixture(path,4);db.close()
            result=read_copies(path,100,'capture')
            self.assertEqual(result['copy_record_bytes_field_sum'],100)
            self.assertEqual(result['selected_copies'][0]['copyCount'],4)
            self.assertEqual(len(result['copy_count_not_single_record_rowids']),1)

    def test_ambiguous_copy_API_remains_explicit_unresolved(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,actor,_=self.fixture(path)
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)',(32,35,actor+100,8,2));db.commit();db.close()
            result=read_copies(path,100,'capture')
            self.assertEqual(result['actor_copy_records'],1)
            self.assertEqual(result['selected_launch_correlated_copy_records'],0)
            self.assertEqual(result['unresolved_launch_copy_records'][0]['bytes_field'],100)

    def test_other_process_same_correlation_does_not_join(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,_,other=self.fixture(path)
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)',(31,34,other+200,8,2));db.commit();db.close()
            result=read_copies(path,100,'capture')
            self.assertEqual(result['selected_launch_correlated_copy_records'],1)
            self.assertEqual(result['unresolved_launch_copy_records'],[])

    def test_absent_copy_table_is_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,_,_=kernel_fixtures.Timeline().fixture(path);db.close()
            result=read_copies(path,100,'capture')
            self.assertFalse(result['copy_table_available'])
            self.assertIsNone(result['actor_copy_records'])
            self.assertIsNone(result['copy_record_bytes_field_sum'])

    def test_copy_overlap_is_not_added_as_active_time(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,actor,_=self.fixture(path)
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)',(32,35,actor+100,9,2))
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?)',(550,650,actor,9,200,2,1));db.commit();db.close()
            result=read_copies(path,100,'capture')
            self.assertEqual(result['copy_record_bytes_field_sum'],300)
            self.assertEqual(result['copy_duration_sum_ns'],200)
            self.assertEqual(result['copy_active_union_ns'],150)


if __name__=='__main__':unittest.main()
