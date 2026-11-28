"""CPU SQLite fixtures exercise ambiguous/async/process matching, not GPUs."""
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from h200_connector_timeline import read


class Timeline(unittest.TestCase):
    def fixture(self,path):
        actor=(1<<48)+(100<<24);other=(1<<48)+(200<<24)
        db=sqlite3.connect(path)
        db.executescript('''
            CREATE TABLE PROCESSES(globalPid INTEGER,pid INTEGER);
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,endGlobalTid INTEGER,text TEXT,textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER);
        ''')
        db.executemany('INSERT INTO PROCESSES VALUES (?,?)',[(actor,100),(other,200)])
        db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'kernel'),(2,'cudaLaunchKernel')])
        db.executemany('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,?)',[(0,100,actor+100,None,'capture',None),(25,38,actor+100,None,'h200_connector/Test.send/0',None)])
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)',(30,35,actor+100,7,2))
        # Async GPU execution intentionally starts AFTER the host range ends.
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)',(300,400,actor,7,1))
        db.commit();return db,actor,other

    def test_async_execution_matches_launch_scope_without_mutating_database(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,_,_=self.fixture(path);db.close()
            digest=hashlib.sha256(path.read_bytes()).hexdigest();result=read(path,100,'capture')
            self.assertEqual(result['selected_launch_correlated_kernel_records'],1)
            self.assertEqual(result['connector_NVTX_linked_kernel_records'],1)
            self.assertEqual(result['selected_kernel_active_union_ns'],100)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
            self.assertFalse(result['scientific_qualified'])

    def test_correlation_collision_in_another_process_is_not_joined(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,_,other=self.fixture(path)
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)',(30,35,other+200,7,2));db.commit();db.close()
            result=read(path,100,'capture')
            self.assertEqual(result['selected_launch_correlated_kernel_records'],1)
            self.assertEqual(result['missing_or_ambiguous_API_kernel_records'],[])

    def test_ambiguous_launch_API_is_preserved_unmatched(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,actor,_=self.fixture(path)
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)',(31,36,actor+100,7,2));db.commit();db.close()
            result=read(path,100,'capture')
            self.assertEqual(result['selected_launch_correlated_kernel_records'],0)
            self.assertEqual(result['missing_or_ambiguous_API_kernel_records'][0]['candidate_API_count'],2)
            self.assertFalse(result['nonempty_selected_launch_coverage'])

    def test_same_time_wrong_thread_NVTX_does_not_attribute_kernel(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';db,actor,_=self.fixture(path)
            db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=?',(actor+101,));db.commit();db.close()
            result=read(path,100,'capture')
            self.assertEqual(result['selected_launch_correlated_kernel_records'],1)
            self.assertEqual(result['connector_NVTX_linked_kernel_records'],0)
            self.assertEqual(result['unmatched_connector_kernel_records'],1)
            self.assertFalse(result['selected_kernels'][0]['launch_on_selected_range_thread'])

    def test_missing_file_is_not_created_by_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'absent.sqlite'
            with self.assertRaises(sqlite3.OperationalError):read(path,100,'capture')
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
