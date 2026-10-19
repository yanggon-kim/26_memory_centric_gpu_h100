#!/usr/bin/env python3
"""CPU-only recovery/scheduling contracts for unattended H200 startup."""
import datetime as dt
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bootstrap_h200_batch as batch
import shoreline_queue as queue


class BatchContracts(unittest.TestCase):
    def test_scheduler_identity_and_closeout_budget(self):
        now = dt.datetime.now().astimezone()
        end = now + dt.timedelta(hours=2)
        state = f'JobState=RUNNING NodeList=node1 EndTime={end.isoformat()}'
        actual_end, cutoff = batch.allocation(state, 'node1.cluster', now, 1800, 1200)
        self.assertEqual(actual_end, end)
        self.assertEqual((end - cutoff).total_seconds(), 1800)
        for bad in [state.replace('RUNNING', 'PREEMPTED'), state.replace('node1', 'node2')]:
            with self.assertRaises(RuntimeError):
                batch.allocation(bad, 'node1.cluster', now, 1800, 1200)
        with self.assertRaises(RuntimeError):
            batch.allocation(state, 'node1.cluster', end - dt.timedelta(minutes=40), 1800, 1200)

    def test_only_completed_queue_commands_reused(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(batch, 'PACKAGE', Path(directory)):
            root = Path(directory) / 'h200/previous'
            batch.write(root / 'session.json', {'slurm_job_id': '10'})
            ledger = root / 'results/campaign_queue.json'
            batch.write(ledger, {'steps': [
                {'id': 'J5', 'status': 'command_complete', 'exit_code': 0},
                {'id': 'J8', 'status': 'running'},
                {'id': 'J7', 'status': 'command_failed', 'exit_code': 1},
                {'id': 'J10', 'status': 'command_complete', 'exit_code': 7},
                {'id': 'J9_analysis', 'status': 'command_complete', 'exit_code': 0}]})
            reused = batch.reused_jobs(root)
            self.assertEqual(set(reused), {'J5'})
            self.assertEqual(reused['J5']['sha256'], batch.digest(ledger))
            legacy = Path(directory) / 'h200/legacy'
            legacy.mkdir()
            batch.write(root / 'session.json', {'slurm_job_id': '10', 'old_session': 'h200/legacy'})
            self.assertEqual(batch.reused_jobs(root), reused)
            child = Path(directory) / 'h200/child'
            batch.write(child / 'session.json', {'slurm_job_id': '11', 'queue_reused_jobs': reused})
            self.assertEqual(batch.reused_jobs(child), reused)
            ledger.write_text('{"steps": []}')
            with self.assertRaises(ValueError):
                batch.reused_jobs(child)

    def test_latest_excludes_self_and_setup_only_attempt(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(batch, 'PACKAGE', Path(directory)):
            for number in (10, 11, 12):
                root = Path(directory) / 'h200' / f'session{number}'
                batch.write(root / 'session.json', {'slurm_job_id': str(number)})
                batch.write(root / 'results/shoreline_case_manifest.json', {})
                if number != 11:
                    batch.write(root / 'results/campaign_queue.json', {'steps': []})
            self.assertEqual(batch.previous_session({'previous_session': 'latest',
                             'minimum_previous_job_id': 10}, '12').name, 'session10')
            with self.assertRaises(ValueError):
                batch.evidence_path('../outside')

    def test_changed_or_missing_raw_artifact_stops_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(batch, 'PACKAGE', Path(directory)):
            previous = Path(directory) / 'h200/previous'
            root = Path(directory) / 'h200/new'
            (root / 'results').mkdir(parents=True)
            batch.write(previous / 'results/prior_capture_evidence.json', {'evidence': []})
            batch.write(previous / 'results/evidence_reuse.json', {'entries': []})
            raw = previous / 'profiles/capture.raw'
            raw.parent.mkdir()
            raw.write_bytes(b'original')
            batch.write(previous / 'results/checkpoint_raw_inventory.json', {'raw_artifacts': [{
                'path': str(raw.relative_to(Path(directory))), 'sha256': batch.digest(raw), 'bytes': 8}]})
            batch.prepare_evidence(previous, root)
            self.assertTrue((root / 'results/batch_evidence_audit.json').exists())
            raw.write_bytes(b'modified')
            with self.assertRaises(RuntimeError):
                batch.prepare_evidence(previous, root)
            raw.unlink()
            with self.assertRaises(FileNotFoundError):
                batch.prepare_evidence(previous, root)

    def test_gpu_queue_skips_prior_disposition_but_runs_remaining_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('environment/source_snapshots', 'logs', 'results'):
                (root / folder).mkdir(parents=True)
            session = root / 'session.json'
            batch.write(session, {'launch_cutoff': '2099-01-01T00:00:00+00:00',
                                 'queue_reused_jobs': {'J5': {'fixture': 'already retained'}}})
            calls = []
            with patch.object(queue.subprocess, 'call', side_effect=lambda command, **kw: calls.append(command) or 0):
                queue.main(SimpleNamespace(session=session, start_at='J5', schedule='gpu-first', phase='all'))
            jobs = [c[c.index('--job') + 1] for c in calls if '--job' in c]
            self.assertEqual(jobs, ['J8', 'J7', 'J10', 'J6', 'J11'])
            scripts = [Path(c[1]).name for c in calls]
            self.assertEqual(scripts[-4:], ['shoreline_analyze.py', 'plot_shoreline.py',
                                          'shoreline_archive_batch.py', 'checkpoint_h200.py'])
            self.assertEqual(batch.read(root / 'results/campaign_queue.json')['status'],
                             'commands_finished_scientific_gates_separate')


if __name__ == '__main__':
    unittest.main()
