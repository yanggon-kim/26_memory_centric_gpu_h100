"""Hash-checked historical output references; never an automatic timing join."""
import hashlib
import json
from pathlib import Path


def output_reference(root, key):
    registry = Path(root) / 'results/evidence_reuse.json'
    if not registry.exists():
        return None
    entries = [e for e in json.loads(registry.read_text())['entries'] if e['key'] == key]
    if not entries:
        return None
    assert len(entries) == 1, 'ambiguous historical reference'
    entry = entries[0]
    assert entry['role'] == 'exact-output-reference-only'
    assert entry['clock_timing_join_permitted'] is False
    package = Path(__file__).resolve().parents[1]
    session = (package / entry['original_session']).resolve()
    assert session.is_relative_to(package / 'h200'), 'reference outside H200 evidence'
    assert session != Path(root).resolve(), 'historical registry cannot alias current session'
    for field, digest in [('artifact', 'sha256'), ('qualification_artifact', 'qualification_sha256')]:
        path = (session / entry[field]).resolve()
        assert path.is_relative_to(session), 'historical path escape'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry[digest], 'historical evidence hash mismatch'
    artifact = session / entry['artifact']
    data = json.loads(artifact.read_text())
    assert data['status'] == 'complete' and len(data['cases']) == 1
    case = data['cases'][0]
    assert case['output_digests_identical'] and case['trials']
    assert all(t['output_ids'] == case['trials'][0]['output_ids'] for t in case['trials']), 'historical outputs differ'
    return artifact
