"""Exact-output and selected-range proof for every prefill profiler replay."""
from pathlib import Path
from h200_priority_campaign import load,sha,valid_tokens
from h200_review_functional_pilots import prefix_coverage


def qualify(base,expected_run,condition,gpu_uuid):
    base=Path(base);publications=sorted(base.glob('pass[0-9][0-9][0-9].json'))
    directories={p.name for p in base.glob('pass[0-9][0-9][0-9]') if p.is_dir()}
    published={p.stem for p in publications};proofs=[]
    expected=expected_run['cases'][0]
    assert expected['batch_size']==1 and expected['generated_tokens']==64
    for path in publications:
        proof={'publication':str(path),'publication_sha256':sha(path),'passed':False}
        try:
            publication=load(path);output=Path(publication['output'])
            assert publication['status']=='complete' and publication['profile_condition']==condition
            assert output.resolve()==(base/path.stem).resolve()
            manifest_path=output/'prefill_resident.json';manifest=load(manifest_path)
            assert manifest['status']=='complete' and manifest['profile_condition']==condition
            assert manifest['protocol']=='resident_prefill_v2'
            native_path=output/condition/'run.json';native=load(native_path)
            assert native['hardware']['uuid']==gpu_uuid and valid_tokens(native,expected['trials'][0]['output_ids'])
            case=native['cases'][0]
            assert case['prompt_ids']==expected['prompt_ids'] and case['prompt_sha256']==expected['prompt_sha256']
            assert (case['batch_size'],case['context_length'],case['generated_tokens'])==(1,expected['context_length'],64)
            assert all(native['config'][key]==expected_run['config'][key]
                for key in ('model_id','model_revision','precision','seed','generated_tokens'))
            interval_path=output/condition/'prefill_interval_output.json';interval=load(interval_path)
            assert interval['prefix']==condition+'_measured0'
            assert interval['prefill']==case['trials'][0]['prefill_and_first_token']
            prefix_coverage(interval['prefill'],expected['context_length'])
            assert interval['first_selected_ids']==[[row[0]] for row in expected['trials'][0]['output_ids']]
            proof.update(passed=True,native=str(native_path),native_sha256=sha(native_path),
                manifest=str(manifest_path),manifest_sha256=sha(manifest_path),
                selected_interval=str(interval_path),selected_interval_sha256=sha(interval_path))
        except (OSError,ValueError,KeyError,AssertionError) as error:proof['error']=str(error)
        proofs.append(proof)
    return {'passed':bool(proofs) and directories==published and all(p['passed'] for p in proofs),
        'passes':proofs,'unpublished_directories':sorted(directories-published),
        'missing_output_directories':sorted(published-directories),
        'scope':'Replay correctness/publication only; profiler counter coverage, units, source/isolation/clock and clean-timing joins remain separate. Replay passes are not independent acquisitions.'}
