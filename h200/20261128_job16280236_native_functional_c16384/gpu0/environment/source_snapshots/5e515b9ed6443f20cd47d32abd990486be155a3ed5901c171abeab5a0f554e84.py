#!/usr/bin/env python3
"""Freeze a simple allocation-penalty null model before held-out shapes exist."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(root,destination):
    root=Path(root).resolve();destination=Path(destination).resolve()
    assert not destination.exists(),'frozen predictions must not be overwritten'
    review_path=root/'results/resident_curve_CPU_review.json'
    review=json.loads(review_path.read_text())
    assert review['status']=='qualified' and review['qualified_timing_curve']
    assert (review['batch'],review['context'])==(1,4096)
    assert review['independent_acquisitions']==3 and review['timed_trials_per_condition']==10
    curve_path=Path(review['curve']);assert sha(curve_path)==review['curve_sha256']
    curve=json.loads(curve_path.read_text())
    assert curve['complete_qualified_curve'] and curve['protocol']=='resident_eager_memory_settle_v3'
    rows=sorted(curve['points'],key=lambda p:p['actual_sm_count'])
    assert [p['actual_sm_count'] for p in rows]==[40,56,80,96,112,132]
    assert all(p['qualified'] for p in rows)
    source_path=Path(review['cases'][0]['run']);source=json.loads(source_path.read_text())
    essential=('controlled_vllm_experiments.py','h200_resident_sweep.py',
               'h200_resident_campaign.py','shoreline_green.py','decode_precondition.py',
               'software_decode_experiments.py','h200_session.py','h200_priority_campaign.py')
    assert all(name in source['script_sha256'] for name in essential)
    clock_bands={}
    for metric in ('sm_clock_mhz','memory_clock_mhz'):
        clocks=[json.loads((root/'results'/(r['id']+'.json')).read_text())['clock']['clocks'][metric]
                for r in review['cases']]
        clock_bands[metric]={'minimum':min(c['minimum'] for c in clocks),
                             'maximum':max(c['maximum'] for c in clocks)}
    model={'schema_version':1,'model_id':'allocation_penalty_null_v2',
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'classification':'analytical predictions from qualified development measurements',
        'hypothesis':'Changing batch or context preserves each tested SM-count penalty when the model, eager decoder and conditioning protocol are matched; ratios use each target device own unrestricted eager stock.',
        'formula':'predicted_ratio(SM, target_shape) = development_paired_ratio(SM); no interpolation',
        'training':{'review':str(review_path),'review_sha256':sha(review_path),
                    'curve':str(curve_path),'curve_sha256':sha(curve_path),
                    'session':review['source_session'],'session_sha256':review['source_session_sha256'],
                    'gpu_uuid':source['hardware']['uuid'],'batch':1,'context':4096,
                    'scope':curve['scope'],'fully_observed_development_data':True},
        'predictions':[{'actual_sm_count':p['actual_sm_count'],'predicted_latency_ratio':p['ratio']['value'],
                        'development_ci95_only':p['ratio']['ci95']} for p in rows],
        'prediction_intervals':None,
        'prediction_interval_reason':'Development confidence intervals do not quantify unseen-shape/device prediction uncertainty.',
        'future_confidence_threshold_predictions':None,
        'threshold_reason':'Future confidence-bound widths are unknown; do not transfer development112/80/40demonstrated counts as new measured thresholds.',
        'predeclared_targets':[{'batch':b,'context':c} for b,c in ((8,4096),(32,4096),(1,16384),(1,32704))],
        'eligibility':{'protocol':'resident_eager_memory_settle_v3',
            'conditioning':source['config']['decode_conditioning'],
            'model_id':source['config']['model_id'],'model_revision':source['config']['model_revision'],
            'precision':'BF16','seed':20261013,'generated_tokens':64,
            'runtime_versions':source['versions'],
            'essential_source_sha256':{name:source['script_sha256'][name] for name in essential},
            'physical_sm_count':132,'hardware_name':source['hardware']['name'],
            'L2_bytes':source['hardware']['l2_bytes'],'power_limit_w':source['hardware']['power_limit_w'],
            'development_clock_bands':clock_bands,
            'target_absolute_clocks_within_development_fraction':.05,
            'target_gate':'complete CPU-qualified3acquisition/10trial curve, own full stock, all actual-decoder placement/strict dualHF/source/isolation/clock gates; all six counts required',
            'temporal_gate':'prediction artifact must predate the first target numerical producer',
            'device_scope':'Each target curve is anchored on its own single physical UUID; do not pool raw timing across devices. Prediction transfer across matched hardware is itself the hypothesis.'},
        'predeclared_score':{'metric':'absolute relative error of predicted versus measured paired latency-ratio point estimates',
            'formula':'abs(predicted_ratio / measured_ratio - 1)',
            'maximum_error_tolerance':.05,'pass_rule':'every one of the six qualified tested counts has error<=0.05',
            'interpretation':'engineering prediction-accuracy criterion, not a significance test or physical-model validation',
            'ineligible_or_incomplete_target':'not judged; retain reasons and every raw observation'},
        'current_queued_batch8':'Job16238357uses the original unconditioned protocol, so it is not an eligible test of this matched conditioned predictor.',
        'already_seen_ancillary_data':'Full optimized C16384captures were already observed and clock-rejected. No restricted conditioned-eager C16384curve has been measured.',
        'scope':'Simple empirical null baseline only. No independent HBM/compute/SRAM rate calibration, causal explanation, physical area or architectural benefit is inferred. Judge on eligible already-planned cases; retain rejected predictions.',
        'script_sha256':sha(Path(__file__))}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(model,indent=2)+'\n')
    return {'frozen_prediction':str(destination),'sha256':sha(destination),'new_GPU_measurements':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args();print(json.dumps(freeze(args.root,args.destination)))
