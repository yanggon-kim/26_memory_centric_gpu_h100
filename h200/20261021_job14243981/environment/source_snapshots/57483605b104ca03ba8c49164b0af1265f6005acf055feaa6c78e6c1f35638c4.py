"""Declared service-reference model; independent features, sealed held-out conditions."""
import hashlib
import json
import math
import statistics

CONDITIONS=[('b1_c128',1,128,'train'),('b1_c1024',1,1024,'train'),
    ('b2_c512',2,512,'train'),('b8_c4096',8,4096,'train'),
    ('b2_c2048',2,2048,'held_out'),('b4_c1536',4,1536,'held_out')]
THRESHOLD=.10


def full_decode_publication(p,reference,batch):
    return (p.get('full_decode_calibration') is True and p.get('decode_iterations')==63 and
        p.get('timed_generated_tokens')==batch*63 and p.get('prompt_sha256')==reference['prompt_sha256'] and
        p.get('output_ids')==reference['trials'][0]['output_ids'] and
        (p.get('post_prefill_conditioning') or {}).get('protocol')=='memory_settle_v3' and
        p.get('prefix','').endswith('_measured0') and p.get('prefill_barrier',{}).get('passed') is True)


def roles(config,batch,context):
    h=config['hidden_size'];i=config['intermediate_size'];hq=config['num_attention_heads']
    d=config.get('head_dim') or h//hq;kv=config.get('num_key_value_heads',hq)*d
    layers=config['num_hidden_layers'];mean_length=context+32
    # Useful work only. Full63-step sums; no issued FLOPs or physical busy-time claim.
    return [('qkv',batch,h,h+2*kv,63*layers),('o',batch,h,h,63*layers),
        ('gate_up',batch,h,2*i,63*layers),('down',batch,i,h,63*layers),
        ('lm_head',batch,h,config['vocab_size'],63),
        ('attention_qk',batch*hq,d,mean_length,63*layers),
        ('attention_av',batch*hq,mean_length,d,63*layers)]


def design_identity():
    return hashlib.sha256(json.dumps({'conditions':CONDITIONS,'threshold':THRESHOLD,
        'model':'T=C_reference+M_reference-eta*min(C_reference,M_reference)+O',
        'main_reference':'tile1024 compute and streaming physical read/write service',
        'uncertainty_scenarios':'tile512; paged128 memory; not statistical confidence intervals',
        'eta_grid':[0,1,1001],'residual':'one shared nonnegative residual',
        'bootstrap_draws':500,'bootstrap_seed':20261013,'eta_interval_width_diagnostic':.5,
        'acquisitions':3,'trials':10,'tokens':64},sort_keys=True).encode()).hexdigest()


def fit(training):
    expected={c[0] for c in CONDITIONS if c[3]=='train'}
    assert {r['id'] for r in training}==expected,'exact declared training set required; no held-out leakage'
    assert all(r['qualified'] for r in training),'training qualification failed'
    assert all(math.isfinite(r[k]) and r[k]>0 for r in training for k in ('C_ms','M_ms','T_ms'))
    best=None
    for index in range(1001):
        eta=index/1000
        residual=max(0,statistics.mean(r['T_ms']-r['C_ms']-r['M_ms']+eta*min(r['C_ms'],r['M_ms']) for r in training))
        errors=[r['C_ms']+r['M_ms']-eta*min(r['C_ms'],r['M_ms'])+residual-r['T_ms'] for r in training]
        loss=sum(e*e for e in errors)/len(errors)
        if best is None or loss<best[0]:best=(loss,eta,residual)
    residual,beta=best[2],best[1]
    rows=[{'id':r['id'],'relative_error':abs(predict({'eta':beta,'O_ms':residual},r)-r['T_ms'])/r['T_ms']} for r in training]
    return {'eta':beta,'O_ms':residual,'training_errors':rows,'design_identity':design_identity(),
        'training_gate':all(r['relative_error']<=THRESHOLD for r in rows),
        'classification':'empirical service-reference model; coefficients are not measured physical overlap',
        'parameter_identifiability':'prediction accuracy alone does not identify eta or physical compute/memory fractions'}


def predict(model,row):
    c,m=row['C_ms'],row['M_ms']
    assert all(math.isfinite(x) and x>0 for x in (c,m))
    return c+m-model['eta']*min(c,m)+model['O_ms']


def validate(model,held,freeze_sha256):
    assert freeze_sha256 and model['design_identity']==design_identity()
    expected={c[0] for c in CONDITIONS if c[3]=='held_out'}
    assert {r['id'] for r in held}==expected,'all predeclared held-out conditions required'
    output=[]
    for row in held:
        p=predict(model,row)
        output.append({'id':row['id'],'qualified':row['qualified'],'predicted_ms':p,
            'measured_ms':row['T_ms'],'relative_error':abs(p-row['T_ms'])/row['T_ms']})
    passed=model['training_gate'] and all(r['qualified'] and r['relative_error']<=THRESHOLD for r in output)
    return {'status':'validated_for_declared_conditioned_graph_family' if passed else 'rejected',
        'held_out':output,'frozen_model_sha256':freeze_sha256,'threshold':THRESHOLD,
        'scope':'useful-work service-reference time predictions only; no exact physical wall-time decomposition or architecture extrapolation'}


def resampled_center(values,rng):
    import numpy as np
    assert len(values)==3 and all(len(a)==10 for a in values)
    chosen=rng.integers(3,size=3)
    return float(np.median([np.median(np.asarray(values[i])[rng.integers(10,size=10)]) for i in chosen]))


def parameter_uncertainty(training,draws=500):
    import numpy as np
    rng=np.random.default_rng(20261013);grid=np.linspace(0,1,1001);parameters=[]
    for _ in range(draws):
        c=np.array([resampled_center(r['C_acquisitions'],rng) for r in training])
        m=np.array([resampled_center(r['M_acquisitions'],rng) for r in training])
        t=np.array([resampled_center(r['T_acquisitions'],rng) for r in training])
        overlap=np.minimum(c,m)[:,None]*grid
        residual=np.maximum(0,(t[:,None]-c[:,None]-m[:,None]+overlap).mean(axis=0))
        errors=c[:,None]+m[:,None]-overlap+residual-t[:,None]
        i=int(np.argmin((errors**2).mean(axis=0)))
        parameters.append({'eta':float(grid[i]),'O_ms':float(residual[i])})
    eta=np.quantile([p['eta'] for p in parameters],[.025,.975]).tolist()
    return {'draws':parameters,'eta_ci95':eta,'O_ms_ci95':np.quantile([p['O_ms'] for p in parameters],[.025,.975]).tolist(),
        'eta_identifiability_diagnostic':eta[1]-eta[0]<=.5,
        'scope':'500 hierarchical acquisition/trial resamples of derived independent-reference and decoder summaries; useful-work reference assumptions are separate, not covered by these intervals'}
