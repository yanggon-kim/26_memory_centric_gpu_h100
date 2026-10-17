"""Acquisition-aware confidence intervals and coupled service-equivalent bounds."""
import math


def hierarchical_interval(acquisitions, seed=20261013, draws=10000):
    import numpy as np
    assert len(acquisitions)>=3 and all(len(a)>=10 for a in acquisitions), 'three acquisitions/ten trials required'
    assert all(math.isfinite(x) and x>0 for a in acquisitions for x in a)
    rng=np.random.default_rng(seed)
    samples=np.empty(draws)
    for i in range(draws):
        chosen=rng.integers(len(acquisitions),size=len(acquisitions))
        medians=[]
        for j in chosen:
            a=np.asarray(acquisitions[j]);medians.append(np.median(a[rng.integers(len(a),size=len(a))]))
        samples[i]=np.median(medians)
    return {'value':float(np.median([np.median(a) for a in acquisitions])),
            'ci95':[float(x) for x in np.quantile(samples,[.025,.975])],
            'bootstrap_draws':draws,'seed':seed,'statistic':'median of acquisition medians',
            'resampling':'acquisitions then trials within chosen acquisition'}


def paired_ratio(numerator,denominator,seed=20261013,draws=10000):
    import numpy as np
    assert len(numerator)==len(denominator)>=3
    assert all(len(a)>=10 for a in numerator+denominator)
    assert all(math.isfinite(x) and x>0 for a in numerator+denominator for x in a)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        ratios=[]
        for j in rng.integers(len(numerator),size=len(numerator)):
            a,b=np.asarray(numerator[j]),np.asarray(denominator[j])
            ratios.append(np.median(a[rng.integers(len(a),size=len(a))])/np.median(b[rng.integers(len(b),size=len(b))]))
        values.append(float(np.exp(np.mean(np.log(ratios)))))
    ratio=float(np.exp(np.mean(np.log([np.median(a)/np.median(b) for a,b in zip(numerator,denominator)]))))
    interval=[float(x) for x in np.quantile(values,[.025,.975])]
    return {'value':ratio,'ci95':interval,'upper_95_latency_regression':interval[1]-1,
            'passes_5_percent_noninferiority':interval[1]<=1.05,
            'bootstrap_draws':draws,'seed':seed,'statistic':'geometric mean of paired acquisition median ratios'}


def latency_bounds(t,c,m,o_low=0,o_high=math.inf):
    assert all(math.isfinite(x) and x>=0 for x in (t,c,m,o_low))
    assert o_high>=o_low
    low=max(o_low,t-c-m);high=min(o_high,t-max(c,m))
    if low>high:
        return {'feasible':False,'reason':'empty coupled residual interval; reject rates/model','T':t,'C':c,'M':m}
    endpoints=[]
    for o in (low,high):
        x=c+m+o-t
        endpoints.append({'O':o,'X':max(0,x),'compute_only':c-x,'memory_only':m-x,
                          'eta':x/min(c,m) if min(c,m)>0 else 0})
    return {'feasible':True,'T':t,'C':c,'M':m,'O_interval':[low,high],
            'coupled_endpoints':endpoints,'classification':'analytical-service-equivalent bounds; not physical busy times'}


def sensitivity(bounds,c_fraction,r,f):
    assert bounds['feasible'] and c_fraction>0 and r>0 and 0<=f<=1
    cn=bounds['C']/c_fraction;mn=bounds['M']*(1-f)/r
    same=[cn+mn-e['eta']*min(cn,mn)+e['O'] for e in bounds['coupled_endpoints']]
    low,high=bounds['O_interval']
    return {'c':c_fraction,'r':r,'f':f,'C_new':cn,'M_new':mn,
            'same_residual_same_eta_time':[min(same),max(same)],
            'relaxed_overlap_time':[max(cn,mn)+low,cn+mn+high],
            'memory_service_improves':f>1-r,'classification':'conditional analytical scenario; physical feasibility/energy unknown'}
