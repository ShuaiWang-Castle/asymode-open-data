"""D06: frozen public-D reachability, objective and oracle diagnostics; no training."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from evaluate_cr_tail import (ROOT, HERE, RUNS, FEATURES, REGIMES, sha, clean,
                              load_outcomes, load_predictions, cohorts, bootstrap_plan, write_new)

OUT = RUNS / 'd06_host_interface_20260929'
RESULT = HERE / 'results/v1/d06_response_audit.json'
DATA_SHA = 'f043bb39e8abd48183670e2acc3c0cecebd7107ea9be0e28771912cfee358c48'


def constraints(n, y0, rcap):
    """A p <= b. Box [0,1] is separately included in primal and dual."""
    a, b = np.zeros((2*n, n)), np.zeros(2*n)
    for t in range(n):
        a[t, t], a[n+t, t], b[n+t] = -1, 1, .515
        if t:
            a[t, t-1], a[n+t, t-1] = 1-rcap, -.485
        else:
            b[t], b[n+t] = -(1-rcap)*y0, .515+.485*y0
    return a, b


def oracle(y, m, y0, rcap):
    """Convex primal with an independently calculated nonnegative-multiplier dual."""
    a, b = constraints(len(y), y0, rcap)
    x0 = np.empty_like(y)
    prev = y0
    for t in range(len(y)):
        prev = np.clip(y[t] if m[t] else prev, (1-rcap)*prev, .515+.485*prev)
        x0[t] = prev
    fun = lambda x: float(np.sum(m*(x-y)**2))
    fit = minimize(fun, x0, jac=lambda x: 2*m*(x-y), method='SLSQP',
                   bounds=[(0., 1.)]*len(y),
                   constraints={'type': 'ineq', 'fun': lambda x: b-a@x, 'jac': lambda x: -a},
                   options={'ftol': 1e-11, 'maxiter': 300})
    # Project tiny numerical violations forward to obtain a genuinely feasible upper bound.
    x = fit.x.copy(); prev = y0
    for t in range(len(y)):
        x[t] = np.clip(x[t], (1-rcap)*prev, .515+.485*prev); prev = x[t]
    upper = fun(x)
    def dual(lam, gradient=False):
        q = a.T@lam
        v = np.where(m, np.clip(y-q/2, 0, 1), (q < 0).astype(float))
        value = fun(v) + q@v - lam@b
        return (float(-value), b-a@v) if gradient else float(value)
    lam = np.maximum(getattr(fit, 'multipliers', np.zeros(len(b))), 0)
    lower = max(0., dual(lam)); refined = False
    if upper-lower > 1e-8:
        d = minimize(lambda l: dual(l, True), lam, jac=True, method='L-BFGS-B',
                     bounds=[(0., None)]*len(lam),
                     options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-9})
        lower = max(lower, dual(np.maximum(d.x, 0))); refined = True
    assert lower <= upper + 1e-9 and np.max(a@x-b) < 1e-10
    return x, dict(lower=max(0., lower), upper=upper, gap=max(0., upper-lower),
                   success=bool(fit.success), iterations=int(fit.nit), dual_refined=refined)


def synthetic_checks():
    y = np.zeros(144); y[2] = .2; m = np.ones(144, bool)
    x, c = oracle(y, m, 0., .5)
    assert abs(c['lower']-.01) < 1e-8 and abs(x[2]-.15) < 1e-6
    _, d = oracle(y, m, 0., 1.); assert d['upper'] < 1e-10
    # Missing hours still bridge the state: cannot drop from .8 to zero after one hidden hour.
    x, c = oracle(np.array([0., 0.]), np.array([False, True]), .8, .5)
    assert abs(c['lower']-.04) < 1e-8 and abs(x[-1]-.2) < 1e-6
    # Nonzero initial state and first-hour upper cap must be enforced.
    _, c = oracle(np.array([1.]), np.array([True]), 0., .5)
    assert abs(c['lower']-.485**2) < 1e-8
    return dict(spike_projection=True, cap_sensitivity=True, missing_bridge=True, initial_upper=True)


def ratio_interval(num, den, plan):
    code, counts = plan['code'], plan['counts']
    n = np.bincount(code, weights=num, minlength=len(plan['levels']))
    d = np.bincount(code, weights=den, minlength=len(plan['levels']))
    bn, bd = counts@n, counts@d; good = bd > 0
    return dict(point=float(n.sum()/d.sum()) if d.sum() else None,
                ci95=np.quantile(bn[good]/bd[good], [.025, .975]) if good.any() else None,
                supporting_groups=int((d > 0).sum()), valid_draws=int(good.sum()))


def weighted_quantile(x, w):
    ix = np.argsort(x); s = np.cumsum(w[ix]);
    return np.interp(np.array([.1, .5, .9])*s[-1], s, x[ix])


def main():
    assert not RESULT.exists(), 'Preserve earlier result'
    tests = synthetic_checks()
    assert sha(FEATURES) == DATA_SHA
    data = load_outcomes(); y, m, w = data['y'], data['m'], data['meta']['w'].astype(float)
    n = len(y); masks = cohorts(data); plan = bootstrap_plan(data['meta'], data['group'])
    host, hr = load_predictions('v1_host_s0', 'W+Cin', data['expected'], n)
    crk, cr = load_predictions('v1_crk_s0', 'CRK+Cin', data['expected'], n)
    assert hr == json.loads((RUNS/'v1_crk_s0/HOST_REFERENCE.json').read_text())
    frozen = {}
    for k in range(1, 6):
        rec = json.loads((RUNS/f'v1_crk_s0/fold{k:02d}/RUNNER_RECEIPT.json').read_text())
        for name, digest in rec['source_sha256'].items():
            assert sha(ROOT/name) == digest, name
        for name, digest in rec['artifact_sha256'].items():
            assert sha(RUNS/f'v1_crk_s0/fold{k:02d}'/name) == digest, name
        frozen[str(k)] = rec
    # Validate exported rates and recurrence without loading or altering model weights.
    recurrence = {}
    for label in ('v1_host_s0', 'v1_crk_s0'):
        errors = []
        for k in range(1, 6):
            with np.load(RUNS/label/f'fold{k:02d}/outer.npz') as z:
                idx = z['idx']; u, r, p = z['u'].astype(float), z['r'].astype(float), z['P'].astype(float)
                assert all(np.isfinite(z[key]).all() for key in ('u','r','raw_logit'))
                assert (u >= 0).all() and (u <= .515+1e-7).all() and (r >= 0).all() and (r <= .5).all()
                prev = np.column_stack((data['y0'][idx], p[:, :-1]))
                error = float(np.max(np.abs(p-(prev+u*(1-prev)-r*prev))))
                assert error < 2e-7; errors.append(error)
        recurrence[label] = errors
    prev = data['y_full'][:, 71:215]
    pair = m & data['obs_full'][:, 71:215]
    down = np.where(pair, np.maximum(.5*prev-y, 0), 0)
    up = np.where(pair, np.maximum(y-(.515+.485*prev), 0), 0)
    strata = [('all', np.ones(n,bool))]
    strata += [('regime/'+r, data['meta']['regime']==r) for r in REGIMES]
    strata += [('fold/'+str(k), data['fold']==k) for k in range(1,6)]
    strata += [('group/'+str(g), data['group']==g) for g in np.unique(data['group'])]
    rows = []
    for cname in ('all','S','J'):
        for sname, ss in strata:
            sel = masks[cname]&ss
            if not sel.any(): continue
            row = dict(cohort=cname, stratum=sname, n=int(sel.sum()), pairs=int(pair[sel].sum()))
            for key, excess in [('lower',down), ('upper',up)]:
                for tol in (1e-7,.01):
                    bad = excess > tol; tag = f'{key}_{tol:g}'
                    row[tag] = dict(pairs=int(bad[sel].sum()), units=int(bad[sel].any(1).sum()),
                        weighted_pair_rate=float(np.sum(w[sel,None]*bad[sel])/np.sum(w[sel,None]*pair[sel])),
                        weighted_unit_rate=float(np.sum(w[sel]*bad[sel].any(1))/w[sel].sum()))
            rows.append(row)
    violation_intervals = {}
    for name in ('all','S','J'):
        ss=masks[name]
        violation_intervals[name] = {f'{key}_{tol:g}': ratio_interval(w*ss*(ex>tol).sum(1), w*ss*pair.sum(1), plan)
            for key, ex in [('lower',down),('upper',up)] for tol in (1e-7,.01)}
    # D05's maximum-jump follow-up and true total-stock violation are distinct definitions.
    jump = np.where(pair,y-prev,-np.inf); jt=jump.argmax(1); ii=np.arange(n); jval=jump[ii,jt]
    follow = masks['J'] & (jt < 143) & m[ii,np.minimum(jt+1,143)]
    post = y[ii,np.minimum(jt+1,143)]; at = y[ii,jt]
    reverse = follow & ((at-post) >= .8*jval)
    boundbad = follow & ((.5*at-post) > 1e-7)
    reversal = dict(eligible=int(follow.sum()), rapid_reversal=int(reverse.sum()),
        true_lower_violation=int(boundbad.sum()), both=int((reverse&boundbad).sum()),
        rapid_weighted=float(w[reverse].sum()/w[follow].sum()),
        true_bound_weighted=float(w[boundbad].sum()/w[follow].sum()),
        bound_given_rapid=float(w[reverse&boundbad].sum()/w[reverse].sum()))
    # Severe false alarms: count exact intersections on observed support.
    ah=np.max(np.where(m,host,-np.inf),1)>=.1; ac=np.max(np.where(m,crk,-np.inf),1)>=.1
    false_intersection = {}
    for key, sel in [('both',ah&ac),('host_only',ah&~ac),('crk_only',ac&~ah),('neither',~ah&~ac)]:
        ss=sel&masks['nonS']; false_intersection[key]=dict(n=int(ss.sum()),weighted_rate=float(w[ss].sum()/w[masks['nonS']].sum()))
    # Original FIT class denominators; no OOF residual is called a fitting residual.
    objective=[]; N=m.sum(1); zero=np.sum(m*y*y,1)
    for k in range(1,6):
        fit=np.zeros(n,bool);fit[data['split']['event'][str(k)]['dev']]=True
        Z={r:float(np.sum(w*zero*fit*(data['meta']['regime']==r))) for r in REGIMES}
        rw=w/np.array([Z[r] for r in data['meta']['regime']])
        for name in ('all','S','J','nonS'):
            ss=fit&masks[name]
            objective.append(dict(fold=k,cohort=name,n=int(ss.sum()),class_zero_SSE=Z,
                design_hour_mass_share=float(np.sum(w*N*ss)/np.sum(w*N*fit)),
                normalized_hour_mass_share=float(np.sum(rw*N*ss)/np.sum(rw*N*fit)),
                design_zero_SSE_share=float(np.sum(w*zero*ss)/np.sum(w*zero*fit)),
                normalized_zero_SSE_share=float(np.sum(rw*zero*ss)/np.sum(rw*zero*fit))))
    # Run the predeclared whole S cohort, preserve every trajectory and certificate.
    indices=np.flatnonzero(masks['S']); projections={}; certs={}; oracle_summaries={}
    host_sse=np.sum(m*(host-y)**2,1); crk_sse=np.sum(m*(crk-y)**2,1)
    truepeak=np.max(np.where(m,y,-np.inf),1)
    for cap in (.5,1.):
        paths=[]; cs=[]
        for j,i in enumerate(indices):
            path,c=oracle(y[i],m[i],data['y0'][i],cap);paths.append(path);cs.append(c)
            if j%100==0: print(f'oracle cap={cap} {j}/{len(indices)}',flush=True)
        paths=np.array(paths); tag=str(cap); projections[tag]=paths;certs[tag]=cs
        lo=np.zeros(n);hi=np.zeros(n)
        lo[indices]=[c['lower'] for c in cs];hi[indices]=[c['upper'] for c in cs]
        oracle_summaries[tag]=dict(max_gap=max(c['gap'] for c in cs),
            sum_gap=sum(c['gap'] for c in cs),failed_solver=sum(not c['success'] for c in cs),
            unresolved_gap_over_1e8=sum(c['gap']>1e-8 for c in cs),
            weighted_RMSE_bracket=[float(np.sqrt(np.sum(w*lo)/np.sum(w*N*masks['S']))),float(np.sqrt(np.sum(w*hi)/np.sum(w*N*masks['S'])))],
            fraction_of_host_SSE_lower=ratio_interval(w*lo,w*host_sse*masks['S'],plan),
            fraction_of_host_SSE_upper=ratio_interval(w*hi,w*host_sse*masks['S'],plan),
            fraction_of_crk_SSE_lower=ratio_interval(w*lo,w*crk_sse*masks['S'],plan),
            peak_ratio_q10_median_q90=weighted_quantile(np.max(np.where(m[indices],paths,-np.inf),1)/truepeak[indices],w[indices]),
            strata=[dict(stratum=name,n=int((ss&masks['S']).sum()),
                     host_SSE=float(np.sum(w*host_sse*ss*masks['S'])),
                     lower_SSE=float(np.sum(w*lo*ss)), upper_SSE=float(np.sum(w*hi*ss)))
                     for name,ss in strata if np.any(ss&masks['S'])])
    with (OUT/'oracle_paths.npz').open('xb') as stream:
        np.savez_compressed(stream,idx=indices,cap05=projections['0.5'],cap1=projections['1.0'])
    write_new(OUT/'oracle_certificates.json',certs)
    ext=json.loads((OUT/'external_manifest.json').read_text()); probes=json.loads((OUT/'probe_replay.json').read_text())
    result=dict(scope_commit='5fa930e',exploratory=True,data_sha256=DATA_SHA,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        external_archive=ext,external_replay=dict(test_count=probes['test_count'],all_passed=probes['all_passed'],sha256=sha(OUT/'probe_replay.json')),
        own_synthetic_tests=tests,recurrence_max_errors=recurrence,host_receipts=hr,crk_receipts=cr,
        frozen_source_hashes=frozen['1']['source_sha256'],frozen_crk_artifacts={k:v['artifact_sha256'] for k,v in frozen.items()},
        violations=rows,violation_cluster_intervals=violation_intervals,reversal_overlap=reversal,
        false_alarm_intersection=false_intersection,fit_objective=objective,oracle=oracle_summaries,
        actual_peak_ratio_q10_median_q90={key:weighted_quantile(np.max(np.where(m[indices],p[indices],-np.inf),1)/truepeak[indices],w[indices]) for key,p in [('host',host),('crk',crk)]},
        local_artifacts={name:sha(OUT/name) for name in ('oracle_paths.npz','oracle_certificates.json')})
    write_new(RESULT,result);print(json.dumps(clean({'oracle':oracle_summaries,'reversal':reversal,'false_alarms':false_intersection}),ensure_ascii=False))


if __name__ == '__main__': main()
