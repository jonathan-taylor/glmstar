from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse
import pytest

from glmnet.paths import CoxNet
from sklearn.model_selection import KFold

from glmnet.cox import CoxFamily, CoxCIndexScorer, c_index, cox_survfit

rng = np.random.default_rng(0)

def get_RCoxNet(Rinfo):
    RGLMNet = Rinfo["RGLMNet"]
    @dataclass
    class RCoxNet(RGLMNet):
        family: str = '"cox"'
        ties: str = 'breslow'
        def __post_init__(self):
            super().__post_init__()
            # Cox has no intercept; R warns if it is passed
            del self.args['intercept']
            self.args['cox.ties'] = f'"{self.ties}"'
    return RCoxNet

def get_glmnet_cox_soln(Rinfo,
                        X,
                        D,
                        start=False,
                        strata=False,
                        sparse=False,
                        extra={},
                        **args):
    """Fit R's glmnet(family="cox"); return lambda, coefs (n_lambda, p), dev.ratio.

    `extra` holds R arguments not handled by RGLMNet, e.g. alpha or lambda.
    """
    rpy = Rinfo["rpy"]
    np_cv_rules = Rinfo["np_cv_rules"]

    parser = get_RCoxNet(Rinfo)(**args)
    args, _ = parser.parse()

    with np_cv_rules.context():
        for k, v in extra.items():
            rpy.r.assign(f'extra.{k}', v)
            args += f', {k}=extra.{k}'
        rpy.r.assign('X', X)
        rpy.r.assign('stop', D['stop'].values)
        rpy.r.assign('status', D['status'].values.astype(float))
        if start:
            rpy.r.assign('start', D['start'].values)
            Y = 'Surv(start, stop, status)'
        else:
            Y = 'Surv(stop, status)'
        if strata:
            # plain str array: pandas >= 3 string columns do not convert to R directly
            rpy.r.assign('strata', np.asarray(D['strata'], dtype=str))
            Y = f'stratifySurv({Y}, strata)'
        Xr = 'Matrix(X, sparse=TRUE)' if sparse else 'X'
        cmd = f'''
suppressMessages({{library(glmnet); library(survival)}})
X = as.matrix(X)
Y = {Y}
G = glmnet({Xr}, Y, {args})
B = as.matrix(G$beta)
L = G$lambda
DR = G$dev.ratio
'''
        rpy.r(cmd)
        B = np.asarray(rpy.r('B'))
        L = np.asarray(rpy.r('L'))
        DR = np.asarray(rpy.r('DR'))
    return L, B.T, DR

def get_data(n,
             p,
             sample_weight,
             offset,
             start=False,
             strata=False):
    """Simulated survival data with tied times."""
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:3] = [1, -0.5, 0.25]
    T = rng.exponential(np.exp(-X @ beta))
    C = rng.exponential(1.5, size=n)
    D = pd.DataFrame({'stop': np.round(np.minimum(T, C), 1) + 0.2,
                      'status': (T <= C).astype(int)})
    family = {'event_id':'stop',
              'status_id':'status'}

    if start:
        D['start'] = np.round(rng.uniform(0, 0.15, size=n), 2) * (rng.uniform(size=n) < 0.5)
        family['start_id'] = 'start'

    if strata:
        D['strata'] = rng.choice(['a', 'b', 'c'], size=n)
        family['strata_id'] = 'strata'

    if offset is not None:
        offset = offset(n)
        offset_id = 'offset'
        D['offset'] = offset
        offsetR = offset
    else:
        offset_id = None
        offsetR = None
    if sample_weight is not None:
        sample_weight = sample_weight(n)
        weight_id = 'weight'
        D['weight'] = sample_weight
        weightsR = sample_weight
    else:
        weight_id = None
        weightsR = None

    col_args = {'weight_id':weight_id,
                'offset_id':offset_id}
    return X, D, family, col_args, weightsR, offsetR

def check_soln(L, R_soln):
    R_lambda, R_coefs, R_dev = R_soln

    assert L.lambda_values_.shape == R_lambda.shape
    assert np.allclose(L.lambda_values_, R_lambda, rtol=1e-8, atol=0)
    assert np.linalg.norm(R_coefs - L.coefs_) / max(np.linalg.norm(L.coefs_), 1) < 1e-8
    assert np.allclose(L.summary_['Fraction Deviance Explained'], R_dev, rtol=0, atol=1e-8)
    assert np.all(L.intercepts_ == 0)

@pytest.mark.parametrize('ties', ['breslow', 'efron'])
@pytest.mark.parametrize('start', [False, True])
def test_coxnet(Rinfo,
                standardize,
                n,
                p,
                sample_weight,
                offset,
                ties,
                start):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D, family, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset, start=start)

    L = CoxNet(family=CoxFamily(tie_breaking=ties, **family),
               standardize=standardize,
               **col_args)
    L.fit(X, D)

    R_soln = get_glmnet_cox_soln(Rinfo,
                                 X,
                                 D,
                                 start=start,
                                 ties=ties,
                                 weights=weightsR,
                                 offset=offsetR,
                                 standardize=standardize)
    check_soln(L, R_soln)

@pytest.mark.parametrize('ties', ['breslow', 'efron'])
@pytest.mark.parametrize('start', [False, True])
def test_coxnet_strata(Rinfo,
                       n,
                       p,
                       sample_weight,
                       ties,
                       start):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D, family, col_args, weightsR, offsetR = get_data(n, p, sample_weight, None,
                                                         start=start, strata=True)

    L = CoxNet(family=CoxFamily(tie_breaking=ties, **family),
               **col_args)
    L.fit(X, D)

    R_soln = get_glmnet_cox_soln(Rinfo,
                                 X,
                                 D,
                                 start=start,
                                 strata=True,
                                 ties=ties,
                                 weights=weightsR)
    check_soln(L, R_soln)

@pytest.mark.parametrize('ties', ['breslow', 'efron'])
def test_coxnet_sparse(Rinfo,
                       standardize,
                       n,
                       p,
                       sample_weight,
                       offset,
                       ties):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D, family, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset, start=True)
    X[X < 0.5] = 0

    L = CoxNet(family=CoxFamily(tie_breaking=ties, **family),
               standardize=standardize,
               **col_args)
    L.fit(scipy.sparse.csc_matrix(X), D)

    R_soln = get_glmnet_cox_soln(Rinfo,
                                 X,
                                 D,
                                 start=True,
                                 sparse=True,
                                 ties=ties,
                                 weights=weightsR,
                                 offset=offsetR,
                                 standardize=standardize)
    check_soln(L, R_soln)

    # sparse and dense paths agree up to the solver's convergence threshold.
    # Only checked for n > p: the two solvers standardize differently, and for p > n
    # the nearly saturated end of the path can pick different active sets (each still
    # matches R, which runs the same C++ code).
    if n > p:
        L_dense = CoxNet(family=CoxFamily(tie_breaking=ties, **family),
                         standardize=standardize,
                         **col_args).fit(X, D)
        assert L.coefs_.shape == L_dense.coefs_.shape
        assert np.linalg.norm(L.coefs_ - L_dense.coefs_) / max(np.linalg.norm(L_dense.coefs_), 1) < 1e-6

def test_coxnet_args(Rinfo,
                     alpha,
                     penalty_factor,
                     df_max,
                     exclude,
                     lower_limits,
                     nlambda,
                     lambda_min_ratio,
                     n=100,
                     p=10):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, D, family, col_args, weightsR, offsetR = get_data(n, p, None, None)

    L = CoxNet(family=CoxFamily(tie_breaking='efron', **family),
               alpha=alpha,
               penalty_factor=penalty_factor,
               df_max=df_max,
               exclude=exclude,
               lower_limits=lower_limits,
               lambda_min_ratio=lambda_min_ratio,
               **col_args)
    if nlambda is not None:
        L.nlambda = nlambda
    L.fit(X, D)

    R_args = {'ties':'efron',
              'df_max':df_max,
              'exclude':exclude,
              'lambda_min_ratio':lambda_min_ratio,
              'nlambda':nlambda,
              'penalty_factor':penalty_factor}
    if lower_limits != -np.inf:
        R_args['lower_limits'] = lower_limits

    R_soln = get_glmnet_cox_soln(Rinfo, X, D, extra={'alpha':alpha}, **R_args)
    check_soln(L, R_soln)

def test_coxnet_lambda_values(Rinfo, n=100, p=10):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D, family, col_args, _, _ = get_data(n, p, None, None)
    lambda_values = np.exp(np.linspace(np.log(0.2), np.log(0.005), 30))

    L = CoxNet(family=CoxFamily(tie_breaking='breslow', **family),
               lambda_values=lambda_values)
    L.fit(X, D)

    R_soln = get_glmnet_cox_soln(Rinfo, X, D, ties='breslow',
                                 extra={'lambda':lambda_values})
    check_soln(L, R_soln)

def test_coxnet_predict(n=100, p=10):

    X, D, family, col_args, _, _ = get_data(n, p, None, None)
    L = CoxNet(family=CoxFamily(**family)).fit(X, D)

    P = L.predict(X)
    nfit = L.coefs_.shape[0]
    assert P.shape == (n, L.nlambda)
    assert np.allclose(P[:, :nfit], X @ L.coefs_.T)

def test_coxnet_input_validation(n=50, p=5):

    X, D, family, col_args, _, _ = get_data(n, p, None, None)

    with pytest.raises(ValueError, match='censored'):
        CoxNet(family=CoxFamily(**family)).fit(X, D.assign(status=0))
    with pytest.raises(ValueError, match='non-positive event times'):
        CoxNet(family=CoxFamily(**family)).fit(X, D.assign(stop=-1.))
    with pytest.raises(ValueError, match='binary'):
        CoxNet(family=CoxFamily(**family)).fit(X, D.assign(status=2))
    with pytest.raises(ValueError, match='DataFrame'):
        CoxNet(family=CoxFamily(**family)).fit(X, D.values)
    with pytest.raises(ValueError, match='intercept'):
        CoxNet(family=CoxFamily(**family), fit_intercept=True).fit(X, D)

@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('strata', [False, True])
def test_c_index(Rinfo, start, strata, n=150):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo["rpy"]
    np_cv_rules = Rinfo["np_cv_rules"]

    X, D, family, _, _, _ = get_data(n, 3, None, None, start=start, strata=strata)
    # rounded so that predictions have ties
    pred = np.round(X[:, 0], 1)
    W = rng.uniform(0.2, 3, size=n)

    with np_cv_rules.context():
        rpy.r.assign('pred', pred)
        rpy.r.assign('W', W)
        rpy.r.assign('stop', D['stop'].values)
        rpy.r.assign('status', D['status'].values.astype(float))
        Y = 'Surv(stop, status)'
        if start:
            rpy.r.assign('start', D['start'].values)
            Y = 'Surv(start, stop, status)'
        if strata:
            rpy.r.assign('strata', np.asarray(D['strata'], dtype=str))
        rpy.r(f'''
suppressMessages({{library(glmnet); library(survival)}})
Y = {Y}
C = Cindex(pred, Y)
C_w = Cindex(pred, Y, weights=W)
''')
        C_R, C_w_R = np.asarray(rpy.r('C'))[0], np.asarray(rpy.r('C_w'))[0]
        if strata:
            C_strat_R = np.asarray(rpy.r('concordance(Y ~ I(-pred) + strata(strata), weights=W)$concordance'))[0]

    S = D['start'] if start else None
    assert np.allclose(c_index(pred, D['stop'], D['status'], start=S), C_R, rtol=0, atol=1e-12)
    assert np.allclose(c_index(pred, D['stop'], D['status'], start=S, sample_weight=W), C_w_R, rtol=0, atol=1e-12)
    if strata:
        assert np.allclose(c_index(pred, D['stop'], D['status'], start=S, strata=D['strata'], sample_weight=W),
                           C_strat_R, rtol=0, atol=1e-12)

    # one value per column
    P = np.column_stack([pred, -pred, np.zeros(n)])
    C = c_index(P, D['stop'], D['status'], start=S)
    assert np.allclose(C, [c_index(P[:, j], D['stop'], D['status'], start=S) for j in range(3)])
    assert np.allclose(C[0] + C[1], 1) and C[2] == 0.5

@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('strata', [False, True])
def test_coxnet_cv_c_index(Rinfo, sample_weight, offset, start, strata, n=200, p=10):
    """Compare C-index cross-validation to R's cv.glmnet(type.measure="C")."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo["rpy"]
    np_cv_rules = Rinfo["np_cv_rules"]

    X, D, family, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset,
                                                         start=start, strata=strata)
    fam = CoxFamily(tie_breaking='breslow', **family)
    L = CoxNet(family=fam, **col_args).fit(X, D)

    cv = KFold(5, shuffle=True, random_state=0)
    foldid = np.empty(n, int)
    for i, (_, test) in enumerate(cv.split(X)):
        foldid[test] = i + 1

    scorer = CoxCIndexScorer.from_family(fam)
    L.cross_validation_path(X, D, cv=cv, scorers=[scorer])
    scores = L.score_path_.scores

    with np_cv_rules.context():
        rpy.r.assign('X', X)
        rpy.r.assign('foldid', foldid)
        rpy.r.assign('stop', D['stop'].values)
        rpy.r.assign('status', D['status'].values.astype(float))
        Y = 'Surv(stop, status)'
        if start:
            rpy.r.assign('start', D['start'].values)
            Y = 'Surv(start, stop, status)'
        if strata:
            rpy.r.assign('strata', np.asarray(D['strata'], dtype=str))
            Y = f'stratifySurv({Y}, strata)'
        rpy.r('rm(list=intersect(c("W", "O"), ls()))')
        args = ''
        if weightsR is not None:
            rpy.r.assign('W', weightsR)
            args += ', weights=W'
        if offsetR is not None:
            rpy.r.assign('O', offsetR)
            args += ', offset=O'
        rpy.r(f'''
suppressMessages({{library(glmnet); library(survival)}})
Y = {Y}
G = cv.glmnet(X, Y, family="cox", cox.ties="breslow", type.measure="C", foldid=foldid,
              keep=TRUE{args})
L = G$lambda
if (exists("O")) {{
  # R scores the C index on X beta, dropping the offset (buildPredmat.coxnetlist);
  # we include it, so redo R's computation (cv.coxnet, cv.glmnet.raw) on X beta + offset
  W = if (exists("W")) W else rep(1, nrow(X))
  cvstuff = glmnet:::cv.coxnet(G$fit.preval + as.vector(O), Y, "C", W, foldid, grouped=TRUE)
  CVM = apply(cvstuff$cvraw, 2, weighted.mean, w=cvstuff$weights, na.rm=TRUE)
  CVSD = sqrt(apply(scale(cvstuff$cvraw, CVM, FALSE)^2, 2, weighted.mean,
                    w=cvstuff$weights, na.rm=TRUE) / (cvstuff$N - 1))
  IDX = glmnet:::getOptcv.glmnet(L, CVM, CVSD, "C-index")$index
}} else {{
  CVM = G$cvm
  CVSD = G$cvsd
  IDX = G$index
}}
IMIN = IDX[1]
I1SE = IDX[2]
''')
        rpy.r('rm(list=intersect(c("W", "O"), ls()))')
        R_cvm, R_cvsd, R_lambda = [np.asarray(rpy.r(v)) for v in ['CVM', 'CVSD', 'L']]
        R_min, R_1se = [int(np.asarray(rpy.r(v))[0]) - 1 for v in ['IMIN', 'I1SE']]

    assert np.allclose(L.lambda_values_, R_lambda, rtol=1e-8, atol=0)
    assert np.allclose(scores['C-index'], R_cvm, rtol=0, atol=1e-6)
    assert np.allclose(scores['SD(C-index)'], R_cvsd, rtol=0, atol=1e-6)
    path = L.score_path_
    assert path.index_best['C-index'] == L.lambda_values_[R_min]
    assert path.index_1se['C-index'] == L.lambda_values_[R_1se]

def _c_index_brute_force(pred, time, status, start, strata, w):
    """O(n^2) C index from the definition, as a check on c_index."""
    num = np.zeros(pred.shape[1])
    den = 0.
    status = status.astype(bool)
    for s in np.unique(strata):
        m = strata == s
        t_s, d_s, start_s, w_s, p_s = time[m], status[m], start[m], w[m], pred[m]
        for t in np.unique(t_s[d_s]):
            dead = d_s & (t_s == t)
            risk = (start_s < t) & ((t_s > t) | ((t_s == t) & ~d_s))
            sign = np.sign(p_s[dead][:, None, :] - p_s[risk][None, :, :])
            num += 0.5 * (w_s[dead].sum() * w_s[risk].sum() +
                          np.einsum('i,j,ijl->l', w_s[dead], w_s[risk], sign))
            den += w_s[dead].sum() * w_s[risk].sum()
    return num / den if den > 0 else np.full(pred.shape[1], np.nan)

@pytest.mark.parametrize('seed', range(40))
def test_c_index_brute_force(seed):

    rng_ = np.random.default_rng(seed)
    n = int(rng_.integers(2, 80))
    # few distinct times and rounded predictions, so that both have ties
    time = rng_.integers(1, 10, n).astype(float)
    status = rng_.binomial(1, 0.6, n)
    start = np.minimum(np.floor(time * rng_.uniform(0, 1, n)), time - 1)
    strata = rng_.integers(0, 3, n)
    w = rng_.uniform(0, 2, n) * (rng_.uniform(size=n) > 0.1)
    P = np.round(rng_.standard_normal((n, 4)), 1)

    for S, G, W in [(None, None, None), (start, None, w), (None, strata, w), (start, strata, w)]:
        C = c_index(P, time, status, start=S, strata=G, sample_weight=W)
        C_bf = _c_index_brute_force(P, time, status,
                                    np.full(n, -np.inf) if S is None else S,
                                    np.zeros(n) if G is None else G,
                                    np.ones(n) if W is None else W)
        assert np.allclose(C, C_bf, rtol=0, atol=1e-12, equal_nan=True)

def _flatten_curves(curves):
    """Stack each curve's non-NaN rows, as R lays out one curve per subject."""
    H = curves.cumhaz
    return np.concatenate([H[~np.isnan(H[:, j]), j] for j in range(H.shape[1])])

@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('strata', [False, True])
@pytest.mark.parametrize('new', [False, True])
def test_coxnet_survfit(Rinfo, sample_weight, offset, start, strata, new, n=120, p=5):
    """Compare CoxNet.survfit to R's survfit.coxnet."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo["rpy"]
    np_cv_rules = Rinfo["np_cv_rules"]

    X, D, family, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset,
                                                         start=start, strata=strata)
    # R's survfit.coxnet always uses Efron's hazard estimate
    L = CoxNet(family=CoxFamily(tie_breaking='efron', **family), **col_args).fit(X, D)
    lam = L.lambda_values_[15]

    newX = X[:7] + 0.5
    new_offset = rng.standard_normal(7) if offsetR is not None else None
    new_strata = np.asarray(D['strata'])[::-1][:7] if strata else None

    if new:
        curves = L.survfit(X, D, lambda_val=lam, newX=newX, new_offset=new_offset,
                           new_strata=new_strata)
    else:
        curves = L.survfit(X, D, lambda_val=lam)

    with np_cv_rules.context():
        rpy.r('rm(list=intersect(c("W", "O", "newO", "newS"), ls()))')
        rpy.r.assign('X', X)
        rpy.r.assign('lam', lam)
        rpy.r.assign('newX', newX)
        rpy.r.assign('stop', D['stop'].values)
        rpy.r.assign('status', D['status'].values.astype(float))
        Y = 'Surv(stop, status)'
        if start:
            rpy.r.assign('start', D['start'].values)
            Y = 'Surv(start, stop, status)'
        if strata:
            rpy.r.assign('strata', np.asarray(D['strata'], dtype=str))
            Y = f'stratifySurv({Y}, strata)'
        args = ''
        if weightsR is not None:
            rpy.r.assign('W', weightsR)
            args += ', weights=as.vector(W)'
        if offsetR is not None:
            rpy.r.assign('O', offsetR)
            args += ', offset=as.vector(O)'
        new_args = ''
        if new:
            new_args = ', newx=newX'
            if new_offset is not None:
                rpy.r.assign('newO', new_offset)
                new_args += ', newoffset=as.vector(newO)'
            if strata:
                rpy.r.assign('newS', np.asarray(new_strata, dtype=str))
                new_args += ', newstrata=newS'
        rpy.r(f'''
suppressMessages({{library(glmnet); library(survival)}})
Y = {Y}
G = glmnet(X, Y, family="cox", cox.ties="efron"{args})
S = survfit(G, s=lam, x=X, y=Y{args}{new_args})
''')
        R = {k: np.asarray(rpy.r(f'as.numeric(S${k})'))
             for k in ['time', 'n.risk', 'n.event', 'n.censor', 'cumhaz']}

    assert np.allclose(L.lambda_values_[15], lam)
    if new and strata:
        # one curve per new subject, each on the times of its stratum
        assert np.allclose(_flatten_curves(curves), R['cumhaz'], rtol=1e-6, atol=1e-8)
    else:
        assert np.allclose(curves.time, R['time'])
        assert np.allclose(curves.n_risk, R['n.risk'])
        assert np.allclose(curves.n_event, R['n.event'])
        assert np.allclose(curves.n_censor, R['n.censor'])
        # R stores the cumhaz matrix column-major
        assert np.allclose(curves.cumhaz.T.reshape(-1), R['cumhaz'], rtol=1e-6, atol=1e-8)

@pytest.mark.parametrize('ties', ['breslow', 'efron'])
def test_cox_survfit_ties(Rinfo, ties, n=80):
    """Compare cox_survfit to survival::survfit.coxph for both tie methods."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo["rpy"]
    np_cv_rules = Rinfo["np_cv_rules"]

    rng_ = np.random.default_rng(5)
    stop = rng_.integers(1, 10, n).astype(float)
    status = rng_.binomial(1, 0.7, n)
    lp = rng_.standard_normal(n)
    W = rng_.uniform(0.5, 2, n)
    new_lp = np.array([-1., 0., 0.7])

    curves = cox_survfit(lp, stop, status, sample_weight=W, new_linear_predictor=new_lp,
                         tie_breaking=ties)
    default = cox_survfit(lp, stop, status, sample_weight=W, tie_breaking=ties)

    with np_cv_rules.context():
        for k, v in dict(stop=stop, status=status.astype(float), lp=lp, W=W, new_lp=new_lp).items():
            rpy.r.assign(k, v)
        rpy.r(f'''
suppressMessages(library(survival))
d = data.frame(stop=as.vector(stop), status=as.vector(status), lp=as.vector(lp))
F = coxph(Surv(stop, status) ~ lp, data=d, weights=as.vector(W), init=1, iter=0, ties="{ties}")
S = survfit(F, newdata=data.frame(lp=as.vector(new_lp)), se.fit=FALSE)
S0 = survfit(F, se.fit=FALSE)
''')
        H = np.asarray(rpy.r('as.numeric(S$cumhaz)'))
        H0 = np.asarray(rpy.r('as.numeric(S0$cumhaz)'))
        surv = np.asarray(rpy.r('as.numeric(S$surv)'))

    assert np.allclose(curves.cumhaz.T.reshape(-1), H, rtol=1e-8, atol=1e-10)
    assert np.allclose(curves.surv.T.reshape(-1), surv, rtol=1e-8, atol=1e-10)
    assert np.allclose(default.cumhaz[:, 0], H0, rtol=1e-8, atol=1e-10)

    # step-function evaluation
    P = curves.predict([0.5, curves.time[2], curves.time[2] + 0.5, 100])
    assert np.allclose(P[0], 1)
    assert np.allclose(P[1], curves.surv[2])
    assert np.allclose(P[2], curves.surv[2])
    assert np.allclose(P[3], curves.surv[-1])

def test_coxnet_survfit_args(n=60, p=4):

    X, D, family, _, _, _ = get_data(n, p, None, lambda n: rng.standard_normal(n), strata=True)
    L = CoxNet(family=CoxFamily(**family), offset_id='offset').fit(X, D)

    curves = L.survfit(X, D, lambda_val=L.lambda_values_[[3, 10]])
    assert isinstance(curves, list) and len(curves) == 2
    assert np.all(np.diff(curves[1].cumhaz[curves[1].strata == 'a', 0]) >= 0)
    assert len(L.survfit(X, D)) == L.lambda_values_.shape[0]

    with pytest.raises(ValueError, match='new_offset'):
        L.survfit(X, D, lambda_val=L.lambda_values_[3], newX=X[:2], new_strata=['a', 'b'])
    with pytest.raises(ValueError, match='new_strata'):
        L.survfit(X, D, lambda_val=L.lambda_values_[3], newX=X[:2], new_offset=[0, 0])
    with pytest.raises(ValueError, match='not seen'):
        L.survfit(X, D, lambda_val=L.lambda_values_[3], newX=X[:2], new_offset=[0, 0],
                  new_strata=['a', 'z'])
