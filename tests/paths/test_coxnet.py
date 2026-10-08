from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse
import pytest

from glmnet.paths import CoxNet
from glmnet.cox import CoxFamily

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
