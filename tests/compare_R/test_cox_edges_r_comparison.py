"""
Cox models against R (glmnet + survival) on combinations of features that
the other Cox comparisons check one at a time: (start, stop] data together
with strata, case weights, an offset and a sparse design, with tied event
times throughout.

- `CoxNet` path (sparse X, start-stop, strata, weights, offset) vs
  `glmnet(Matrix(x, sparse=TRUE), stratifySurv(Surv(start, stop, status), strata))`:
  lambda, coefficients, dev.ratio, and `predict(newx=, newoffset=)`.
- `CoxFamilySpec.deviance` vs `coxnet.deviance(pred=)` and
  `coxnet.deviance(x=, beta=, offset=)`, for both tie methods.
- `c_index` / `CoxCIndexScorer` vs `Cindex` on a stratified response
  (R's `Cindex` ignores strata).
- `cross_validation_path` deviance vs `cv.glmnet(grouped=FALSE)`.
- `CoxLM` vs `coxph(Surv(start, stop, status) ~ . + strata() + offset())`:
  coefficients, standard errors, partial log-likelihood.
"""
import numpy as np
import pandas as pd
import scipy.sparse
import pytest
from sklearn.model_selection import KFold

from glmnet.paths import CoxNet
from glmnet.paths.fastnet import FastNetControl
from glmnet.cox import CoxFamily, CoxFamilySpec, CoxLM, CoxCIndexScorer, c_index
from glmnet.glm import GLMControl

N, P = 240, 8


def get_data(seed=0, sparse=False):
    """(start, stop] data in three strata, with weights, an offset and tied times."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, P))
    if sparse:
        X[X < 0.3] = 0
    beta = np.zeros(P)
    beta[:3] = [1, -0.5, 0.25]
    T = rng.exponential(np.exp(-X @ beta))
    C = rng.exponential(1.5, size=N)
    stop = np.round(np.minimum(T, C), 1) + 0.2        # rounded: tied times
    D = pd.DataFrame({'stop': stop,
                      'status': (T <= C).astype(int),
                      'start': np.round(rng.uniform(0, 0.15, size=N), 2) * (rng.uniform(size=N) < 0.5),
                      'strata': rng.choice(['a', 'b', 'c'], size=N),
                      'weight': rng.uniform(0.3, 3, size=N),
                      'offset': 0.3 * rng.standard_normal(N)})
    # ties in event times, within strata
    assert D[D.status == 1].groupby('strata')['stop'].apply(lambda s: s.duplicated().any()).all()
    return X, D


def family(ties='breslow', strata=True):
    return CoxFamily(tie_breaking=ties,
                     event_id='stop',
                     status_id='status',
                     start_id='start',
                     strata_id='strata' if strata else None)


def assign_R(Rinfo, D, **arrays):
    """Assign the survival data (and any other arrays) as plain R vectors / matrices,
    and build `Y = stratifySurv(Surv(start, stop, status), strata)`."""
    rpy = Rinfo['rpy']
    with Rinfo['np_cv_rules'].context():
        for k in ['stop', 'start', 'weight', 'offset']:
            rpy.r.assign(k, D[k].values.astype(float))
        rpy.r.assign('status', D['status'].values.astype(float))
        # plain str array: pandas >= 3 string columns do not convert to R directly
        rpy.r.assign('strata', np.asarray(D['strata'], dtype=str))
        for k, v in arrays.items():
            rpy.r.assign(k, v)
    # plain vectors: numpy arrays arrive with a dim attribute, which breaks
    # Matrix's `[` inside cv.glmnet
    rpy.r('''suppressMessages({library(Matrix); library(glmnet); library(survival)})
for (v in c("stop", "start", "status", "weight", "offset", "strata")) assign(v, as.vector(get(v)))
Y0 = Surv(start, stop, status)
Y = stratifySurv(Y0, strata)''')


def get(Rinfo, expr):
    with Rinfo['np_cv_rules'].context():
        return np.asarray(Rinfo['rpy'].r(expr), float)


@pytest.mark.parametrize('sparse', [False, True])
def test_coxnet_start_strata_weights_offset(Rinfo, sparse):
    """The path, and predict(newx=, newoffset=), with every feature at once."""
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D = get_data(1, sparse=sparse)
    Xfit = scipy.sparse.csc_matrix(X) if sparse else X
    L = CoxNet(family=family(), weight_id='weight', offset_id='offset').fit(Xfit, D)
    assert L.family.tie_breaking == 'breslow'    # R glmnet's default cox.ties

    rng = np.random.default_rng(2)
    newX = X[:15] + 0.5 * rng.standard_normal((15, P))
    if sparse:
        newX[np.abs(newX) < 0.3] = 0
    new_offset = rng.standard_normal(15)

    assign_R(Rinfo, D, X=X, newX=newX, newO=new_offset)
    Xr, newXr = ('Matrix(X, sparse=TRUE)', 'Matrix(newX, sparse=TRUE)') if sparse else ('X', 'newX')
    Rinfo['rpy'].r(f'''newO = as.vector(newO)
G = glmnet({Xr}, Y, family="cox", cox.ties="breslow", weights=weight, offset=offset)
link = predict(G, newx={newXr}, newoffset=newO)
resp = predict(G, newx={newXr}, newoffset=newO, type="response")''')
    R_lambda = get(Rinfo, 'G$lambda')
    R_coefs = get(Rinfo, 'as.matrix(G$beta)').T
    R_dev = get(Rinfo, 'G$dev.ratio')

    assert L.lambda_values_.shape == R_lambda.shape
    np.testing.assert_allclose(L.lambda_values_, R_lambda, rtol=1e-8, atol=0)
    assert np.linalg.norm(L.coefs_ - R_coefs) / max(np.linalg.norm(L.coefs_), 1) < 1e-8
    np.testing.assert_allclose(L.summary_['Fraction Deviance Explained'], R_dev, rtol=0, atol=1e-8)

    newXp = scipy.sparse.csr_matrix(newX) if sparse else newX
    nlam = R_lambda.shape[0]
    link = L.predict(newXp, offset=new_offset)[:, :nlam]
    np.testing.assert_allclose(link, get(Rinfo, 'link'), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(L.predict(newXp, prediction_type='response', offset=new_offset)[:, :nlam],
                               get(Rinfo, 'resp'), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('ties', ['breslow', 'efron'])
@pytest.mark.parametrize('strata', [False, True])
def test_coxnet_deviance(Rinfo, ties, strata):
    """CoxFamilySpec.deviance vs coxnet.deviance(pred=) and (x=, beta=, offset=)."""
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D = get_data(3, sparse=True)
    W = D['weight'].values
    O = D['offset'].values
    # coefficients from a path, and a predictor with ties
    L = CoxNet(family=family(strata=strata), weight_id='weight', offset_id='offset').fit(X, D)
    B = L.coefs_[::10]                            # (k, p)
    pred = np.round(X[:, 0] - X[:, 1], 1)

    fam = CoxFamilySpec(event_data=D, tie_breaking=ties, event_id='stop', status_id='status',
                        start_id='start', strata_id='strata' if strata else None)
    dev = lambda eta, w: fam.deviance(None, eta, w)

    assign_R(Rinfo, D, X=X, B=B.T, pred=pred)
    y = 'Y' if strata else 'Y0'
    Rinfo['rpy'].r(f'''pred = as.vector(pred)
d_raw = coxnet.deviance(pred=pred, y={y}, weights=weight, std.weights=FALSE, cox.ties="{ties}")
d_std = coxnet.deviance(pred=pred, y={y}, weights=weight, cox.ties="{ties}")
d_unw = coxnet.deviance(pred=pred, y={y}, cox.ties="{ties}")
d_xb = coxnet.deviance(x=X, beta=B, offset=offset, y={y}, weights=weight, std.weights=FALSE, cox.ties="{ties}")
d_sp = coxnet.deviance(x=Matrix(X, sparse=TRUE), beta=B, offset=offset, y={y}, weights=weight,
                       std.weights=FALSE, cox.ties="{ties}")''')

    np.testing.assert_allclose(dev(pred, W), get(Rinfo, 'd_raw')[0], rtol=1e-10)
    # std.weights=TRUE (the default) rescales the weights to sum to n
    np.testing.assert_allclose(dev(pred, W * N / W.sum()), get(Rinfo, 'd_std')[0], rtol=1e-10)
    np.testing.assert_allclose(dev(pred, np.ones(N)), get(Rinfo, 'd_unw')[0], rtol=1e-10)

    eta = X @ B.T + O[:, None]
    ours = np.array([dev(eta[:, j], W) for j in range(B.shape[0])])
    np.testing.assert_allclose(ours, get(Rinfo, 'd_xb'), rtol=1e-10)
    np.testing.assert_allclose(ours, get(Rinfo, 'd_sp'), rtol=1e-10)

    if ties == 'breslow':
        # the path's dev.ratio is 1 - deviance / (deviance at the offset alone)
        k = np.arange(0, L.coefs_.shape[0], 10)
        np.testing.assert_allclose(1 - ours / dev(O, W),
                                   L.summary_['Fraction Deviance Explained'].values[k],
                                   rtol=0, atol=1e-8)


def test_cindex_stratified_response(Rinfo):
    """R's Cindex ignores the strata of a stratifySurv response, as c_index does by default;
    weights, start-stop data and ties in both pred and the event times."""
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D = get_data(4)
    pred = np.round(X[:, 0] - 0.5 * X[:, 1], 1)
    assert pd.Series(pred).duplicated().sum() > 50
    W = D['weight'].values

    assign_R(Rinfo, D, pred=pred)
    Rinfo['rpy'].r('''pred = as.vector(pred)
C_s = Cindex(pred, Y, weights=weight)
C_0 = Cindex(pred, Y0, weights=weight)
C_strat = concordance(Y0 ~ I(-pred) + strata(strata), weights=weight)$concordance''')
    C_s, C_0, C_strat = [get(Rinfo, v)[0] for v in ['C_s', 'C_0', 'C_strat']]

    assert C_s == C_0
    C = c_index(pred, D['stop'], D['status'], start=D['start'], sample_weight=W)
    np.testing.assert_allclose(C, C_s, rtol=0, atol=1e-12)
    np.testing.assert_allclose(c_index(pred, D['stop'], D['status'], start=D['start'],
                                       strata=D['strata'], sample_weight=W),
                               C_strat, rtol=0, atol=1e-12)

    # the scorer on one "fold" holding all the data
    fam = family()
    split = np.arange(N)
    value, wsum = CoxCIndexScorer.from_family(fam).score_fn(split, D, pred, W)
    np.testing.assert_allclose(value, C_s, rtol=0, atol=1e-12)
    np.testing.assert_allclose(wsum, W.sum())
    value, _ = CoxCIndexScorer.from_family(fam, stratified=True).score_fn(split, D, pred, W)
    np.testing.assert_allclose(value, C_strat, rtol=0, atol=1e-12)


@pytest.mark.parametrize('strata', [False, True])
def test_coxnet_cv_deviance_ungrouped(Rinfo, strata):
    """cross_validation_path's 'Cox Deviance' vs cv.glmnet(grouped=FALSE): the deviance
    of each held-out fold on its own, per unit weight, with start-stop data, weights
    and an offset."""
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D = get_data(5)
    L = CoxNet(family=family(strata=strata), weight_id='weight', offset_id='offset').fit(X, D)
    cv = KFold(5, shuffle=True, random_state=0)
    foldid = np.empty(N, int)
    for i, (_, test) in enumerate(cv.split(X)):
        foldid[test] = i + 1
    L.cross_validation_path(X, D, cv=cv)
    scores = L.score_path_.scores

    assign_R(Rinfo, D, X=X, foldid=foldid)
    y = 'Y' if strata else 'Y0'
    Rinfo['rpy'].r(f'''foldid = as.integer(as.vector(foldid))
CV = cv.glmnet(X, {y}, family="cox", cox.ties="breslow", weights=weight, offset=offset,
               foldid=foldid, grouped=FALSE)''')
    R_lambda, R_cvm, R_cvsd = [get(Rinfo, f'CV${k}') for k in ['lambda', 'cvm', 'cvsd']]

    np.testing.assert_allclose(L.lambda_values_, R_lambda, rtol=1e-8, atol=0)
    np.testing.assert_allclose(scores['Cox Deviance'], R_cvm, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(scores['SD(Cox Deviance)'], R_cvsd, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('ties', ['breslow', 'efron'])
def test_coxlm_start_strata_weights(Rinfo, ties, use_offset):
    """Unpenalized CoxLM vs coxph with (start, stop] data, strata, weights (and an offset).
    `use_offset` is parametrized by conftest."""
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    X, D = get_data(6)
    X = X[:, :4]
    fit = CoxLM(family=family(ties),
                weight_id='weight',
                offset_id='offset' if use_offset else None,
                summarize=True,
                control=GLMControl(epsnr=1e-12, mxitnr=100)).fit(X, D)

    assign_R(Rinfo, D, X=X)
    off = ' + offset(offset)' if use_offset else ''
    Rinfo['rpy'].r(f'''d = data.frame(X, start=start, stop=stop, status=status, strata=strata,
                offset=offset)
f = coxph(Surv(start, stop, status) ~ X1 + X2 + X3 + X4 + strata(strata){off}, data=d,
          weights=weight, ties="{ties}", robust=FALSE,
          control=coxph.control(eps=1e-14, iter.max=100))''')

    np.testing.assert_allclose(fit.coef_, get(Rinfo, 'coef(f)'), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fit.summary_['std err'].values, get(Rinfo, 'sqrt(diag(vcov(f)))'),
                               rtol=1e-6)
    np.testing.assert_allclose(fit.loglike_, get(Rinfo, 'f$loglik[2]')[0], rtol=1e-10)

    # strata matter here: the unstratified fit differs
    unstrat = CoxLM(family=family(ties, strata=False), weight_id='weight',
                    offset_id='offset' if use_offset else None).fit(X, D)
    assert np.abs(unstrat.coef_ - fit.coef_).max() > 1e-3
