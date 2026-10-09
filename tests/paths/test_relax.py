import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
import statsmodels.api as sm

from glmnet import GaussNet, LogNet, FishNet, MultiClassNet, MultiGaussNet, CoxNet, GLMNet
from glmnet.paths.fastnet import FastNetControl
from glmnet.glmnet import GLMNetControl
from glmnet.cox import CoxFamily
from glmnet.scorer import RelaxedScorePath

N, P = 150, 10
FAMILIES = ['gaussian', 'binomial', 'poisson', 'multinomial', 'mgaussian', 'cox']
CONTROL = FastNetControl(thresh=1e-14)


def get_data(family, n=N, p=P, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    eta = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    if family == 'gaussian':
        D = pd.DataFrame({'Y': eta + rng.standard_normal(n)})
    elif family == 'binomial':
        D = pd.DataFrame({'Y': rng.binomial(1, expit(eta)).astype(float)})
    elif family == 'poisson':
        D = pd.DataFrame({'Y': rng.poisson(np.exp(eta / 2)).astype(float)})
    elif family == 'multinomial':
        logits = np.column_stack([eta, -eta, np.zeros(n)])
        prob = np.exp(logits) / np.exp(logits).sum(1)[:, None]
        D = pd.DataFrame({'Y': [rng.choice(3, p=p_) for p_ in prob]})
    elif family == 'mgaussian':
        D = pd.DataFrame({'Y1': eta + rng.standard_normal(n),
                          'Y2': -eta + rng.standard_normal(n)})
    else:
        D = pd.DataFrame({'stop': rng.exponential(np.exp(-eta)),
                          'status': rng.binomial(1, 0.8, n)})
    D['w'] = rng.uniform(0.5, 2, n)
    return X, D


def get_estimator(family, **args):
    args = dict(control=CONTROL, weight_id='w', **args)
    if family == 'gaussian':
        return GaussNet(response_id='Y', **args)
    if family == 'binomial':
        return LogNet(response_id='Y', **args)
    if family == 'poisson':
        return FishNet(response_id='Y', **args)
    if family == 'multinomial':
        return MultiClassNet(response_id='Y', **args)
    if family == 'mgaussian':
        return MultiGaussNet(response_id=['Y1', 'Y2'], **args)
    return CoxNet(family=CoxFamily(event_id='stop', status_id='status'), **args)


def assign_R(Rinfo, family, X, D):
    rpy = Rinfo['rpy']
    rpy.r.assign('X', X)
    rpy.r.assign('wv', D['w'].values)
    rpy.r('w = as.vector(wv)')
    if family == 'cox':
        rpy.r.assign('st', D['stop'].values)
        rpy.r.assign('d', D['status'].values.astype(float))
        rpy.r('Y = Surv(as.vector(st), as.vector(d))')
    elif family == 'mgaussian':
        rpy.r.assign('Yv', D[['Y1', 'Y2']].values)
        rpy.r('Y = as.matrix(Yv)')
    elif family == 'multinomial':
        rpy.r.assign('Yv', D['Y'].values.astype(float))
        rpy.r('Y = factor(as.vector(Yv))')
    else:
        rpy.r.assign('Yv', D['Y'].values.astype(float))
        rpy.r('Y = as.vector(Yv)')
    rpy.r('suppressMessages({library(glmnet); library(survival)})')


def R_link(Rinfo, gamma):
    link = np.asarray(Rinfo['rpy'].r(f'predict(fit, X, gamma={gamma}, type="link")'))
    if link.ndim == 3:
        # R returns (n, K, nlambda)
        link = np.transpose(link, (0, 2, 1))
    return link


def center(link, family):
    # multinomial intercepts are only identified up to a constant
    return link - link.mean(-1, keepdims=True) if family == 'multinomial' else link


@pytest.mark.parametrize('family', FAMILIES)
# with p > n - 3 some active sets are too large to refit
@pytest.mark.parametrize('shape', [(N, P), (25, 40)])
def test_relax_matches_R(Rinfo, family, shape):
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    n, p = shape
    if p > n and family == 'multinomial':
        # the unpenalized refits do not converge (separable classes), and
        # R's glmnet(relax=TRUE) stops with an error; see
        # test_relax_refit_not_converged
        pytest.skip("R's relaxed multinomial fit errors on this data")
    X, D = get_data(family, n=n, p=p)
    L = get_estimator(family, relax=True, nlambda=30).fit(X, D)
    if p > n and family in ['gaussian', 'poisson', 'mgaussian']:
        # (the other paths stop before an active set has n - 3 variables)
        assert L.relaxed_omitted_.any()

    rpy = Rinfo['rpy']
    with Rinfo['np_cv_rules'].context():
        assign_R(Rinfo, family, X, D)
        # both compute the lambda values (passing them to R would solve
        # exactly at lambda_max, where roundoff can leave tiny coefficients)
        rpy.r(f'fit = glmnet(X, Y, family="{family}", weights=w, nlambda=30, relax=TRUE, thresh=1e-14)')
        lam_R = np.asarray(rpy.r('fit$lambda'))
        nlam = min(len(lam_R), len(L.lambda_values_))
        np.testing.assert_allclose(L.lambda_values_[:nlam], lam_R[:nlam], rtol=1e-8)
        for gamma in [0, 0.5, 1]:
            link_R = center(R_link(Rinfo, gamma), family)
            link = center(L.predict(X, prediction_type='link', gamma=gamma), family)
            # the relaxed Cox fits with n=25 and ~20 variables are ill-conditioned
            tol = 1e-3 if (family == 'cox' and p > n) else 1e-5
            np.testing.assert_allclose(link[:, :nlam], link_R[:, :nlam], rtol=tol, atol=tol)


def test_relax_gamma_one_is_lasso():
    X, D = get_data('binomial')
    L = get_estimator('binomial', relax=True).fit(X, D)
    plain = get_estimator('binomial').fit(X, D)
    np.testing.assert_allclose(L.predict(X, gamma=1), plain.predict(X))
    np.testing.assert_allclose(L.coefs_, plain.coefs_)


def test_relax_blend_and_interpolation():
    X, D = get_data('poisson')
    L = get_estimator('poisson', relax=True).fit(X, D)
    gamma = 0.3
    coefs = gamma * L.coefs_ + (1 - gamma) * L.relaxed_coefs_
    intercepts = gamma * L.intercepts_ + (1 - gamma) * L.relaxed_intercepts_
    nfit = coefs.shape[0] # predict pads to nlambda columns
    np.testing.assert_allclose(L.predict(X, prediction_type='link', gamma=gamma)[:, :nfit],
                               X @ coefs.T + intercepts[None, :])
    # interpolating the blend is blending the interpolations
    grid = np.sort(np.random.default_rng(0).uniform(L.lambda_values_.min(),
                                                    L.lambda_values_.max(), 5))
    # (gamma=0 is floored at 1e-5, as in R)
    c0, i0 = L._interpolate(L.relaxed_coefs_, L.relaxed_intercepts_, grid)
    c1, i1 = L.interpolate_coefs(grid, gamma=1)
    cg, ig = L.interpolate_coefs(grid, gamma=gamma)
    np.testing.assert_allclose(cg, gamma * c1 + (1 - gamma) * c0)
    np.testing.assert_allclose(ig, gamma * i1 + (1 - gamma) * i0)


def test_relaxed_fit_is_unpenalized_on_active_set():
    # the relaxed fit at each lambda is the unpenalized GLM on its active set,
    # here for the IRLS GLMNet
    X, D = get_data('binomial')
    L = GLMNet(family=sm.families.Binomial(), response_id='Y', relax=True, nlambda=20,
               control=GLMNetControl(thresh=1e-12)).fit(X, D.drop(columns='w'))
    y = D['Y'].values
    nfit = L.coefs_.shape[0]
    for k in [3, nfit // 2, nfit - 1]:
        active = np.nonzero(L.coefs_[k])[0]
        if len(active) == 0:
            continue
        glm = sm.GLM(y, sm.add_constant(X[:, active]), family=sm.families.Binomial()).fit()
        np.testing.assert_allclose(L.relaxed_coefs_[k, active], glm.params[1:], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(L.relaxed_intercepts_[k], glm.params[0], rtol=1e-4, atol=1e-6)
        assert np.all(np.delete(L.relaxed_coefs_[k], active) == 0)


def test_relax_refit_not_converged():
    # with separable classes the unpenalized refit has no solution: the
    # relaxed fit keeps the lasso solution there, with a warning
    X, D = get_data('multinomial', n=25, p=40)
    with pytest.warns(UserWarning, match='did not converge'):
        L = get_estimator('multinomial', relax=True, nlambda=30).fit(X, D)
    same = np.all(L.relaxed_coefs_ == L.coefs_, axis=(1, 2))
    assert same[1:].any() and not same.all()


def test_relax_maxp_extends_last_refit():
    X, D = get_data('gaussian', n=30, p=40)
    L = get_estimator('gaussian', relax=True, relax_maxp=5).fit(X, D)
    omitted = L.relaxed_omitted_
    assert omitted.any() and not omitted.all()
    assert np.all((L.coefs_[omitted] != 0).sum(1) > 5)
    last = np.nonzero(~omitted)[0].max()
    np.testing.assert_allclose(L.relaxed_coefs_[omitted], np.repeat(L.relaxed_coefs_[[last]],
                                                                    omitted.sum(), 0))


def test_gamma_requires_relax():
    X, D = get_data('gaussian')
    L = get_estimator('gaussian').fit(X, D)
    with pytest.raises(ValueError, match='relax=True'):
        L.predict(X, gamma=0.5)
    with pytest.raises(ValueError, match='relax=True'):
        L.cross_validation_path(X, D, cv=3, gamma=[0, 1])
    L = get_estimator('gaussian', relax=True).fit(X, D)
    with pytest.raises(ValueError, match=r'gamma should be in \[0, 1\]'):
        L.predict(X, gamma=1.5)


R_SCORE = {'gaussian': ('Mean Squared Error', 'mse'),
           'binomial': ('Binomial Deviance', 'deviance'),
           'poisson': ('Poisson Deviance', 'deviance')}


@pytest.mark.parametrize('family', list(R_SCORE))
@pytest.mark.parametrize('fixed_lambda', [True, False])
def test_relax_cv_matches_R(Rinfo, family, fixed_lambda):
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    X, D = get_data(family, n=200, seed=1)
    rng = np.random.default_rng(2)
    foldid = np.arange(200) % 5 + 1
    rng.shuffle(foldid)
    cv = [(np.nonzero(foldid != k)[0], np.nonzero(foldid == k)[0]) for k in range(1, 6)]
    gamma = [0, 0.25, 0.5, 0.75, 1]
    score, measure = R_SCORE[family]

    if fixed_lambda:
        lam = get_estimator(family, nlambda=30).fit(X, D).lambda_values_
        L = get_estimator(family, relax=True, lambda_values=lam).fit(X, D)
    else:
        L = get_estimator(family, relax=True, nlambda=30).fit(X, D)
    predictions, rsp = L.cross_validation_path(X, D, cv=cv, gamma=gamma)
    assert isinstance(rsp, RelaxedScorePath)
    assert predictions.shape[:2] == (200, len(gamma))

    rpy = Rinfo['rpy']
    with Rinfo['np_cv_rules'].context():
        assign_R(Rinfo, family, X, D)
        rpy.r.assign('fv', foldid.astype(float))
        rpy.r('foldid = as.vector(fv)')
        lam_arg = 'lambda=lam' if fixed_lambda else 'nlambda=30'
        if fixed_lambda:
            rpy.r.assign('lam', L.lambda_values_)
        rpy.r(f'cvf = cv.glmnet(X, Y, family="{family}", weights=w, {lam_arg}, relax=TRUE, '
              f'gamma=c(0, .25, .5, .75, 1), foldid=foldid, type.measure="{measure}", thresh=1e-14)')
        for i in range(len(gamma)):
            cvm_R = np.asarray(rpy.r(f'cvf$relaxed$statlist[[{i + 1}]]$cvm'))
            cvm = np.asarray(rsp.score_paths[i].scores[score])
            m = min(len(cvm), len(cvm_R))
            np.testing.assert_allclose(cvm[:m], cvm_R[:m], rtol=1e-5)
        for which in ['min', '1se']:
            lam_R = float(rpy.r(f'cvf$relaxed$lambda.{which}')[0])
            gamma_R = float(rpy.r(f'cvf$relaxed$gamma.{which}')[0])
            index = rsp.index_best if which == 'min' else rsp.index_1se
            np.testing.assert_allclose(index.loc[score, 'lambda'], lam_R, rtol=1e-8)
            assert index.loc[score, 'gamma'] == gamma_R


def test_relax_cv_lasso_scores_unchanged():
    # score_path_ of a relaxed CV is the lasso's, as without relax
    X, D = get_data('binomial')
    cv = 4
    plain = get_estimator('binomial').fit(X, D)
    plain.cross_validation_path(X, D, cv=cv)
    L = get_estimator('binomial', relax=True).fit(X, D)
    L.cross_validation_path(X, D, cv=cv, gamma=[0, 0.5])
    pd.testing.assert_frame_equal(L.score_path_.scores, plain.score_path_.scores)
    np.testing.assert_array_equal(L.relaxed_score_path_.gamma, [0, 0.5])


def test_relaxed_score_path_plot():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    X, D = get_data('gaussian')
    L = get_estimator('gaussian', relax=True).fit(X, D)
    _, rsp = L.cross_validation_path(X, D, cv=3)
    fig, ax = plt.subplots()
    rsp.plot(ax=ax)
    assert len(ax.get_lines()) == len(rsp.gamma) + 2 # a curve per gamma, best and 1SE
    plt.close(fig)
