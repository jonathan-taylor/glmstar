import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from glmnet import GaussNet, LogNet
from glmnet.paths.fastnet import FastNetControl

N, P = 200, 10
CONTROL = FastNetControl(thresh=1e-14)


def get_data(family, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, P))
    eta = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    if family == 'gaussian':
        y = eta + rng.standard_normal(N)
    else:
        y = rng.binomial(1, expit(eta)).astype(float)
    foldid = np.arange(N) % 5 + 1
    rng.shuffle(foldid)
    cv = [(np.nonzero(foldid != k)[0], np.nonzero(foldid == k)[0]) for k in range(1, 6)]
    return X, y, foldid, cv


ESTIMATORS = {'gaussian': GaussNet, 'binomial': LogNet}
R_MEASURE = {'gaussian': ('Mean Squared Error', 'mse'),
             'binomial': ('Binomial Deviance', 'deviance')}


def fit_cv(family, relax):
    X, y, foldid, cv = get_data(family)
    cls = ESTIMATORS[family]
    lam = cls(nlambda=30, control=CONTROL).fit(X, y).lambda_values_
    L = cls(lambda_values=lam, relax=relax, control=CONTROL).fit(X, y)
    L.cross_validation_path(X, y, cv=cv)
    return L, X, y, foldid


@pytest.mark.parametrize('family', list(ESTIMATORS))
@pytest.mark.parametrize('relax', [False, True])
def test_cv_choice_matches_R(Rinfo, family, relax):
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']
    L, X, y, foldid = fit_cv(family, relax)
    score, measure = R_MEASURE[family]

    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('X', X)
        rpy.r.assign('Yv', y)
        rpy.r('Y = as.vector(Yv)')
        rpy.r.assign('fv', foldid.astype(float))
        rpy.r('foldid = as.vector(fv)')
        rpy.r.assign('lam', L.lambda_values_)
        rpy.r('suppressMessages(library(glmnet))')
        relax_R = 'TRUE' if relax else 'FALSE'
        rpy.r(f'cvf = cv.glmnet(X, Y, family="{family}", lambda=lam, foldid=foldid, '
              f'type.measure="{measure}", relax={relax_R}, thresh=1e-14)')
        for which, s in [('1se', 'lambda.1se'), ('best', 'lambda.min')]:
            if relax:
                lam_R = float(rpy.r(f'cvf$relaxed${s}')[0])
                gamma_R = float(rpy.r(f'cvf$relaxed$gamma.{s.split(".")[1]}')[0])
                coef_R = np.asarray(rpy.r(f'as.numeric(coef(cvf, s="{s}", gamma="gamma.{s.split(".")[1]}"))'))
                link_R = np.asarray(rpy.r(f'predict(cvf, X, s="{s}", gamma="gamma.{s.split(".")[1]}")')).ravel()
            else:
                lam_R = float(rpy.r(f'cvf${s}')[0])
                gamma_R = 1.
                coef_R = np.asarray(rpy.r(f'as.numeric(coef(cvf, s="{s}"))'))
                link_R = np.asarray(rpy.r(f'predict(cvf, X, s="{s}")')).ravel()

            lam_, gamma_ = L.cv_choice(which=which, score=score)
            np.testing.assert_allclose(lam_, lam_R, rtol=1e-8)
            assert gamma_ == gamma_R

            coef, intercept = L.cv_coefs(which=which, score=score)
            np.testing.assert_allclose(np.r_[intercept, coef], coef_R, rtol=1e-6, atol=1e-8)
            link = L.cv_predict(X, which=which, score=score, prediction_type='link')
            np.testing.assert_allclose(link, link_R, rtol=1e-6, atol=1e-8)


def test_cv_choice_defaults_and_errors():
    X, y, _, cv = get_data('gaussian')
    L = GaussNet().fit(X, y)
    with pytest.raises(ValueError, match='cross_validation_path'):
        L.cv_coefs()
    L.cross_validation_path(X, y, cv=cv)
    # default: 1se for the family's default score
    lam_, gamma_ = L.cv_choice()
    assert lam_ == L.score_path_.index_1se['Mean Squared Error'] and gamma_ == 1
    np.testing.assert_allclose(L.cv_predict(X), L.predict(X, interpolation_grid=lam_))
    with pytest.raises(ValueError, match="'1se' or 'best'"):
        L.cv_choice(which='min')
    with pytest.raises(ValueError, match='no cross-validated score'):
        L.cv_choice(score='nonsense')


def test_exact_coefs_relaxed():
    # the relaxed solution at a lambda off the grid is the unpenalized fit
    # on the active set there
    X, y, _, _ = get_data('gaussian')
    L = GaussNet(relax=True, control=CONTROL).fit(X, y)
    lam = np.sqrt(L.lambda_values_[10] * L.lambda_values_[11])
    lasso, _ = L.exact_coefs(X, y, lam)
    relaxed, relaxed_intercept = L.exact_coefs(X, y, lam, gamma=0)
    active = np.nonzero(lasso)[0]
    ols = GaussNet(lambda_values=np.array([0.]), control=CONTROL,
                   exclude=[j for j in range(P) if j not in active]).fit(X, y)
    # gamma=0 is floored at 1e-5, as in R
    g = 1e-5
    np.testing.assert_allclose(relaxed, g * lasso + (1 - g) * ols.coefs_[-1], rtol=1e-8, atol=1e-10)
    blend, _ = L.exact_coefs(X, y, lam, gamma=0.5)
    np.testing.assert_allclose(blend, 0.5 * lasso + 0.5 * ols.coefs_[-1], rtol=1e-8, atol=1e-10)


def test_relaxed_coef_path():
    X, y, _, _ = get_data('binomial')
    L = LogNet(relax=True, control=CONTROL).fit(X, y)
    path = L.relaxed_coef_path(gamma=0.25)
    np.testing.assert_allclose(path.coefs, 0.25 * L.coefs_ + 0.75 * L.relaxed_coefs_)
    np.testing.assert_allclose(L.relaxed_coef_path(gamma=1).coefs, L.coefs_)
    plain = LogNet(control=CONTROL).fit(X, y)
    with pytest.raises(ValueError, match='relax=True'):
        plain.relaxed_coef_path()


def test_relaxed_plots():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    X, y, _, cv = get_data('gaussian')
    L = GaussNet(relax=True, control=CONTROL).fit(X, y)
    _, rsp = L.cross_validation_path(X, y, cv=cv)
    fig, ax = plt.subplots()
    rsp.plot(ax=ax, se_bands=True)
    assert len(ax.collections) == len(rsp.gamma) # a band per gamma
    plt.close(fig)
    fig, ax = plt.subplots()
    L.relaxed_coef_path(gamma=0).plot(ax=ax)
    plt.close(fig)
