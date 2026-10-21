"""
GAMs for the Cox family (glmstar's `CoxFamily`): local scoring with the
diagonal of the Hessian of the partial likelihood as working weights, whose
fixed point is the penalized partial likelihood estimate. Checked against
glmstar's `CoxLM` and R's `coxph` for parametric terms, with ties (Efron and
Breslow), start times, strata, weights and an offset.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scatter_smooth')

from glmnet.cox import CoxFamily, CoxLM
from glmnet.gam import GAM, GAMControl, s, linear, factor

# the parametric fits converge on the objective; with df terms (matched at
# each step's weights) local scoring stops on the change in eta
TIGHT = GAMControl(epsnr=1e-16, mxitnr=500, bf_tol=1e-13, bf_maxit=5000)
SMOOTH = GAMControl(epsnr=1e-12, mxitnr=500, bf_tol=1e-13, bf_maxit=5000)
OPTIONS = {'plain': {},
           'start': dict(start_id='start'),
           'strata': dict(strata_id='st'),
           'all': dict(start_id='start', strata_id='st')}


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    n = 300
    D = pd.DataFrame({'x': np.round(rng.uniform(0, 1, n), 2),
                      'z': rng.normal(0, 1, n),
                      'c': rng.choice(list('abc'), n),
                      'st': rng.integers(0, 2, n),
                      'w': rng.uniform(0.5, 2, n),
                      'o': rng.normal(0, 0.3, n)})
    eta = np.sin(2 * np.pi * D['x']) + 0.5 * D['z'] + D['c'].map({'a': 0, 'b': .5, 'c': -.5})
    T = rng.exponential(1 / np.exp(eta))
    C = rng.exponential(2, n)
    # rounded: tied event times
    D['time'] = np.round(np.minimum(T, C), 1) + 0.1
    D['status'] = (T <= C).astype(int)
    D['start'] = np.where(rng.uniform(size=n) < 0.3, D['time'] * rng.uniform(0, 0.8, n), 0.)
    return D


def _family(ties, option):
    return CoxFamily(event_id='time', status_id='status', tie_breaking=ties, **OPTIONS[option])


def _gam(D, terms, ties='efron', option='plain', weights=False, offset=False, control=TIGHT):
    return GAM(terms=terms, family=_family(ties, option), control=control,
               weight_id='w' if weights else None,
               offset_id='o' if offset else None).fit(D, D)


def _centered(eta):
    eta = np.asarray(eta, float)
    return eta - eta.mean()


@pytest.mark.parametrize('ties', ['efron', 'breslow'])
@pytest.mark.parametrize('option', list(OPTIONS))
@pytest.mark.parametrize('weights', [False, True])
@pytest.mark.parametrize('with_offset', [False, True])
def test_parametric_is_coxlm(data, ties, option, weights, with_offset):
    # linear terms and a factor: the Cox model (CoxLM, glmstar's), up to the
    # constant of the linear predictor
    G = _gam(data, [linear('x'), linear('z'), factor('c')], ties, option, weights, with_offset)
    X = pd.get_dummies(data[['x', 'z', 'c']], columns=['c'], drop_first=True, dtype=float)
    L = CoxLM(family=_family(ties, option), weight_id='w' if weights else None,
              offset_id='o' if with_offset else None).fit(X, data)
    eta_L = X.to_numpy() @ L.coef_
    np.testing.assert_allclose(_centered(G.predict(data, 'link')), _centered(eta_L),
                               atol=1e-8)
    assert G.intercept_ == 0
    np.testing.assert_allclose(G.edf_.values, [1, 1, 2])
    n = data.shape[0]
    np.testing.assert_allclose(G.df_resid_, n - 4)
    np.testing.assert_allclose(G.aic_, -2 * G.loglike_ + 2 * 4)
    assert G.dispersion_ == 1


@pytest.mark.parametrize('ties', ['efron', 'breslow'])
@pytest.mark.parametrize('option', ['plain', 'all'])
def test_parametric_coxph(Rinfo, data, ties, option):
    # against R's coxph: the coefficients and the partial log-likelihood
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("survival", quietly=TRUE))')[0]:
        pytest.skip('requires the R package survival')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data)
    G = _gam(data, [linear('x'), linear('z')], ties, option, weights=True, offset=True)
    surv = 'Surv(start, time, status)' if option == 'all' else 'Surv(time, status)'
    strata = ' + strata(st)' if option == 'all' else ''
    rpy.r(f'f <- survival::coxph({surv} ~ x + z{strata} + offset(o), data=D, weights=w, '
          f'ties="{ties}", control=survival::coxph.control(eps=1e-11, iter.max=100))')
    coef_R = np.asarray(rpy.r('coef(f)'))
    slopes = [np.polyfit(data[c], G.fits_.iloc[:, j], 1)[0] for j, c in enumerate(['x', 'z'])]
    np.testing.assert_allclose(slopes, coef_R, rtol=1e-7)
    np.testing.assert_allclose(G.loglike_, rpy.r('f$loglik[2]')[0], rtol=1e-10)
    np.testing.assert_allclose(G.null_deviance_ / -2 + (G.loglike_ + G.deviance_ / 2),
                               rpy.r('f$loglik[1]')[0], rtol=1e-10)


@pytest.mark.parametrize('ties', ['efron', 'breslow'])
@pytest.mark.parametrize('option', list(OPTIONS))
def test_smooth(data, ties, option):
    # smooths with df: their df at the final working weights; a constant
    # offset and the scale of the weights leave the fit unchanged
    terms = [s('x', df=4), linear('z'), factor('c')]
    G = _gam(data, terms, ties, option, weights=True, control=SMOOTH)
    np.testing.assert_allclose(G.edf_.values, [4, 1, 2], rtol=1e-6)
    eta = G.predict(data, 'link')
    np.testing.assert_allclose(eta, G.fits_.sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(G.predict(data), np.exp(eta))
    np.testing.assert_allclose(G.fits_.mean(axis=0), 0, atol=1e-10)

    D2 = data.assign(o=3., w=10 * data['w'])
    H = _gam(D2, terms, ties, option, weights=True, offset=True, control=SMOOTH)
    np.testing.assert_allclose(H.fits_, G.fits_, atol=1e-7)
    np.testing.assert_allclose(H.edf_, G.edf_, rtol=1e-6)


def test_cox_errors(data):
    with pytest.raises(ValueError, match='DataFrame'):
        GAM(terms=[s('x', df=4)], family=_family('efron', 'plain')).fit(data, data['time'].values)
    G = _gam(data, [s('x', df=4)], control=SMOOTH)
    from glmnet.glm import residuals
    with pytest.raises(NotImplementedError):
        residuals(G, 'deviance')


# the joint solve: Newton steps with the full Hessian of the partial
# likelihood (df terms: their smoothing parameters re-matched at each step
# move the fit by about 1e-10, so a looser tolerance)
JOINT = GAMControl(solver='joint', epsnr=1e-12, mxitnr=100)
JOINT_DF = GAMControl(solver='joint', epsnr=1e-10, mxitnr=100)


@pytest.mark.parametrize('ties', ['efron', 'breslow'])
@pytest.mark.parametrize('option', list(OPTIONS))
@pytest.mark.parametrize('terms', ['lam', 'df'])
def test_joint_is_backfit(data, ties, option, terms):
    if terms == 'lam':
        # backfitting converges linearly: tight, but above its roundoff
        backfit = GAMControl(epsnr=1e-14, mxitnr=2000, bf_tol=1e-13, bf_maxit=5000)
        T_, controls = [s('x', lam=1e-3), linear('z'), factor('c')], (backfit, JOINT)
    else:
        T_, controls = [s('x', df=4), s('z', df=3), factor('c')], (SMOOTH, JOINT_DF)
    B = _gam(data, T_, ties, option, weights=True, offset=True, control=controls[0])
    J = _gam(data, T_, ties, option, weights=True, offset=True, control=controls[1])
    assert J.n_iter_ < B.n_iter_
    np.testing.assert_allclose(J.fits_, B.fits_, atol=1e-6)
    np.testing.assert_allclose(J.edf_, B.edf_, rtol=1e-6)
    np.testing.assert_allclose(J.deviance_, B.deviance_, rtol=1e-9)
    assert J.intercept_ == 0
    # the influence edf of the linear terms and the factor
    np.testing.assert_allclose(J.edf_influence_.values[-1], 2, rtol=1e-8)


@pytest.mark.parametrize('ties', ['efron', 'breslow'])
@pytest.mark.parametrize('option', list(OPTIONS))
def test_parametric_covariance_coxph(Rinfo, data, ties, option):
    # linear terms: the posterior covariance is the inverse information, R's
    # coxph's model-based variance (weights, offset, start times and strata;
    # with weights that are not integers, coxph's vcov is the robust one, its
    # naive.var the model-based)
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("survival", quietly=TRUE))')[0]:
        pytest.skip('requires the R package survival')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data)
    G = _gam(data, [linear('x'), linear('z')], ties, option, weights=True, offset=True,
             control=JOINT)
    surv = 'Surv(start, time, status)' if 'start_id' in OPTIONS[option] else 'Surv(time, status)'
    strata = ' + strata(st)' if 'strata_id' in OPTIONS[option] else ''
    rpy.r(f'f <- survival::coxph({surv} ~ x + z{strata} + offset(o), data=D, weights=w, '
          f'ties="{ties}", control=survival::coxph.control(eps=1e-11, iter.max=100))')
    np.testing.assert_allclose(G.posterior_covariance_[1:, 1:],
                               np.asarray(rpy.r('if (is.null(f$naive.var)) vcov(f) else f$naive.var')),
                               rtol=1e-6)
    assert np.all(G.posterior_covariance_[0] == 0)


def test_cox_mgcv(Rinfo, data, use_offset):
    # (use_offset: parametrized by conftest)
    # mgcv's cox.ph (Breslow ties; its weights are the status, so no case
    # weights) with sp fixed and our knots: the fit (up to a constant), edf,
    # the standard errors of the terms and the log-likelihood. mgcv's sp is
    # lam * n * S.scale
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', data)
    rpy.r('D$c <- factor(D$c)')
    lam = (1e-3, 3e-2)
    G = _gam(data, [s('x', lam=lam[0]), s('z', lam=lam[1]), factor('c')], 'breslow', 'plain',
             offset=use_offset, control=JOINT)
    kx, kz = G.terms_[0].knots_, G.terms_[1].knots_
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('kx', kx)
        rpy.r.assign('kz', kz)
    off = ' + offset(o)' if use_offset else ''

    def mgcv(sp):
        rpy.r(f"""m <- mgcv::gam(time ~ s(x, bs='cr', k={len(kx)}) + s(z, bs='cr', k={len(kz)})
                                  + c{off}, family=mgcv::cox.ph(), weights=status, data=D,
                   knots=list(x=kx, z=kz), sp=c({float(sp[0])!r}, {float(sp[1])!r}),
                   control=mgcv::gam.control(epsilon=1e-12, maxit=200))""")

    mgcv((1., 1.))
    scale = [rpy.r(f'm$smooth[[{i}]]$S.scale')[0] for i in (1, 2)]
    n = data.shape[0]
    mgcv([l * n * c for l, c in zip(lam, scale)])
    eta, eta_R = G.state_.link_parameter, np.asarray(rpy.r('m$linear.predictors'))
    np.testing.assert_allclose(eta - eta.mean(), eta_R - eta_R.mean(), atol=1e-7)
    edf_R = np.asarray(rpy.r('sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))'))
    np.testing.assert_allclose(G.edf_influence_.values[:2], edf_R, rtol=1e-7)
    np.testing.assert_allclose(G.loglike_, rpy.r('logLik(m)')[0], rtol=1e-9)

    new = data.iloc[::7]
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('new', new)
    rpy.r('new$c <- factor(new$c, levels=levels(D$c))')
    terms_R = rpy.r('predict(m, newdata=new, type="terms", se.fit=TRUE)')
    names_R = list(rpy.r('colnames(predict(m, newdata=new, type="terms"))'))
    for j, name in [(0, 's(x)'), (1, 's(z)')]:
        fit, se = G.partial_effect(j, new[G.terms_[j].column], return_std=True)
        fit_R = np.asarray(terms_R[0])[:, names_R.index(name)]
        np.testing.assert_allclose(fit - fit.mean(), fit_R - fit_R.mean(), atol=1e-7)
        np.testing.assert_allclose(se, np.asarray(terms_R[1])[:, names_R.index(name)],
                                   rtol=1e-6, atol=1e-10)
