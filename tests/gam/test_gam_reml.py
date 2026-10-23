"""
Smoothing parameter selection by REML and ML (`GAM(method=...)`): the
criterion's gradient, the Gaussian mixed model's (exact) REML and ML against
statsmodels' MixedLM, the consistency of the final fit, and errors.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')
jax = pytest.importorskip('jax')

from glmnet.gam import GAM, GAMControl, s, lo, re, linear, categorical
from glmnet.gam.reml import LaplaceCriterion

TIGHT = dict(epsnr=1e-12, mxitnr=100)
GROUPS = [f'g{i:02d}' for i in range(20)]


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(10)
    n = 400
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n), 2),
                      'b': np.round(rng.uniform(-2, 2, n), 1),
                      'c': rng.choice(list('xyz'), n),
                      'g': rng.choice(GROUPS, n),
                      'v': rng.uniform(0, 2, n),
                      'o': rng.normal(0, 0.2, n),
                      'w': rng.uniform(0.5, 2, n)})
    effects = dict(zip(GROUPS, rng.normal(0, 0.8, len(GROUPS))))
    slopes = dict(zip(GROUPS, rng.normal(0, 0.4, len(GROUPS))))
    eta = (np.sin(2 * np.pi * D['a']) + 0.3 * D['b']**2 + D['c'].map({'x': 0, 'y': .5, 'z': -.5}) +
           D['g'].map(effects) + D['v'] * D['g'].map(slopes))
    D['y'] = eta + 0.5 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-(eta + D['o'] - 0.5))))
    D['yp'] = rng.poisson(np.exp(0.4 * eta + D['o']))
    D['yg'] = rng.gamma(4, np.exp(0.3 * eta + D['o']) / 4)
    return D


FAMILIES = {'gaussian': (sm.families.Gaussian(), 'y'),
            'binomial': (sm.families.Binomial(), 'yb'),
            'poisson': (sm.families.Poisson(), 'yp'),
            'Gamma (log)': (sm.families.Gamma(sm.families.links.Log()), 'yg'),
            'Gamma (inverse)': (sm.families.Gamma(), 'yg'),
            'inverse Gaussian (log)': (sm.families.InverseGaussian(sm.families.links.Log()), 'yg')}


def _terms():
    return [s('a'), s('b'), categorical('c'), re('g'), re('g', by='v')]


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', list(FAMILIES))
def test_gradient(data, family, method):
    # the gradient (through the implicit derivative of the coefficients)
    # against central differences of the criterion, away from the optimum,
    # with prior weights and an offset
    fam, y = FAMILIES[family]
    terms = [s('a'), s('b'), categorical('c'), re('g')] if 'inverse' in family else _terms()
    G = GAM(terms=terms, family=fam, method=method, control=GAMControl(**TIGHT),
            response_id=y, weight_id='w', offset_id='o').fit(data, data[[y, 'w', 'o']])
    estimated = [j for j, t in enumerate(G.terms_) if t.penalty_matrix() is not None]
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, estimated, method=method)
        rng = np.random.default_rng(0)
        x = C.start() + rng.uniform(-1, 1, C.n_params)
        value, grad = C.value_and_grad(x)
        eps = 1e-5
        fd = np.array([(C.value_and_grad(x + eps * e)[0] - C.value_and_grad(x - eps * e)[0]) / (2 * eps)
                       for e in np.identity(len(x))])
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-5)
    # and at the optimum it vanishes
    np.testing.assert_allclose(G.smoothing_.gradient, 0, atol=1e-5)
    assert G.smoothing_.converged


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_gaussian_random_intercept_is_mixedlm(data, method):
    # a Gaussian random intercept model: REML and ML are exact, those of the
    # linear mixed model (statsmodels' MixedLM)
    G = GAM(terms=[linear('a'), categorical('c'), re('g')], method=method,
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    X = sm.add_constant(pd.get_dummies(data[['a', 'c']], drop_first=True).astype(float))
    M = sm.MixedLM(data['y'], X, groups=data['g']).fit(reml=method == 'REML', method='bfgs',
                                                       gtol=1e-10)
    vc = G.variance_components_
    np.testing.assert_allclose(vc.loc['scale', 'variance'], M.scale, rtol=1e-5)
    np.testing.assert_allclose(vc.loc['re(g)', 'variance'], M.cov_re.iloc[0, 0], rtol=1e-4)
    # the fitted values (BLUPs) and the slope
    blup = pd.Series({k: v.iloc[0] for k, v in M.random_effects.items()})
    np.testing.assert_allclose(G.predict(data), M.fittedvalues, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(G.terms_[0].coef_, M.fe_params['a'], rtol=1e-5)
    np.testing.assert_allclose(G.terms_[2].effects_.loc[GROUPS].values, blup.loc[GROUPS].values,
                               atol=1e-5)
    if method == 'ML':
        # the criterion is minus the log-likelihood
        np.testing.assert_allclose(-G.reml_score_, M.llf, rtol=1e-8)


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson', 'Gamma (log)'])
def test_final_fit_is_the_fixed_fit(data, family, method):
    # the fit at the estimated smoothing parameters is the fixed fit there
    # (joint or backfitted), with mgcv's edf; the variance components are
    # the dispersion over the smoothing parameters
    fam, y = FAMILIES[family]
    G = GAM(terms=_terms(), family=fam, method=method, control=GAMControl(**TIGHT)).fit(data, data[y])
    lam = dict(zip(G.smoothing_.terms, G.smoothing_.lam))
    assert G.smoothing_.terms == ['s(a)', 's(b)', 're(g)', 're(g, by=v)']
    fixed = [s('a', lam=lam['s(a)']), s('b', lam=lam['s(b)']), categorical('c'),
             re('g', lam=lam['re(g)']), re('g', by='v', lam=lam['re(g, by=v)'])]
    for solver in ['joint', 'backfit']:
        F = GAM(terms=fixed, family=fam, control=GAMControl(**TIGHT, solver=solver,
                                                            bf_tol=1e-13, bf_maxit=5000)).fit(data, data[y])
        np.testing.assert_allclose(F.state_.linear_predictor, G.state_.linear_predictor,
                                   rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(F.edf_influence_.values, G.edf_.values, rtol=1e-6)
    n = data.shape[0]
    vc = G.variance_components_
    for name in lam:
        np.testing.assert_allclose(vc.loc[name, 'variance'], G.dispersion_ / (lam[name] * n),
                                   rtol=1e-10)
    np.testing.assert_allclose(G.df_resid_, n - 1 - G.edf_.sum())
    if family in ('binomial', 'poisson'):
        assert G.dispersion_ == 1


def test_weights_scale_and_rows(data):
    # Gaussian prior weights are precisions: scaling them scales the
    # dispersion, the fit is unchanged
    fits = [GAM(terms=[s('a'), categorical('c'), re('g')], method='REML', control=GAMControl(**TIGHT),
                response_id='y', weight_id='w').fit(data, data[['y', 'w']].assign(w=c * data['w']))
            for c in (1., 3.)]
    np.testing.assert_allclose(fits[1].state_.linear_predictor, fits[0].state_.linear_predictor,
                               rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fits[1].dispersion_, 3 * fits[0].dispersion_, rtol=1e-6)
    # rows of zero weight are left out (with terms whose bases do not
    # depend on them: a spline's knots are at all the distinct values)
    W = data.assign(w=np.where(np.arange(data.shape[0]) % 5 == 0, 0., data['w']))
    G0 = GAM(terms=[linear('a'), categorical('c'), re('g')], method='REML', control=GAMControl(**TIGHT),
             response_id='y', weight_id='w').fit(W, W[['y', 'w']])
    keep = W['w'] > 0
    G1 = GAM(terms=[linear('a'), categorical('c'), re('g')], method='REML', control=GAMControl(**TIGHT),
             response_id='y', weight_id='w').fit(W[keep], W.loc[keep, ['y', 'w']])
    # (the score differs by a constant: the zero weight rows are among those
    # the terms are centered over, which parametrizes the unpenalized part)
    np.testing.assert_allclose(G0.smoothing_.lam_raw, G1.smoothing_.lam_raw, rtol=1e-4)
    np.testing.assert_allclose(G0.predict(W[keep]), G1.predict(W[keep]), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', ['binomial', 'poisson'])
def test_given_dispersion_fixed_family(data, family, method):
    # a dispersion given for the binomial or Poisson (a quasi-likelihood's)
    # is used by the criterion, not only by the covariance: the fit is that
    # with prior weights 1 / phi and dispersion 1, its smoothing parameters
    # (on the prior weights' scale) phi times those
    fam, y = FAMILIES[family]
    phi = 3.
    terms = [s('a'), categorical('c'), re('g')]
    G = GAM(terms=terms, family=fam, method=method,
            control=GAMControl(**TIGHT)).fit(data, data[y], dispersion=phi)
    W = GAM(terms=terms, family=fam, method=method, control=GAMControl(**TIGHT),
            response_id=y, weight_id='w').fit(data, data[[y]].assign(w=1 / phi))
    assert G.smoothing_.dispersion == phi and G.dispersion_ == phi
    np.testing.assert_allclose(G.state_.linear_predictor, W.state_.linear_predictor,
                               rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(G.smoothing_.lam_raw, phi * W.smoothing_.lam_raw, rtol=1e-4)
    np.testing.assert_allclose(G.posterior_covariance_, W.posterior_covariance_,
                               rtol=1e-3, atol=1e-6)
    # and differs from the fit at dispersion 1
    D = GAM(terms=terms, family=fam, method=method,
            control=GAMControl(**TIGHT)).fit(data, data[y])
    assert not np.allclose(D.smoothing_.lam_raw, G.smoothing_.lam_raw, rtol=1e-2)


def test_df_and_lam_terms_are_fixed(data):
    # a term with lam keeps it, one with df has it matched at the prior
    # weights; the others are estimated
    G = GAM(terms=[s('a', df=5), s('b', lam=1e-2), re('g')], family=sm.families.Poisson(),
            method='REML', control=GAMControl(**TIGHT)).fit(data, data['yp'])
    assert G.smoothing_.terms == ['re(g)']
    assert G.terms_[1].lam == 1e-2
    P = GAM(terms=[s('a', df=5)], control=GAMControl(**TIGHT)).fit(data, data['y'])
    # (Gaussian at the prior weights: the df=5 spline's smoothing parameter)
    np.testing.assert_allclose(G.terms_[0].lam, P.terms_[0].lam_, rtol=1e-6)


def test_random_effects_reml(data):
    # a random intercept and slope by REML: the effects named by the groups,
    # the standard errors from the posterior covariance at REML's dispersion
    G = GAM(terms=[s('a'), categorical('c'), re('g'), re('g', by='v')], method='REML',
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    assert list(G.terms_[2].effects_.index) == GROUPS
    assert list(G.variance_components_.index) == ['s(a)', 're(g)', 're(g, by=v)', 'scale']
    # standard errors at the REML dispersion
    _, se = G.predict(data.iloc[:5], return_std=True)
    rows = G.design_.rows(G._columns(G._design(data.iloc[:5]), G.terms_))
    np.testing.assert_allclose(se**2, np.einsum('ij,jk,ik->i', rows, G.posterior_covariance_, rows), rtol=1e-10)


def test_errors(data):
    with pytest.raises(ValueError, match="method should be"):
        GAM(terms=[s('a')], method='GCV').fit(data, data['y'])
    with pytest.raises(ValueError, match='need a basis'):
        GAM(terms=[lo('a'), s('b')], method='REML').fit(data, data['y'])
    with pytest.raises(ValueError, match='no smoothing parameter to estimate'):
        GAM(terms=[s('a', df=4), categorical('c')], method='REML').fit(data, data['y'])
    with pytest.raises(NotImplementedError, match='Tweedie'):
        GAM(terms=[s('a')], family=sm.families.Tweedie(var_power=1.5),
            method='REML').fit(data, data['yp'] + 0.5)
    # method='fixed': s() without arguments has df=4
    G = GAM(terms=[s('a')]).fit(data, data['y'])
    np.testing.assert_allclose(G.edf_.iloc[0], 4, rtol=1e-6)


def test_without_jax(data, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'jax' or name.startswith('jax.'):
            raise ImportError('no jax')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ImportError, match='pip install jax'):
        GAM(terms=[s('a')], method='REML').fit(data, data['y'])
    # the fixed method does not need it
    GAM(terms=[s('a')]).fit(data, data['y'])


def test_reml_is_experimental(data):
    # REML and ML warn that they are experimental; the fixed method does not
    import warnings
    from glmnet.gam import ExperimentalWarning
    for method in ['REML', 'ML']:
        with pytest.warns(ExperimentalWarning, match=f"method='{method}'.*experimental"):
            GAM(terms=[s('a')], method=method, control=GAMControl(**TIGHT)).fit(data, data['y'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ExperimentalWarning)
        GAM(terms=[s('a', df=4)]).fit(data, data['y'])
    assert issubclass(ExperimentalWarning, UserWarning)
