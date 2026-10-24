"""
`glmnet.gam.GAM` against R:

- fixed smoothing parameters (`s(x, lam=...)`) against `mgcv::gam` with
  `bs='cr'` and the knots at the distinct values (the same natural cubic
  splines) and `sp` fixed. mgcv rescales the penalties (`smooth$S.scale`),
  so its `sp` is `lam * sum(w) * S.scale`; the fits agree to rounding;
- fixed degrees of freedom (`s(x, df=...)`) against R's `gam::gam`, which
  also matches df at the weights of each working problem; R matches df to
  about 1e-4, so the fits agree to about 1e-4. With fewer than 50 distinct
  values R's `gam` puts its knots at all of them.

Gaussian, binomial and Poisson, with prior weights, a factor and an offset.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GAMControl, s, lo, linear, categorical

CONTROL = GAMControl(epsnr=1e-12, mxitnr=100)
FAMILIES = {'gaussian': (sm.families.Gaussian, 'y'),
            'binomial': (sm.families.Binomial, 'yb'),
            'poisson': (sm.families.Poisson, 'yp')}


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(1)
    n = 300
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n) * 40) / 40,   # 41 distinct values
                      'b': np.round(rng.uniform(-2, 2, n), 1),        # 41
                      'c': rng.choice(list('xyz'), n),
                      'o': rng.normal(0, 0.2, n),
                      'w': rng.uniform(0.5, 2, n)})
    eta = np.sin(2 * np.pi * D['a']) + 0.25 * D['b']**2 + D['c'].map({'x': 0, 'y': .5, 'z': -.5})
    D['y'] = eta + 0.5 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-(eta + D['o'] - 0.5))))
    D['yp'] = rng.poisson(np.exp(0.3 * eta + D['o']))
    return D


def _R(Rinfo, D, package):
    rpy = Rinfo['rpy']
    if not rpy.r(f'isTRUE(requireNamespace("{package}", quietly=TRUE))')[0]:
        pytest.skip(f'requires the R package {package}')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', D)
    rpy.r('D$c <- factor(D$c)')
    return rpy


@pytest.mark.parametrize('family', list(FAMILIES))
def test_fixed_lambda_mgcv(Rinfo, data, family, use_offset):
    if family == 'gaussian' and use_offset:
        pytest.skip('the Gaussian is checked without the offset')
    rpy = _R(Rinfo, data, 'mgcv')
    klass, yname = FAMILIES[family]
    lam = (1e-4, 3e-3)
    cols = [yname, 'w'] + (['o'] if use_offset else [])
    G = GAM(terms=[s('a', lam=lam[0]), s('b', lam=lam[1]), categorical('c')],
            family=klass(), control=CONTROL, response_id=yname, weight_id='w',
            offset_id='o' if use_offset else None).fit(data, data[cols])

    off = ' + offset(o)' if use_offset else ''
    ka, kb = data['a'].nunique(), data['b'].nunique()

    def mgcv(sp):
        rpy.r(f"""m <- mgcv::gam({yname} ~ s(a, bs='cr', k={ka}) + s(b, bs='cr', k={kb}) + c{off},
                   data=D, weights=w, family={family}(),
                   knots=list(a=sort(unique(D$a)), b=sort(unique(D$b))),
                   sp=c({float(sp[0])!r}, {float(sp[1])!r}),
                   control=mgcv::gam.control(epsilon=1e-12, maxit=200))""")

    sw = data['w'].sum()
    mgcv((1., 1.))
    scale = [rpy.r(f'm$smooth[[{i}]]$S.scale')[0] for i in (1, 2)]
    mgcv([l * sw * c for l, c in zip(lam, scale)])

    eta_R = np.asarray(rpy.r('m$linear.predictors'))
    eta = G.state_.link_parameter
    np.testing.assert_allclose(eta, eta_R, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(G.deviance_, rpy.r('deviance(m)')[0], rtol=1e-9)

    # the curves, centered as mgcv's (mean 0 over the rows)
    pred = rpy.r('predict(m, type="terms")')
    np.testing.assert_allclose(G.fits_['s(a, lam=0.0001)'], np.asarray(pred)[:, 1], rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('family', list(FAMILIES))
def test_fixed_df_gam(Rinfo, data, family):
    rpy = _R(Rinfo, data, 'gam')
    klass, yname = FAMILIES[family]
    G = GAM(terms=[s('a', df=4), s('b', df=3), categorical('c')], family=klass(),
            control=CONTROL, response_id=yname, weight_id='w').fit(data, data[[yname, 'w']])
    # gam's s() is a special of its formulas only by name: attach the package
    rpy.r('suppressPackageStartupMessages(library(gam))')
    rpy.r(f"""g <- gam::gam({yname} ~ s(a, df=4) + s(b, df=3) + c, data=D,
               weights=w, family={family},
               control=gam::gam.control(epsilon=1e-12, bf.epsilon=1e-12,
                                        maxit=100, bf.maxit=200))""")
    eta_R = np.asarray(rpy.r('g$additive.predictors'))
    np.testing.assert_allclose(G.state_.linear_predictor, eta_R, rtol=1e-3, atol=2e-4)
    np.testing.assert_allclose(G.deviance_, rpy.r('deviance(g)')[0], rtol=1e-5)
    # R's nl.df leaves out the linear part
    np.testing.assert_allclose(G.edf_.iloc[:2] - 1, np.asarray(rpy.r('g$nl.df')), atol=1e-3)
    np.testing.assert_allclose(G.df_resid_, rpy.r('g$df.residual')[0], atol=1e-3)


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_many_distinct_values_gam(Rinfo, family):
    # 50 or more distinct values: R's gam (as smooth.spline) thins the knots
    # (.nknots.smspl, about 190 here) and so does scatter_smooth. gam rounds x
    # to 6 significant digits, so the data are rounded to 5 decimals.
    rng = np.random.default_rng(2)
    n = 3000
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n), 5),
                      'b': np.round(rng.uniform(-2, 2, n), 4),
                      'c': rng.choice(list('xyz'), n)})
    eta = np.sin(2 * np.pi * D['a']) + 0.25 * D['b']**2 + D['c'].map({'x': 0, 'y': .5, 'z': -.5})
    D['y'] = eta + 0.5 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    rpy = _R(Rinfo, D, 'gam')
    klass, yname = FAMILIES[family]
    G = GAM(terms=[s('a', df=4), s('b', df=3), categorical('c')], family=klass(),
            control=GAMControl(epsnr=1e-10, mxitnr=100)).fit(D, D[yname])
    assert len(G.terms_[0].knots_) < 200
    rpy.r('suppressPackageStartupMessages(library(gam))')
    rpy.r(f"""g <- gam::gam({yname} ~ s(a, df=4) + s(b, df=3) + c, data=D, family={family},
               control=gam::gam.control(epsilon=1e-12, bf.epsilon=1e-12,
                                        maxit=100, bf.maxit=200))""")
    eta_R = np.asarray(rpy.r('g$additive.predictors'))
    np.testing.assert_allclose(G.state_.linear_predictor, eta_R, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(G.deviance_, rpy.r('deviance(g)')[0], rtol=1e-5)
    np.testing.assert_allclose(G.edf_.iloc[:2] - 1, np.asarray(rpy.r('g$nl.df')), atol=1e-3)


# more comparisons with R's gam::gam (fixed degrees of freedom). R matches
# each spline's df at the working weights to about 1e-4, so the fits agree
# to about 1e-4 (1e-3 for the binomial's rounder curvature)

GAM_CONTROL = 'gam::gam.control(epsilon=1e-12, bf.epsilon=1e-12, maxit=100, bf.maxit=500)'


def _gam(rpy, formula, family='gaussian', extra=''):
    rpy.r('suppressPackageStartupMessages(library(gam))')
    rpy.r(f'g <- gam::gam({formula}, data=D, family={family}{extra}, control={GAM_CONTROL})')
    return {k: np.asarray(rpy.r(v)) for k, v in
            [('eta', 'g$additive.predictors'), ('deviance', 'deviance(g)'),
             ('null', 'g$null.deviance'), ('dispersion', 'summary(g)$dispersion'),
             ('aic', 'AIC(g)'), ('df_resid', 'g$df.residual'), ('nl_df', 'g$nl.df')]}


def _check_gam(G, R, atol=2e-4):
    # R's additive predictors include the offset
    np.testing.assert_allclose(G.state_.link_parameter, R['eta'], rtol=1e-3, atol=atol)
    np.testing.assert_allclose(G.deviance_, R['deviance'][0], rtol=1e-4)
    np.testing.assert_allclose(G.df_resid_, R['df_resid'][0], atol=2e-3)
    np.testing.assert_allclose(G.dispersion_, R['dispersion'][0], rtol=1e-4)
    np.testing.assert_allclose(G.aic_, R['aic'][0], atol=5e-3)


@pytest.fixture(scope='module')
def kyphosis(Rinfo):
    rpy = _R(Rinfo, pd.DataFrame({'c': ['x']}), 'gam')
    rpy.r('data(kyphosis, package="gam")')
    with Rinfo['np_cv_rules'].context():
        K = rpy.r('kyphosis')
    K = pd.DataFrame(K)
    K['y'] = (K['Kyphosis'] == 'present').astype(float)
    for c in ['Age', 'Number', 'Start']:
        K[c] = K[c].astype(float)
    return K


def test_kyphosis_gam(Rinfo, kyphosis):
    # the classic example of R's gam: binomial, s() with its default df=4
    from glmnet.glm import residuals
    rpy = Rinfo['rpy']
    K = kyphosis
    G = GAM(terms=[s('Age'), s('Number'), s('Start')], family=sm.families.Binomial(),
            control=CONTROL).fit(K, K['y'])
    rpy.r('D <- kyphosis')
    R = _gam(rpy, 'Kyphosis ~ s(Age) + s(Number) + s(Start)', 'binomial')
    _check_gam(G, R, atol=2e-3)
    np.testing.assert_allclose(G.null_deviance_, R['null'][0], rtol=1e-10)
    np.testing.assert_allclose(G.edf_.values - 1, R['nl_df'], atol=1e-3)

    # predictions at new data, and the terms (R's are centered by a constant)
    new = K.iloc[::7]
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('new', new)
    for kind in ['link', 'response']:
        np.testing.assert_allclose(G.predict(new, kind),
                                   np.asarray(rpy.r(f'predict(g, newdata=new, type="{kind}")')),
                                   atol=2e-3)
    terms_R = np.asarray(rpy.r('predict(g, type="terms")'))
    np.testing.assert_allclose(np.asarray(G.fits_), terms_R - terms_R.mean(0), atol=2e-3)
    for kind in ['deviance', 'pearson', 'response']:
        np.testing.assert_allclose(residuals(G, kind),
                                   np.asarray(rpy.r(f'residuals(g, type="{kind}")')), atol=2e-3)


GAM_FAMILIES = {
    'gaussian, weights': (sm.families.Gaussian(), 'gaussian', 'y', True, False),
    'poisson, weights and offset': (sm.families.Poisson(), 'poisson', 'yp', True, True),
    'Gamma, log link': (sm.families.Gamma(sm.families.links.Log()), 'Gamma(link="log")', 'yg', False, False),
    'Gamma, inverse link': (sm.families.Gamma(), 'Gamma', 'yg', False, False),
}


@pytest.fixture(scope='module')
def data_gam(data):
    D = data.copy()
    rng = np.random.default_rng(4)
    eta = np.sin(2 * np.pi * D['a']) + 0.25 * D['b']**2
    D['yg'] = rng.gamma(5, np.exp(0.3 * eta) / 5)
    D['d'] = rng.uniform(0, 1, D.shape[0])
    D['a2'] = np.round(D['a'] + 0.05 * rng.standard_normal(D.shape[0]), 3)
    return D


@pytest.mark.parametrize('case', list(GAM_FAMILIES))
def test_families_gam(Rinfo, data_gam, case):
    family, Rfamily, yname, weights, offset = GAM_FAMILIES[case]
    D = data_gam
    rpy = _R(Rinfo, D, 'gam')
    cols = [yname] + (['w'] if weights else []) + (['o'] if offset else [])
    terms = [s('a', df=4), s('b', df=3)] + ([categorical('c')] if 'inverse' not in case else [])
    G = GAM(terms=terms, family=family, control=CONTROL, response_id=yname,
            weight_id='w' if weights else None,
            offset_id='o' if offset else None).fit(D, D[cols])
    formula = (f'{yname} ~ s(a, 4) + s(b, 3)' + (' + c' if 'inverse' not in case else '') +
               (' + offset(o)' if offset else ''))
    R = _gam(rpy, formula, Rfamily, ', weights=w' if weights else '')
    _check_gam(G, R)
    np.testing.assert_allclose(G.edf_.values[:2] - 1, R['nl_df'], atol=1e-3)
    # the null deviance: R's gam leaves out the offset, glmstar (as R's glm)
    # fits the intercept with it
    null = f'glm({yname} ~ 1{" + offset(o)" if offset else ""}, data=D, family={Rfamily}{", weights=w" if weights else ""})$null.deviance'
    np.testing.assert_allclose(G.null_deviance_, rpy.r(null)[0], rtol=1e-8)


@pytest.mark.parametrize('case', ['linear terms', 'df near 1 and large', 'concurvity'])
def test_terms_gam(Rinfo, data_gam, case):
    D = data_gam
    rpy = _R(Rinfo, D, 'gam')
    terms, formula = {
        'linear terms': ([s('a', df=4), linear('b'), linear('d')], 'y ~ s(a, 4) + b + d'),
        'df near 1 and large': ([s('a', df=1.2), s('b', df=10)], 'y ~ s(a, 1.2) + s(b, 10)'),
        # a2 is a with noise: modified backfitting fits their linear parts jointly
        'concurvity': ([s('a', df=4), s('a2', df=4)], 'y ~ s(a, 4) + s(a2, 4)'),
    }[case]
    G = GAM(terms=terms, control=CONTROL).fit(D, D['y'])
    R = _gam(rpy, formula)
    _check_gam(G, R)
    if case == 'linear terms':
        coef = dict(zip(rpy.r('names(coef(g))'), np.asarray(rpy.r('coef(g)'))))
        np.testing.assert_allclose([G.terms_[1].coef_, G.terms_[2].coef_],
                                   [coef['b'], coef['d']], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('degree', [1, 2])
@pytest.mark.parametrize('span', [0.3, 0.5])
def test_lo_is_R_loess(Rinfo, data_gam, degree, span):
    # a single lo() term is the loess smooth (scatter_smooth's is R's loess
    # computed exactly, surface="direct"), recentered: backfitting centers
    # it and the intercept is the mean of y (loess reproduces constants but
    # does not keep the mean). Its df is the trace of the smoother
    D = data_gam
    rpy = _R(Rinfo, D, 'stats')
    G = GAM(terms=[lo('d', span=span, degree=degree)], control=CONTROL).fit(D, D['y'])
    rpy.r(f'l <- loess(y ~ d, data=D, span={span}, degree={degree}, family="gaussian", '
          'control=loess.control(surface="direct", statistics="exact"))')
    f = np.asarray(rpy.r('fitted(l)'))
    np.testing.assert_allclose(G.predict(D), f - f.mean() + D['y'].mean(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(G.edf_.iloc[0] + 1, rpy.r('l$trace.hat')[0], rtol=1e-8)


@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_lo_mixed_gam(Rinfo, data_gam, family):
    # gam::lo is its own loess (gam's Fortran), about 0.05 from R's loess for
    # these spans: the fits agree to that
    D = data_gam
    rpy = _R(Rinfo, D, 'gam')
    klass, yname = FAMILIES[family]
    G = GAM(terms=[lo('d', span=0.5), s('b', df=3), categorical('c')], family=klass(),
            control=CONTROL).fit(D, D[yname])
    R = _gam(rpy, f'{yname} ~ lo(d, span=0.5) + s(b, 3) + c', family)
    np.testing.assert_allclose(G.state_.link_parameter, R['eta'], atol=5e-2)
    np.testing.assert_allclose(G.deviance_, R['deviance'][0], rtol=2e-3)
    np.testing.assert_allclose(G.df_resid_, R['df_resid'][0], atol=0.5)


@pytest.mark.parametrize('family', list(FAMILIES))
def test_covariance_mgcv(Rinfo, data, family):
    # the Bayesian covariance against mgcv's (sp fixed, as above, and a random
    # effect, bs="re"): edf, the standard errors of predictions (link and
    # response), of the terms and from Ve. The dispersions differ (R's gam's
    # residual df here, mgcv's n - sum(edf)), so they are divided out
    from glmnet.gam import random_effect as re
    D = data.copy()
    rng = np.random.default_rng(5)
    groups = [f'g{i:02d}' for i in range(15)]
    D['g'] = rng.choice(groups, D.shape[0])
    D[FAMILIES[family][1]] = D[FAMILIES[family][1]] + (family == 'gaussian') * D['g'].map(
        dict(zip(groups, rng.normal(0, 0.5, 15))))
    rpy = _R(Rinfo, D, 'mgcv')
    rpy.r('D$g <- factor(D$g)')
    klass, yname = FAMILIES[family]
    lam = (1e-4, 3e-3, 0.02)
    use_offset = family != 'gaussian'
    cols = [yname, 'w'] + (['o'] if use_offset else [])
    G = GAM(terms=[s('a', lam=lam[0]), s('b', lam=lam[1]), categorical('c'), re('g', lam=lam[2])],
            family=klass(), control=GAMControl(epsnr=1e-12, mxitnr=100, solver='joint'),
            response_id=yname, weight_id='w',
            offset_id='o' if use_offset else None).fit(D, D[cols])

    off = ' + offset(o)' if use_offset else ''
    ka, kb = D['a'].nunique(), D['b'].nunique()

    def mgcv(sp):
        rpy.r(f"""m <- mgcv::gam({yname} ~ s(a, bs='cr', k={ka}) + s(b, bs='cr', k={kb}) + c
                                  + s(g, bs='re'){off},
                   data=D, weights=w, family={family}(),
                   knots=list(a=sort(unique(D$a)), b=sort(unique(D$b))),
                   sp=c({float(sp[0])!r}, {float(sp[1])!r}, {float(sp[2])!r}),
                   control=mgcv::gam.control(epsilon=1e-12, maxit=200))""")

    mgcv((1., 1., 1.))
    scale = [rpy.r(f'm$smooth[[{i}]]$S.scale')[0] for i in (1, 2, 3)]
    mgcv([l * D['w'].sum() * c for l, c in zip(lam, scale)])
    np.testing.assert_allclose(G.state_.link_parameter, np.asarray(rpy.r('m$linear.predictors')),
                               rtol=1e-8, atol=1e-8)
    edf_R = np.asarray(rpy.r('sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))'))
    np.testing.assert_allclose(G.edf_influence_.values[[0, 1, 3]], edf_R, rtol=1e-8)

    new = D.iloc[::9]
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('new', new)
    phi, phi_R = G.dispersion_, rpy.r('m$sig2')[0]
    offset = new['o'] if use_offset else None
    for kind in ['link', 'response']:
        fit, se = G.predict(new, kind, offset=offset, return_std=True)
        pred_R = rpy.r(f'predict(m, newdata=new, se.fit=TRUE, type="{kind}")')
        np.testing.assert_allclose(fit, np.asarray(pred_R[0]), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(se / np.sqrt(phi), np.asarray(pred_R[1]) / np.sqrt(phi_R),
                                   rtol=1e-7, atol=1e-10)
    terms_R = rpy.r('predict(m, newdata=new, type="terms", se.fit=TRUE)')
    names_R = list(rpy.r('colnames(predict(m, newdata=new, type="terms"))'))
    for j, name in [(0, 's(a)'), (1, 's(b)'), (3, 's(g)')]:
        _, se = G.partial_effect(j, new[G.terms_[j].columns], return_std=True)
        np.testing.assert_allclose(se / np.sqrt(phi),
                                   np.asarray(terms_R[1])[:, names_R.index(name)] / np.sqrt(phi_R),
                                   rtol=1e-7, atol=1e-10)
    # Ve: through the design rows (mgcv's lpmatrix)
    rows = G.design_.rows(G._columns(new, G.terms_))
    Xp, Ve_R = np.asarray(rpy.r('predict(m, newdata=new, type="lpmatrix")')), np.asarray(rpy.r('m$Ve'))
    np.testing.assert_allclose(np.einsum('ij,jk,ik->i', rows, G.frequentist_covariance_, rows) / phi,
                               np.einsum('ij,jk,ik->i', Xp, Ve_R, Xp) / phi_R, rtol=1e-7, atol=1e-12)
