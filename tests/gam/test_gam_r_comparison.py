"""
`ISLP.gam.GAM` against R:

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

from ISLP.gam import GAM, GAMControl, s, factor

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
    G = GAM(terms=[s('a', lam=lam[0]), s('b', lam=lam[1]), factor('c')],
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
    G = GAM(terms=[s('a', df=4), s('b', df=3), factor('c')], family=klass(),
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
    G = GAM(terms=[s('a', df=4), s('b', df=3), factor('c')], family=klass(),
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
