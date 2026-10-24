"""
REML and ML (`GLMM(method=...)`) against R: mgcv's `gam(method="REML"/"ML")`
(splines `bs="cr"` with our knots, random effects `bs="re"`), and mixed
models: lme4's `lmer` (REML and ML, exact for the Gaussian), `glmer`
(the Laplace approximation, our ML) and nlme's `lme`.

The REML criterion is defined up to a constant that depends on the
parametrization of the unpenalized coefficients (ours: sum-to-zero
constraints; mgcv's and lme4's: treatment contrasts), so REML scores are
compared by their differences; ML scores are compared directly.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')
jax = pytest.importorskip('jax')

from glmnet.gam import GAM, GLMM, GAMControl, s, random_effect as re, linear, categorical
from glmnet.gam.reml import LaplaceCriterion

TIGHT = dict(epsnr=1e-12, mxitnr=100)
GROUPS = [f'g{i:02d}' for i in range(20)]


def _R(Rinfo, D, *packages):
    rpy = Rinfo['rpy']
    for package in packages:
        if not rpy.r(f'isTRUE(requireNamespace("{package}", quietly=TRUE))')[0]:
            pytest.skip(f'requires the R package {package}')
    if D is not None:
        with Rinfo['np_cv_rules'].context():
            rpy.r.assign('D', D)
    return rpy


def _get(Rinfo, expr):
    with Rinfo['np_cv_rules'].context():
        return np.asarray(Rinfo['rpy'].r(expr))


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(11)
    n = 500
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n) * 40) / 40,
                      'b': np.round(rng.uniform(-2, 2, n), 1),
                      'c': rng.choice(list('xyz'), n),
                      'g': rng.choice(GROUPS, n),
                      'v': np.round(rng.uniform(0, 2, n), 2),
                      'o': rng.normal(0, 0.2, n),
                      'w': rng.uniform(0.5, 2, n)})
    effects = dict(zip(GROUPS, rng.normal(0, 0.8, len(GROUPS))))
    slopes = dict(zip(GROUPS, rng.normal(0, 0.5, len(GROUPS))))
    eta = (np.sin(2 * np.pi * D['a']) + 0.3 * D['b']**2 + D['c'].map({'x': 0, 'y': .5, 'z': -.5}) +
           D['g'].map(effects) + D['v'] * D['g'].map(slopes))
    D['y'] = eta + 0.5 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-(eta + D['o'] - 0.5))))
    D['yp'] = rng.poisson(np.exp(0.4 * eta + D['o']))
    D['yg'] = rng.gamma(4, np.exp(0.3 * eta + D['o']) / 4)
    return D


# family, R family, response, prior weights, offset
MGCV = {'gaussian, weights': (sm.families.Gaussian(), 'gaussian()', 'y', True, False),
        'binomial, offset': (sm.families.Binomial(), 'binomial()', 'yb', False, True),
        'poisson, weights and offset': (sm.families.Poisson(), 'poisson()', 'yp', True, True),
        'Gamma (log), offset': (sm.families.Gamma(sm.families.links.Log()), 'Gamma(link="log")',
                                'yg', False, True)}


def _mgcv(rpy, D, y, Rfamily, weights, offset, method, sp=None):
    ka, kb = D['a'].nunique(), D['b'].nunique()
    extra = (', weights=w' if weights else '') + (f', sp=c({", ".join(repr(float(v)) for v in sp)})'
                                                   if sp is not None else '')
    off = ' + offset(o)' if offset else ''
    rpy.r(f"""m <- mgcv::gam({y} ~ s(a, bs='cr', k={ka}) + s(b, bs='cr', k={kb}) + c + s(g, bs='re')
                              + s(g, v, bs='re'){off},
               data=D, family={Rfamily}, method='{method}'{extra},
               knots=list(a=sort(unique(D$a)), b=sort(unique(D$b))),
               control=mgcv::gam.control(epsilon=1e-12, maxit=200, newton=list(conv.tol=1e-10)))""")


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('case', list(MGCV))
def test_mgcv(Rinfo, data, case, method):
    family, Rfamily, y, weights, offset = MGCV[case]
    D = data
    rpy = _R(Rinfo, D, 'mgcv')
    rpy.r('D$c <- factor(D$c); D$g <- factor(D$g)')
    cols = [y] + (['w'] if weights else []) + (['o'] if offset else [])
    G = GLMM(terms=[s('a'), s('b'), categorical('c'), re('g'), re('g', by='v')], family=family,
            method=method, control=GAMControl(**TIGHT), response_id=y,
            weight_id='w' if weights else None,
            offset_id='o' if offset else None).fit(D, D[cols])
    _mgcv(rpy, D, y, Rfamily, weights, offset, method)
    sw = D['w'].sum() if weights else D.shape[0]
    scale = np.array([rpy.r(f'm$smooth[[{i}]]$S.scale')[0] for i in range(1, 5)])
    sp_R = _get(Rinfo, 'm$sp')
    edf_R = _get(Rinfo, 'sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))')
    # the smoothing parameters, on our scale, but at the boundary (a
    # variance 0, where the criterion is flat in it)
    inner = edf_R > 0.05
    np.testing.assert_allclose(G.smoothing_.lam[inner], (sp_R / scale / sw)[inner], rtol=2e-3)
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'm$linear.predictors'),
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(G.edf_.values[[0, 1, 3, 4]], edf_R, rtol=1e-3, atol=1e-3)
    # the dispersion: the REML or ML estimate (mgcv's reml.scale; its sig2 is
    # that for the Gaussian REML, Fletcher's estimate for other families)
    phi_R = _get(Rinfo, 'm$reml.scale')[0]
    np.testing.assert_allclose(G.dispersion_, phi_R, rtol=1e-4)
    score_R = _get(Rinfo, 'm$gcv.ubre')[0]
    if method == 'ML':
        # (ours is at least as small: mgcv stops sooner on a flat criterion)
        np.testing.assert_allclose(G.reml_score_, score_R, rtol=1e-6)
        assert G.reml_score_ <= score_R + 1e-6 * abs(score_R)
    else:
        # the criterion at twice mgcv's smoothing parameters, less at them (the
        # dispersion at its minimum given them): the constant cancels
        _mgcv(rpy, D, y, Rfamily, weights, offset, method, sp=2 * sp_R)
        diff_R = _get(Rinfo, 'm$gcv.ubre')[0] - score_R
        with jax.enable_x64(True):
            C = LaplaceCriterion(G, [0, 1, 3, 4], method=method)

            def profiled(rho):
                if not C.free_dispersion:
                    return C.value_and_grad(rho)[0]
                from scipy.optimize import minimize_scalar
                return minimize_scalar(lambda t: C.value_and_grad(np.append(rho, t))[0],
                                       bracket=(np.log(phi_R) - 0.1, np.log(phi_R) + 0.1),
                                       tol=1e-12).fun
            rho = np.log(sp_R / scale)
            diff = profiled(rho + np.log(2)) - profiled(rho)
        np.testing.assert_allclose(diff, diff_R, rtol=1e-5, atol=1e-7)
        _mgcv(rpy, D, y, Rfamily, weights, offset, method)
    # standard errors at the estimates (mgcv's Vp, at its dispersion)
    new = D.iloc[::23]
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('new', new)
    _, se = G.predict(new, 'link', offset=new['o'] if offset else None, return_std=True)
    se_R = _get(Rinfo, 'predict(m, newdata=new, se.fit=TRUE)$se.fit')
    sig2_R = _get(Rinfo, 'm$sig2')[0]
    np.testing.assert_allclose(se / np.sqrt(G.dispersion_), se_R / np.sqrt(sig2_R), rtol=2e-3)
    # the variance components: mgcv's gam.vcomp (standard deviations, on the
    # penalties' original scale), off the boundary
    vc_R = _get(Rinfo, 'v <- mgcv::gam.vcomp(m, rescale=TRUE); if (is.list(v)) v$all else v')
    sd = G.variance_components_['sd'].values[:4]
    np.testing.assert_allclose(sd[inner], vc_R[inner], rtol=1e-3)


@pytest.fixture(scope='module')
def sleepstudy(Rinfo):
    rpy = _R(Rinfo, None, 'lme4')
    with Rinfo['np_cv_rules'].context():
        S = pd.DataFrame(rpy.r('lme4::sleepstudy'))
    S['Subject'] = S['Subject'].astype(str)
    S['Days'] = S['Days'].astype(float)
    rpy.r('S <- lme4::sleepstudy')
    return S


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_lmer_sleepstudy(Rinfo, sleepstudy, method):
    # independent random intercepts and slopes: (1|Subject) + (0 + Days|Subject).
    # Gaussian: the criterion is exact, minus lmer's log-likelihood (ours
    # centers Days, which does not change the REML's determinant)
    rpy = Rinfo['rpy']
    S = sleepstudy
    G = GLMM(terms=[linear('Days'), re('Subject'), re('Subject', by='Days')], method=method,
            control=GAMControl(**TIGHT)).fit(S, S['Reaction'])
    rpy.r(f"""l <- lme4::lmer(Reaction ~ Days + (1|Subject) + (0 + Days|Subject), data=S,
                              REML={'TRUE' if method == 'REML' else 'FALSE'},
                              control=lme4::lmerControl(optCtrl=list(ftol_abs=1e-12, xtol_abs=1e-10)))""")
    vc = G.variance_components_
    sd_R = _get(Rinfo, 'as.data.frame(lme4::VarCorr(l))$sdcor')      # Subject, Subject.1, Residual
    np.testing.assert_allclose(vc['sd'].values, sd_R, rtol=1e-4)
    np.testing.assert_allclose(G.terms_[0].coef_, _get(Rinfo, 'lme4::fixef(l)')[1], rtol=1e-6)
    np.testing.assert_allclose(G.predict(S), _get(Rinfo, 'fitted(l)'), rtol=1e-5)
    ranef = _get(Rinfo, 'as.matrix(lme4::ranef(l)$Subject)')
    subjects = list(rpy.r('rownames(lme4::ranef(l)$Subject)'))
    np.testing.assert_allclose(G.terms_[1].effects_.loc[subjects].values, ranef[:, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(G.terms_[2].effects_.loc[subjects].values, ranef[:, 1], rtol=1e-4, atol=1e-4)
    # (ours at least as large: lmer stops sooner)
    loglik_R = _get(Rinfo, 'as.numeric(logLik(l))')[0]
    np.testing.assert_allclose(-G.reml_score_, loglik_R, rtol=1e-8)
    assert -G.reml_score_ >= loglik_R - 1e-8 * abs(loglik_R)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_lme_orthodont(Rinfo, method):
    # nlme's lme: a random intercept with a factor (Sex) and a slope (age)
    rpy = _R(Rinfo, None, 'nlme')
    with Rinfo['np_cv_rules'].context():
        O = pd.DataFrame(rpy.r('as.data.frame(nlme::Orthodont)'))
    for c in ['Subject', 'Sex']:
        O[c] = O[c].astype(str)
    G = GLMM(terms=[linear('age'), categorical('Sex'), re('Subject')], method=method,
            control=GAMControl(**TIGHT)).fit(O, O['distance'])
    rpy.r(f"""o <- nlme::lme(distance ~ age + Sex, random=~1|Subject, data=nlme::Orthodont,
                             method='{method}', control=nlme::lmeControl(tolerance=1e-10))""")
    sd_R = _get(Rinfo, 'as.numeric(nlme::VarCorr(o)[, "StdDev"])')        # Subject, Residual
    np.testing.assert_allclose(G.variance_components_['sd'].values, sd_R, rtol=1e-5)
    np.testing.assert_allclose(G.terms_[0].coef_, _get(Rinfo, 'nlme::fixef(o)')[1], rtol=1e-7)
    np.testing.assert_allclose(G.predict(O), _get(Rinfo, 'fitted(o)'), rtol=1e-7)
    if method == 'ML':
        np.testing.assert_allclose(-G.reml_score_, _get(Rinfo, 'as.numeric(logLik(o))')[0], rtol=1e-8)


@pytest.mark.parametrize('nAGQ', [0, 1])
def test_glmer_cbpp(Rinfo, nAGQ):
    # binomial counts: incidence of size, a herd random intercept. Our ML (as
    # mgcv's) is the Laplace approximation with the fixed effects at the
    # penalized likelihood's mode, glmer's nAGQ=0; with nAGQ=1 (its default)
    # glmer also maximizes the approximation in them: close
    rpy = _R(Rinfo, None, 'lme4')
    with Rinfo['np_cv_rules'].context():
        C = pd.DataFrame(rpy.r('lme4::cbpp'))
    for c in ['herd', 'period']:
        C[c] = C[c].astype(str)
    C['p'] = C['incidence'] / C['size']
    G = GLMM(terms=[categorical('period'), re('herd')], family=sm.families.Binomial(), method='ML',
            control=GAMControl(**TIGHT), response_id='p', weight_id='size').fit(C, C[['p', 'size']])
    rpy.r(f"""b <- lme4::glmer(cbind(incidence, size - incidence) ~ period + (1|herd), data=lme4::cbpp,
                              family=binomial, nAGQ={nAGQ},
                              control=lme4::glmerControl(optCtrl=list(ftol_abs=1e-12, xtol_abs=1e-10)))""")
    tol = dict(rtol=1e-4, atol=1e-5) if nAGQ == 0 else dict(rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(G.variance_components_.loc['random_effect(herd)', 'sd'],
                               _get(Rinfo, 'as.data.frame(lme4::VarCorr(b))$sdcor')[0], **tol)
    np.testing.assert_allclose(G.predict(C, 'link'), _get(Rinfo, 'predict(b, type="link")'), **tol)
    if nAGQ == 0:
        # (ours at least as large: glmer stops sooner)
        loglik_R = _get(Rinfo, 'as.numeric(logLik(b))')[0]
        np.testing.assert_allclose(-G.reml_score_, loglik_R, rtol=1e-6)
        assert -G.reml_score_ >= loglik_R - 1e-8 * abs(loglik_R)


@pytest.mark.parametrize('nAGQ', [0, 1])
def test_glmer_poisson(Rinfo, data, nAGQ):
    # Poisson with an offset: a random intercept and a fixed slope (nAGQ as
    # for cbpp)
    D = data
    rpy = _R(Rinfo, D, 'lme4')
    G = GLMM(terms=[linear('b'), categorical('c'), re('g')], family=sm.families.Poisson(), method='ML',
            control=GAMControl(**TIGHT), response_id='yp', offset_id='o').fit(D, D[['yp', 'o']])
    rpy.r(f"""p <- lme4::glmer(yp ~ b + c + (1|g) + offset(o), data=D, family=poisson, nAGQ={nAGQ},
                              control=lme4::glmerControl(optCtrl=list(ftol_abs=1e-12, xtol_abs=1e-10)))""")
    tol = dict(rtol=1e-4, atol=1e-5) if nAGQ == 0 else dict(rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(G.variance_components_.loc['random_effect(g)', 'sd'],
                               _get(Rinfo, 'as.data.frame(lme4::VarCorr(p))$sdcor')[0], **tol)
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'predict(p, type="link")'), **tol)
    np.testing.assert_allclose(G.terms_[0].coef_, _get(Rinfo, 'lme4::fixef(p)')[1], **tol)
    if nAGQ == 0:
        loglik_R = _get(Rinfo, 'as.numeric(logLik(p))')[0]
        np.testing.assert_allclose(-G.reml_score_, loglik_R, rtol=1e-6)
        assert -G.reml_score_ >= loglik_R - 1e-8 * abs(loglik_R)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_mgcv_wage(Rinfo, method):
    # real data (ISLP's Wage): wage ~ s(year) + s(age) + education, the
    # splines' knots ours (age's thinned, as R's smooth.spline) given to mgcv
    ISLP = pytest.importorskip('ISLP')
    Wage = ISLP.load_data('Wage')
    G = GLMM(terms=[s('year'), s('age'), categorical('education')],
            method=method, control=GAMControl(**TIGHT)).fit(Wage, Wage['wage'])
    knots = {t.name: t.knots_ for t in G.terms_[:2]}
    W = Wage[['wage', 'year', 'age']].assign(education=Wage['education'].astype(str))
    rpy = _R(Rinfo, W, 'mgcv')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('ky', knots['s(year)'])
        rpy.r.assign('ka', knots['s(age)'])
    rpy.r(f"""D$education <- factor(D$education)
              m <- mgcv::gam(wage ~ s(year, bs='cr', k=length(ky)) + s(age, bs='cr', k=length(ka)) + education,
                             data=D, method='{method}', knots=list(year=ky, age=ka),
                             control=mgcv::gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))""")
    # (year's spline is nearly linear: a large smoothing parameter, where the
    # criterion is flat)
    np.testing.assert_allclose(G.predict(Wage), _get(Rinfo, 'fitted(m)'), rtol=5e-5)
    edf_R = _get(Rinfo, 'sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))')
    np.testing.assert_allclose(G.edf_.values[:2], edf_R, rtol=1e-3)
    np.testing.assert_allclose(G.dispersion_, _get(Rinfo, 'm$reml.scale')[0], rtol=1e-6)
    if method == 'ML':
        # (ours is lower: mgcv stops sooner)
        score_R = _get(Rinfo, 'm$gcv.ubre')[0]
        np.testing.assert_allclose(G.reml_score_, score_R, rtol=1e-6)
        assert G.reml_score_ <= score_R + 1e-9 * abs(score_R)
