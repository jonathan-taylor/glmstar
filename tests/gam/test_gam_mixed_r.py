"""
Classical mixed models as GAMs with `re()` terms (docs/gam_mixed_models.md),
against R: lme4's `lmer` (REML and ML) and `glmer` (nAGQ=0: our ML, exact;
nAGQ=1: close), nlme's `lme` and mgcv's `gam` with `bs="re"`.

The data are simulated with the designs of classical examples: a one-way
layout of unequal groups; crossed (24 plates x 6 samples, as lme4's
Penicillin); nested (10 batches x 3 casks x 2, as Pastes); longitudinal
(18 subjects x 10 days, as sleepstudy); binomial counts (15 herds x 4
periods, as cbpp); overdispersed Poisson counts (an observation-level random
effect); and a smooth with a random intercept (a GAMM).

For the Gaussian the REML criterion is exact, and with no factors among the
fixed effects (whose sum-to-zero constraints change its constant) it is
minus lmer's REML log-likelihood; ML is compared directly.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytest.importorskip('scatter_smooth')
pytest.importorskip('jax')

from glmnet.gam import GAM, GAMControl, s, re, linear, categorical

TIGHT = dict(epsnr=1e-12, mxitnr=100)
LMER_CONTROL = 'lme4::lmerControl(optCtrl=list(ftol_abs=1e-12, xtol_abs=1e-10))'
# (bobyqa, glmer's optimizer for nAGQ=0, with a tight tolerance, and tight
# inner iterations: at the defaults glmer stops 2e-4 short in the sd)
GLMER_CONTROL = ("lme4::glmerControl(optimizer='bobyqa', tolPwrss=1e-12, "
                 "optCtrl=list(rhoend=1e-10))")
# mgcv's gam.vcomp: the standard deviations, with the scale's
VCOMP = 'v <- mgcv::gam.vcomp(m, rescale=TRUE); if (is.list(v)) v$vc[, 1] else v[, 1]'
MGCV_CONTROL = 'mgcv::gam.control(epsilon=1e-12, maxit=200, newton=list(conv.tol=1e-10))'


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


def _effects(rng, labels, sd):
    labels = np.unique(labels)
    return dict(zip(labels, rng.normal(0, sd, len(labels))))


# the data (as on the page)

def oneway():
    rng = np.random.default_rng(1)
    sizes = rng.integers(2, 41, 30)
    D = pd.DataFrame({'school': np.repeat([f's{i:02d}' for i in range(30)], sizes)})
    D['score'] = 50 + D['school'].map(_effects(rng, D['school'], 3)) + rng.normal(0, 8, D.shape[0])
    return D


def crossed():
    rng = np.random.default_rng(2)
    D = pd.DataFrame({'plate': np.repeat([f'p{i:02d}' for i in range(24)], 6),
                      'sample': np.tile(list('ABCDEF'), 24)})
    D['diameter'] = (23 + D['plate'].map(_effects(rng, D['plate'], 0.85))
                     + D['sample'].map(_effects(rng, D['sample'], 1.9))
                     + rng.normal(0, 0.55, D.shape[0]))
    return D


def nested():
    rng = np.random.default_rng(3)
    D = pd.DataFrame({'batch': np.repeat([f'B{i:02d}' for i in range(10)], 6),
                      'cask': np.tile(np.repeat(list('abc'), 2), 10)})
    D['sample'] = D['batch'] + ':' + D['cask']
    D['strength'] = (60 + D['batch'].map(_effects(rng, D['batch'], 1.3))
                     + D['sample'].map(_effects(rng, D['sample'], 2.9))
                     + rng.normal(0, 0.8, D.shape[0]))
    return D


def longitudinal():
    rng = np.random.default_rng(4)
    D = pd.DataFrame({'subject': np.repeat([f'S{i:02d}' for i in range(18)], 10),
                      'days': np.tile(np.arange(10.), 18)})
    D['reaction'] = (250 + 10 * D['days'] + D['subject'].map(_effects(rng, D['subject'], 25))
                     + D['days'] * D['subject'].map(_effects(rng, D['subject'], 6))
                     + rng.normal(0, 25, D.shape[0]))
    return D


def herds():
    rng = np.random.default_rng(5)
    D = pd.DataFrame({'herd': np.repeat([f'h{i:02d}' for i in range(15)], 4),
                      'period': np.tile(list('1234'), 15)})
    D['size'] = rng.integers(5, 35, D.shape[0])
    eta = (-1.4 + D['period'].map({'1': 0, '2': -1, '3': -1.1, '4': -1.6})
           + D['herd'].map(_effects(rng, D['herd'], 0.65)))
    D['incidence'] = rng.binomial(D['size'], 1 / (1 + np.exp(-eta)))
    D['p'] = D['incidence'] / D['size']
    return D


def counts():
    rng = np.random.default_rng(6)
    n = 200
    D = pd.DataFrame({'x': np.round(rng.uniform(0, 1, n), 3),
                      'obs': [f'o{i:03d}' for i in range(n)]})
    D['y'] = rng.poisson(np.exp(1 + D['x'] + rng.normal(0, 0.5, n)))
    return D


def growth():
    rng = np.random.default_rng(7)
    D = pd.DataFrame({'subject': np.repeat([f'S{i:02d}' for i in range(30)], 12),
                      'time': np.tile(np.linspace(0, 1, 12), 30)})
    D['y'] = (np.sin(2 * np.pi * D['time']) + D['subject'].map(_effects(rng, D['subject'], 0.7))
              + rng.normal(0, 0.3, D.shape[0]))
    return D


def _gaussian(D, terms, y, method):
    return GAM(terms=terms, method=method, control=GAMControl(**TIGHT)).fit(D, D[y])


def _check_lmer(Rinfo, G, D, formula, method, terms, grouping, rtol_sd=1e-4):
    """
    Against `lmer(formula)`: the standard deviations (`terms`: our variance
    components in lmer's order), the fitted values, the BLUPs (`grouping`:
    our term's index, lmer's ranef as an R expression and a function of its
    row names giving our levels) and the
    log-likelihood.
    """
    rpy = Rinfo['rpy']
    rpy.r(f"l <- lme4::lmer({formula}, data=D, REML={'TRUE' if method == 'REML' else 'FALSE'}, "
          f"control={LMER_CONTROL})")
    vc = G.variance_components_
    np.testing.assert_allclose(vc['sd'].loc[terms + ['scale']].values,
                               _get(Rinfo, 'as.data.frame(lme4::VarCorr(l))$sdcor'), rtol=rtol_sd)
    np.testing.assert_allclose(G.predict(D), _get(Rinfo, 'fitted(l)'), rtol=1e-6)
    for j, expr, relabel in grouping:
        levels = [relabel(v) for v in rpy.r(f'rownames({expr})')]
        np.testing.assert_allclose(G.terms_[j].effects_.loc[levels].values,
                                   _get(Rinfo, f'{expr}[, 1]'), rtol=1e-4, atol=1e-4)
    # (ours at least as large: lmer stops sooner)
    loglik_R = _get(Rinfo, 'as.numeric(logLik(l))')[0]
    np.testing.assert_allclose(-G.reml_score_, loglik_R, rtol=1e-8)
    assert -G.reml_score_ >= loglik_R - 1e-8 * abs(loglik_R)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_oneway(Rinfo, method):
    D = oneway()
    rpy = _R(Rinfo, D, 'lme4')
    G = _gaussian(D, [re('school')], 'score', method)
    _check_lmer(Rinfo, G, D, 'score ~ 1 + (1|school)', method, ['re(school)'],
                [(0, 'lme4::ranef(l)$school', str)])
    np.testing.assert_allclose(G.intercept_, _get(Rinfo, 'lme4::fixef(l)')[0], rtol=1e-7)


def test_oneway_mgcv(Rinfo):
    D = oneway()
    rpy = _R(Rinfo, D, 'mgcv')
    G = _gaussian(D, [re('school')], 'score', 'REML')
    rpy.r(f"""D$school <- factor(D$school)
              m <- mgcv::gam(score ~ s(school, bs='re'), data=D, method='REML', control={MGCV_CONTROL})""")
    np.testing.assert_allclose(G.variance_components_['sd'].values,
                               _get(Rinfo, VCOMP), rtol=1e-4)
    np.testing.assert_allclose(G.predict(D), _get(Rinfo, 'fitted(m)'), rtol=1e-6)
    # the posterior standard errors of the fitted values (mgcv's Vp)
    _, se = G.predict(D, return_std=True)
    np.testing.assert_allclose(se, _get(Rinfo, 'predict(m, se.fit=TRUE)$se.fit'), rtol=1e-3)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_crossed(Rinfo, method):
    D = crossed()
    _R(Rinfo, D, 'lme4')
    G = _gaussian(D, [re('plate'), re('sample')], 'diameter', method)
    _check_lmer(Rinfo, G, D, 'diameter ~ 1 + (1|plate) + (1|sample)', method,
                ['re(plate)', 're(sample)'],
                [(0, 'lme4::ranef(l)$plate', str), (1, 'lme4::ranef(l)$sample', str)])


def test_crossed_mgcv(Rinfo):
    D = crossed()
    rpy = _R(Rinfo, D, 'mgcv')
    G = _gaussian(D, [re('plate'), re('sample')], 'diameter', 'REML')
    rpy.r(f"""D$plate <- factor(D$plate); D$sample <- factor(D$sample)
              m <- mgcv::gam(diameter ~ s(plate, bs='re') + s(sample, bs='re'), data=D,
                             method='REML', control={MGCV_CONTROL})""")
    np.testing.assert_allclose(G.variance_components_['sd'].values,
                               _get(Rinfo, VCOMP), rtol=1e-4)
    np.testing.assert_allclose(G.predict(D), _get(Rinfo, 'fitted(m)'), rtol=1e-6)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_nested_lmer(Rinfo, method):
    # (1|batch/cask) is (1|batch) + (1|batch:cask): lmer orders the terms
    # by their number of levels, the casks first
    D = nested()
    _R(Rinfo, D, 'lme4')
    G = _gaussian(D, [re('batch'), re('sample')], 'strength', method)
    _check_lmer(Rinfo, G, D, 'strength ~ 1 + (1|batch/cask)', method,
                ['re(sample)', 're(batch)'],
                [(0, 'lme4::ranef(l)$batch', str),
                 (1, 'lme4::ranef(l)$`cask:batch`', lambda v: ':'.join(v.split(':')[::-1]))])


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_nested_lme(Rinfo, method):
    D = nested()
    rpy = _R(Rinfo, D, 'nlme')
    G = _gaussian(D, [re('batch'), re('sample')], 'strength', method)
    rpy.r(f"""o <- nlme::lme(strength ~ 1, random=~1|batch/cask, data=D, method='{method}',
                             control=nlme::lmeControl(tolerance=1e-10))""")
    # VarCorr's rows: batch (its variance, sd), cask (...), residual
    sd_R = _get(Rinfo, 'as.numeric(nlme::VarCorr(o)[c(2, 4, 5), "StdDev"])')
    np.testing.assert_allclose(G.variance_components_['sd'].values, sd_R, rtol=1e-4)
    np.testing.assert_allclose(G.predict(D), _get(Rinfo, 'fitted(o)'), rtol=1e-6)
    np.testing.assert_allclose(-G.reml_score_, _get(Rinfo, 'as.numeric(logLik(o))')[0], rtol=1e-7)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_slopes(Rinfo, method):
    # independent random intercepts and slopes: (1|subject) + (0 + days|subject),
    # lmer's (days||subject)
    D = longitudinal()
    _R(Rinfo, D, 'lme4')
    G = _gaussian(D, [linear('days'), re('subject'), re('subject', by='days')], 'reaction', method)
    _check_lmer(Rinfo, G, D, 'reaction ~ days + (days||subject)', method,
                ['re(subject)', 're(subject, by=days)'], [])
    ranef = _get(Rinfo, 'as.matrix(lme4::ranef(l)$subject)')
    subjects = list(Rinfo['rpy'].r('rownames(lme4::ranef(l)$subject)'))
    np.testing.assert_allclose(G.terms_[1].effects_.loc[subjects].values, ranef[:, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(G.terms_[2].effects_.loc[subjects].values, ranef[:, 1], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(G.terms_[0].coef_, _get(Rinfo, 'lme4::fixef(l)')[1], rtol=1e-6)
    # its standard error: the posterior covariance's, lmer's vcov (GLS)
    block = G.design_.slices[0]
    np.testing.assert_allclose(np.sqrt(G.posterior_covariance_[block, block]).item(),
                               _get(Rinfo, 'sqrt(as.matrix(vcov(l))[2, 2])')[0], rtol=1e-4)


def test_slopes_mgcv(Rinfo):
    D = longitudinal()
    rpy = _R(Rinfo, D, 'mgcv')
    G = _gaussian(D, [linear('days'), re('subject'), re('subject', by='days')], 'reaction', 'REML')
    rpy.r(f"""D$subject <- factor(D$subject)
              m <- mgcv::gam(reaction ~ days + s(subject, bs='re') + s(subject, days, bs='re'),
                             data=D, method='REML', control={MGCV_CONTROL})""")
    np.testing.assert_allclose(G.variance_components_['sd'].values,
                               _get(Rinfo, VCOMP), rtol=1e-4)
    np.testing.assert_allclose(G.predict(D), _get(Rinfo, 'fitted(m)'), rtol=1e-6)


def _binomial(D, method):
    return GAM(terms=[categorical('period'), re('herd')], family=sm.families.Binomial(), method=method,
               control=GAMControl(**TIGHT), response_id='p', weight_id='size').fit(D, D[['p', 'size']])


@pytest.mark.parametrize('nAGQ', [0, 1])
def test_glmer_binomial(Rinfo, nAGQ):
    # our ML is the Laplace approximation at the penalized likelihood's
    # mode, glmer's nAGQ=0; nAGQ=1 also maximizes it in the fixed effects
    D = herds()
    rpy = _R(Rinfo, D, 'lme4')
    G = _binomial(D, 'ML')
    rpy.r(f"""b <- lme4::glmer(cbind(incidence, size - incidence) ~ period + (1|herd), data=D,
                              family=binomial, nAGQ={nAGQ}, control={GLMER_CONTROL})""")
    tol = dict(rtol=1e-4, atol=1e-5) if nAGQ == 0 else dict(rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(G.variance_components_.loc['re(herd)', 'sd'],
                               _get(Rinfo, 'as.data.frame(lme4::VarCorr(b))$sdcor')[0], **tol)
    np.testing.assert_allclose(G.predict(D, 'link'), _get(Rinfo, 'predict(b, type="link")'), **tol)
    if nAGQ == 0:
        # (with nAGQ=1 the intercept moves, and with it all the BLUPs)
        herds_ = list(rpy.r('rownames(lme4::ranef(b)$herd)'))
        np.testing.assert_allclose(G.terms_[1].effects_.loc[herds_].values,
                                   _get(Rinfo, 'lme4::ranef(b)$herd[, 1]'), **tol)
        # (ours at least as large: glmer stops sooner)
        loglik_R = _get(Rinfo, 'as.numeric(logLik(b))')[0]
        np.testing.assert_allclose(-G.reml_score_, loglik_R, rtol=1e-6)
        assert -G.reml_score_ >= loglik_R - 1e-8 * abs(loglik_R)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_mgcv_binomial(Rinfo, method):
    # mgcv's Laplace REML and ML (no PQL): the same criterion
    D = herds()
    rpy = _R(Rinfo, D, 'mgcv')
    G = _binomial(D, method)
    rpy.r(f"""D$herd <- factor(D$herd); D$period <- factor(D$period)
              m <- mgcv::gam(p ~ period + s(herd, bs='re'), weights=size, data=D, family=binomial,
                             method='{method}', control={MGCV_CONTROL})""")
    np.testing.assert_allclose(G.variance_components_.loc['re(herd)', 'sd'],
                               _get(Rinfo, VCOMP)[0], rtol=1e-4)
    np.testing.assert_allclose(G.predict(D, 'link'), _get(Rinfo, 'predict(m, type="link")'),
                               rtol=1e-5, atol=1e-6)
    if method == 'ML':
        np.testing.assert_allclose(G.reml_score_, _get(Rinfo, 'm$gcv.ubre')[0], rtol=1e-6)


@pytest.mark.parametrize('nAGQ', [0, 1])
def test_glmer_olre(Rinfo, nAGQ):
    # an observation-level random effect: Poisson-lognormal counts
    D = counts()
    rpy = _R(Rinfo, D, 'lme4')
    G = GAM(terms=[linear('x'), re('obs')], family=sm.families.Poisson(), method='ML',
            control=GAMControl(**TIGHT)).fit(D, D['y'])
    rpy.r(f"""p <- lme4::glmer(y ~ x + (1|obs), data=D, family=poisson, nAGQ={nAGQ},
                              control={GLMER_CONTROL})""")
    tol = dict(rtol=1e-4, atol=1e-5) if nAGQ == 0 else dict(rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(G.variance_components_.loc['re(obs)', 'sd'],
                               _get(Rinfo, 'as.data.frame(lme4::VarCorr(p))$sdcor')[0], **tol)
    np.testing.assert_allclose(G.terms_[0].coef_, _get(Rinfo, 'lme4::fixef(p)')[1], **tol)
    np.testing.assert_allclose(G.predict(D, 'link'), _get(Rinfo, 'predict(p, type="link")'), **tol)
    if nAGQ == 0:
        loglik_R = _get(Rinfo, 'as.numeric(logLik(p))')[0]
        np.testing.assert_allclose(-G.reml_score_, loglik_R, rtol=1e-6)
        assert -G.reml_score_ >= loglik_R - 1e-8 * abs(loglik_R)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_gamm_mgcv(Rinfo, method):
    # a smooth of time and a random intercept per subject; our knots given
    # to mgcv's cubic regression spline
    D = growth()
    G = _gaussian(D, [s('time'), re('subject')], 'y', method)
    rpy = _R(Rinfo, D, 'mgcv')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('kt', G.terms_[0].knots_)
    rpy.r(f"""D$subject <- factor(D$subject)
              m <- mgcv::gam(y ~ s(time, bs='cr', k=length(kt)) + s(subject, bs='re'), data=D,
                             method='{method}', knots=list(time=kt), control={MGCV_CONTROL})""")
    np.testing.assert_allclose(G.predict(D), _get(Rinfo, 'fitted(m)'), rtol=1e-5, atol=1e-6)
    edf_R = _get(Rinfo, 'sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))')
    np.testing.assert_allclose(G.edf_.values, edf_R, rtol=1e-3)
    # the random intercept's and the residual standard deviations
    vc_R = _get(Rinfo, VCOMP)
    np.testing.assert_allclose(G.variance_components_['sd'].values[1:], vc_R[1:], rtol=1e-4)
    np.testing.assert_allclose(G.dispersion_, _get(Rinfo, 'm$reml.scale')[0], rtol=1e-5)
    if method == 'ML':
        np.testing.assert_allclose(G.reml_score_, _get(Rinfo, 'm$gcv.ubre')[0], rtol=1e-6)
