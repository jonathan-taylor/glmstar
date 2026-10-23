"""
Gaussian process smooths (`gp()`) against mgcv's `s(..., bs="gp")`.

mgcv's basis is `[C(x, K) U, t(x)]` with the penalty `diag(L)`, `U` and `L`
the leading eigenvectors and eigenvalues of the correlations of the knots,
found by Lanczos iterations (`slanczos`) to a tolerance of `sqrt(eps)`
(relative to the largest eigenvalue), so that its eigenvectors of small
eigenvalues are accurate to about 1e-6. Ours are exact (LAPACK): the bases
agree with the exact eigendecomposition (`eigen(E)` in R) to rounding, and
fits with mgcv's own to about 1e-4. With as many knots as `k` less the
trend's dimension, mgcv uses the kernel at the knots without the
eigendecomposition, the same space as ours: there the fits at fixed `sp`
(`lam * sum(w) * S.scale`), the edf, the standard errors and REML/ML agree
to rounding (or mgcv's convergence).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from glmnet.gam import GAM, GLMM, GAMControl, gaussian_process as gp, categorical, random_effect as re

TIGHT = dict(epsnr=1e-12, mxitnr=100)
GROUPS = [f'g{i}' for i in range(10)]


def _model(method, **kwargs):
    # fixed smoothing is a GAM; estimated, a GLMM
    return GAM(**kwargs) if method == 'fixed' else GLMM(method=method, **kwargs)



def _R(Rinfo, D):
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    rpy.r('suppressMessages(library(mgcv))')
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', D)
    rpy.r('D$c <- factor(D$c); D$g <- factor(D$g)')
    return rpy


def _get(Rinfo, expr):
    with Rinfo['np_cv_rules'].context():
        return np.asarray(Rinfo['rpy'].r(expr))


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(7)
    n = 400
    D = pd.DataFrame({'x': np.round(rng.uniform(0, 1, n), 2),
                      'u': np.round(rng.uniform(0, 1, n), 2),
                      'v': np.round(rng.uniform(0, 1, n), 2),
                      'c': rng.choice(list('abc'), n),
                      'g': rng.choice(GROUPS, n),
                      'o': rng.normal(0, 0.2, n),
                      'w': rng.uniform(0.5, 2, n)})
    effects = dict(zip(GROUPS, rng.normal(0, 0.5, len(GROUPS))))
    eta = (np.sin(2 * np.pi * D['x']) + np.sin(3 * D['u']) * np.cos(4 * D['v']) +
           D['c'].map({'a': 0, 'b': .5, 'c': -.3}) + D['g'].map(effects))
    D['y'] = eta + 0.3 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-(2 * eta + D['o']))))
    D['yp'] = rng.poisson(np.exp(0.8 * eta + D['o']))
    return D


def _knots(D, cols, nk, seed=0):
    # nk distinct rows of the data
    U = D[cols].drop_duplicates()
    return U.iloc[np.sort(np.random.default_rng(seed).choice(U.shape[0], nk, replace=False))]


def _Rknots(K):
    return 'list(' + ', '.join(f'{c}=c({", ".join(repr(float(v)) for v in K[c])})' for c in K) + ')'


# columns, mgcv's m, our arguments, k
BASES = {'1-D Matern 1.5': (['x'], 'NA', {}, 15),
         '1-D Matern 2.5, range 0.3': (['x'], 'c(4, 0.3)', dict(nu=2.5, range=0.3), 20),
         '2-D Matern 1.5': (['u', 'v'], 'NA', {}, 40),
         '2-D power exponential, stationary': (['u', 'v'], 'c(-2, 0.5, 1.5)',
                                               dict(kernel='exponential', power=1.5, range=0.5,
                                                    trend=False), 30)}


@pytest.mark.parametrize('case', list(BASES))
def test_basis_mgcv(Rinfo, data, case):
    cols, m, kw, k = BASES[case]
    rpy = _R(Rinfo, data)
    rpy.r(f"sm <- smoothCon(s({', '.join(cols)}, bs='gp', k={k}, m={m}), data=D, "
          "absorb.cons=FALSE, scale.penalty=FALSE)[[1]]")
    term = gp(*cols, k=k, **kw).setup(data[cols].to_numpy())
    B = term.basis()
    X_R = _get(Rinfo, 'sm$X')
    np.testing.assert_allclose(term.shift_, _get(Rinfo, 'sm$shift'))
    np.testing.assert_allclose(term.range_, _get(Rinfo, 'sm$gp.defn')[1], rtol=1e-12)
    K_R = _get(Rinfo, 'sm$knt')
    np.testing.assert_allclose(np.sort(term.knots_ - term.shift_, 0), np.sort(K_R, 0), atol=1e-12)
    # mgcv's eigenvalues (its penalty) are ours
    ev_R = np.diag(_get(Rinfo, 'sm$S[[1]]'))[:term.rank_]
    np.testing.assert_allclose(term.eigenvalues_, ev_R, rtol=1e-8)
    # the same space, to the precision of mgcv's Lanczos eigenvectors
    coef = np.linalg.lstsq(B, X_R, rcond=None)[0]
    assert np.abs(X_R - B @ coef).max() < 1e-4 * np.abs(X_R).max()
    # and to rounding with the exact eigendecomposition
    rpy.r(f"""E <- mgcv:::gpE(sm$knt, sm$knt, sm$gp.defn); e <- eigen(E, symmetric=TRUE)
              x <- sweep(as.matrix(D[, c({', '.join(repr(c) for c in cols)}), drop=FALSE]), 2, sm$shift)
              X <- cbind(mgcv:::gpE(x, sm$knt, sm$gp.defn) %*% e$vectors[, 1:{term.rank_}],
                         mgcv:::gpT(x, sm$gp.defn))""")
    X = _get(Rinfo, 'X')
    # ours: the kernel's columns scaled by the eigenvalues' square roots (up
    # to signs, the eigenvectors')
    P = X[:, :term.rank_] / np.sqrt(term.eigenvalues_)
    signs = np.sign(np.sum(P * B[:, :term.rank_], 0))
    np.testing.assert_allclose(B[:, :term.rank_] * signs, P, rtol=1e-6, atol=1e-9 * np.abs(P).max())
    np.testing.assert_allclose(B[:, term.rank_:], X[:, term.rank_:], atol=1e-12)
    # at new points (mgcv's Predict.matrix)
    new = pd.DataFrame({c: np.linspace(-0.1, 1.1, 25) for c in cols})
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('new', new)
    X_new = _get(Rinfo, 'PredictMat(sm, new)')
    coef = np.linalg.lstsq(B, X_R, rcond=None)[0]
    np.testing.assert_allclose(term.basis(new.to_numpy()) @ coef, X_new, atol=1e-4 * np.abs(X_new).max())


# family, R family, response, weights, offset
FAMILIES = {'gaussian, weights': (sm.families.Gaussian(), 'gaussian()', 'y', True, False),
            'binomial, offset': (sm.families.Binomial(), 'binomial()', 'yb', False, True),
            'poisson, weights and offset': (sm.families.Poisson(), 'poisson()', 'yp', True, True)}


def _fit_both(Rinfo, D, family, method, lam=None, eigen=False):
    """
    gp(x) (1-D) and gp(u, v) (2-D), a factor and a random intercept; mgcv's
    knots-as-basis case (k = trend's dimension + number of knots) unless
    `eigen` (the default knots, its Lanczos eigenvectors).
    """
    klass, Rfamily, y, weights, offset = FAMILIES[family]
    rpy = _R(Rinfo, D)
    if eigen:
        k1, k2, knots = 15, 40, {}
        Rknots = ''
    else:
        K1, K2 = _knots(D, ['x'], 15), _knots(D, ['u', 'v'], 35)
        k1, k2 = 17, 38
        knots = [dict(knots=K1.to_numpy()), dict(knots=K2.to_numpy())]
        Rknots = f', knots=c({_Rknots(K1)}, {_Rknots(K2)})'
    kw = knots or [{}, {}]
    lam = lam or (None, None, None)
    terms = [gp('x', k=k1, lam=lam[0], **kw[0]), gp('u', 'v', k=k2, lam=lam[1], **kw[1]),
             categorical('c'), re('g', lam=lam[2])]
    cols = [y] + (['w'] if weights else []) + (['o'] if offset else [])
    G = _model(method, terms=terms, family=klass, response_id=y,
            weight_id='w' if weights else None, offset_id='o' if offset else None,
            control=GAMControl(**TIGHT)).fit(D, D[cols])
    sw = D['w'].sum() if weights else D.shape[0]

    def mgcv(sp=None, start=None, scale=1.):
        extra = (', weights=w' if weights else '') + (
            f', sp=c({", ".join(repr(float(v)) for v in sp)})' if sp is not None else '')
        if start is not None:
            extra += (f', in.out=list(sp=c({", ".join(repr(float(v)) for v in start)}), '
                      f'scale={float(scale)!r})')
        off = ' + offset(o)' if offset else ''
        rpy.r(f"""m <- gam({y} ~ s(x, bs='gp', k={k1}) + s(u, v, bs='gp', k={k2}) + c
                              + s(g, bs='re'){off}, data=D, family={Rfamily},
                   method='{method if method != 'fixed' else 'REML'}'{extra}{Rknots},
                   control=gam.control(epsilon=1e-12, maxit=200, newton=list(conv.tol=1e-10)))""")
    if lam[0] is not None:
        mgcv((1., 1., 1.))
        scale = [rpy.r(f'm$smooth[[{i}]]$S.scale')[0] for i in (1, 2, 3)]
        mgcv([l * sw * c for l, c in zip(lam, scale)])
    else:
        mgcv()
    return G, rpy, sw, mgcv


LAM = (1e-3, 1e-4, 0.05)


@pytest.mark.parametrize('family', list(FAMILIES))
def test_fixed_sp_mgcv(Rinfo, data, family):
    G, rpy, _, _ = _fit_both(Rinfo, data, family, 'fixed', lam=LAM)
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'm$linear.predictors'),
                               rtol=1e-7, atol=1e-7)
    edf_R = _get(Rinfo, 'sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))')
    np.testing.assert_allclose(G.edf_influence_.values[[0, 1, 3]], edf_R, rtol=1e-6)
    # the curves (centered as mgcv's) and the standard errors (Vp, the
    # dispersions divided out)
    new = data.iloc[::13].copy()
    new['x'] = np.linspace(-0.05, 1.05, new.shape[0])
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('new', new)
    rpy.r('new$c <- factor(new$c, levels=levels(D$c)); new$g <- factor(new$g, levels=levels(D$g))')
    terms_R = rpy.r('predict(m, newdata=new, type="terms", se.fit=TRUE)')
    names_R = list(rpy.r('colnames(predict(m, newdata=new, type="terms"))'))
    phi, phi_R = G.dispersion_, _get(Rinfo, 'm$sig2')[0]
    for j, name in [(0, 's(x)'), (1, 's(u,v)')]:
        fit, se = G.partial_effect(j, new, return_std=True)
        i = names_R.index(name)
        np.testing.assert_allclose(fit, np.asarray(terms_R[0])[:, i], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(se / np.sqrt(phi), np.asarray(terms_R[1])[:, i] / np.sqrt(phi_R),
                                   rtol=1e-6, atol=1e-9)
    offset = new['o'] if FAMILIES[family][4] else None
    fit, se = G.predict(new, 'link', offset=offset, return_std=True)
    pred_R = rpy.r('predict(m, newdata=new, se.fit=TRUE)')
    np.testing.assert_allclose(fit, np.asarray(pred_R[0]), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(se / np.sqrt(phi), np.asarray(pred_R[1]) / np.sqrt(phi_R), rtol=1e-6)


def test_fixed_sp_mgcv_eigen(Rinfo, data):
    # mgcv's default knots (all the distinct values) and its Lanczos
    # eigenvectors: the fits agree to their precision
    G, rpy, _, _ = _fit_both(Rinfo, data, 'binomial, offset', 'fixed', lam=LAM, eigen=True)
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'm$linear.predictors'),
                               rtol=1e-3, atol=1e-3)
    edf_R = _get(Rinfo, 'sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))')
    np.testing.assert_allclose(G.edf_influence_.values[[0, 1, 3]], edf_R, rtol=1e-3)


@pytest.mark.parametrize('method', ['REML', 'ML'])
@pytest.mark.parametrize('family', list(FAMILIES))
def test_reml_mgcv(Rinfo, data, family, method):
    pytest.importorskip('jax')
    G, rpy, sw, mgcv = _fit_both(Rinfo, data, family, method)
    scale = np.array([rpy.r(f'm$smooth[[{i}]]$S.scale')[0] for i in (1, 2, 3)])
    if method == 'ML':
        score_R = _get(Rinfo, 'm$gcv.ubre')[0]
        if G.reml_score_ < score_R - 1e-4 * abs(score_R):
            # mgcv stopped at a worse optimum (the binomial: s(x) linear, its
            # smoothing parameter at infinity, ML 2.4 larger than ours):
            # from ours, it stays there
            assert family == 'binomial, offset'
            mgcv(start=G.smoothing_.lam * sw * scale, scale=G.dispersion_)
    sp_R = _get(Rinfo, 'm$sp')
    edf_R = _get(Rinfo, 'sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))')
    # (off the boundary: a smoothing parameter at infinity in effect, the
    # trend's edf, where the criterion is flat in it)
    inner = edf_R > np.array([1, 2, 0]) + 0.05
    np.testing.assert_allclose(G.smoothing_.lam[inner], (sp_R / scale / sw)[inner], rtol=2e-3)
    np.testing.assert_allclose(G.state_.link_parameter, _get(Rinfo, 'm$linear.predictors'),
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(G.edf_.values[[0, 1, 3]], edf_R, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(G.dispersion_, _get(Rinfo, 'm$reml.scale')[0], rtol=1e-4)
    if method == 'ML':
        # (ours at least as small)
        score_R = _get(Rinfo, 'm$gcv.ubre')[0]
        np.testing.assert_allclose(G.reml_score_, score_R, rtol=1e-6)
        assert G.reml_score_ <= score_R + 1e-6 * abs(score_R)
