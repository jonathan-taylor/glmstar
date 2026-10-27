"""
Sandwich (robust) covariances of a GAM's coefficients:
`sandwich_covariance(meat=None, groups=None)`, `A^{-1} M A^{-1}` with the
bread `A = X' W X + S_lambda` and the meat `M` from `score_contributions()`.

Without penalties they are statsmodels' HC0 and cluster-robust covariances
(without the small-sample correction). The parametrizations differ (ours:
centered linear columns and sum-to-zero constraints; statsmodels': the
columns as given, treatment contrasts), so the covariances are compared on
the linear predictor at the training rows, `X V X'`, which does not depend
on the parametrization.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytestmark = [pytest.mark.filterwarnings("ignore::glmnet.gam.ExperimentalWarning"),
              pytest.mark.filterwarnings("ignore:cov_type not fully supported")]

pytest.importorskip('scatter_smooth')

from glmnet.gam import GAM, GLMM, GAMControl, s, linear, categorical, random_effect as re
from glmnet.cox import CoxFamily

TIGHT = GAMControl(epsnr=1e-12, mxitnr=100)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(3)
    n = 300
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n), 2),
                      'b': np.round(rng.uniform(-2, 2, n), 1),
                      'c': rng.choice(list('xyz'), n),
                      'g': rng.choice([f'g{i}' for i in range(25)], n),
                      'w': rng.uniform(0.5, 2, n)})
    effects = dict(zip(sorted(D['g'].unique()), rng.normal(0, 0.5, 25)))
    eta = (np.sin(2 * np.pi * D['a']) + 0.4 * D['b'] + D['c'].map({'x': 0, 'y': .5, 'z': -.5})
           + D['g'].map(effects))
    # heteroscedastic, so that HC0 differs from the model-based covariance
    D['y'] = eta + 0.3 * (1 + np.abs(D['b'])) * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    D['yp'] = rng.poisson(np.exp(0.5 * eta))
    return D


def _sm_design(D):
    return np.column_stack([np.ones(D.shape[0]), D['a'], D['b'],
                            D['c'] == 'y', D['c'] == 'z']).astype(float)


FAMILIES = {'gaussian': (sm.families.Gaussian(), 'y'),
            'poisson': (sm.families.Poisson(), 'yp'),
            'binomial': (sm.families.Binomial(), 'yb')}


def _linear_gam(D, family, y, weights=False):
    return GAM(terms=[linear('a'), linear('b'), categorical('c')], family=family,
               control=TIGHT, response_id=y,
               weight_id='w' if weights else None).fit(D, D[[y, 'w']])


def _on_rows(G, V):
    X = G.design_.X
    return X @ V @ X.T


@pytest.mark.parametrize('weights', [False, True])
@pytest.mark.parametrize('case', list(FAMILIES))
def test_statsmodels_hc0_cluster(data, case, weights):
    family, y = FAMILIES[case]
    D = data
    G = _linear_gam(D, family, y, weights)
    assert np.abs(G.penalty_matrix_).max() == 0
    Xs = _sm_design(D)
    kw = dict(var_weights=D['w'].values) if weights else {}
    model = sm.GLM(D[y].values.astype(float), Xs, family=family, **kw)
    hc0 = model.fit(cov_type='HC0', tol=1e-12)
    np.testing.assert_allclose(G.state_.link_parameter, Xs @ hc0.params, rtol=1e-7, atol=1e-8)
    V = G.sandwich_covariance()
    ref = Xs @ hc0.cov_params() @ Xs.T
    scale = np.abs(ref).max()
    np.testing.assert_allclose(_on_rows(G, V), ref, atol=1e-6 * scale)
    # the slope of a linear term is the same coefficient in both
    sl = G.design_.slices[1]
    np.testing.assert_allclose(V[sl, sl][0, 0], hc0.cov_params()[2, 2], rtol=1e-6)
    # and HC0 differs from the model-based covariance
    assert not np.allclose(V, G.frequentist_covariance_, rtol=1e-2)

    cl = model.fit(cov_type='cluster', tol=1e-12,
                   cov_kwds={'groups': pd.factorize(D['g'])[0], 'use_correction': False})
    Vc = G.sandwich_covariance(groups=D['g'])
    ref = Xs @ cl.cov_params() @ Xs.T
    np.testing.assert_allclose(_on_rows(G, Vc), ref, atol=1e-6 * np.abs(ref).max())
    # groups of one row each: HC0
    np.testing.assert_allclose(G.sandwich_covariance(groups=np.arange(D.shape[0])), V,
                               atol=1e-12 * np.abs(V).max())


@pytest.mark.parametrize('case', list(FAMILIES))
def test_penalized_pieces(data, case):
    family, y = FAMILIES[case]
    D = data
    G = GAM(terms=[s('a', df=5), linear('b'), categorical('c'), re('g', lam=0.05)],
            family=family, control=TIGHT, response_id=y, weight_id='w').fit(D, D[[y, 'w']])
    X, W, S = G.design_.X, G.working_weights_, G.penalty_matrix_
    assert np.abs(S).max() > 0
    beta = G.design_.gamma(G.state_)
    np.testing.assert_allclose(X @ beta, G.state_.linear_predictor, atol=1e-8)
    # the score of the penalized objective sums to S_lambda beta at the fit
    sc = G.score_contributions()
    assert sc.shape == X.shape
    np.testing.assert_allclose(sc.sum(0), S @ beta, atol=1e-8 * np.abs(S @ beta).max() + 1e-12)
    # the bread: posterior_covariance_ = dispersion A^{-1} / sum(w)
    A = X.T @ (W[:, None] * X) + S
    np.testing.assert_allclose(G.dispersion_ * np.linalg.pinv(A) / G.weight_total_,
                               G.posterior_covariance_, rtol=1e-8,
                               atol=1e-10 * np.abs(G.posterior_covariance_).max())
    # the model-based meat gives the frequentist covariance (mgcv's Ve)
    meat = G.dispersion_ * X.T @ (W[:, None] * X) / G.weight_total_
    Ve = G.frequentist_covariance_
    np.testing.assert_allclose(G.sandwich_covariance(meat=meat), Ve, atol=1e-10 * np.abs(Ve).max())
    # a meat of one's own, HC0's: HC0
    hc0 = G.sandwich_covariance()
    np.testing.assert_allclose(G.sandwich_covariance(meat=sc.T @ sc), hc0, rtol=1e-12,
                               atol=1e-14 * np.abs(hc0).max())
    # a grouped bootstrap of the group sums of the score: near the cluster meat
    g = pd.factorize(D['g'])[0]
    sums = np.zeros((g.max() + 1, sc.shape[1]))
    np.add.at(sums, g, sc)
    rng = np.random.default_rng(0)
    totals = np.array([sums[rng.integers(0, len(sums), len(sums))].sum(0) for _ in range(2000)])
    boot = G.sandwich_covariance(meat=np.cov(totals.T))
    cluster = G.sandwich_covariance(groups=D['g'])
    d = np.sqrt(np.diag(cluster))[:6]
    np.testing.assert_allclose(np.sqrt(np.diag(boot))[:6], d, rtol=0.15)
    with pytest.raises(ValueError, match='meat'):
        G.sandwich_covariance(meat=np.eye(3))
    with pytest.raises(ValueError, match='not both'):
        G.sandwich_covariance(meat=meat, groups=D['g'])
    with pytest.raises(ValueError, match='label per training row'):
        G.sandwich_covariance(groups=D['g'][:10])


def test_standard_errors(data):
    D = data
    G = GAM(terms=[s('a', df=5), linear('b'), categorical('c')], control=TIGHT).fit(D, D['y'])
    new = D.iloc[::37]
    _, se = G.predict(new, return_std=True)
    _, se_p = G.predict(new, return_std=True, covariance='posterior')
    np.testing.assert_allclose(se, se_p)
    _, se_s = G.predict(new, return_std=True, covariance='sandwich')
    V = G.sandwich_covariance(groups=D['g'])
    _, se_g = G.predict(new, return_std=True, covariance=V)
    rows = G.design_.rows(G._columns(G._design(new), G.terms_))
    np.testing.assert_allclose(se_g, np.sqrt(np.einsum('ij,jk,ik->i', rows, V, rows)))
    assert not np.allclose(se, se_s, rtol=1e-2) and not np.allclose(se, se_g, rtol=1e-2)
    _, se_f = G.predict(new, return_std=True, covariance='frequentist')
    assert not np.allclose(se, se_f, rtol=1e-4)
    # back to the posterior by default
    np.testing.assert_allclose(G.predict(new, return_std=True)[1], se)

    x = np.linspace(0.1, 0.9, 5)
    _, pe = G.partial_effect('s(a, df=5)', x, return_std=True)
    _, pe_s = G.partial_effect('s(a, df=5)', x, return_std=True, covariance='sandwich')
    assert not np.allclose(pe, pe_s, rtol=1e-2)
    np.testing.assert_allclose(G.partial_effect('s(a, df=5)', x, return_std=True)[1], pe)
    with pytest.raises(ValueError, match='covariance should be'):
        G.predict(new, return_std=True, covariance='robust')
    with pytest.raises(ValueError, match='covariance should be'):
        G.predict(new, return_std=True, covariance=np.eye(2))


def test_glmm(data):
    D = data
    G = GLMM(terms=[s('a'), linear('b'), re('g')], control=TIGHT).fit(D, D['y'])
    sc = G.score_contributions()
    beta = G.design_.gamma(G.state_)
    Sb = G.penalty_matrix_ @ beta
    np.testing.assert_allclose(sc.sum(0), Sb, atol=1e-7 * np.abs(Sb).max())
    X, W = G.design_.X, G.working_weights_
    meat = G.dispersion_ * X.T @ (W[:, None] * X) / G.weight_total_
    Ve = G.frequentist_covariance_
    np.testing.assert_allclose(G.sandwich_covariance(meat=meat), Ve, atol=1e-10 * np.abs(Ve).max())


def test_max_dense_coef(data):
    D = data
    G = GAM(terms=[s('a', df=5), linear('b'), categorical('c')],
            control=GAMControl(max_dense_coef=5)).fit(D, D['y'])
    for call in (G.score_contributions, G.sandwich_covariance,
                 lambda: G.predict(D, return_std=True, covariance='sandwich')):
        with pytest.raises(ValueError, match='max_dense_coef'):
            call()


def test_max_dense_design(data):
    # the dense N x P design is not formed above max_dense_design: the
    # covariances are None, the standard errors raise
    D = data
    terms = [s('a', df=5), linear('b'), categorical('c')]
    ref = GAM(terms=terms).fit(D, D['y'])
    n_coef = 1 + sum(term.n_coef() for term in ref.terms_)   # (as max_dense_coef counts)
    G = GAM(terms=terms, control=GAMControl(max_dense_design=D.shape[0] * n_coef - 1)).fit(D, D['y'])
    assert G.design_ is None and G.posterior_covariance_ is None
    for call in (G.score_contributions, G.sandwich_covariance,
                 lambda: G.predict(D, return_std=True)):
        with pytest.raises(ValueError, match='max_dense_design'):
            call()
    # nor by dir() and scikit-learn's HTML display (a notebook's), which
    # evaluate the properties
    from sklearn.utils import estimator_html_repr
    dir(G), estimator_html_repr(G)
    assert G._covariance['design'] is None
    # at the limit, as before
    H = GAM(terms=terms, control=GAMControl(max_dense_design=D.shape[0] * n_coef)).fit(D, D['y'])
    np.testing.assert_allclose(H.posterior_covariance_, ref.posterior_covariance_)

def test_cox(data):
    D = data.copy()
    rng = np.random.default_rng(1)
    D['time'] = rng.exponential(np.exp(-0.4 * D['b']))
    D['status'] = rng.binomial(1, 0.8, D.shape[0])
    G = GAM(terms=[linear('b'), categorical('c')], family=CoxFamily(event_id='time', status_id='status'),
            control=GAMControl(solver='joint')).fit(D, D[['time', 'status']])
    with pytest.raises(NotImplementedError, match='risk sets'):
        G.score_contributions()
    with pytest.raises(NotImplementedError, match='risk sets'):
        G.sandwich_covariance()
    # a meat of one's own: the model-based one gives the frequentist covariance
    Ve = G.frequentist_covariance_
    P = Ve.shape[0]
    Ainv = G.posterior_covariance_ * G.weight_total_ / G.dispersion_
    meat = np.linalg.pinv(Ainv) * G.dispersion_ / G.weight_total_
    np.testing.assert_allclose(G.sandwich_covariance(meat=meat)[1:, 1:], Ve[1:, 1:],
                               rtol=1e-6, atol=1e-10 * np.abs(Ve).max())
    assert G.sandwich_covariance(meat=np.eye(P)).shape == (P, P)


def test_mgcv(Rinfo, data):
    """
    mgcv's `vcov(m, sandwich=TRUE, freq=TRUE)` (`gam.sandwich`) is HC0's
    sandwich times `n / (n - sum(edf))` (unit prior weights: mgcv's meat
    leaves them out); without `freq=TRUE` it adds `Vp - Ve`.
    """
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    D = data
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', D)
    rpy.r('D$c <- factor(D$c)')
    G = GLMM(terms=[s('a'), s('b'), categorical('c')], control=TIGHT).fit(D, D['y'])
    ka, kb = D['a'].nunique(), D['b'].nunique()
    rpy.r(f"""m <- mgcv::gam(y ~ s(a, bs='cr', k={ka}) + s(b, bs='cr', k={kb}) + c,
               data=D, method='REML', knots=list(a=sort(unique(D$a)), b=sort(unique(D$b))),
               control=mgcv::gam.control(epsilon=1e-12, maxit=200, newton=list(conv.tol=1e-10)))
           Xp <- predict(m, type='lpmatrix')""")
    with Rinfo['np_cv_rules'].context():
        on_rows = lambda expr: np.asarray(rpy.r(f'diag(Xp %*% {expr} %*% t(Xp))'))
        Vs_R = on_rows('vcov(m, sandwich=TRUE, freq=TRUE)')
        Vs_R_default = on_rows('vcov(m, sandwich=TRUE)')
        edf_R = np.asarray(rpy.r('sum(m$edf)'))[0]
    n = D.shape[0]
    X = G.design_.X
    diag = lambda V: np.einsum('ij,jk,ik->i', X, V, X)
    V = G.sandwich_covariance()
    np.testing.assert_allclose(diag(V) * n / (n - edf_R), Vs_R, rtol=5e-3)
    np.testing.assert_allclose(diag(V * n / (n - edf_R) + G.posterior_covariance_
                                    - G.frequentist_covariance_), Vs_R_default, rtol=5e-3)
