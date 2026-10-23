"""
Gaussian process smooths (`gp()`, mgcv's `bs="gp"`): their construction,
backfitting against the joint solve, df targeting, predictions, plots and
errors; and, with a knot at every distinct value and no trend, the full
rank GP regression of scikit-learn's `GaussianProcessRegressor` at fixed
hyperparameters (its posterior mean and standard deviations, and its log
marginal likelihood against our REML).
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from glmnet.gam import GAM, GAMControl, gaussian_process as gp, categorical, re, linear, plot_term
from glmnet.gam.gp import gp_correlation, GPTerm

TIGHT = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(3)
    n = 300
    D = pd.DataFrame({'x': np.round(rng.uniform(0, 1, n), 2),
                      'u': rng.uniform(0, 1, n),
                      'v': rng.uniform(0, 1, n),
                      'c': rng.choice(list('abc'), n),
                      'g': rng.choice([f'g{i}' for i in range(8)], n),
                      'o': rng.normal(0, 0.2, n),
                      'w': rng.uniform(0.5, 2, n)})
    eta = (np.sin(2 * np.pi * D['x']) + np.sin(3 * D['u']) * np.cos(3 * D['v']) +
           D['c'].map({'a': 0, 'b': .4, 'c': -.4}))
    D['y'] = eta + 0.3 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-(eta + D['o']))))
    D['yp'] = rng.poisson(np.exp(0.5 * eta + D['o']))
    return D


FAMILIES = {'gaussian': (sm.families.Gaussian, 'y'),
            'binomial': (sm.families.Binomial, 'yb'),
            'poisson': (sm.families.Poisson, 'yp')}
TERMS = {
    'curve, surface and factor': lambda: [gp('x', k=12, lam=1e-3), gp('u', 'v', k=25, lam=1e-4),
                                          categorical('c')],
    'df, stationary, random effect': lambda: [gp('x', df=4, nu=2.5), gp('u', 'v', k=20, trend=False,
                                                                       df=8), re('g', lam=0.05)],
}


@pytest.mark.parametrize('terms', list(TERMS))
@pytest.mark.parametrize('family', list(FAMILIES))
def test_backfit_is_joint(data, family, terms):
    klass, y = FAMILIES[family]
    kw = dict(response_id=y, weight_id='w', offset_id='o')
    fits = {(solver, modified): GAM(terms=TERMS[terms](), family=klass(),
                                    control=GAMControl(**TIGHT, solver=solver, bf_modified=modified),
                                    **kw).fit(data, data[[y, 'w', 'o']])
            for solver, modified in [('joint', True), ('backfit', True), ('backfit', False)]}
    J = fits['joint', True]
    for G in fits.values():
        np.testing.assert_allclose(G.state_.link_parameter, J.state_.link_parameter,
                                   rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(G.edf_, J.edf_, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(G.edf_influence_, J.edf_influence_, rtol=1e-6, atol=1e-7)
    # the edf of the smoother, as the influence matrix's (one term each)
    np.testing.assert_allclose(J.predict(data, 'link', offset=data['o']), J.state_.link_parameter,
                               rtol=1e-10, atol=1e-10)


def test_df(data):
    # df: the trace of the smoother, less 1, at the final weights
    for family, y in [(sm.families.Gaussian(), 'y'), (sm.families.Binomial(), 'yb')]:
        G = GAM(terms=[gp('x', df=5), gp('u', 'v', k=30, df=10)], family=family,
                control=GAMControl(**TIGHT)).fit(data, data[y])
        np.testing.assert_allclose(G.edf_.values, [5, 10], rtol=1e-8)
    # a single Gaussian term: its edf is also the influence matrix's
    G = GAM(terms=[gp('u', 'v', k=30, df=7)]).fit(data, data['y'])
    np.testing.assert_allclose(G.edf_influence_.values, [7], rtol=1e-8)


def test_predict_new_data(data):
    G = GAM(terms=[gp('x', k=12, lam=1e-3), gp('u', 'v', k=25, lam=1e-4), categorical('c')],
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    np.testing.assert_allclose(G.predict(data), G.state_.link_parameter, rtol=1e-10, atol=1e-10)
    rng = np.random.default_rng(1)
    new = pd.DataFrame({'x': rng.uniform(-0.1, 1.1, 50), 'u': rng.uniform(0, 1, 50),
                        'v': rng.uniform(0, 1, 50), 'c': rng.choice(list('abc'), 50)})
    pred, se = G.predict(new, 'link', return_std=True)
    rows = G.design_.rows(G._columns(new, G.terms_))
    gamma = G.design_.gamma(G.state_)
    np.testing.assert_allclose(pred, rows @ gamma, rtol=1e-9, atol=1e-9)
    assert np.all(se > 0)
    # a term's function: its basis at the new points times its coefficients
    term = G.terms_[1]
    f = G.partial_effect(1, new)
    np.testing.assert_allclose(f, term.basis(new[['u', 'v']].to_numpy()) @ term.coef_)
    # numpy input, the columns by position
    X = data[['x', 'u', 'v']].to_numpy()
    H = GAM(terms=[gp(0, k=12, lam=1e-3), gp(1, 2, k=25, lam=1e-4)],
            control=GAMControl(**TIGHT)).fit(X, data['y'].to_numpy())
    K = GAM(terms=[gp('x', k=12, lam=1e-3), gp('u', 'v', k=25, lam=1e-4)],
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    np.testing.assert_allclose(H.predict(X), K.predict(data), rtol=1e-10)


def test_construction(data):
    # the basis spans the trend and the kernel at the knots' eigenvectors,
    # with the penalty the identity on the latter
    term = gp('u', 'v', k=25).setup(data[['u', 'v']].to_numpy())
    B = term.basis()
    assert B.shape == (data.shape[0], 25) and term.rank_ == 22 and term.null_dim_ == 3
    X = data[['u', 'v']].to_numpy()
    np.testing.assert_allclose(term.shift_, X.mean(0))
    from scipy.spatial.distance import cdist, pdist
    np.testing.assert_allclose(term.range_, pdist(X).max())
    # (the knots are the distinct rows: all of them here)
    assert term.knots_.shape == X.shape
    C = gp_correlation(cdist(term.knots_, term.knots_) / term.range_, 3)
    ev, U = np.linalg.eigh(C)
    np.testing.assert_allclose(term.eigenvalues_, ev[::-1][:22], rtol=1e-8)
    np.testing.assert_allclose(term.penalty_matrix(), np.diag([1.] * 22 + [0.] * 3))
    # its penalized columns at the knots: U L^{1/2}, orthogonal
    P = B[:, :22]
    np.testing.assert_allclose(P.T @ P, np.diag(term.eigenvalues_), atol=1e-8)
    # the penalty is the norm of the kernel expansion: c' S c = beta' C beta
    # for f = C(., K) beta, beta = U L^{-1/2} c
    c = np.random.default_rng(0).standard_normal(22)
    beta = term._W @ c
    np.testing.assert_allclose(beta @ C @ beta, c @ c, rtol=1e-8)
    # the default basis dimension (mgcv's): 12 in one dimension, 33 in two
    assert gp('x').setup(data['x'].to_numpy()).basis().shape[1] == 12
    assert gp('u', 'v').setup(X).basis().shape[1] == 33
    # mgcv's m codes
    x = data['x'].to_numpy()
    for kw, m in [(dict(nu=2.5, range=0.3), (4, 0.3)), (dict(kernel='spherical', trend=False), (-1,)),
                  (dict(kernel='exponential', power=1.5), (2, -1, 1.5)),
                  (dict(nu=3.5), (5,)), (dict(nu=0.5), (2, 0, 1))]:
        a, b = gp('x', **kw).setup(x), gp('x', m=m).setup(x)
        np.testing.assert_allclose(a.basis(), b.basis())
        assert a.range_ == b.range_ and a.trend_ == b.trend_


def test_correlations():
    d = np.linspace(0, 2, 21)
    np.testing.assert_allclose(gp_correlation(d, 3), (1 + d) * np.exp(-d))
    np.testing.assert_allclose(gp_correlation(d, 4), (1 + d + d**2 / 3) * np.exp(-d))
    np.testing.assert_allclose(gp_correlation(d, 5), (1 + d + 2 * d**2 / 5 + d**3 / 15) * np.exp(-d))
    np.testing.assert_allclose(gp_correlation(d, 2, 2), np.exp(-d**2))
    np.testing.assert_allclose(gp_correlation(d, 1), np.where(d <= 1, 1 - 1.5 * d + 0.5 * d**3, 0))
    # in scikit-learn's parametrization: the length scale sqrt(2 nu) rho
    from sklearn.gaussian_process.kernels import Matern, RBF
    r = d[:, None]
    for nu, code in [(0.5, 2), (1.5, 3), (2.5, 4), (3.5, 5)]:
        K = Matern(length_scale=np.sqrt(2 * nu), nu=nu)(r, np.zeros((1, 1)))[:, 0]
        np.testing.assert_allclose(gp_correlation(d, code), K, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(gp_correlation(d, 2, 2), RBF(1 / np.sqrt(2))(r, np.zeros((1, 1)))[:, 0])


def test_errors(data):
    x = data['x'].to_numpy()
    with pytest.raises(ValueError, match='kernel should be'):
        gp('x', kernel='cubic')
    with pytest.raises(ValueError, match='nu should be'):
        gp('x', nu=2)
    with pytest.raises(ValueError, match='power'):
        gp('x', kernel='exponential', power=3)
    with pytest.raises(ValueError, match='m\\[0\\]'):
        gp('x', m=(7,))
    with pytest.raises(ValueError, match='range should be positive'):
        gp('x', range=-1)
    with pytest.raises(ValueError, match='one of df and lam'):
        gp('x', df=3, lam=1.)
    with pytest.raises(ValueError, match='more than'):
        gp('x', k=200).setup(x)            # 101 distinct values
    with pytest.raises(ValueError, match='at least 3'):
        gp('x', k=2).setup(x)
    with pytest.raises(ValueError, match='df should be in'):
        GAM(terms=[gp('x', k=12, df=20)]).fit(data, data['y'])
    with pytest.raises(ValueError, match='give lam or df'):
        GAM(terms=[gp('x')]).fit(data, data['y'])
    # a kernel of low numerical rank (the Gaussian at a long range)
    with pytest.warns(UserWarning, match='numerical rank'):
        term = gp('x', kernel='exponential', power=2, range=5, k=60).setup(x)
    assert term.rank_ < 58 and term.basis().shape[1] == term.rank_ + 2
    assert term.eigenvalues_.min() > 1e-12 * term.eigenvalues_.max()


def test_names():
    assert gp('x').name == 'gaussian_process(x)'
    assert (gp('u', 'v', nu=2.5, range=0.5, k=40).name ==
            'gaussian_process(u, v, nu=2.5, range=0.5, k=40)')
    assert gp('x', trend=False, df=4).name == 'gaussian_process(x, trend=False, df=4)'
    assert gp('x', kernel='spherical').name == "gaussian_process(x, kernel='spherical')"
    assert gp('x', name='f').name == 'f'
    assert gp('u', 'v').columns == ['u', 'v']


def test_plots(data):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    G = GAM(terms=[gp('x', k=12, lam=1e-3), gp('u', 'v', k=25, lam=1e-4)],
            control=GAMControl(**TIGHT)).fit(data, data['y'])
    ax = plot_term(G, 0)
    curve = ax.get_lines()[0]
    np.testing.assert_allclose(curve.get_ydata(), G.partial_effect(0, curve.get_xdata()))
    ax = plot_term(G, 1, n_grid=30, too_far=None)
    assert ax.get_xlabel() == 'u' and ax.get_ylabel() == 'v'
    axes = G.plot()
    assert len(axes) == 2
    plt.close('all')
    H = GAM(terms=[gp('x', 'u', 'v', k=20, lam=1e-3)]).fit(data, data['y'])
    with pytest.raises(ValueError, match='more than two columns'):
        plot_term(H, 0)


# scikit-learn's GP regression: a knot at every distinct value (full rank),
# no trend (the GAM's intercept with a flat prior, sklearn's a constant
# kernel of large variance). Our fit minimizes sum(w (y - f)^2) / (2 sum(w))
# + lam / 2 ||f||^2_C, the posterior mode (and mean) of the GP with
# covariance tau2 C and noise variances sigma2 / w when sigma2 / tau2 = lam
# sum(w). mgcv's Matern(nu) of range rho is sklearn's of length scale
# sqrt(2 nu) rho.

FLAT = 1e5          # the variance of sklearn's intercept

SKLEARN = {'1-D': (['x'], 1.5, 0.3), '1-D, nu=2.5': (['x'], 2.5, 0.2),
           '2-D': (['u', 'v'], 1.5, 0.4)}


def _sklearn_fit(D, cols, nu, rho, sigma2, tau2, w):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    kernel = (ConstantKernel(tau2, 'fixed') *
              Matern(length_scale=np.sqrt(2 * nu) * rho, nu=nu, length_scale_bounds='fixed') +
              ConstantKernel(FLAT, 'fixed'))
    return GaussianProcessRegressor(kernel=kernel, alpha=sigma2 / w, optimizer=None,
                                    normalize_y=False).fit(D[cols].to_numpy(), D['y'].to_numpy())


@pytest.mark.parametrize('weights', [False, True])
@pytest.mark.parametrize('case', list(SKLEARN))
def test_sklearn_gpr(data, case, weights):
    jax = pytest.importorskip('jax')
    from glmnet.gam.reml import LaplaceCriterion
    cols, nu, rho = SKLEARN[case]
    D = data.iloc[:150]
    w = D['w'].to_numpy() if weights else np.ones(D.shape[0])
    nk = D[cols].drop_duplicates().shape[0]
    sigma2, lam = 0.09, 2e-4
    tau2 = sigma2 / (lam * w.sum())

    def fit(lam):
        return GAM(terms=[gp(*cols, nu=nu, range=rho, trend=False, k=nk + 1, lam=lam)],
                   weight_id='w' if weights else None, response_id='y',
                   control=GAMControl(**TIGHT)).fit(D, D[['y', 'w']], dispersion=sigma2)

    G = fit(lam)
    R = _sklearn_fit(D, cols, nu, rho, sigma2, tau2, w)
    rng = np.random.default_rng(2)
    new = pd.DataFrame({c: rng.uniform(-0.05, 1.05, 40) for c in cols})
    term = G.terms_[0]
    K = term.knots_ - term.shift_
    from scipy.spatial.distance import cdist
    for X in [D, new]:
        mean_R, sd_R = R.predict(X[cols].to_numpy(), return_std=True)
        mean, sd = G.predict(X, 'link', return_std=True)
        np.testing.assert_allclose(mean, mean_R, rtol=1e-6, atol=1e-6)
        # the posterior sd of the latent function (intercept included): ours
        # is that of the function in the span of the kernel at the knots, the
        # GP's adds the prior variance of the part orthogonal to them, 0 at
        # the knots, tau2 (1 - c' C^{-1} c) elsewhere (c the correlations
        # with the knots)
        c = gp_correlation(cdist(X[cols].to_numpy() - term.shift_, K) / rho, term.kernel_code_)
        CK = gp_correlation(cdist(K, K) / rho, term.kernel_code_)
        orth = tau2 * np.maximum(1 - np.einsum('ij,ji->i', c, np.linalg.solve(CK, c.T)), 0)
        np.testing.assert_allclose(sd**2 + orth, sd_R**2, rtol=1e-5, atol=1e-10)
        if X is D:
            np.testing.assert_allclose(sd, sd_R, rtol=1e-5)
    # the log marginal likelihood: the intercept's flat prior integrated, our
    # REML (exact for the Gaussian), less log(2 pi FLAT) / 2 for sklearn's
    # proper prior; at two smoothing parameters
    with jax.enable_x64(True):
        C = LaplaceCriterion(G, [0], method='REML', dispersion=sigma2)
        for lam_k in [lam, 10 * lam]:
            reml = C.value_and_grad(np.array([np.log(lam_k * w.sum())]))[0]
            R = _sklearn_fit(D, cols, nu, rho, sigma2, sigma2 / (lam_k * w.sum()), w)
            np.testing.assert_allclose(-reml - np.log(2 * np.pi * FLAT) / 2,
                                       R.log_marginal_likelihood_value_, rtol=1e-7)


def test_basis_in_blocks():
    # the basis is computed in blocks of rows (the correlations with the
    # knots are never formed for all rows): the same as in one
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (500, 2))
    term = gp(0, 1).setup(X)
    full = term.basis()
    term.block_rows = 7
    np.testing.assert_allclose(term.basis(X), full, rtol=1e-10, atol=1e-12)
