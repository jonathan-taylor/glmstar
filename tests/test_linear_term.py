import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import statsmodels.api as sm

from glmnet import GLMNet, GaussNet
from glmnet.glmnet import GLMNetControl
from glmnet.elnet import ElNet

rng = np.random.default_rng(0)
N, P = 100, 8


def _data(family='gaussian', weights=False):
    X = rng.standard_normal((N, P)) * rng.uniform(0.5, 3, P)
    eta = 0.5 * X[:, 0] / X[:, 0].std() - 0.5 * X[:, 1] / X[:, 1].std()
    if family == 'binomial':
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    elif family == 'poisson':
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = eta + rng.standard_normal(N)
    w = rng.uniform(0.5, 2, N) if weights else np.ones(N)
    return X, pd.DataFrame({'y': y, 'w': w})


FAMILIES = {'gaussian': sm.families.Gaussian(),
            'binomial': sm.families.Binomial(),
            'poisson': sm.families.Poisson()}


def _glmnet(family='gaussian', **kwargs):
    return GLMNet(family=FAMILIES[family], response_id='y', weight_id='w', **kwargs)


def _kkt(fit, X, Y, G, alpha=1.0, pf=None, family='gaussian'):
    """
    Largest violation of the KKT conditions of
    -loglik / sum(w) + G^T beta + lambda * penalty(beta) along the path
    (canonical links, standardize=False).
    """
    pf = np.ones(X.shape[1]) if pf is None else pf
    pf = pf * X.shape[1] / pf.sum()
    v = np.asarray(Y['w']) / Y['w'].sum()
    y = np.asarray(Y['y'])
    worst = 0
    for lam, beta, a0 in zip(fit.lambda_values_, fit.coefs_, fit.intercepts_):
        eta = a0 + X @ beta
        mu = FAMILIES[family].link.inverse(eta)
        grad = X.T @ (v * (y - mu)) - G - lam * (1 - alpha) * pf * beta
        l1 = lam * alpha * pf
        viol = np.where(beta != 0,
                        np.fabs(grad - l1 * np.sign(beta)),
                        np.maximum(np.fabs(grad) - l1, 0))
        worst = max(worst, viol.max(), np.fabs((v * (y - mu)).sum()))
    return worst


# standardize is parametrized by conftest.py
@pytest.mark.parametrize('sparse', [False, True])
def test_gaussian_matches_shifted_response(standardize, sparse):
    # with delta weighted-centered and G = -X^T V delta (V the normalized weights),
    # 1/2 sum v (y - eta)^2 + G^T beta = 1/2 sum v (y + delta - eta)^2 + const
    X, Y = _data(weights=True)
    v = Y['w'] / Y['w'].sum()
    delta = rng.standard_normal(N)
    delta -= (v * delta).sum()
    G = -X.T @ (v * delta)
    Y2 = Y.copy()
    Y2['y'] = Y['y'] + delta
    X_ = scipy.sparse.csc_array(X) if sparse else X

    lin = _glmnet(linear_term=G, standardize=standardize).fit(X_, Y)
    shifted = _glmnet(standardize=standardize).fit(X_, Y2)

    # the paths stop at different lambdas: the deviance is that of y, resp. y + delta
    k = min(len(lin.lambda_values_), len(shifted.lambda_values_))
    assert k > 10
    np.testing.assert_allclose(lin.lambda_values_[:k], shifted.lambda_values_[:k], rtol=1e-10)
    np.testing.assert_allclose(lin.coefs_[:k], shifted.coefs_[:k], atol=1e-8)
    np.testing.assert_allclose(lin.intercepts_[:k], shifted.intercepts_[:k], atol=1e-8)


def _wls(X, y, w):
    """
    Weighted least squares with an intercept, as R's `lm(y ~ X, weights=w)`:
    coefficients (without intercept) and sigma^2 = sum w r^2 / (n - p - 1).
    """
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X1 * sw[:, None], y * sw, rcond=None)[0]
    resid = y - X1 @ beta
    sigma2 = (w * resid**2).sum() / (X.shape[0] - X1.shape[1])
    return beta[1:], sigma2


# standardize is parametrized by conftest.py
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('sparse', [False, True])
def test_offset_is_linear_term(standardize, weighted, sparse):
    # an offset o = X beta_hat + noise, with beta_hat the (weighted) least squares
    # fit on the original scale and noise ~ N(0, sigma_hat^2 / w):
    #   1/2 sum v (y - o - a0 - X beta)^2
    #     = 1/2 sum v (y - a0' - X beta)^2 + G^T beta + const,
    # G = X^T V (o - o_bar), o_bar = sum v o, a0' = a0 + o_bar, v = w / sum(w),
    # whatever the standardization of the design
    X, Y = _data(weights=weighted)
    w = np.asarray(Y['w'])
    beta_hat, sigma2 = _wls(X, np.asarray(Y['y']), w)
    o = X @ beta_hat + rng.standard_normal(N) * np.sqrt(sigma2 / w)
    v = w / w.sum()
    o_bar = (v * o).sum()
    G = X.T @ (v * (o - o_bar))
    Y['o'] = o
    X_ = scipy.sparse.csc_array(X) if sparse else X

    offset = _glmnet(offset_id='o', standardize=standardize).fit(X_, Y)
    lin = _glmnet(linear_term=G, standardize=standardize).fit(X_, Y)
    np.testing.assert_allclose(lin.lambda_max_, offset.lambda_max_, rtol=1e-10)

    # the offset explains most of y, so the default path stops early:
    # compare on a fixed grid (no stopping rule)
    lambdas = offset.lambda_max_ * np.exp(np.linspace(0, np.log(1e-3), 30))
    # the two fits take different numerical routes to the same solution, so they agree
    # only up to the solver's convergence: tighten it
    kw = dict(standardize=standardize, lambda_values=lambdas,
              control=GLMNetControl(thresh=1e-14))
    offset = _glmnet(offset_id='o', **kw).fit(X_, Y)
    lin = _glmnet(linear_term=G, **kw).fit(X_, Y)
    assert offset.coefs_.shape == lin.coefs_.shape == (30, P)
    np.testing.assert_allclose(lin.coefs_, offset.coefs_, atol=1e-6)
    np.testing.assert_allclose(lin.intercepts_ - o_bar, offset.intercepts_, atol=1e-6)


@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson'])
@pytest.mark.parametrize('mix', [1.0, 0.5])  # alpha is parametrized by conftest.py
def test_kkt(family, mix):
    X, Y = _data(family, weights=True)
    G = 0.05 * rng.standard_normal(P)
    fit = _glmnet(family, linear_term=G, alpha=mix, standardize=False).fit(X, Y)
    assert len(fit.lambda_values_) > 10
    assert _kkt(fit, X, Y, G, alpha=mix, family=family) < 5e-5
    # and it is not the solution without the linear term
    assert _kkt(fit, X, Y, 0 * G, alpha=mix, family=family) > 1e-2


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_unpenalized_variable(family):
    # the linear term of an unpenalized variable enters the initial fit and lambda_max
    X, Y = _data(family, weights=True)
    G = 0.05 * rng.standard_normal(P)
    G[0] = 0.3
    pf = np.ones(P)
    pf[0] = 0
    fit = _glmnet(family, linear_term=G, penalty_factor=pf, standardize=False).fit(X, Y)
    assert _kkt(fit, X, Y, G, pf=pf, family=family) < 5e-5
    # at lambda_max only the unpenalized variable is nonzero, and the KKT bound is tight
    assert np.fabs(fit.coefs_[0][1:]).max() < 1e-12 and fit.coefs_[0][0] != 0
    v = Y['w'] / Y['w'].sum()
    mu = FAMILIES[family].link.inverse(fit.intercepts_[0] + X @ fit.coefs_[0])
    score = X.T @ (v * (Y['y'] - mu)) - G
    pf_ = pf * P / pf.sum()
    np.testing.assert_allclose(np.fabs(score[1:] / pf_[1:]).max(), fit.lambda_values_[0], rtol=1e-4)


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_zero_linear_term_is_none(family):
    X, Y = _data(family)
    none = _glmnet(family).fit(X, Y)
    zero = _glmnet(family, linear_term=np.zeros(P)).fit(X, Y)
    np.testing.assert_array_equal(none.lambda_values_, zero.lambda_values_)
    np.testing.assert_array_equal(none.coefs_, zero.coefs_)
    np.testing.assert_array_equal(none.intercepts_, zero.intercepts_)


def test_bad_shape():
    X, Y = _data()
    with pytest.raises(ValueError, match='linear_term should have shape'):
        _glmnet(linear_term=np.ones(P - 1)).fit(X, Y)


def _unbounded_check(fit, c):
    # the path stopped before the first lambda below c, with a warning
    assert len(fit.lambda_values_) > 0
    assert fit.lambda_values_.min() >= c


def test_unbounded_zero_column():
    # zero column with G_j = c: unbounded below (exactly, along e_j) once lambda < |c|
    X, Y = _data()
    X[:, 3] = 0
    G = np.zeros(P)
    c = 0.05
    G[3] = c
    with pytest.warns(UserWarning, match='unbounded below'):
        fit = _glmnet(linear_term=G, standardize=False).fit(X, Y)
    _unbounded_check(fit, c)
    # bounded with limits, or with a ridge penalty
    lower = np.full(P, -np.inf)
    lower[3] = -1
    fit = _glmnet(linear_term=G, standardize=False, lower_limits=lower).fit(X, Y)
    assert fit.lambda_values_.min() < c
    assert np.all(fit.coefs_[fit.lambda_values_ < c][:, 3] == -1)


def test_unbounded_duplicated_column():
    # duplicated columns with G = (c, -c): along (-1, 1) the loss is constant and
    # G^T beta + lambda |beta|_1 decreases once lambda < c (detected at maxit)
    X, Y = _data()
    X[:, 3] = X[:, 2]
    G = np.zeros(P)
    c = 0.05
    G[2], G[3] = c, -c
    glmnet = _glmnet(linear_term=G, standardize=False)
    glmnet.control.maxit = 2000
    with pytest.warns(UserWarning, match='unbounded below'):
        fit = glmnet.fit(X, Y)
    _unbounded_check(fit, c)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _glmnet(linear_term=G, standardize=False, alpha=0.5).fit(X, Y)


def test_elnet_unbounded_raises():
    X, Y = _data()
    X[:, 3] = 0
    G = np.zeros(P)
    G[3] = 1
    with pytest.raises(ValueError, match='unbounded below'):
        ElNet(lambda_val=0.1, linear_term=G, standardize=False).fit(X, np.asarray(Y['y']))


def test_cross_validation_and_relax():
    # clone() carries linear_term to the CV folds and the relaxed refits
    X, Y = _data('binomial', weights=True)
    G = 0.05 * rng.standard_normal(P)
    fit = _glmnet('binomial', linear_term=G, standardize=False, relax=True).fit(X, Y)
    fit.cross_validation_path(X, Y, cv=3)
    # relaxed fits: unpenalized, so the gradient vanishes on the active set
    v = Y['w'] / Y['w'].sum()
    for beta, a0 in zip(fit.relaxed_coefs_[1:], fit.relaxed_intercepts_[1:]):
        active = beta != 0
        mu = FAMILIES['binomial'].link.inverse(a0 + X @ beta)
        grad = X.T @ (v * (Y['y'] - mu)) - G
        assert np.fabs(grad[active]).max() < 5e-5


def test_path_solvers_refuse():
    X, Y = _data()
    with pytest.raises(NotImplementedError):
        GaussNet(linear_term=np.ones(P), response_id='y').fit(X, Y)


PROBIT = sm.families.Binomial(link=sm.families.links.Probit())


def test_intercept_kkt():
    # KKT conditions of -loglik / sum(w) + G^T beta + G_0 beta_0 + lambda * penalty,
    # for a canonical and a non-canonical link
    from glmnet import LinearTerm
    X, Y = _data('binomial', weights=True)
    v = np.asarray(Y['w']) / Y['w'].sum()
    y = np.asarray(Y['y'])
    lam = 0.02
    for fam in [sm.families.Binomial(), PROBIT]:
        lt = LinearTerm(coef=0.02 * rng.standard_normal(P), intercept=0.05)
        fit = GLMNet(family=fam, response_id='y', weight_id='w', standardize=False,
                     lambda_values=np.array([lam]), linear_term=lt,
                     control=GLMNetControl(epsnr=1e-12, thresh=1e-14)).fit(X, Y)
        b, a0 = fit.coefs_[0], fit.intercepts_[0]
        eta = a0 + X @ b
        mu = fam.link.inverse(eta)
        r = v * (y - mu) * fam.link.inverse_deriv(eta) / fam.variance(mu)
        grad = X.T @ r - lt.coef
        viol = np.where(b != 0, np.fabs(grad - lam * np.sign(b)), np.maximum(np.fabs(grad) - lam, 0))
        assert viol.max() < 1e-6
        np.testing.assert_allclose(r.sum(), lt.intercept, atol=1e-6)


def test_intercept_requires_intercept():
    from glmnet import LinearTerm
    X, Y = _data('gaussian')
    with pytest.raises(ValueError, match='intercept'):
        _glmnet(fit_intercept=False, linear_term=LinearTerm(np.zeros(P), 0.1)).fit(X, Y)


@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_perturbed_response_is_linear_term(family):
    # for the canonical link the response y + omega is the linear term
    # (-X^T V omega, -sum(v omega)), with its intercept entry
    from glmnet import LinearTerm
    from glmnet.inference import glmstar_problem
    X, Y = _data(family, weights=True)
    v = np.asarray(Y['w']) / Y['w'].sum()
    omega = 0.3 * rng.standard_normal(N)
    if family == 'poisson':
        omega = np.maximum(omega, -Y['y'].values)
    tight = GLMNetControl(epsnr=1e-13, thresh=1e-15)
    Y_prime = Y.assign(y=Y['y'] + omega)
    direct = _glmnet(family, lambda_values=np.array([0.05]), control=tight).fit(X, Y_prime)
    lt = LinearTerm(coef=-X.T @ (v * omega), intercept=-(v * omega).sum())
    via_lt = _glmnet(family, lambda_values=np.array([0.05]), linear_term=lt, control=tight).fit(X, Y)
    np.testing.assert_allclose(direct.coefs_[0], via_lt.coefs_[0], atol=1e-8)
    np.testing.assert_allclose(direct.intercepts_[0], via_lt.intercepts_[0], atol=1e-8)
    # and the problems agree (Q_hat at the LASSO solution does not depend on the response)
    p1 = glmstar_problem(direct, X, Y_prime, information='lasso')
    p2 = glmstar_problem(via_lt, X, Y, information='lasso')
    np.testing.assert_allclose(p1.G_hat, p2.G_hat, atol=1e-8)
    np.testing.assert_allclose(p1.Q_hat, p2.Q_hat, rtol=1e-6)


def test_probit_is_not_logit():
    # statsmodels' Probit subclasses Logit
    from glmnet.inference import resolve_family
    assert resolve_family(sm.families.Binomial()).canonical
    fam = resolve_family(PROBIT)
    assert fam.name == 'binomial' and not fam.canonical


def test_probit_information():
    # at the MLE (lambda = 0) the gradient vanishes and n Q_hat is statsmodels' Fisher information
    from glmnet.inference import problem_from_solution
    X, Y = _data('binomial')
    y = np.asarray(Y['y'])
    mle = sm.GLM(y, sm.add_constant(X), family=PROBIT).fit(tol=1e-12)
    problem = problem_from_solution(X, y, mle.params[1:], mle.params[0], 0., family=PROBIT,
                             information='lasso')
    assert np.abs(problem.G_hat).max() < 1e-8
    np.testing.assert_allclose(np.linalg.inv(N * problem.Q_hat), mle.cov_params(), rtol=1e-5)


@pytest.mark.parametrize('method', ['replicates', 'gaussian'])
def test_score_perturbation_covariance(method):
    # Var(omega) = scalar_noise * dispersion * Q / sum(w), Q the Fisher information at eta
    from glmnet.inference import score_perturbation, problem_from_solution
    X = rng.standard_normal((50, 3))
    eta = 0.3 + X @ np.array([0.5, -0.3, 0.])
    w = rng.integers(1, 4, 50).astype(float)
    draws = np.array([(lambda lt: np.r_[lt.intercept, lt.coef])(
        score_perturbation(X, eta, PROBIT, 0.25, weights=w, dispersion=2., method=method, rng=rng))
        for _ in range(20000)])
    # Q_hat at the LASSO solution 0 + eta: the information at eta
    problem = problem_from_solution(X, np.full(50, 0.5), np.zeros(3), 0., 0., family=PROBIT, weights=w,
                             offset=eta, information='lasso')
    Sigma = 0.25 * 2. * problem.Q_hat / w.sum()
    # within 5 Monte Carlo standard errors (about sqrt(Sigma_jj Sigma_kk / ndraw))
    se = np.sqrt(np.outer(np.diag(Sigma), np.diag(Sigma)) / draws.shape[0])
    assert np.all(np.abs(np.cov(draws.T) - Sigma) < 5 * se)
    assert np.all(np.abs(draws.mean(0)) < 5 * np.sqrt(np.diag(Sigma) / draws.shape[0]))


def test_randomized_inference_problem():
    from glmnet import LinearTerm
    from glmnet.inference import glmstar_problem, glm_inference_problem, problem_from_solution, score_perturbation
    X, Y = _data('binomial')
    y = np.asarray(Y['y'])
    lam = 0.03
    pilot = sm.GLM(y, sm.add_constant(X), family=PROBIT).fit()
    lt = score_perturbation(X, sm.add_constant(X) @ pilot.params, PROBIT, 0.25, rng=rng)
    fit = GLMNet(family=PROBIT, lambda_values=np.array([lam]), linear_term=lt,
                 control=GLMNetControl(epsnr=1e-12, thresh=1e-14)).fit(X, y)
    problem = glmstar_problem(fit, X, y)
    assert problem.kkt_violation() < 1e-4 * lam
    np.testing.assert_allclose(problem.linear_term, np.r_[lt.intercept, lt.coef])
    with pytest.raises(ValueError, match='scalar_noise'):
        glm_inference_problem(problem, X, y, PROBIT)
    info = glm_inference_problem(problem, X, y, PROBIT, scalar_noise=0.25)
    assert info.scalar_noise == 0.25
    # Z_full linearizes the score of y without the linear term: with Q_hat at the LASSO
    # solution, it is the same as for the problem without it at the same solution
    kw = dict(family=PROBIT, scaling=fit.design_.scaling_, y_scale=1., information='lasso')
    with_lt = problem_from_solution(X, y, fit.coefs_[0], fit.intercepts_[0], lam, linear_term=lt, **kw)
    without = problem_from_solution(X, y, fit.coefs_[0], fit.intercepts_[0], lam, **kw)
    np.testing.assert_allclose(with_lt.G_hat - without.G_hat, np.r_[lt.intercept, lt.coef], atol=1e-14)
    info = glm_inference_problem(with_lt, X, y, PROBIT, scalar_noise=0.25)
    info_0 = glm_inference_problem(without, X, y, PROBIT, scalar_noise=0.25)
    np.testing.assert_allclose(info.Z_full, info_0.Z_full, rtol=1e-12, atol=1e-14)


def test_lambda_zero_is_glm():
    # lambda_values=[0] is an unpenalized GLM fit: no warnings, and the
    # constructor's parameters are left as they were
    import warnings
    X, Y = _data('binomial')
    y = np.asarray(Y['y'])
    G = GLMNet(family=PROBIT, lambda_values=np.array([0.]),
               control=GLMNetControl(epsnr=1e-12, thresh=1e-14))
    params = G.get_params()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        G.fit(X, y)
    assert G.get_params()['lambda_min_ratio'] is params['lambda_min_ratio']
    glm = sm.GLM(y, sm.add_constant(X), family=PROBIT).fit(tol=1e-12)
    np.testing.assert_allclose(np.r_[G.intercepts_[0], G.coefs_[0]], glm.params, atol=1e-6)


def test_inference_problem_with_score_covariance():
    # Sigma replaces dispersion * Q_hat / sum(w), e.g. for dependent observations
    from glmnet.inference import glmstar_inference_problem
    X, Y = _data('gaussian')
    fit = _glmnet('gaussian', lambda_values=np.array([0.1])).fit(X, Y)
    X1 = np.column_stack([np.ones(N), X])
    Sigma_eps = 0.5 ** np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    Sigma = X1.T @ Sigma_eps @ X1 / N**2
    info = glmstar_inference_problem(fit, X, Y, Sigma=Sigma)
    assert info.Sigma is Sigma and np.isnan(info.dispersion)
    default = glmstar_inference_problem(fit, X, Y)
    np.testing.assert_allclose(info.Z_full, default.Z_full)
    np.testing.assert_allclose(default.Sigma, default.dispersion * default.problem.Q_hat / N)


def test_inference_problem_with_score_covariance_operator():
    # Sigma may be a LinearOperator, e.g. the sandwich X1^T diag(r^2) X1 / n^2 of a
    # misspecified model; lassoinf gives the same intervals as with the dense matrix
    from scipy.sparse.linalg import LinearOperator, aslinearoperator
    from glmnet.inference import glmstar_inference_problem, score_perturbation
    from glmnet.inference.glmnet_problem import XTVXOperator
    lassoinf = pytest.importorskip('lassoinf')
    X, Y = _data('binomial')
    y = np.asarray(Y['y'])
    X1 = np.column_stack([np.ones(N), X])
    eta = X1 @ sm.GLM(y, X1, family=sm.families.Binomial()).fit().params
    omega = score_perturbation(X, eta, sm.families.Binomial(), 1 / 9, rng=0)
    fit = GLMNet(family=sm.families.Binomial(), lambda_values=np.array([0.02]),
                 linear_term=omega, control=GLMNetControl(epsnr=1e-12, thresh=1e-14)).fit(X, y)
    r = y - 1 / (1 + np.exp(-eta))
    dense = X1.T @ (X1 * r[:, None]**2) / N**2
    op = XTVXOperator(X1, aslinearoperator(scipy.sparse.diags(r**2 / N**2)))
    info_dense = glmstar_inference_problem(fit, X, y, scalar_noise=1 / 9, Sigma=dense)
    info_op = glmstar_inference_problem(fit, X, y, scalar_noise=1 / 9, Sigma=op)
    assert isinstance(info_op.Sigma, LinearOperator) and np.isnan(info_op.dispersion)
    np.testing.assert_allclose(info_op.Z_full, info_dense.Z_full)
    summaries = [lassoinf.LassoInference(**info.lasso_args(), level=0.9).summary_
                 for info in (info_dense, info_op)]
    assert len(summaries[0]) > 0
    # the same up to the tolerance (1e-6) of the root-finding that inverts the intervals
    pd.testing.assert_frame_equal(summaries[0], summaries[1], rtol=1e-6)
