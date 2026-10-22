"""
Smoothing parameter selection by (Laplace approximate) REML or ML, as mgcv's
`gam(method="REML")` and `gam(method="ML")` (Wood, 2011): the smoothing
parameters of the terms' quadratic penalties are those of Gaussian priors
on their coefficients (random effects), and are chosen to maximize the
(restricted) marginal likelihood, with the coefficients integrated out by
Laplace's method. For a Gaussian response (identity link) the approximation
is exact: REML is the REML of the mixed model. A random effect term's
variance is the dispersion over its smoothing parameter.

The inner fit, the penalized likelihood at given smoothing parameters, is
`GAM`'s local scoring with the joint solve; the criterion and its gradient,
through the implicit derivative of the fitted coefficients, are computed by
JAX (an optional dependency: `pip install glmstar[gam]`, or `pip install jax`).

On the scale of the prior weights (`D` the deviance, `S` the penalty matrix,
`sum_k lam_k S_k` over the terms' penalties, several for some terms,
`H = X' W X + S` the Hessian of the penalized deviance over 2 at its
minimum, with `W` the observed information weights, `phi` the dispersion,
`l_s` the saturated log-likelihood, `M` the dimension of the penalties'
null space), the negative log REML is

    (D + b' S b) / (2 phi) - l_s(phi) + log|H| / 2 - log|S|_+ / 2 - M log(2 pi phi) / 2

and ML integrates only the penalized coefficients: `H` restricted to them,
and no last term.

For the Cox family (as mgcv's `cox.ph`) the likelihood is the partial
likelihood, with no intercept and no dispersion (`CoxLaplaceCriterion`, in
numpy, from coxdev's information operator).

References
----------
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and
  marginal likelihood estimation of semiparametric generalized linear
  models. *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N., Pya, N. and Säfken, B. (2016). Smoothing parameter and
  model selection for general smooth models. *Journal of the American
  Statistical Association* 111(516), 1548–1563.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from statsmodels.genmod.families import family as sm_family, links as sm_links

from .joint import JointDesign


def _jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        raise ImportError('method="REML" and method="ML" use JAX: '
                          'pip install jax (or glmstar[gam])') from e
    return jax, jnp


def _jax_family(family):
    """
    The pieces of a (statsmodels) family the criterion needs, in JAX: the
    inverse link, the unit deviance, the saturated log-likelihood (as a
    function of the dispersion) and whether the dispersion is fixed at 1.
    """
    jax, jnp = _jax()
    from jax.scipy.special import xlogy, gammaln
    from jax.scipy.stats import norm

    link = family.link
    if isinstance(link, sm_links.Identity):
        linkinv = lambda eta: eta
    elif isinstance(link, sm_links.Log):
        linkinv = jnp.exp
    elif isinstance(link, sm_links.Logit):
        linkinv = jax.nn.sigmoid
    elif isinstance(link, sm_links.Probit):
        linkinv = norm.cdf
    elif isinstance(link, sm_links.CLogLog):
        linkinv = lambda eta: -jnp.expm1(-jnp.exp(eta))
    elif isinstance(link, sm_links.InverseSquared):
        linkinv = lambda eta: 1 / jnp.sqrt(eta)
    elif isinstance(link, sm_links.InversePower) and getattr(link, 'power', -1.) == -1.:
        linkinv = lambda eta: 1 / eta
    else:
        raise NotImplementedError(f'REML and ML: the link {type(link).__name__} is not supported')

    if isinstance(family, sm_family.Gaussian):
        dev = lambda y, mu: (y - mu)**2
        ls = lambda y, a, phi: jnp.sum(-0.5 * jnp.log(2 * np.pi * phi / a))
        fixed = False
    elif isinstance(family, sm_family.Binomial):
        dev = lambda y, mu: 2 * (xlogy(y, y) - xlogy(y, mu) + xlogy(1 - y, 1 - y) -
                                 xlogy(1 - y, 1 - mu))
        # y the proportion of a trials (a = 1 for 0/1 responses)
        ls = lambda y, a, phi: jnp.sum(gammaln(a + 1) - gammaln(a * y + 1) - gammaln(a * (1 - y) + 1) +
                                       a * (xlogy(y, y) + xlogy(1 - y, 1 - y)))
        fixed = True
    elif isinstance(family, sm_family.Poisson):
        dev = lambda y, mu: 2 * (xlogy(y, y) - xlogy(y, mu) - (y - mu))
        ls = lambda y, a, phi: jnp.sum(a * (xlogy(y, y) - y - gammaln(y + 1)))
        fixed = True
    elif isinstance(family, sm_family.Gamma):
        dev = lambda y, mu: 2 * (-jnp.log(y / mu) + (y - mu) / mu)

        def ls(y, a, phi):
            k = a / phi
            return jnp.sum(k * jnp.log(k) - jnp.log(y) - k - gammaln(k))
        fixed = False
    elif isinstance(family, sm_family.InverseGaussian):
        dev = lambda y, mu: (y - mu)**2 / (mu**2 * y)
        ls = lambda y, a, phi: jnp.sum(-0.5 * jnp.log(2 * np.pi * phi * y**3 / a))
        fixed = False
    else:
        raise NotImplementedError(f'REML and ML: the family {type(family).__name__} is not supported')
    return linkinv, dev, ls, fixed


@dataclass
class SmoothingSelection(object):
    """
    The result of the smoothing parameter selection: the method, the
    estimated terms (their indices) and their smoothing parameters (`lam`,
    on the scale of the GAM's objective; `lam_raw`, on the scale of the
    prior weights, as mgcv's `sp` but for its rescaling of the penalties),
    the dispersion, the criterion at its minimum (`score`, mgcv's
    `gcv.ubre`), its gradient and the optimizer's report.
    """
    method: str
    terms: list
    lam: np.ndarray
    lam_raw: np.ndarray
    dispersion: float
    score: float
    gradient: np.ndarray
    converged: bool
    n_evaluations: int
    message: str = ''


class LaplaceCriterion(object):
    """
    The (negative log) Laplace approximate REML or ML of a GAM, as a function
    of the log smoothing parameters of its estimated terms and, when free,
    the log dispersion. `gam` is a `GAM` set up for fitting (its terms set
    up, its response and weights recorded); its local scoring gives the
    coefficients at each value.
    """

    # the precision of the gradient (for convergence) and of the criterion
    # (relative), and the step of the differences of the gradient
    grad_tol = 1e-5
    value_tol = 1e-12
    hess_eps = 1e-5

    def __init__(self, gam, estimated, method='REML', dispersion=None):
        self.gam, self.method = gam, method
        self.terms = gam.terms_
        self.estimated = list(estimated)
        jax, jnp = _jax()
        self._jax, self._jnp = jax, jnp
        self.linkinv, self.dev, self.ls, fixed = _jax_family(gam._family.base)
        self.free_dispersion = not fixed and dispersion is None
        self.fixed_dispersion = 1. if fixed else dispersion

        D = self.design = JointDesign(self.terms)
        y = np.asarray(gam._response, float)
        a = np.asarray(gam._weight, float)
        keep = a > 0
        self.y, self.a = y[keep], a[keep]
        self.keep = keep
        offset = gam._offset
        self.offset = np.zeros(keep.sum()) if offset is None else np.asarray(offset, float)[keep]
        self.weight_total = a.sum()

        self._set_structure(D, keep)
        self.n_params = self.n_rho + self.free_dispersion
        self._warm = None
        self.n_evaluations = 0

        dev = self.dev
        linkinv = self.linkinv
        # half the second derivative of a unit deviance in eta: the observed
        # information weight (Fisher's for canonical links)
        unit = lambda eta, y: dev(y, linkinv(eta))
        self._h = jax.vmap(jax.grad(jax.grad(unit)), in_axes=(0, 0))

    def _set_structure(self, D, keep, intercept=True):
        """
        The coordinates of the criterion and its penalties. Each penalized
        block is rotated to the eigenvectors of its term's total penalty
        (the sum of its penalties): the null space (unpenalized) is then a
        set of coordinates, the range the others. In them, a term's single
        penalty is diagonal (its eigenvalues: `M`); the penalties of a term
        with several are dense on its range coordinates (`multi`), and
        `log|S|_+` of the term is that of their weighted sum there.

        The penalties (all the terms' together, in order) are indexed by
        `k`: `penalty_terms[k]` is the term of the penalty `k`, and
        `est_index` the estimated ones (all those of the estimated terms).
        """
        P = D.n_coef
        T = np.identity(P)
        self.penalty_terms, self.penalty_names = [], []
        single, multi, multi_terms = [], [], []
        range_coords = []
        for j, (sl, Ss) in enumerate(zip(D.slices, D.penalty_sets)):
            if not Ss:
                continue
            total = sum(Ss)
            ev, V = np.linalg.eigh((total + total.T) / 2)
            on = ev > ev.max() * 1e-10
            T[sl, sl] = V
            idx = np.arange(P)[sl][on]
            range_coords.append(idx)
            if len(Ss) == 1:
                single.append((len(self.penalty_terms), idx, ev[on]))
                self.penalty_terms.append(j)
                self.penalty_names.append(self.terms[j].name)
            else:
                ks = []
                for m, S in enumerate(Ss):
                    U = V[:, on].T @ S @ V[:, on]
                    multi.append((len(self.penalty_terms), idx, (U + U.T) / 2))
                    ks.append(len(self.penalty_terms))
                    self.penalty_terms.append(j)
                    self.penalty_names.append(f'{self.terms[j].name}[{m}]')
                multi_terms.append(ks)
        n_pen = len(self.penalty_terms)
        if not intercept:
            T = T[:, 1:]
            single = [(k, idx - 1, ev) for k, idx, ev in single]
            multi = [(k, idx - 1, U) for k, idx, U in multi]
            range_coords = [idx - 1 for idx in range_coords]
            P -= 1
        self.T = T
        self.X = (D.X @ T)[keep]
        # the single penalties: their eigenvalues in the coordinates
        self.single = np.array([k for k, _, _ in single], int)
        self.M = np.zeros((len(single), P))
        for b, (_, idx, ev) in enumerate(single):
            self.M[b, idx] = ev
        self.rank = np.array([len(ev) for _, _, ev in single], int)
        self.log_ev = np.array([np.log(ev).sum() for _, _, ev in single])
        self.multi, self.multi_terms = multi, multi_terms
        self._multi_of = {k: i for i, (k, _, _) in enumerate(multi)}
        self.penalized = (np.unique(np.concatenate(range_coords)) if range_coords
                          else np.zeros(0, int))
        self.null_dim = P - len(self.penalized)

        estimated = set(self.estimated)
        if not estimated <= set(self.penalty_terms):
            raise ValueError('only penalized terms have smoothing parameters to estimate')
        self.est_index = np.array([k for k in range(n_pen) if self.penalty_terms[k] in estimated], int)
        self.n_rho = len(self.est_index)
        # the fixed smoothing parameters (on the prior weights' scale)
        lam_fixed = np.zeros(n_pen)
        for j in sorted(set(self.penalty_terms) - estimated):
            ks = [k for k in range(n_pen) if self.penalty_terms[k] == j]
            lam_fixed[ks] = np.asarray(self.terms[j].penalty_lams(), float) * self.weight_total
        self.lam_fixed = lam_fixed

    def _penalty_matrix(self, lam, np_=None):
        """The penalty matrix in the coordinates (jnp, or numpy with `np_=np`)."""
        xp = self._jnp if np_ is None else np_
        S = xp.diag(lam[self.single] @ self.M) if len(self.single) else xp.zeros((self.M.shape[1],) * 2)
        for k, idx, U in self.multi:
            if xp is np:
                S[np.ix_(idx, idx)] += lam[k] * U
            else:
                S = S.at[xp.ix_(idx, idx)].add(lam[k] * U)
        return S

    def _logdet_S(self, lam):
        jnp = self._jnp
        value = jnp.sum(self.rank * jnp.log(lam[self.single])) + jnp.sum(self.log_ev)
        for ks in self.multi_terms:
            value = value + jnp.linalg.slogdet(sum(lam[k] * self.multi[self._multi_of[k]][2]
                                                   for k in ks))[1]
        return value

    def _dS_beta(self, lam, beta):
        """The derivatives of the penalty's gradient `S beta` in the estimated `rho_k`: (K, P)."""
        out = np.zeros((self.n_rho, beta.shape[0]))
        where = {k: i for i, k in enumerate(self.single)}
        for i, k in enumerate(self.est_index):
            if k in where:
                out[i] = lam[k] * self.M[where[k]] * beta
            else:
                _, idx, U = self.multi[self._multi_of[k]]
                out[i, idx] = lam[k] * U @ beta[idx]
        return out

    def _rotate(self, H, lam, np_=None):
        """
        `H` (jnp, or numpy with `np_=np`) in coordinates where the penalties
        of each term with several are diagonal at `lam`: its range
        coordinates rotated to the eigenvectors of their weighted sum (as
        Wood's (2011) stable reparametrization; the criterion does not
        change, so the rotation is held fixed for its derivatives). With a
        dominant penalty, the scale of each direction is then on the
        diagonal, and equilibration conditions `H`. The rotations are
        returned, `(coordinates, eigenvectors)` per term.
        """
        xp = self._jnp if np_ is None else np_
        rotations = []
        for ks in self.multi_terms:
            idx = self.multi[self._multi_of[ks[0]]][1]
            A = sum(lam[k] * self.multi[self._multi_of[k]][2] for k in ks)
            if xp is np:
                V = np.linalg.eigh(A)[1]
                H = H.copy()
                H[:, idx] = H[:, idx] @ V
                H[idx, :] = V.T @ H[idx, :]
            else:
                V = self._jax.lax.stop_gradient(xp.linalg.eigh(A)[1])
                H = H.at[:, idx].set(H[:, idx] @ V)
                H = H.at[idx, :].set(V.T @ H[idx, :])
            rotations.append((idx, V))
        return H, rotations

    def _solve_H(self, H, lam, b):
        """`H^{-1} b` (numpy), in the rotated coordinates of `_rotate`, equilibrated."""
        H, rotations = self._rotate(np.asarray(H), np.asarray(lam), np_=np)
        b = np.array(b, float)
        for idx, V in rotations:
            b[idx] = V.T @ b[idx]
        d = 1 / np.sqrt(np.abs(np.diag(H)))
        x = d[:, None] * np.linalg.solve(d[:, None] * H * d[None, :], d[:, None] * b)
        for idx, V in rotations:
            x[idx] = V @ x[idx]
        return x

    def _penalty_diag(self, k):
        """The diagonal of the penalty `k` on its coordinates (for the starting values)."""
        where = {k_: i for i, k_ in enumerate(self.single)}
        if k in where:
            on = self.M[where[k]] > 0
            return np.nonzero(on)[0], self.M[where[k]][on]
        _, idx, U = self.multi[self._multi_of[k]]
        return idx, np.diag(U)

    # the smoothing parameters and dispersion from the parameters

    def _split(self, params):
        jnp = self._jnp
        rho = params[:self.n_rho]
        lam = jnp.asarray(self.lam_fixed).at[self.est_index].set(jnp.exp(rho))
        phi = jnp.exp(params[-1]) if self.free_dispersion else self.fixed_dispersion
        return lam, phi

    def _hessian(self, params, beta):
        jnp = self._jnp
        lam, _ = self._split(params)
        eta = self.X @ beta + self.offset
        h = 0.5 * self._h(eta, self.y) * self.a
        return self.X.T @ (h[:, None] * self.X) + self._penalty_matrix(lam)

    def criterion(self, params, beta):
        """The criterion at `params`, with the coefficients `beta` (in the criterion's coordinates)."""
        jnp = self._jnp
        lam, phi = self._split(params)
        eta = self.X @ beta + self.offset
        mu = self.linkinv(eta)
        Dp = jnp.sum(self.a * self.dev(self.y, mu)) + beta @ self._penalty_matrix(lam) @ beta
        H, _ = self._rotate(self._hessian(params, beta), lam)
        if self.method == 'ML':
            idx = self.penalized
            H = H[jnp.ix_(idx, idx)]
        # equilibrated: with a large smoothing parameter H is ill conditioned
        d = 1 / jnp.sqrt(jnp.abs(jnp.diag(H)))
        logdet_H = jnp.linalg.slogdet(d[:, None] * H * d[None, :])[1] - 2 * jnp.sum(jnp.log(d))
        logdet_S = self._logdet_S(lam)
        value = Dp / (2 * phi) - self.ls(self.y, self.a, phi) + logdet_H / 2 - logdet_S / 2
        if self.method == 'REML':
            value = value - self.null_dim * jnp.log(2 * np.pi * phi) / 2
        return value

    def _set_lams(self, params):
        lam, _ = self._split(params)
        lam = np.asarray(lam)
        terms = np.asarray(self.penalty_terms)
        for j in sorted(set(terms[self.est_index])):
            self.terms[j].set_penalty_lams(lam[terms == j] / self.weight_total)

    def coef(self, params):
        """
        The coefficients at `params` (in the criterion's coordinates), by the
        GAM's local scoring, warm started from the last fit.
        """
        self._set_lams(params)
        gam = self.gam
        # (a fit that does not converge, at smoothing parameters the search
        # tries far from the optimum, is not a warm start)
        state = gam._local_scoring(state=deepcopy(self._warm), control=self._inner_control(),
                                   warn=False)
        if gam.converged_:
            self._warm = state
        return self.T.T @ self.design.gamma(state)

    def _inner_control(self):
        from copy import copy
        control = copy(self.gam.control)
        control.epsnr = min(control.epsnr, 1e-10)
        control.convergence = 'eta'
        # (Fisher scoring converges linearly for non-canonical links)
        control.mxitnr = max(control.mxitnr, 500)
        return control

    def value_and_grad(self, params):
        """The criterion and its gradient (through the fitted coefficients' implicit derivative)."""
        jax, jnp = self._jax, self._jnp
        self.n_evaluations += 1
        params = np.asarray(params, float)
        beta = self.coef(params)
        if not self.gam.converged_:
            return self._unconverged(params)
        value, (g_params, g_beta) = jax.value_and_grad(self.criterion, argnums=(0, 1))(params, beta)
        H = np.asarray(self._hessian(params, beta))
        lam, _ = self._split(params)
        lam = np.asarray(lam)
        g = np.array(g_params, float)
        # d beta / d rho_k = -H^{-1} (lam_k S_k beta): the minimum's equations
        # differentiated
        dbeta = -self._solve_H(H, lam, self._dS_beta(lam, np.asarray(beta)).T)  # (P, K)
        g[:self.n_rho] += np.asarray(g_beta) @ dbeta
        self._max_value = max(getattr(self, '_max_value', -np.inf), float(value))
        return float(value), g

    def _unconverged(self, params):
        """
        The criterion where the inner fit did not converge: with smoothing
        parameters near 0 the penalized likelihood can be maximized only at
        very large coefficients (a smooth with as many knots as events, for
        Cox), and the criterion at an unconverged fit is meaningless (it can
        be very negative). A value larger than any seen, with a gradient
        towards larger smoothing parameters, so that the search turns back.
        """
        worst = getattr(self, '_max_value', 0.)
        g = np.zeros(len(params))
        g[:self.n_rho] = -1e3
        return abs(worst) + 1e6, g

    def start(self):
        """
        Starting values, as mgcv's: each smoothing parameter matches the mean
        diagonal of its block of X' W X (at the intercept-only fit's weights)
        to that of its penalty; the dispersion, the deviance of that fit over
        the number of observations.
        """
        jnp = self._jnp
        eta0 = np.full(self.y.shape[0], float(self.gam._null_state.intercept)) + self.offset
        h = 0.5 * np.asarray(self._h(jnp.asarray(eta0), jnp.asarray(self.y))) * self.a
        xwx = np.einsum('ij,i,ij->j', self.X, h, self.X)
        rho = []
        for k in self.est_index:
            idx, d = self._penalty_diag(k)
            rho.append(np.log(xwx[idx].mean() / max(d.mean(), np.finfo(float).tiny)))
        params = list(rho)
        if self.free_dispersion:
            mu0 = np.asarray(self.linkinv(jnp.asarray(eta0)))
            d = np.asarray(self.dev(jnp.asarray(self.y), jnp.asarray(mu0)))
            params.append(np.log(np.sum(self.a * d) / self.y.shape[0]))
        return np.array(params, float)


class CoxLaplaceCriterion(LaplaceCriterion):
    """
    The Laplace approximate REML or ML of a Cox GAM (as mgcv's `cox.ph`):
    the penalized partial likelihood, no intercept and no dispersion,

        -l_p(b) + b' S b / 2 + log|H| / 2 - log|S|_+ / 2 - M log(2 pi) / 2

    with `H = X' I X + S`, `I` the information of the partial likelihood in
    the linear predictor (coxdev's operator), on the scale of the prior
    weights. In numpy: the derivative of `log|H|` along the coefficients'
    implicit derivative is a central difference of the information operator
    (it needs the third derivatives of the partial likelihood).
    """
    # the partial likelihood's criterion is precise to about 1e-11
    # (relative), its gradient (from a difference of the information) to
    # about 1e-5
    grad_tol = 1e-4
    value_tol = 1e-10
    hess_eps = 1e-3

    def __init__(self, gam, estimated, method='REML', dispersion=None):
        self.gam, self.method = gam, method
        self.terms = gam.terms_
        self.estimated = list(estimated)
        self.free_dispersion, self.fixed_dispersion = False, 1.
        self.family = gam._family
        D = self.design = JointDesign(self.terms)
        a = self.a = np.asarray(gam._weight, float)
        self.weight_total = a.sum()
        self.keep = np.ones(a.shape[0], bool)
        offset = gam._offset
        self.offset = np.zeros(a.shape[0]) if offset is None else np.asarray(offset, float)
        self._coxdev = self.family._coxdev_at(gam._normed_weight)

        # as LaplaceCriterion, without the column of ones (no intercept)
        self._set_structure(D, self.keep, intercept=False)
        if self.multi:
            raise NotImplementedError('REML and ML for the Cox family: terms with several '
                                      'penalties are not supported yet')
        # (every penalty is a single one: the rows of M are the penalties)
        self.est_blocks = self.est_index
        self.n_params = self.n_rho
        self._warm = None
        self.n_evaluations = 0

    def _split(self, params):
        lam = self.lam_fixed.copy()
        lam[self.est_blocks] = np.exp(np.asarray(params, float)[:self.n_rho])
        return lam, 1.

    def _eta(self, beta):
        return self.X @ beta + self.offset

    def _XIX(self, eta):
        I = self._coxdev.information(eta, self.a)
        return self.X.T @ (I @ self.X)

    def _hessian(self, params, beta):
        lam, _ = self._split(params)
        return self._XIX(self._eta(beta)) + np.diag(lam @ self.M)

    def _logdet_H(self, H):
        if self.method == 'ML':
            H = H[np.ix_(self.penalized, self.penalized)]
        # equilibrated, its eigenvalues bounded below (far from the optimum,
        # with a smoothing parameter near 0, H can be numerically singular)
        d = np.sqrt(np.maximum(np.diag(H), np.finfo(float).tiny))
        ev = np.linalg.eigvalsh(H / d[:, None] / d[None, :])
        ev = np.maximum(ev, ev.max() * 1e-15)
        return float(np.sum(np.log(ev)) + 2 * np.sum(np.log(d)))

    def criterion(self, params, beta):
        lam, _ = self._split(params)
        result = self._coxdev(self._eta(beta), self.a)
        value = (result.deviance / 2 - result.loglik_sat + np.sum(lam @ self.M * beta**2) / 2 +
                 self._logdet_H(self._hessian(params, beta)) / 2 -
                 (np.sum(self.rank * np.log(lam)) + np.sum(self.log_ev)) / 2)
        if self.method == 'REML':
            value -= self.null_dim * np.log(2 * np.pi) / 2
        return float(value)

    def coef(self, params):
        self._set_lams(params)
        gam = self.gam
        control = self._inner_control()
        state = gam._local_scoring(state=deepcopy(self._warm), control=control, warn=False)
        converged = gam.converged_
        if converged:
            # one more Newton step: the coefficients to rounding (the
            # objective, on which the fit stops, is flat at its minimum)
            from copy import copy
            one = copy(control)
            one.mxitnr = 1
            state = gam._local_scoring(state=state, control=one, warn=False)
            self._warm = state
        gam.converged_ = converged
        return self.T.T @ self.design.gamma(state)

    def _inner_control(self):
        # Newton's steps (the full Hessian) converge quadratically on the
        # objective; the linear predictor can move by rounding along nearly
        # flat directions (a smoothing parameter near 0), so not on it
        from copy import copy
        control = copy(self.gam.control)
        control.epsnr = min(control.epsnr, 1e-13)
        control.convergence = 'objective'
        control.mxitnr = max(control.mxitnr, 100)
        return control

    def value_and_grad(self, params):
        self.n_evaluations += 1
        params = np.asarray(params, float)
        K = self.n_rho
        beta = self.coef(params)
        if not self.gam.converged_:
            return self._unconverged(params)
        lam, _ = self._split(params)
        s = lam @ self.M
        eta = self._eta(beta)
        result = self._coxdev(eta, self.a)
        H = self._XIX(eta) + np.diag(s)
        value = self.criterion(params, beta)
        idx = self.penalized if self.method == 'ML' else np.arange(H.shape[0])
        Hinv_sub = np.linalg.inv(H[np.ix_(idx, idx)])
        # the derivative in beta of the penalized deviance over 2 (0 at the
        # minimum, up to the inner fit's convergence)
        g_beta = -self.X.T @ (-result.gradient / 2) + s * beta
        dS = lam[self.est_blocks, None] * self.M[self.est_blocks]       # (K, P)
        dbeta = -np.linalg.solve(H, (dS * beta[None, :]).T)            # (P, K)
        g = np.empty(K)
        for k in range(K):
            # with beta held: the penalty and log|H| (and log|S|)
            g[k] = (np.sum(dS[k] * beta**2) / 2 +
                    np.sum(np.diag(Hinv_sub) * dS[k][idx]) / 2 -
                    self.rank[self.est_blocks[k]] / 2)
            # through beta: the information along X dbeta_k
            v = self.X @ dbeta[:, k]
            eps = 1e-4 / max(np.max(np.abs(v)), 1e-12)
            dI = (self._XIX(eta + eps * v) - self._XIX(eta - eps * v)) / (2 * eps)
            g[k] += g_beta @ dbeta[:, k] + np.sum(Hinv_sub * dI[np.ix_(idx, idx)].T) / 2
        self._max_value = max(getattr(self, '_max_value', -np.inf), float(value))
        return value, g

    def start(self):
        """As `LaplaceCriterion.start`: the diagonal of the information at the offset."""
        h = self._coxdev(self.offset, self.a).diag_hessian / 2
        xwx = np.einsum('ij,i,ij->j', self.X, h, self.X)
        rho = []
        for b in self.est_blocks:
            on = self.M[b] > 0
            rho.append(np.log(xwx[on].mean() / self.M[b][on].mean()))
        return np.array(rho, float)


def _newton(crit, x, bounds, maxit=20, tol=1e-8):
    """
    Newton's method from `x` (the quasi-Newton optimum), its Hessian by
    central differences of the gradient (step `crit.hess_eps`), with step
    halving: the optimum to the precision of the gradient. Near the
    optimum a step's decrease of the criterion can be below its precision
    (`crit.value_tol`, relative): a step that does not increase it beyond
    that, and halves the gradient, is taken. Parameters on their bounds (a
    smoothing parameter at 0 or infinity in effect) are held there.
    """
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    value, grad = crit.value_and_grad(x)
    for _ in range(maxit):
        free = ~(((x <= lo + 1e-8) & (grad > 0)) | ((x >= hi - 1e-8) & (grad < 0)))
        if np.max(np.abs(grad[free]), initial=0) < tol:
            return x, value, grad, True
        k, eps = len(x), crit.hess_eps
        H = np.empty((k, k))
        for i in range(k):
            e = np.zeros(k)
            e[i] = eps
            H[:, i] = (crit.value_and_grad(x + e)[1] - crit.value_and_grad(x - e)[1]) / (2 * eps)
        H = (H + H.T) / 2
        Hf = H[np.ix_(free, free)]
        ev = np.linalg.eigvalsh(Hf)
        if ev.min() <= 0:
            Hf = Hf + (1e-6 - ev.min()) * np.identity(Hf.shape[0])
        step = np.zeros(k)
        step[free] = -np.linalg.solve(Hf, grad[free])
        t = 1.
        while t > 1e-8:
            x_new = np.clip(x + t * step, lo, hi)
            v_new, g_new = crit.value_and_grad(x_new)
            if v_new <= value + 1e-12 * abs(value):
                break
            free_new = free & ~(((x_new <= lo + 1e-8) & (g_new > 0)) |
                                ((x_new >= hi - 1e-8) & (g_new < 0)))
            if (v_new <= value + crit.value_tol * abs(value) and
                np.max(np.abs(g_new[free_new]), initial=0) <=
                np.max(np.abs(grad[free]), initial=0) / 2):
                break
            t /= 2
        else:
            # no decrease: at the precision of the criterion
            return x, value, grad, bool(np.max(np.abs(grad[free]), initial=0) < crit.grad_tol)
        x, value, grad = x_new, v_new, g_new
    free = ~(((x <= lo + 1e-8) & (grad > 0)) | ((x >= hi - 1e-8) & (grad < 0)))
    return x, value, grad, bool(np.max(np.abs(grad[free]), initial=0) < crit.grad_tol)


def select_smoothing(gam, dispersion=None):
    """
    Choose the smoothing parameters of `gam`'s estimated terms (penalized
    terms with neither `lam` nor `df`; terms given `df` have their smoothing
    parameter matched at the prior weights, then held fixed) by REML or ML
    (`gam.method`), and set them on its terms. `gam` is set up for fitting.
    """
    terms = gam.terms_
    if any(term.basis() is None for term in terms):
        raise ValueError('REML and ML need a basis for every term (not lo())')
    normed = gam._normed_weight
    estimated = []
    for j, term in enumerate(terms):
        if term.penalty_matrix() is None:
            continue
        if getattr(term, 'lam', None) is None and getattr(term, 'df', None) is None:
            estimated.append(j)
        elif getattr(term, 'df', None) is not None:
            # df: matched at the prior weights, then fixed
            term.set_weights(normed)
            term.lam, term.df = float(term.penalty_lam()), None
    if not estimated:
        raise ValueError(f'method="{gam.method}": no smoothing parameter to estimate '
                         '(every penalized term has lam or df)')
    # placeholders, set at each evaluation
    for j in estimated:
        terms[j].set_penalty_lams(np.ones(len(terms[j].penalty_matrices())))
    for term in terms:
        term.set_weights(normed)

    if getattr(gam, '_cox', False):
        from contextlib import nullcontext
        context, Criterion = nullcontext(), CoxLaplaceCriterion
    else:
        jax, _ = _jax()
        context, Criterion = jax.enable_x64(True), LaplaceCriterion
    with context:
        crit = Criterion(gam, estimated, method=gam.method, dispersion=dispersion)
        x0 = crit.start()
        K = crit.n_rho
        bounds = [(r - 25, r + 25) for r in x0[:K]] + [(None, None)] * (len(x0) - K)
        result = minimize(crit.value_and_grad, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=500, ftol=1e-14, gtol=1e-8))
        x, value, grad, converged = _newton(crit, result.x, bounds)
        lam, phi = crit._split(x)
        lam, phi = np.asarray(lam, float)[crit.est_index], float(phi)
        # the terms at the optimum
        crit.coef(x)
        converged = converged and gam.converged_
    return SmoothingSelection(method=gam.method,
                              terms=[crit.penalty_names[k] for k in crit.est_index],
                              lam=lam / crit.weight_total,
                              lam_raw=lam,
                              dispersion=phi,
                              score=value,
                              gradient=grad,
                              converged=converged,
                              n_evaluations=crit.n_evaluations,
                              message=str(result.message))
