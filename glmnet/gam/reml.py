"""
**Experimental.** The fits, smoothing parameters and variance components
are checked against mgcv, lme4 and nlme, but the interface may change and
the optimization of the criterion is less tested than the rest of
`glmnet.gam`: check `GAM.smoothing_.converged` and compare with fixed
smoothing parameters or with R. `GAM(method='REML')` warns
(`ExperimentalWarning`).

Smoothing parameter selection by (Laplace approximate) REML or ML, as mgcv's
`gam(method="REML")` and `gam(method="ML")` (Wood, 2011): the smoothing
parameters of the terms' quadratic penalties are those of Gaussian priors
on their coefficients (random effects), and are chosen to maximize the
(restricted) marginal likelihood, with the coefficients integrated out by
Laplace's method. For a Gaussian response (identity link) the approximation
is exact: REML is the REML of the mixed model. A random effect term's
variance is the dispersion over its smoothing parameter.

The inner fit, the penalized likelihood at given smoothing parameters, is
`GAM`'s local scoring with the joint solve. The criterion and its gradient,
through the implicit derivative of the fitted coefficients, need only four
operations on the Hessian `H`: its log determinant, solves, the traces
`tr(H^{-1} S_k)` and the leverages `x_i' H^{-1} x_i` (`glmnet.gam.linalg`;
`DenseHessian` forms `H`, and other implementations need not). JAX (an
optional dependency: `pip install glmstar[gam]`, or `pip install jax`) gives
the derivatives of the family's unit deviance.

On the scale of the prior weights (`D` the deviance, `S` the penalty matrix,
`sum_k lam_k S_k` over the terms' penalties, several for some terms,
`H = X' W X + S` the Hessian of the penalized deviance over 2 at its
minimum, with `W` the observed information weights, `phi` the dispersion,
`l_s` the saturated log-likelihood, `M` the dimension of the penalties'
null space), the negative log REML is

    (D + b' S b) / (2 phi) - l_s(phi) + log|H| / 2 - log|S|_+ / 2 - M log(2 pi phi) / 2

and ML integrates only the penalized coefficients: `H` restricted to them,
and no last term.

From the formulas to the code
-----------------------------
The notation is that of the design page. `LaplaceCriterion` evaluates `V`
in four steps, each a method:

1. `coef(params)`: the inner fit. It sets the smoothing parameters on the
   terms and runs `GAM`'s local scoring to `b = argmin D_p(b)`, warm started.
2. `_evaluation(params, beta)`: the penalty side. It gives `S_lambda`, the
   pieces `lambda_k S_k` (the derivatives of `S_lambda` in `rho_k`),
   `log|S_lambda|_+` and its derivatives, from the penalty objects
   (`glmnet.gam.penalties`).
3. `_factor(ev)`: the likelihood side. It gives the unit deviance `d_i` and
   its derivatives in `eta`, `d'_i, d''_i, d'''_i` (`dev, d1, d2, d3`, by
   JAX), the weights `W = diag(a_i d''_i / 2)`, and the factored
   `H = X' W X + S_lambda` (`H_int`: `H_PP` for ML).
4. `_value(ev)`: `V`, term by term (see its comments).

Symbols and their names in the code:

    rho_k = log lambda_k          params[:n_rho]          (estimated; fixed: lam_fixed)
    log phi                       params[n_rho]           (if free_dispersion)
    log theta (negative binomial) params[-1]              (if estimate_theta)
    a_i (prior weights)           self.a
    X (in the criterion's coords) self.X, ev['X']
    beta-hat                      ev['beta']
    S_lambda                      ev['S']
    lambda_k S_k                  ev['pieces'][k]
    log|S_lambda|_+               ev['logdet_S']
    D_p = D + b' S_lambda b       ev['Dp']
    l_s(phi), dl_s/dphi           ls, ev['dls']
    M                             self.null_dim
    H, H_PP                       ev['H'], ev['H_int']

The gradient (`value_and_grad`) is the total derivative through `b`, by the
implicit function theorem (`b` solves `dD_p/db = 0`):

    dV/drho_k = dV/drho_k |_b + dV/db |_b . db/drho_k,    db/drho_k = -H^{-1} lambda_k S_k b

    dV/drho_k |_b = b' lambda_k S_k b / (2 phi) + tr(H^{-1} lambda_k S_k) / 2 - d log|S_lambda|_+ / drho_k / 2
    dV/db |_b     = [X' (a d') / 2 + S_lambda b] / phi      (0 at the optimum: kept for the inner
                                                             fit's tolerance)
                    + X' (a d''' . lev) / 4                  (log|H| through W; lev_i = x_i' H^{-1} x_i)

    dV/dlog phi   = -D_p / (2 phi) - phi dl_s/dphi - M / 2   (REML; ML without M / 2)

No term of the gradient needs more of `H` than its four operations
(`glmnet.gam.linalg`): `log|H|`, solves, `tr(H^{-1} A)` and the leverages.
The Hessian of `V`, for the outer Newton steps (`_newton`), is central
differences of this gradient.

Two parameters' names are easily confused here. `theta` (`_theta`,
`theta_fixed`) is the vector of the penalties' parameters
(`glmnet.gam.penalties`): `log lambda_k`, or the log-Cholesky entries of a
random effects' precision. `log_theta` (`t` in the family functions) is the
log of the negative binomial's theta, a parameter of the family.

The documentation's page "REML and ML for GAMs: the design" derives the
criterion, computes it by hand in JAX for a small example, and describes
the choices made here (the coordinates, the inner fit, the search).

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
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from statsmodels.genmod.families import family as sm_family, links as sm_links

from .joint import JointDesign
from .linalg import DenseHessian
from .penalties import (ScaledPenalty, SumPenalty, CholeskyPenalty, CovariancePenalty,
                        covariance_penalty)

# the penalties of correlated random effects (their covariance's parameters)
COVARIANCE = (CholeskyPenalty, CovariancePenalty)


class ExperimentalWarning(UserWarning):
    """The feature is experimental: its interface and results may change."""


def _jax():
    # (JAX is optional, needed only for REML and ML: imported when used)
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

    # (each function takes `t`, the log of the negative binomial's theta,
    # which the others ignore)
    if isinstance(family, sm_family.Gaussian):
        dev = lambda y, mu, t: (y - mu)**2
        ls = lambda y, a, phi, t: jnp.sum(-0.5 * jnp.log(2 * np.pi * phi / a))
        fixed = False
    elif isinstance(family, sm_family.Binomial):
        dev = lambda y, mu, t: 2 * (xlogy(y, y) - xlogy(y, mu) + xlogy(1 - y, 1 - y) -
                                    xlogy(1 - y, 1 - mu))
        # y the proportion of a trials (a = 1 for 0/1 responses)
        ls = lambda y, a, phi, t: jnp.sum(gammaln(a + 1) - gammaln(a * y + 1) -
                                          gammaln(a * (1 - y) + 1) +
                                          a * (xlogy(y, y) + xlogy(1 - y, 1 - y)))
        fixed = True
    elif isinstance(family, sm_family.Poisson):
        dev = lambda y, mu, t: 2 * (xlogy(y, y) - xlogy(y, mu) - (y - mu))
        ls = lambda y, a, phi, t: jnp.sum(a * (xlogy(y, y) - y - gammaln(y + 1)))
        fixed = True
    elif isinstance(family, sm_family.NegativeBinomial):
        # theta = exp(t): variance mu + mu^2 / theta
        def dev(y, mu, t):
            theta = jnp.exp(t)
            return 2 * (xlogy(y, y) - xlogy(y, mu) - (y + theta) * (jnp.log(y + theta) -
                                                                    jnp.log(mu + theta)))

        def ls(y, a, phi, t):
            theta = jnp.exp(t)
            return jnp.sum(a * (gammaln(y + theta) - gammaln(theta) - gammaln(y + 1) +
                                theta * t + xlogy(y, y) - (y + theta) * jnp.log(y + theta)))
        fixed = True
    elif isinstance(family, sm_family.Gamma):
        dev = lambda y, mu, t: 2 * (-jnp.log(y / mu) + (y - mu) / mu)

        def ls(y, a, phi, t):
            k = a / phi
            return jnp.sum(k * jnp.log(k) - jnp.log(y) - k - gammaln(k))
        fixed = False
    elif isinstance(family, sm_family.InverseGaussian):
        dev = lambda y, mu, t: (y - mu)**2 / (mu**2 * y)
        ls = lambda y, a, phi, t: jnp.sum(-0.5 * jnp.log(2 * np.pi * phi * y**3 / a))
        fixed = False
    else:
        raise NotImplementedError(f'REML and ML: the family {type(family).__name__} is not supported')
    if getattr(family, 'quasi', False):
        # a quasi-likelihood (`glmnet.gam.families`): a free dispersion, and
        # mgcv's "saturated log-likelihood", a Gaussian's in the dispersion
        ls = lambda y, a, phi, t: jnp.sum(-0.5 * jnp.log(phi / a))
        fixed = False
    return linkinv, dev, ls, fixed


@dataclass
class SmoothingSelection(object):
    """
    The result of the smoothing parameter selection: the method, the
    estimated terms (their indices) and their smoothing parameters (`lam`,
    on the scale of the GAM's objective; `lam_raw`, on the scale of the
    prior weights, as mgcv's `sp` but for its rescaling of the penalties),
    the dispersion, the criterion at its minimum (`score`, mgcv's
    `gcv.ubre`), its gradient and the optimizer's report. `theta` are the
    estimated parameters of the penalties (`glmnet.gam.penalties`): the
    logs of `lam_raw`, and for correlated random effects those of the
    Cholesky factor of their precision, whose `lam` are NaN.
    `negative_binomial_theta` is the estimated theta of a negative binomial
    (`glmnet.gam.families.NegativeBinomial(theta=None)`), else None.
    """
    method: str
    terms: list
    lam: np.ndarray
    lam_raw: np.ndarray
    theta: np.ndarray
    dispersion: float
    score: float
    gradient: np.ndarray
    converged: bool
    n_evaluations: int
    message: str = ''
    negative_binomial_theta: Optional[float] = None


class LaplaceCriterion(object):
    """
    The (negative log) Laplace approximate REML or ML of a GAM, as a function
    of the log smoothing parameters of its estimated terms and, when free,
    the log dispersion and the log of the negative binomial's theta (in that
    order: `params = (rho, log phi, log theta)`). `gam` is a `GAM` set up for fitting (its terms set
    up, its response and weights recorded); its local scoring gives the
    coefficients at each value.
    """

    # the precision of the gradient (for convergence) and of the criterion
    # (relative), and the step of the differences of the gradient
    grad_tol = 1e-5
    value_tol = 1e-12
    hess_eps = 1e-5

    # the negative binomial's theta estimated (`glmnet.gam.families`), and
    # its log (0, ignored, for the other families)
    estimate_theta = False
    log_theta = 0.

    def __init__(self, gam, estimated, method='REML', dispersion=None):
        self.gam, self.method = gam, method
        self.terms = gam.terms_
        self.estimated = list(estimated)
        jax, jnp = _jax()
        self._jax, self._jnp = jax, jnp
        base = gam._family.base
        self.linkinv, self.dev, self.ls, fixed = _jax_family(base)
        self.estimate_theta = getattr(base, 'estimate_theta', False)
        # the log theta (the negative binomial's; 0 for the others)
        self.log_theta = (float(np.log(1 / base.alpha))
                          if isinstance(base, sm_family.NegativeBinomial) else 0.)
        # a given dispersion is used for every family (for the binomial and
        # Poisson, a quasi-likelihood's); otherwise 1 for those
        self.free_dispersion = not fixed and dispersion is None
        self.fixed_dispersion = dispersion if dispersion is not None else (1. if fixed else None)

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
        self.n_params = self.n_rho + self.free_dispersion + self.estimate_theta
        self._warm = None
        self.n_evaluations = 0

        dev = self.dev
        linkinv = self.linkinv
        # the unit deviance as a function of eta, d(y, g^{-1}(eta)): all of
        # the family the criterion uses is it, its derivatives in eta (d',
        # d'', d''') and l_s. W_ii = a_i d''_i / 2 is then the observed
        # information for any link (Fisher's for a canonical one)
        unit = lambda eta, y, t: dev(y, linkinv(eta), t)
        self._h = jax.vmap(jax.grad(jax.grad(unit)), in_axes=(0, 0, None))
        # the first three derivatives of the unit deviance in eta, and that
        # of the saturated log-likelihood in the dispersion: all the
        # criterion needs of the family
        d1 = jax.grad(unit)
        d2 = jax.grad(d1)
        d3 = jax.grad(d2)
        self._derivatives = jax.jit(jax.vmap(lambda e, y, t: (unit(e, y, t), d1(e, y, t),
                                                              d2(e, y, t), d3(e, y, t)),
                                             in_axes=(0, 0, None)))
        # and, for theta, those of the deviance and its first two in eta in
        # log theta
        dt = [jax.grad(f, argnums=2) for f in (unit, d1, d2)]
        self._theta_derivatives = jax.jit(jax.vmap(lambda e, y, t: tuple(f(e, y, t) for f in dt),
                                                   in_axes=(0, 0, None)))
        ls = self.ls
        self._ls = jax.jit(lambda y, a, phi, t: (ls(y, a, phi, t), jax.grad(ls, argnums=2)(y, a, phi, t),
                                                 jax.grad(ls, argnums=3)(y, a, phi, t)))

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

        In the math: the coefficients are `beta = T gamma`, `T` orthogonal and
        block diagonal, and the criterion works in `gamma` throughout (with
        `X T` as the design, `self.X`). `V` is invariant to the change. In
        these coordinates the null space of `S_lambda` is the set of
        coordinates `penalized` leaves out, so `M` (`null_dim`) is a count,
        and `H_PP` for ML is `H` restricted to `penalized`. `log|S_lambda|_+`
        of a single penalty is `r_j log lambda_j + sum log(eigenvalues)`,
        exactly, however large `lambda_j` is.
        """
        P = D.n_coef
        T = np.identity(P)
        self.penalty_terms, self.penalty_names = [], []
        single, multi, multi_terms, cholesky = [], [], [], []
        range_coords = []
        for j, (sl, Ss) in enumerate(zip(D.slices, D.penalty_sets)):
            if not Ss:
                continue
            term = self.terms[j]
            if getattr(term, 'penalty_varies', False):
                # correlated random effects: their precision, by its
                # covariance structure, on the coordinates as they are (group
                # by group)
                idx = np.arange(P)[sl]
                range_coords.append(idx)
                model = covariance_penalty(getattr(term, 'covariance', 'unstructured'),
                                           idx.reshape((-1, term.q_)), term.name,
                                           term.effect_names)
                cholesky.append((len(self.penalty_terms), model))
                self.penalty_terms.extend([j] * model.n_params)
                self.penalty_names.extend(model.names)
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
            for _, model in cholesky:
                model.idx = model.idx - 1
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
        # the terms' penalties as objects (`glmnet.gam.penalties`), in the
        # order of their parameters: (term, penalty, its parameters' slice)
        self.models = []
        for k, idx, ev in single:
            self.models.append((self.penalty_terms[k],
                                ScaledPenalty(idx, ev, [self.penalty_names[k]]), slice(k, k + 1)))
        for ks in multi_terms:
            idx = multi[self._multi_of[ks[0]]][1]
            self.models.append((self.penalty_terms[ks[0]],
                                SumPenalty(idx, [multi[self._multi_of[k]][2] for k in ks],
                                           [self.penalty_names[k] for k in ks]),
                                slice(ks[0], ks[-1] + 1)))
        for k, model in cholesky:
            self.models.append((self.penalty_terms[k], model, slice(k, k + model.n_params)))
        self.models.sort(key=lambda m: m[2].start)
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
            if not getattr(self.terms[j], 'penalty_varies', False):
                lam_fixed[ks] = np.asarray(self.terms[j].penalty_lams(), float) * self.weight_total
        self.lam_fixed = lam_fixed
        # the parameters of the penalties (`glmnet.gam.penalties`): the logs
        # of the smoothing parameters, or of a precision's Cholesky factor
        theta = np.log(np.where(lam_fixed > 0, lam_fixed, 1.))
        for j, model, sl in getattr(self, 'models', []):
            if isinstance(model, COVARIANCE) and j not in estimated:
                theta[sl] = model.theta(self.terms[j].precision * self.weight_total)
        self.theta_fixed = theta

    def _evaluation(self, params, beta):
        """
        The criterion's pieces at `params` and `beta`, in the coordinates in
        which the penalties evaluate themselves (`glmnet.gam.penalties`;
        Wood's stable reparametrization for a sum of penalties): the design
        `X`, coefficients `beta`, the penalty matrix `S`, the derivatives of
        the penalty in each parameter `k` (`pieces[k]`), and `log|S|_+` and
        its derivatives.

        In the math: `S = S_lambda = sum_k lambda_k S_k`; for a smoothing
        parameter, `pieces[k] = dS_lambda / drho_k = lambda_k S_k`, and
        `dlogdet_S[k] = d log|S_lambda|_+ / drho_k` (`r_k` for a single
        penalty). A penalty may evaluate in its own coordinates,
        `beta = V u` (Wood's stable basis for several penalties; lme4's
        `b = L u` for a covariance): then `X` becomes `X V`, `S` becomes
        `V' S V`, and `logdet_V = log|det V|` corrects `log|H|`, while
        `logdet_S` stays that of the original coordinates.
        """
        lam, phi = self._split(params)
        lam, phi = np.asarray(lam, float), float(phi)
        theta = self._theta(params)
        X, beta = self.X.copy(), np.array(beta, float)
        P = X.shape[1]
        S = np.zeros((P, P))
        pieces, logdet, dlogdet, logdet_V = {}, 0., np.zeros(len(lam)), 0.
        for _, model, sl in self.models:
            e = model.evaluate(theta[sl])
            if e.V is not None:
                idx = model.idx
                if idx.ndim == 1:
                    X[:, idx] = X[:, idx] @ e.V
                    beta[idx] = e.V_inverse @ beta[idx]
                else:
                    # the same change in each group's coordinates
                    X[:, idx] = X[:, idx] @ e.V
                    beta[idx] = beta[idx] @ e.V_inverse.T
                logdet_V += e.logdet_V
            e.matrix.add_to(S)
            for k, piece in zip(range(sl.start, sl.stop), e.derivatives):
                pieces[k] = piece
            logdet += e.logdet
            dlogdet[sl] = e.dlogdet
        return dict(lam=lam, phi=phi, log_theta=self._log_theta(params), X=X, beta=beta, S=S,
                    pieces=pieces, logdet_S=logdet, dlogdet_S=dlogdet, logdet_V=logdet_V)

    def _split(self, params):
        params = np.asarray(params, float)
        lam = self.lam_fixed.copy()
        lam[self.est_index] = np.exp(params[:self.n_rho])
        phi = float(np.exp(params[self.n_rho])) if self.free_dispersion else self.fixed_dispersion
        return lam, phi

    def _log_theta(self, params):
        """The log of the negative binomial's theta at `params` (fixed, or the last)."""
        return float(params[-1]) if self.estimate_theta else self.log_theta

    # the factorization of H (a class with DenseHessian's operations)
    Hessian = DenseHessian

    def _factor(self, ev):
        """
        Factor `H` at the evaluation `ev` (`_evaluation`): adds `H` and the family's pieces.

        In the math, at `eta = X b + offset`: `dev` is the unit deviances
        `d_i`, and `d1, d2, d3` their derivatives in `eta_i`. `H = X' W X +
        S_lambda` with `W = diag(a_i d''_i / 2)`, the Hessian of `D_p / 2`.
        For ML, `H_int = H_PP`, the block of the penalized coordinates (the
        unpenalized ones are not integrated out). For REML, `H_int = H`.
        """
        jnp = self._jnp
        eta = ev['X'] @ ev['beta'] + self.offset
        args = (jnp.asarray(eta), jnp.asarray(self.y), ev['log_theta'])
        # d_i, d'_i, d''_i, d'''_i (derivatives in eta_i)
        dev, d1, d2, d3 = (np.asarray(v, float) for v in self._derivatives(*args))
        if not all(np.all(np.isfinite(v)) for v in (dev, d1, d2, d3)):
            raise ValueError('the deviance or its derivatives are not finite')
        ev.update(dev=dev, d1=d1, d3=d3, H=self.Hessian(ev['X'], 0.5 * self.a * d2, ev['S']))
        if self.estimate_theta:
            ev['dev_t'], ev['d1_t'], ev['d2_t'] = (np.asarray(v, float)
                                                   for v in self._theta_derivatives(*args))
        ev['H_int'] = ev['H'].restrict(self.penalized) if self.method == 'ML' else ev['H']
        return ev

    def _polish(self, ev):
        """
        One Newton step from the inner fit's coefficients, in the stable
        coordinates and with the criterion's factorization of `H`: the
        coefficients to its precision, and independent of the inner fit's
        path (its warm start), on which the criterion would otherwise depend
        a little where a smoothing parameter is large.

        In the math: `b <- b - H^{-1} g`, with `g = X' (a d') / 2 + S_lambda b`
        the gradient of `D_p / 2` (and `H` its Hessian).
        """
        grad = 0.5 * ev['X'].T @ (self.a * ev['d1']) + ev['S'] @ ev['beta']
        ev['beta'] = ev['beta'] - ev['H'].solve(grad)
        return self._factor(ev)

    def _value(self, ev):
        """
        The criterion `V` at the evaluation `ev` (`_factor`'s), term by term:

            V = D_p / (2 phi) - l_s(phi) + log|H| / 2 - log|S_lambda|_+ / 2 [- M log(2 pi phi) / 2]

        (the last term for REML only; ML's `log|H|` is `log|H_PP|`).
        """
        lam, phi, beta = ev['lam'], ev['phi'], ev['beta']
        # D_p = sum_i a_i d_i + b' S_lambda b
        Dp = float(np.sum(self.a * ev['dev']) + beta @ ev['S'] @ beta)
        jnp = self._jnp
        # l_s and its derivatives in phi and in log theta
        ls, dls, dls_t = (float(v) for v in self._ls(jnp.asarray(self.y), jnp.asarray(self.a), phi,
                                                     ev['log_theta']))
        # (log|H| in the original coordinates: less that of the change of them,
        # log|H_orig| / 2 = log|H| / 2 - log|det V|)
        value = (Dp / (2 * phi) - ls + ev['H_int'].logdet() / 2 - ev['logdet_V'] -
                 ev['logdet_S'] / 2)
        if self.method == 'REML':
            # the flat prior on the M unpenalized coefficients
            value -= self.null_dim * np.log(2 * np.pi * phi) / 2
        ev.update(Dp=Dp, dls=dls, dls_t=dls_t)
        return float(value)

    def criterion(self, params, beta):
        """The criterion at `params`, with the coefficients `beta` (in the criterion's coordinates)."""
        params = np.asarray(params, float)
        return self._value(self._factor(self._evaluation(params, beta)))

    def value_and_grad(self, params):
        """
        The criterion and its gradient. In the smoothing parameters, with
        `beta` held, and through `beta`, the implicit derivative of the
        fitted coefficients, `d beta / d rho_k = -H^{-1} lam_k S_k beta`;
        `log|H|` changes with `beta` through the weights `h`, whose
        derivatives (`h'`, the third derivatives of the deviance) weight the
        leverages `x_i' H^{-1} x_i`. Only `H`'s operations (`DenseHessian`)
        are needed: its log determinant, solves, traces and leverages.
        """
        self.n_evaluations += 1
        params = np.asarray(params, float)
        beta = self.coef(params)
        if not self.gam.converged_:
            return self._unconverged(params)
        try:
            ev = self._polish(self._factor(self._evaluation(params, beta)))
            value = self._value(ev)
        except (ValueError, np.linalg.LinAlgError):
            # the fitted means at 0 or 1 in floating point (a separated
            # binomial fit, far from the optimum): the deviance's derivatives
            # are not finite there, as at an unconverged fit
            return self._unconverged(params)
        if not np.isfinite(value):
            return self._unconverged(params)
        lam, phi, X, beta, H, H_int = (ev[k] for k in ('lam', 'phi', 'X', 'beta', 'H', 'H_int'))
        g = np.zeros(self.n_params)
        # dV/drho_k with b held: the three terms of V that contain lambda_k,
        #   b' lambda_k S_k b / (2 phi) + tr(H^{-1} lambda_k S_k) / 2 - (d log|S_lambda|_+ / drho_k) / 2
        dS_beta = np.zeros((self.n_rho, beta.shape[0]))            # lam_k S_k beta
        for i, k in enumerate(self.est_index):
            piece = ev['pieces'][k]
            dS_beta[i] = piece.apply(beta)
            g[i] = (beta @ dS_beta[i] / (2 * phi) + piece.trace(H_int) / 2 -
                    ev['dlogdet_S'][k] / 2)
        # dV/db with rho held: from D_p / (2 phi), [X' (a d') + 2 S_lambda b] / (2 phi),
        # which is 0 at the minimum (up to the inner fit's convergence); and
        # from log|H| / 2, through W(b): d log|H| / db_j = tr(H^{-1} X' dW/db_j X)
        # = sum_i (a_i d'''_i / 2) x_ij lev_i, with the leverages
        # lev_i = x_i' H^{-1} x_i
        g_beta = (X.T @ (self.a * ev['d1']) + 2 * ev['S'] @ beta) / (2 * phi)
        g_beta += X.T @ (0.5 * self.a * ev['d3'] * H_int.leverages()) / 2
        # the implicit derivative db/drho_k = -H^{-1} lambda_k S_k b, one solve per k
        dbeta = -H.solve(dS_beta.T)                                  # (P, K)
        g[:self.n_rho] += g_beta @ dbeta
        if self.free_dispersion:
            # dV/dlog phi: b does not depend on phi (it cancels from the
            # inner problem), so only the explicit terms,
            #   -D_p / (2 phi) - phi dl_s/dphi [- M / 2]
            g[self.n_rho] = (-ev['Dp'] / (2 * phi) - phi * ev['dls'] -
                             (self.null_dim / 2 if self.method == 'REML' else 0.))
        if self.estimate_theta:
            # log theta: in the deviance, the saturated log-likelihood and
            # log|H| (through the weights) with beta held, and through beta,
            # d beta / d log theta = -H^{-1} X' (a d1_t) / 2
            g[-1] = (np.sum(self.a * ev['dev_t']) / (2 * phi) - ev['dls_t'] +
                     np.sum(0.5 * self.a * ev['d2_t'] * H_int.leverages()) / 2 +
                     g_beta @ -H.solve(0.5 * X.T @ (self.a * ev['d1_t'])))
        self._max_value = max(getattr(self, '_max_value', -np.inf), value)
        return value, g

    def off_diagonal(self):
        """The positions, among the estimated parameters, of Cholesky factors' off-diagonal entries."""
        position = {k: i for i, k in enumerate(self.est_index)}
        off = []
        for _, model, sl in self.models:
            if isinstance(model, COVARIANCE):
                for k, o in zip(range(sl.start, sl.stop), model.off_diagonal_params):
                    if o and k in position:
                        off.append(position[k])
        return np.array(off, int)

    def _theta(self, params):
        """The penalties' parameters: the fixed, and the estimated (`params`)."""
        theta = self.theta_fixed.copy()
        theta[self.est_index] = np.asarray(params, float)[:self.n_rho]
        return theta

    def _set_lams(self, params):
        """Set the estimated terms' smoothing parameters (or precisions) at `params`."""
        theta = self._theta(params)
        estimated = set(np.asarray(self.penalty_terms)[self.est_index])
        for j, model, sl in self.models:
            if j not in estimated:
                continue
            if isinstance(model, COVARIANCE):
                self.terms[j].set_precision(model.precision(theta[sl]) / self.weight_total)
            else:
                self.terms[j].set_penalty_lams(np.exp(theta[sl]) / self.weight_total)

    def coef(self, params):
        """
        The coefficients at `params` (in the criterion's coordinates), by the
        GAM's local scoring, warm started from the last fit.
        """
        self._set_lams(params)
        gam = self.gam
        if self.estimate_theta:
            gam._family.base.set_theta(np.exp(self._log_theta(params)))
        # (a fit that does not converge, at smoothing parameters the search
        # tries far from the optimum, is not a warm start)
        try:
            state = gam._local_scoring(state=deepcopy(self._warm), control=self._inner_control(),
                                       warn=False)
        except ValueError:
            # (step halving cannot decrease the objective, at extreme
            # parameters: as an unconverged fit)
            gam.converged_ = False
            return np.zeros(self.X.shape[1])
        if gam.converged_:
            self._warm = state
        return self.T.T @ self.design.gamma(state)

    def _inner_control(self):
        control = copy(self.gam.control)
        # 1e-8, not tighter: the polish (`_polish`, a Newton step, which
        # squares the error) gives the coefficients the criterion's
        # precision, and a large smoothing parameter's rounding can move the
        # linear predictor by more than 1e-10 at each step
        control.epsnr = 1e-8
        control.convergence = 'eta'
        # (Fisher scoring converges linearly for non-canonical links)
        control.mxitnr = max(control.mxitnr, 500)
        return control

    def _unconverged(self, params):
        """
        The criterion where the inner fit did not converge: with smoothing
        parameters near 0 the penalized likelihood can be maximized only at
        very large coefficients (a smooth with as many knots as events, for
        Cox), and the criterion at an unconverged fit is meaningless (it can
        be very negative). A value larger than any seen, with a gradient
        towards more penalty (larger smoothing parameters; a smaller
        covariance of correlated random effects: its penalty object's
        `more_penalty`), so that the search turns back.
        """
        worst = getattr(self, '_max_value', 0.)
        theta = self._theta(params)
        direction = np.ones(len(theta))
        for _, model, sl in self.models:
            if hasattr(model, 'more_penalty'):
                direction[sl] = model.more_penalty(theta[sl])
        g = np.zeros(len(params))
        g[:self.n_rho] = -1e3 * direction[self.est_index]
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
        h = 0.5 * np.asarray(self._h(jnp.asarray(eta0), jnp.asarray(self.y), self.log_theta)) * self.a
        xwx = np.einsum('ij,i,ij->j', self.X, h, self.X)
        start = np.zeros(len(self.lam_fixed))
        for _, model, sl in self.models:
            start[sl] = model.start(xwx)
        params = list(start[self.est_index])
        if self.free_dispersion:
            mu0 = np.asarray(self.linkinv(jnp.asarray(eta0)))
            d = np.asarray(self.dev(jnp.asarray(self.y), jnp.asarray(mu0), self.log_theta))
            params.append(np.log(np.sum(self.a * d) / self.y.shape[0]))
        if self.estimate_theta:
            # the moments' estimate at the intercept-only fit: var = mu + mu^2 / theta
            mu0 = np.asarray(self.linkinv(jnp.asarray(eta0)))
            excess = np.sum(self.a * ((self.y - mu0)**2 - mu0))
            theta0 = np.sum(self.a * mu0**2) / excess if excess > 0 else 100.
            params.append(np.log(np.clip(theta0, 1e-2, 1e4)))
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
    # the partial likelihood's criterion is precise to about 1e-9 (relative:
    # the inner fit, warm started, depends a little on the search's path,
    # about 3e-10 at the optimum in the tests), its gradient (from a
    # difference of the information) to about 1e-5
    grad_tol = 1e-4
    value_tol = 1e-9
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
        self.est_blocks = self.est_index
        self.n_params = self.n_rho
        self._warm = None
        self.n_evaluations = 0

    def _eta(self, beta):
        return self.X @ beta + self.offset

    def _XIX(self, X, eta):
        I = self._coxdev.information(eta, self.a)
        return X.T @ (I @ X)

    def _cox_evaluation(self, params, beta):
        """
        `_evaluation` (in the penalties' coordinates), and the partial
        likelihood and the factored `H = X' I X + S` there.
        """
        ev = self._evaluation(params, beta)
        X = ev['X']
        for polish in (True, False):
            eta = X @ ev['beta'] + self.offset
            ev['eta'] = eta
            ev['result'] = self._coxdev(eta, self.a)
            H = DenseHessian.from_matrix(self._XIX(X, eta) + ev['S'], X)
            if polish:
                # one Newton step with this factorization (as
                # `LaplaceCriterion._polish`): the coefficients to its
                # precision, independent of the inner fit's path
                grad = X.T @ (ev['result'].gradient / 2) + ev['S'] @ ev['beta']
                ev['beta'] = ev['beta'] - H.solve(grad)
        ev['H'] = H
        ev['H_int'] = H.restrict(self.penalized) if self.method == 'ML' else H
        return ev

    def _cox_value(self, ev):
        # V with phi = 1: the partial likelihood's deviance over 2 less its
        # saturated log-likelihood is -l_p(b), then the penalty, log|H| / 2,
        # -log|S_lambda|_+ / 2 and (REML) -M log(2 pi) / 2, as `_value`
        beta = ev['beta']
        value = (ev['result'].deviance / 2 - ev['result'].loglik_sat + beta @ ev['S'] @ beta / 2 +
                 ev['H_int'].logdet() / 2 - ev['logdet_V'] - ev['logdet_S'] / 2)
        if self.method == 'REML':
            value -= self.null_dim * np.log(2 * np.pi) / 2
        return float(value)

    def criterion(self, params, beta):
        return self._cox_value(self._cox_evaluation(np.asarray(params, float), beta))

    def coef(self, params):
        self._set_lams(params)
        gam = self.gam
        control = self._inner_control()
        try:
            state = gam._local_scoring(state=deepcopy(self._warm), control=control, warn=False)
        except ValueError:
            gam.converged_ = False
            return np.zeros(self.X.shape[1])
        converged = gam.converged_
        if converged:
            self._warm = state
        gam.converged_ = converged
        return self.T.T @ self.design.gamma(state)

    def _inner_control(self):
        # Newton's steps (the full Hessian) converge quadratically on the
        # objective; the linear predictor can move by rounding along nearly
        # flat directions (a smoothing parameter near 0), so not on it. To
        # 1e-10, not tighter: the polish (`_cox_evaluation`) gives the
        # coefficients the criterion's precision, and with a large
        # smoothing parameter the objective's rounding stops step halving
        control = copy(self.gam.control)
        control.epsnr = 1e-10
        control.convergence = 'objective'
        control.mxitnr = max(control.mxitnr, 100)
        return control

    def value_and_grad(self, params):
        """
        As `LaplaceCriterion.value_and_grad`, with the information of the
        partial likelihood: the derivative of `log|H|` along the implicit
        derivative of the coefficients is a central difference of the
        information along `X d beta` (it needs the partial likelihood's
        third derivatives).
        """
        self.n_evaluations += 1
        params = np.asarray(params, float)
        beta = self.coef(params)
        if not self.gam.converged_:
            return self._unconverged(params)
        try:
            ev = self._cox_evaluation(params, beta)
            value = self._cox_value(ev)
        except (ValueError, np.linalg.LinAlgError):
            return self._unconverged(params)
        if not np.isfinite(value):
            return self._unconverged(params)
        X, beta, H, H_int, eta = (ev[k] for k in ('X', 'beta', 'H', 'H_int', 'eta'))
        K = self.n_rho
        dS_beta = np.zeros((K, beta.shape[0]))
        g = np.zeros(K)
        for i, k in enumerate(self.est_index):
            piece = ev['pieces'][k]
            dS_beta[i] = piece.apply(beta)
            g[i] = beta @ dS_beta[i] / 2 + piece.trace(H_int) / 2 - ev['dlogdet_S'][k] / 2
        # the derivative in beta of the penalized deviance over 2 (0 at the
        # minimum, up to the inner fit's convergence)
        g_beta = X.T @ (ev['result'].gradient / 2) + ev['S'] @ beta
        dbeta = -H.solve(dS_beta.T)                                  # (P, K)
        # log|H| through b: H = X' I(eta) X + S_lambda, with I the partial
        # likelihood's information (not diagonal), so along db/drho_k,
        #   d log|H| = tr(H^{-1} X' [dI(eta) along X db/drho_k] X),
        # the directional derivative of I by a central difference in eta (it
        # stands in for the Gaussian family's X' (a d''' . lev) / 4)
        idx = H_int.coords
        for i in range(K):
            v = X @ dbeta[:, i]
            eps = 1e-4 / max(np.max(np.abs(v)), 1e-12)
            dI = (self._XIX(X, eta + eps * v) - self._XIX(X, eta - eps * v)) / (2 * eps)
            g[i] += g_beta @ dbeta[:, i] + H_int.trace(idx, dI[np.ix_(idx, idx)]) / 2
        self._max_value = max(getattr(self, '_max_value', -np.inf), value)
        return value, g

    def start(self):
        """As `LaplaceCriterion.start`: the diagonal of the information at the offset."""
        h = self._coxdev(self.offset, self.a).diag_hessian / 2
        xwx = np.einsum('ij,i,ij->j', self.X, h, self.X)
        start = np.zeros(len(self.lam_fixed))
        for _, model, sl in self.models:
            start[sl] = model.start(xwx)
        return start[self.est_index]


def _newton(crit, x, bounds, maxit=100, tol=1e-8):
    """
    Newton's method from `x` (the quasi-Newton optimum), its Hessian by
    central differences of the gradient (step `crit.hess_eps`), with step
    halving: the optimum to the precision of the gradient. Near the
    optimum a step's decrease of the criterion can be below its precision
    (`crit.value_tol`, relative): a step that does not increase it beyond
    that, and halves the gradient, is taken. Parameters on their bounds (a
    smoothing parameter at 0 or infinity in effect) are held there. Towards
    an optimum at infinity (a singular covariance of correlated random
    effects: a log-Cholesky diagonal going to minus infinity) each step
    moves the parameter by about 1, hence up to `maxit` steps.
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


def estimated_terms(gam):
    """
    The indices of the terms of the set-up `gam` whose penalty parameters a
    method is to choose: the penalized terms with neither `lam` nor `df`
    (nor, for correlated random effects, a given `precision`). Terms given
    `df` have their smoothing parameter matched at the prior weights, then
    fixed (as mgcv's `sp`).
    """
    normed = gam._normed_weight
    estimated = []
    for j, term in enumerate(gam.terms_):
        if term.penalty_matrix() is None:
            continue
        if getattr(term, 'penalty_varies', False):
            if term.precision is None:
                estimated.append(j)
            continue
        if getattr(term, 'lam', None) is None and getattr(term, 'df', None) is None:
            estimated.append(j)
        elif getattr(term, 'df', None) is not None:
            term.set_weights(normed)
            term.lam, term.df = float(term.penalty_lam()), None
    return estimated


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
    estimated = estimated_terms(gam)
    if not estimated:
        raise ValueError(f'method="{gam._method.name}": no smoothing parameter to estimate '
                         '(every penalized term has lam or df)')
    # placeholders, set at each evaluation
    for j in estimated:
        if getattr(terms[j], 'penalty_varies', False):
            terms[j].set_precision(np.identity(terms[j].q_))
        else:
            terms[j].set_penalty_lams(np.ones(len(terms[j].penalty_matrices())))
    for term in terms:
        term.set_weights(normed)

    if getattr(gam, '_cox', False):
        context, Criterion = nullcontext(), CoxLaplaceCriterion
    else:
        jax, _ = _jax()
        context, Criterion = jax.enable_x64(True), LaplaceCriterion
    # the search for argmin V over (rho, log phi, log theta): from mgcv's
    # starting values, L-BFGS-B within +-25 of them in each rho (lambda from
    # 0 to infinity in effect), then Newton steps (`_newton`, the Hessian of
    # V by differences of its gradient) to the gradient's precision
    with context:
        crit = Criterion(gam, estimated, method=gam._method.name, dispersion=dispersion)
        x0 = crit.start()
        K = crit.n_rho
        bounds = [(r - 25, r + 25) for r in x0[:K]] + [(None, None)] * (len(x0) - K)
        if getattr(crit, 'estimate_theta', False):
            # (theta to infinity: the Poisson; held at the bound)
            bounds[-1] = (x0[-1] - 15, x0[-1] + 15)
        off = crit.off_diagonal() if hasattr(crit, 'off_diagonal') else np.zeros(0, int)
        if len(off):
            # correlated random effects: first with their covariances
            # diagonal (the off-diagonal entries of their Cholesky factors
            # held at 0), as independent random effects, then all. From the
            # start, the off-diagonals' gradient can be large, and a first
            # step to the edge of the box can lead the search into the basin
            # of a singular covariance, a local minimum
            diagonal = list(bounds)
            for i in off:
                diagonal[i] = (0., 0.)
            x0 = x0.copy()
            x0[off] = 0.
            x0 = minimize(crit.value_and_grad, x0, jac=True, method='L-BFGS-B', bounds=diagonal,
                          options=dict(maxiter=500, ftol=1e-12, gtol=1e-6)).x
        result = minimize(crit.value_and_grad, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=500, ftol=1e-14, gtol=1e-8))
        x, value, grad, converged = _newton(crit, result.x, bounds)
        lam, phi = crit._split(x)
        lam, phi = np.asarray(lam, float)[crit.est_index], float(phi)
        theta = (crit._theta(x)[crit.est_index] if hasattr(crit, '_theta')
                 else np.log(lam))
        # (a precision's parameters are not smoothing parameters)
        linear = np.array([not getattr(terms[crit.penalty_terms[k]], 'penalty_varies', False)
                           for k in crit.est_index])
        lam = np.where(linear, lam, np.nan)
        # the terms at the optimum
        crit.coef(x)
        converged = converged and gam.converged_
    return SmoothingSelection(method=gam._method.name,
                              terms=[crit.penalty_names[k] for k in crit.est_index],
                              lam=lam / crit.weight_total,
                              lam_raw=lam,
                              theta=theta,
                              dispersion=phi,
                              score=value,
                              gradient=grad,
                              converged=converged,
                              n_evaluations=crit.n_evaluations,
                              message=str(result.message),
                              negative_binomial_theta=(float(np.exp(x[-1]))
                                                       if getattr(crit, 'estimate_theta', False)
                                                       else None))
