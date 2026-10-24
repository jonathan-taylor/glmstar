"""
Families with a parameter that `GLMM(method='REML')` or `'ML'` estimates
with the smoothing parameters, beyond statsmodels':

- `QuasiPoisson` and `QuasiBinomial` (R's `quasipoisson`, `quasibinomial`):
  the Poisson's and binomial's mean and variance functions, with a free
  dispersion `phi` (variance `phi mu`, `phi mu (1 - mu)`). Under REML and ML
  the dispersion is estimated with the smoothing parameters, as mgcv's: the
  quasi-likelihood's "saturated log-likelihood" is that of a Gaussian,
  `-(n/2) log phi + sum(log w)/2`. With fixed smoothing it is Pearson's
  estimate, as R's `glm`. They have no likelihood: no log-likelihood or AIC.
- `NegativeBinomial(theta=None)` (mgcv's `nb()`; `theta` given: its
  `negbin(theta)`, or statsmodels' `NegativeBinomial(alpha=1/theta)`): the
  variance `mu + mu^2 / theta`; with `theta` None it is estimated by REML or
  ML with the smoothing parameters, its log a parameter of the criterion.
- `JaxFamily`: a family of one's own, its inverse link and unit deviance
  written in JAX, the rest (the variance function, the derivatives for
  local scoring, REML and ML) by `jax.grad`. Any family with a method
  `jax_pieces()` (`glmnet.gam.reml._jax_family`) has REML and ML.
"""
import warnings

import numpy as np

from statsmodels.genmod.families import family as sm_family, links as sm_links, varfuncs
from statsmodels.tools.sm_exceptions import ValueWarning


class QuasiPoisson(sm_family.Poisson):
    """The Poisson's mean and variance, with a free dispersion (R's `quasipoisson`)."""
    quasi = True


class QuasiBinomial(sm_family.Binomial):
    """The binomial's mean and variance, with a free dispersion (R's `quasibinomial`)."""
    quasi = True


class NegativeBinomial(sm_family.NegativeBinomial):
    """
    The negative binomial, variance `mu + mu^2 / theta` (statsmodels'
    `alpha = 1 / theta`), log link by default. With `theta` None it is
    estimated by `GLMM(method='REML')` or `'ML'` (as mgcv's `nb()`; the fitted
    value is the GAM's `theta_`); otherwise fixed (mgcv's `negbin(theta)`).
    """

    def __init__(self, theta=None, link=None, check_link=True):
        self.estimate_theta = theta is None
        with warnings.catch_warnings():
            # (statsmodels warns of its default alpha)
            warnings.simplefilter('ignore', ValueWarning)
            super().__init__(link=link, alpha=1. if theta is None else 1 / float(theta),
                             check_link=check_link)

    @property
    def theta(self):
        return 1 / self.alpha

    def set_theta(self, theta):
        """Set `theta` (the variance function's too)."""
        self.alpha = 1 / float(theta)
        self.variance = varfuncs.NegativeBinomial(alpha=self.alpha)

    def __repr__(self):
        theta = 'estimated' if self.estimate_theta else f'{self.theta:g}'
        return f'NegativeBinomial(theta={theta})'


def _jax_x64():
    # (JAX in double precision, as REML's criterion; imported when used)
    from .reml import _jax
    jax, jnp = _jax()
    return jax, jnp, jax.enable_x64(True)


class _JaxLink(sm_links.Link):
    """
    The link of a `JaxFamily`: its inverse given in JAX, the derivatives of
    the inverse by `jax.grad`, the link the inverse's (by bisection, but
    for `link` given).
    """

    def __init__(self, linkinv, link=None, name=None):
        self.linkinv, self.link, self.name = linkinv, link, name

    def __repr__(self):
        return f'link({self.name or getattr(self.linkinv, "__name__", "linkinv")})'

    def __getstate__(self):
        # (the jitted functions are made again when used)
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_jit')}

    def _jitted(self):
        if not hasattr(self, '_jit'):
            jax, jnp, x64 = _jax_x64()
            g = jax.grad(self.linkinv)
            g2 = jax.grad(g)
            self._jit = tuple(jax.jit(jax.vmap(f)) for f in (self.linkinv, g, g2))
        return self._jit

    def _eval(self, k, z):
        jax, jnp, x64 = _jax_x64()
        z = np.asarray(z, float)
        with x64:
            return np.asarray(self._jitted()[k](z.reshape(-1)), float).reshape(z.shape)

    def inverse(self, z):
        return self._eval(0, z)

    def inverse_deriv(self, z):
        return self._eval(1, z)

    def inverse_deriv2(self, z):
        return self._eval(2, z)

    def __call__(self, p):
        p = np.asarray(p, float)
        if self.link is not None:
            jax, jnp, x64 = _jax_x64()
            with x64:
                return np.asarray(self.link(jnp.asarray(p)), float)
        # the inverse of the (monotone) inverse link by bisection: a bracket
        # about 0, doubled until it holds p, then halved
        target = p.reshape(-1)
        lo, hi = -np.ones_like(target), np.ones_like(target)
        increasing = self.inverse(np.array([1.]))[0] >= self.inverse(np.array([-1.]))[0]
        sign = 1. if increasing else -1.
        for _ in range(64):
            below = sign * (self.inverse(lo) - target) > 0
            above = sign * (self.inverse(hi) - target) < 0
            if not (below.any() or above.any()):
                break
            lo = np.where(below, 2 * lo, lo)
            hi = np.where(above, 2 * hi, hi)
        for _ in range(200):
            mid = (lo + hi) / 2
            low = sign * (self.inverse(mid) - target) < 0
            lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
            if np.all(hi - lo <= 1e-15 * np.maximum(1, np.abs(mid))):
                break
        return ((lo + hi) / 2).reshape(p.shape)

    def deriv(self, p):
        return 1 / self.inverse_deriv(self(p))

    def deriv2(self, p):
        eta = self(p)
        return -self.inverse_deriv2(eta) / self.inverse_deriv(eta)**3


class _JaxVariance(object):
    """
    The variance function of a `JaxFamily`, of its unit deviance: `V(mu) =
    2 / d''(mu, mu)`, `d''` the second derivative of `d(y, mu)` in `mu`, as
    `d(y, mu) = 2 int_mu^y (y - u) / V(u) du` for a (quasi) exponential
    family.
    """

    def __init__(self, dev):
        self.dev = dev

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_jit')}

    def _jitted(self):
        if not hasattr(self, '_jit'):
            jax, jnp, x64 = _jax_x64()
            d2 = jax.grad(jax.grad(self.dev, argnums=1), argnums=1)
            V = lambda mu: 2 / d2(mu, mu)
            self._jit = tuple(jax.jit(jax.vmap(f)) for f in (V, jax.grad(V)))
        return self._jit

    def _eval(self, k, mu):
        jax, jnp, x64 = _jax_x64()
        mu = np.asarray(mu, float)
        with x64:
            return np.asarray(self._jitted()[k](mu.reshape(-1)), float).reshape(mu.shape)

    def __call__(self, mu):
        return self._eval(0, mu)

    def deriv(self, mu):
        return self._eval(1, mu)


class JaxFamily(sm_family.Family):
    """
    A family of one's own, written in JAX: its inverse link and unit
    deviance (and, for the likelihood and a free dispersion, its saturated
    log-likelihood). The rest follows from them by `jax.grad`: the link's
    derivative, the variance function (`V(mu) = 2 / d''(mu, mu)`, `d''` the
    unit deviance's second derivative in the mean) for the fit's local
    scoring (`GAM`), and the derivatives of `d(y, linkinv(eta))` in `eta`
    for REML and ML (`GLMM`).

    Parameters
    ----------
    linkinv : function
        `mu = linkinv(eta)`, of a scalar, monotone (e.g.
        `jax.scipy.stats.norm.cdf`, the probit's).
    unit_deviance : function
        `d(y, mu)`, of scalars: `sum_i a_i d(y_i, mu_i)` is the deviance,
        `a` the prior weights. In `jax.numpy` (traceable by JAX).
    saturated_loglike : function, optional
        `ls(y, a, phi)`, of the arrays of the responses and prior weights
        and the dispersion: the saturated log-likelihood, summed, the
        log-likelihood being `ls - sum_i a_i d(y_i, mu_i) / (2 phi)`.
        Needed if the dispersion is free; with it fixed, None is 0 (a
        constant of the criterion, as the Bernoulli's is 0).
    fixed_dispersion : bool
        Whether the dispersion is fixed at 1 (True: a binomial or Poisson)
        or estimated (False).
    link : function, optional
        `eta = link(mu)`, in JAX. By default the inverse of `linkinv`, by
        bisection (it is used only to start the fit).
    name : str, optional
        The family's name, for display.

    Examples
    --------
    The probit by hand, the same as `sm.families.Binomial(sm.families.links.Probit())`:

    >>> from jax.scipy.special import xlogy
    >>> from jax.scipy.stats import norm
    >>> probit = JaxFamily(norm.cdf,
    ...                    lambda y, mu: 2 * (xlogy(y, y) - xlogy(y, mu) + xlogy(1 - y, 1 - y)
    ...                                       - xlogy(1 - y, 1 - mu)),
    ...                    name='probit')
    """

    def __init__(self, linkinv, unit_deviance, saturated_loglike=None, fixed_dispersion=True,
                 link=None, name=None):
        if not callable(linkinv) or not callable(unit_deviance):
            raise TypeError('JaxFamily needs the functions linkinv(eta) and unit_deviance(y, mu)')
        if saturated_loglike is not None and not callable(saturated_loglike):
            raise TypeError('saturated_loglike should be a function ls(y, a, phi), or None')
        if saturated_loglike is None and not fixed_dispersion:
            raise ValueError('a free dispersion (fixed_dispersion=False) needs '
                             'saturated_loglike(y, a, phi): REML and ML differentiate it in phi')
        self.unit_deviance = unit_deviance
        self.saturated_loglike = saturated_loglike
        self.fixed_dispersion = bool(fixed_dispersion)
        self.name = name
        super().__init__(_JaxLink(linkinv, link=link, name=name), _JaxVariance(unit_deviance),
                         check_link=False)

    def __repr__(self):
        return f'JaxFamily({self.name})' if self.name else 'JaxFamily()'

    def jax_pieces(self):
        """
        `(linkinv, dev, ls, fixed)` for REML and ML (`glmnet.gam.reml._jax_family`):
        the functions of `t` (the negative binomial's log theta), which they ignore.
        """
        d, ls = self.unit_deviance, self.saturated_loglike
        dev = lambda y, mu, t: d(y, mu)
        if ls is None:
            sat = lambda y, a, phi, t: 0. * phi
        else:
            sat = lambda y, a, phi, t: ls(y, a, phi)
        return self.link.linkinv, dev, sat, self.fixed_dispersion

    def _valid_mean(self, mu):
        # (any mean of a finite positive variance: not only positive ones)
        return True

    def _jitted_deviance(self):
        if not hasattr(self, '_jit_dev'):
            jax, jnp, x64 = _jax_x64()
            self._jit_dev = jax.jit(jax.vmap(self.unit_deviance))
        return self._jit_dev

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_jit')}

    def _resid_dev(self, endog, mu):
        jax, jnp, x64 = _jax_x64()
        y, mu = np.broadcast_arrays(np.asarray(endog, float), np.asarray(mu, float))
        with x64:
            return np.asarray(self._jitted_deviance()(y.reshape(-1), mu.reshape(-1)),
                              float).reshape(y.shape)

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        """
        The log-likelihood of each observation, `ls_i - a_i d_i / (2 phi)`: NaN
        without `saturated_loglike` (as a quasi-likelihood's: no AIC).
        """
        jax, jnp, x64 = _jax_x64()
        y, mu, a = np.broadcast_arrays(np.asarray(endog, float), np.asarray(mu, float),
                                       np.asarray(var_weights, float))
        if self.saturated_loglike is None:
            return np.full(y.shape, np.nan)
        ls = self.saturated_loglike
        with x64:
            sat = np.asarray(jax.vmap(lambda yi, ai: ls(yi[None], ai[None], scale))(
                y.reshape(-1), a.reshape(-1)), float).reshape(y.shape)
        return sat - a * self._resid_dev(y, mu) / (2 * scale)

    def starting_mu(self, y):
        return (np.asarray(y, float) + np.mean(y)) / 2
