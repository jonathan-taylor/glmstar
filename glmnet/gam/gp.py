"""
Gaussian process smooths: the low rank Gaussian process (kriging) smoother
of Kammann & Wand (2003), as mgcv's `s(..., bs="gp")` (Wood, 2017, section
5.8.2), in one or more columns (a spatial surface in two).

The function is `f(x) = sum_i beta_i C(x, x_i) + alpha' t(x)`: a combination
of the correlation function `C` centered at the knots `x_i` (by default the
distinct values, or rows, of the columns), plus a linear trend `t(x) = (1,
x)` (or only the constant), penalized by `beta' C beta`, the squared norm of
the first part in the reproducing kernel Hilbert space of `C`. As a Bayesian
model, the first part is a Gaussian process with covariance proportional to
`C`, the trend a fixed effect. The rank is reduced (Wood, 2003) by keeping
the eigenvectors of `C` at the knots of its `k - M` largest eigenvalues (`M`
the dimension of the trend): with `C(K, K) = U L U'`, the basis is
`[C(x, K) U L^{-1/2}, t(x)]` and the penalty the identity on the first
block. (mgcv's basis is `[C(x, K) U, t(x)]` with penalty `L`: the same
functions and penalty.)

References
----------
- Kammann, E. E. and Wand, M. P. (2003). Geoadditive models. *Journal of
  the Royal Statistical Society, Series C* 52(1), 1–18.
- Wood, S. N. (2003). Thin plate regression splines. *Journal of the Royal
  Statistical Society, Series B* 65(1), 95–114.
- Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
"""
from dataclasses import dataclass
from typing import Optional, Union
import warnings

import numpy as np
from scipy.linalg import eigh, solve_triangular
from scipy.optimize import brentq
from scipy.spatial.distance import cdist

from .terms import _deprecated_names, Smoother, _name

# mgcv's codes (abs(m[1])) of the correlation functions
_KERNELS = {1: 'spherical', 2: 'power exponential', 3: 'Matern(1.5)', 4: 'Matern(2.5)',
            5: 'Matern(3.5)'}
_MATERN = {1.5: 3, 2.5: 4, 3.5: 5}
# mgcv's default basis dimension beyond the trend, by dimension (1, 2, more)
_DEFAULT_K = (10, 30, 100)


def gp_correlation(d, kernel_code, power=1.):
    """
    The correlation functions of mgcv's `bs="gp"` at the scaled distances
    `d = distance / range` (mgcv's `gpE`): 1, spherical, `(1 - 1.5 d + 0.5
    d^3) 1{d <= 1}`; 2, power exponential, `exp(-d^power)`; 3, 4, 5, Matern
    with smoothness 1.5, 2.5 and 3.5, `exp(-d) (1 + d)`, `exp(-d) (1 + d +
    d^2 / 3)` and `exp(-d) (1 + d + 2 d^2 / 5 + d^3 / 15)`. (In the usual
    parametrization of the Matern, as scikit-learn's, with length scale
    `ell`, the range is `ell / sqrt(2 nu)`.)
    """
    d = np.asarray(d, float)
    if kernel_code == 1:
        return (1 - 1.5 * d + 0.5 * d**3) * (d <= 1)
    if kernel_code == 2:
        return np.exp(-d**power)
    e = np.exp(-d)
    if kernel_code == 3:
        return e * (1 + d)
    if kernel_code == 4:
        return e * (1 + d + d**2 / 3)
    if kernel_code == 5:
        return e * (1 + d + 0.4 * d**2 + d**3 / 15)
    raise ValueError(f'unknown kernel code {kernel_code}')


def _distances(x, y):
    return cdist(x, y)


@dataclass
class GaussianProcessSmoother(Smoother):
    """
    A Gaussian process smooth of one or more columns (`column`, a name or a
    list of them): the term `gaussian_process(...)`, mgcv's `s(..., bs="gp")`; see
    `glmnet.gam.gp`.

    Parameters
    ----------
    kernel: str
        `'matern'` (the default, smoothness `nu` 1.5, 2.5 or 3.5; 0.5 is the
        exponential), `'exponential'` (the power exponential,
        `exp(-(d / range)^power)`, `0 < power <= 2`; 2 is the Gaussian, or
        squared exponential) or `'spherical'`; in mgcv's parametrization
        (`gp_correlation`).
    nu, power: float
        Of the Matern and the power exponential.
    range: float
        The range `rho`: correlations are functions of `distance / rho`
        (Euclidean distances between rows of the columns, which should be
        on comparable scales). By default the largest distance between the
        knots (Kammann & Wand's, as mgcv's).
    trend: bool
        With the linear trend `(1, x)` (the default, as mgcv's `m[1] > 0`),
        unpenalized; False, only the constant (mgcv's `m[1] < 0`): a
        stationary process, the GAM's intercept the mean.
    k: int
        The basis dimension, with the trend: mgcv's `k` (by default `M +
        10`, `M + 30`, `M + 100` in 1, 2 and more dimensions, `M` the
        trend's dimension). At most `M` plus the number of knots, which is
        the full rank (kriging, with a knot at every distinct value). The
        eigenvalues of a smooth kernel at a long range fall off fast: those
        below 1e-12 of the largest (rounding) are left out, with a warning.
    knots: array
        The knots (rows of values of the columns). By default the distinct
        rows of the training data, or `max_knots` of them drawn at random
        (seed `seed`) when there are more (as mgcv, whose draw differs).
    m: tuple
        mgcv's `m`: `(code, range, power)`, `abs(code)` selecting the
        kernel (1 spherical, 2 power exponential, 3-5 Matern 1.5-3.5), its
        sign the trend, `range <= 0` the default; overrides `kernel`, `nu`,
        `power`, `range` and `trend`.
    df, lam:
        The smoothing, as `s()`: `lam` fixed (on the scale of the GAM's
        objective, `lam / 2` times the penalty), or `df` matched at each
        working problem's weights (the trace of the smoother less 1); with
        neither, estimated by `GLMM(method='REML')` or `'ML'`. (mgcv's `sp`
        is `lam * sum(weights) * S.scale`.)
    """
    kernel: str = 'matern'
    nu: float = 1.5
    power: float = 1.
    range: Optional[float] = None
    trend: bool = True
    k: Optional[int] = None
    knots: Optional[np.ndarray] = None
    m: Optional[tuple] = None
    df: Optional[float] = None
    lam: Optional[float] = None
    max_knots: int = 2000
    seed: int = 1

    _label = 'gaussian_process'
    rank_tol = 1e-12       # the eigenvalues kept, relative to the largest

    def __post_init__(self):
        if self.df is not None and self.lam is not None:
            raise ValueError('give one of df and lam')
        self.kernel_code_, self.power_, self.range_given_, self.trend_ = self._kernel_spec()

    def _kernel_spec(self):
        if self.m is not None:
            m = list(np.atleast_1d(self.m))
            code = int(round(m[0]))
            if abs(code) not in _KERNELS:
                raise ValueError(f'gp: abs(m[0]) should be in 1, ..., 5 ({_KERNELS}), got {m[0]}')
            rho = m[1] if len(m) > 1 and m[1] > 0 else None
            power = m[2] if len(m) > 2 else 1.
            kernel_code, trend = abs(code), code > 0
        else:
            kind = str(self.kernel).lower()
            power, rho, trend = self.power, self.range, self.trend
            if kind == 'matern':
                if self.nu == 0.5:
                    kernel_code, power = 2, 1.
                elif self.nu in _MATERN:
                    kernel_code = _MATERN[self.nu]
                else:
                    raise ValueError(f'gp: nu should be one of 0.5, 1.5, 2.5, 3.5, got {self.nu}')
            elif kind in ('exponential', 'power exponential', 'power_exponential'):
                kernel_code = 2
            elif kind == 'spherical':
                kernel_code = 1
            else:
                raise ValueError(f"gp: kernel should be 'matern', 'exponential' or "
                                 f"'spherical', got {self.kernel!r}")
            if rho is not None and not rho > 0:
                raise ValueError(f'gp: range should be positive, got {rho}')
        if kernel_code == 2 and not 0 < power <= 2:
            raise ValueError(f'gp: the power of the power exponential should be in (0, 2], got {power}')
        return kernel_code, float(power), rho, bool(trend)

    @property
    def has_objective(self):
        return self.lam is not None

    @property
    def kernel_name(self):
        return _KERNELS[self.kernel_code_]

    def _as_matrix(self, x):
        x = np.asarray(x, float)
        return x.reshape((x.shape[0], -1)) if x.ndim > 1 else x.reshape((-1, self.dim_))

    def setup(self, x):
        x = np.asarray(x, float)
        X = x.reshape((x.shape[0], -1))
        n, d = X.shape
        self.dim_ = d
        if d == 1:
            self.x_ = X[:, 0]          # (for the plots of a curve)
        unique = np.unique(X, axis=0)
        if self.knots is not None:
            knots = np.asarray(self.knots, float).reshape((-1, d))
        elif n > self.max_knots and unique.shape[0] > self.max_knots:
            rng = np.random.default_rng(self.seed)
            knots = unique[np.sort(rng.choice(unique.shape[0], self.max_knots, replace=False))]
        else:
            knots = unique
        M = d + 1 if self.trend_ else 1
        # (the default, mgcv's, at most the full rank)
        bs_dim = (self.k if self.k is not None else
                  min(d + 1 + _DEFAULT_K[min(d, 3) - 1], M + knots.shape[0]))
        if bs_dim < M + 1:
            raise ValueError(f'{self.name}: k should be at least {M + 1} (the trend has '
                             f'dimension {M})')
        rank = bs_dim - M
        if rank > knots.shape[0]:
            raise ValueError(f'{self.name}: k={bs_dim} is more than the trend\'s dimension '
                             f'({M}) plus the number of knots ({knots.shape[0]})')
        self.shift_ = X.mean(0)
        self.knots_ = knots
        K = knots - self.shift_
        dist = _distances(K, K)
        self.range_ = float(self.range_given_ if self.range_given_ is not None else dist.max())
        C = gp_correlation(dist / self.range_, self.kernel_code_, self.power_)
        nk = C.shape[0]
        ev, U = eigh(C, subset_by_index=[nk - rank, nk - 1])
        ev, U = ev[::-1], U[:, ::-1]
        keep = ev > ev.max() * self.rank_tol
        if not keep.all():
            # the eigenvectors beyond the numerical rank are rounding (a
            # smooth kernel at a long range): left out
            warnings.warn(f'{self.name}: the correlation matrix of the knots has numerical rank '
                          f'{keep.sum()} < k - {M} = {rank} (eigenvalues below {self.rank_tol:g} '
                          f'of the largest): the basis has dimension {keep.sum() + M}; '
                          'reduce k, or the range')
            ev, U, rank = ev[keep], U[:, keep], int(keep.sum())
        self.eigenvalues_ = ev
        self._W = U / np.sqrt(ev)
        self.rank_ = rank
        self.null_dim_ = M
        self._basis = self._basis_values(X)
        self._X = X - self.shift_
        self._coef = self._nl_coef = None
        self._fit = None
        return self

    def _trend(self, Xs):
        if self.trend_:
            return np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        return np.ones((Xs.shape[0], 1))

    # rows of the basis computed at a time: the correlations with the knots
    # are (rows, knots), the basis only (rows, k)
    block_rows = 4096

    def _basis_values(self, X):
        Xs = X - self.shift_
        K = self.knots_ - self.shift_
        B = np.empty((Xs.shape[0], self._W.shape[1]))
        for start in range(0, Xs.shape[0], self.block_rows):
            rows = slice(start, start + self.block_rows)
            C = gp_correlation(_distances(Xs[rows], K) / self.range_, self.kernel_code_,
                               self.power_)
            B[rows] = C @ self._W
        return np.hstack([B, self._trend(Xs)])

    def basis(self, x=None):
        if x is None:
            return self._basis
        return self._basis_values(self._as_matrix(x))

    def penalty_matrix(self):
        P = np.zeros((self.rank_ + self.null_dim_,) * 2)
        P[np.arange(self.rank_), np.arange(self.rank_)] = 1.
        return P

    # the penalized weighted least squares fits at the current weights: with
    # sqrt(W) B = [P, N] (penalized, trend), N = Q R, and the part of P
    # orthogonal to N, (I - Q Q') P = U s V', the fit is that of the ridge
    # regression on it, with the trend unpenalized

    def set_weights(self, w):
        super().set_weights(w)
        self._decomposition = None
        if self.df is not None:
            self.lam_ = self._lam_for_df(self.df)
        elif self.lam is not None:
            self.lam_ = self.lam
        else:
            raise ValueError(f'{self.name}: give lam or df (or estimate it with a GLMM)')

    def _decompose(self):
        if self._decomposition is None:
            sq = np.sqrt(self.w_)
            Bw = sq[:, None] * self._basis
            P, N = Bw[:, :self.rank_], Bw[:, self.rank_:]
            Q, R = np.linalg.qr(N)
            if np.abs(np.diag(R)).min() <= 1e-10 * np.abs(np.diag(R)).max():
                raise ValueError(f'{self.name}: the trend is not identified by the rows of '
                                 'positive weight')
            Pt = P - Q @ (Q.T @ P)
            U, s, Vt = np.linalg.svd(Pt, full_matrices=False)
            self._decomposition = (sq, P, Q, R, U, s, Vt)
        return self._decomposition

    def _lam_for_df(self, df):
        *_, s, _ = self._decompose()
        s2 = s[s > s.max() * 1e-10]**2
        M = self.null_dim_
        target = df + 1 - M
        if not 0 < target < len(s2):
            raise ValueError(f'{self.name}: df should be in ({M - 1}, {M - 1 + len(s2)})')
        scale = np.log(s2.max())
        t = brentq(lambda t: (s2 / (s2 + np.exp(t + scale))).sum() - target, -80, 40, xtol=1e-12)
        return float(np.exp(t + scale))

    def _solve(self, r):
        # the coefficients of the fit to r (uncentered)
        sq, P, Q, R, U, s, Vt = self._decompose()
        rw = sq * r
        delta = Vt.T @ ((s / (s**2 + self.lam_)) * (U.T @ rw))
        alpha = solve_triangular(R, Q.T @ (rw - P @ delta))
        return np.concatenate([delta, alpha])

    def smooth(self, r):
        coef = self._solve(r)
        f = self._basis @ coef
        center = f.mean()
        coef[self.rank_] -= center          # the constant's coefficient
        self._coef = coef
        return f - center

    def coef(self):
        return self._coef

    def linear_columns(self):
        return self._X if self.trend_ else None

    def smooth_nonlinear(self, r):
        coef = self._solve(r)
        f = self._basis @ coef
        # less its weighted least squares fit on the trend
        sq = np.sqrt(self.w_)
        T = self._basis[:, self.rank_:]
        a = np.linalg.lstsq(sq[:, None] * T, sq * f, rcond=None)[0]
        coef[self.rank_:] -= a
        self._nl_coef = coef
        return f - T @ a

    def combine_coef(self, beta, center):
        coef = self._nl_coef.copy()
        if self.trend_:
            coef[self.rank_ + 1:] += beta
        coef[self.rank_] -= center
        return coef

    def penalty_lam(self):
        return self.lam_

    def penalty(self, f, coef=None):
        if self.lam is None or coef is None:
            return 0.
        return self.lam * float(np.sum(np.asarray(coef)[:self.rank_]**2)) / 2

    def edf(self):
        *_, s, _ = self._decompose()
        return float(self.null_dim_ + np.sum(s**2 / (s**2 + self.lam_)) - 1)

    def finalize(self, f, r, coef=None):
        self.coef_ = np.zeros(self.rank_ + self.null_dim_) if coef is None else np.asarray(coef, float)
        return self

    def predict(self, x):
        return self.basis(x) @ self.coef_


def gaussian_process(*columns, kernel='matern', nu=1.5, range=None, power=1., trend=True, k=None,
                     knots=None, m=None, df=None, lam=None, max_knots=2000, seed=1, name=None):
    """
    A Gaussian process smooth of the columns (see `GaussianProcessSmoother`): mgcv's
    `s(*columns, bs="gp", k=k, m=m)`. With neither `df` nor `lam`, its
    smoothing parameter is estimated (`GLMM(method='REML')` or `'ML'`).

    >>> gaussian_process('age', df=5)
    >>> gaussian_process('lon', 'lat', nu=2.5, range=0.5)
    """
    if not columns:
        raise ValueError('gp needs at least one column')
    column = columns[0] if len(columns) == 1 else list(columns)
    term = GaussianProcessSmoother(column, kernel=kernel, nu=nu, power=power, range=range, trend=trend, k=k,
                  knots=knots, m=m, df=df, lam=lam, max_knots=max_knots, seed=seed)
    label = ', '.join(str(c) for c in columns)
    term.label = name or _name('gaussian_process', label,
                               kernel=None if kernel == 'matern' else repr(kernel),
                               nu=None if nu == 1.5 or kernel != 'matern' else nu,
                               power=None if power == 1. else power,
                               range=range, trend=None if trend else False, k=k,
                               m=None if m is None else tuple(m), df=df, lam=lam)
    return term


# the former name (glmstar 0.5.0), deprecated
__getattr__ = _deprecated_names({'GPTerm': 'GaussianProcessSmoother'}, globals(), __name__)
