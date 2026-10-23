"""
Terms of a generalized additive model: each is a weighted smoother of one
column of `X`, as in Hastie & Tibshirani's local scoring.

A term is set up on its column of the training data (`setup`), told the
weights of the current working (IRLS) problem (`set_weights`), and then asked
to smooth partial residuals at those weights (`smooth`), as often as
backfitting needs. Its fits are centered by the backfitter, to mean 0 over the
training rows (unweighted, as `mgcv`), the intercept taking up the rest.

A term may also keep coefficients of its last smooth (`coef()`, centered as
the fit; for a spline, those of its basis), linear in the fit. The backfitter
keeps them with the fits, so that IRLS can average them when it halves a step.
After the fit, `finalize` gives the term its final fit (and coefficients),
from which it predicts at new values (`predict`).

For modified backfitting (Buja, Hastie & Tibshirani, 1989) a term also gives
the columns of the functions its smoother reproduces, other than the
constant (`linear_columns()`: `x` for a spline or a linear term; the
indicators for a factor), which are fitted jointly with those of the other
terms, and smooths only the rest (`smooth_nonlinear`). A term that is itself a
projection on its columns (linear, factor) is not smoothed then. This needs a
symmetric smoother (in the weighted inner product), for which it has the same
fixed point as plain backfitting; loess is not, and gives no columns.

For the joint (mgcv-style) solve of the working problems, and the Bayesian
covariance of a fit, a term also gives its design operator: its basis
(`basis(x)`, the columns of the functions it fits, at the training rows or
at new values), the matrix of its quadratic penalty (`penalty_matrix()`,
None if unpenalized) and its smoothing parameter at the current weights
(`penalty_lam()`). Loess has none. A term with `centered = False` (a random
effect, whose penalty identifies it) is not centered.

`penalty(f, coef)` is the term's contribution to the objective, so that local
scoring can check that its steps decrease it. A term whose smoothing does not
minimize a fixed objective (loess, or a spline fitted to a fixed `df` whose
smoothing parameter changes with the weights) has `has_objective = False`.

References
----------
- Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman & Hall.
- Buja, A., Hastie, T. and Tibshirani, R. (1989). Linear smoothers and
  additive models. *The Annals of Statistics* 17(2), 453–510.
- Green, P. J. and Silverman, B. W. (1994). *Nonparametric Regression and
  Generalized Linear Models: A Roughness Penalty Approach*. Chapman & Hall.
- Cleveland, W. S., Grosse, E. and Shyu, W. M. (1992). Local regression
  models. In J. M. Chambers and T. J. Hastie (eds.), *Statistical Models in
  S*, chapter 8. Wadsworth & Brooks/Cole.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
"""
from dataclasses import dataclass, field
from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.optimize import brentq


def _scatter_smooth():
    # (imported when a spline or loess term is set up: a clear error without it)
    try:
        import scatter_smooth
    except ImportError as e:
        raise ImportError('the smoothing spline and loess terms of glmnet.gam use the '
                          'scatter_smooth package: pip install scatter-smooth') from e
    return scatter_smooth


@dataclass
class Term(object):
    """
    A term of a GAM, a function of the column `column` of `X` (a name of a
    DataFrame's column, or an integer index).
    """
    column: Union[str, int]

    has_objective = True
    centered = True

    @property
    def columns(self):
        """The columns of `X` the term uses (one, but for `re(..., by=...)`)."""
        return self.column

    @property
    def name(self):
        label = getattr(self, 'label', None)
        return label if label else f'{self._label}({self.column})'

    def setup(self, x):
        """Set the term up on its column `x` of the training data."""
        raise NotImplementedError

    def set_weights(self, w):
        """The weights of the next smooths, those of the working problem."""
        self.w_ = np.asarray(w, float)

    def smooth(self, r):
        """The term's weighted fit to `r`, at the training rows."""
        raise NotImplementedError

    def coef(self):
        """Coefficients of the last smooth, centered as its fit (None: the term has none)."""
        return None

    def linear_columns(self):
        """
        Columns (at the training rows, without the constant) of the functions
        the term's smoother reproduces, for modified backfitting; None if none.
        """
        return None

    @property
    def nonlinear(self):
        """Whether the term has a part beyond its `linear_columns` (to backfit)."""
        return True

    def smooth_nonlinear(self, r):
        """
        The smooth of `r` less its weighted least squares fit on the constant
        and `linear_columns()`: the smoother's part beyond them.
        """
        f = self.smooth(r)
        return f - _weighted_fit(self.linear_columns(), f, self.w_)

    def combine_coef(self, beta, center):
        """
        The coefficients of the fit `linear_columns() @ beta` plus the last
        `smooth_nonlinear`, less `center` (None: the term has none).
        """
        return None

    def penalty(self, f, coef=None):
        """The term's penalty at its fit `f` (at the training rows), with coefficients `coef`."""
        return 0.

    def basis(self, x=None):
        """
        The term's basis at the training rows (`x` None) or at `x`: the
        columns of the functions it fits (None: it has none, as loess).
        """
        return None

    def penalty_matrix(self):
        """The matrix of the term's quadratic penalty on its basis' coefficients (None: unpenalized)."""
        return None

    def penalty_lam(self):
        """The term's smoothing parameter, at the current weights (after `set_weights`)."""
        return 0.

    # several penalties: `sum_m lam_m c' S_m c / 2`. A term with one penalty
    # (the default) gives it as a list of one; a term with several gives
    # their sum (at smoothing parameters 1) as `penalty_matrix()`, its
    # structure, and has no single `penalty_lam()`

    def penalty_matrices(self):
        """The matrices `S_m` of the term's penalties (empty: unpenalized)."""
        S = self.penalty_matrix()
        return [] if S is None else [S]

    def penalty_lams(self):
        """The smoothing parameters of the penalties, at the current weights."""
        return [self.penalty_lam()] if self.penalized else []

    # cheap questions about the basis and the penalty, which do not form
    # them (a random effect's are of the size of its levels: of its basis,
    # rows by levels; of its penalty, levels squared)

    @property
    def has_basis(self):
        """Does the term have a basis (`basis()`; loess does not)?"""
        return type(self).basis is not Term.basis

    @property
    def penalized(self):
        """Is the term penalized (has it a `penalty_matrix()`)?"""
        return self.penalty_matrix() is not None

    def n_coef(self):
        """The number of columns of its basis (set up; 0 without one)."""
        return self.basis().shape[1] if self.has_basis else 0

    def n_penalties(self):
        """The number of its penalties (set up), the length of `penalty_matrices()`."""
        return len(self.penalty_matrices())

    def set_penalty_lams(self, lams):
        """Fix the smoothing parameters, one per penalty (as REML and ML do)."""
        self.lam = float(lams[0])

    def edf(self):
        """The degrees of freedom of the term, at the current weights, without the constant."""
        raise NotImplementedError

    def finalize(self, f, r, coef=None):
        """
        Fix the fitted function from its (centered) fit `f` at the training
        rows, its coefficients `coef` and the partial residuals `r` it was
        fitted to.
        """
        raise NotImplementedError

    def predict(self, x):
        """The fitted function at `x`."""
        raise NotImplementedError


def _weighted_fit(X, r, w):
    """The weighted least squares fit of `r` on the constant and the columns `X` (or None)."""
    sw = w.sum()
    rbar = (w * r).sum() / sw
    if X is None:
        return np.full_like(r, rbar)
    Xc = X - (w[:, None] * X).sum(0) / sw
    sq = np.sqrt(w)
    beta = np.linalg.lstsq(sq[:, None] * Xc, sq * (r - rbar), rcond=None)[0]
    return rbar + Xc @ beta


def _is_interaction(column):
    return isinstance(column, (list, tuple))


def _group_label(column):
    return ':'.join(str(c) for c in column) if _is_interaction(column) else column


def _interaction(x):
    """The labels of the combinations of the columns of `x`: `'a:b'`."""
    g = x[:, 0].astype(str)
    for k in range(1, x.shape[1]):
        g = np.char.add(np.char.add(g, ':'), x[:, k].astype(str))
    return g


def _level_codes(levels, x):
    """The positions of the values `x` in the sorted `levels`; -1 for values not among them."""
    x = np.asarray(x).reshape(-1)
    codes = np.minimum(np.searchsorted(levels, x), len(levels) - 1)
    return np.where(levels[codes] == x, codes, -1)


def _weighted_ls(x, r, w):
    """The weighted least squares fit of `r` on `(1, x)`: (intercept, slope)."""
    sw = w.sum()
    xbar, rbar = (w * x).sum() / sw, (w * r).sum() / sw
    xc = x - xbar
    sxx = (w * xc**2).sum()
    slope = (w * xc * (r - rbar)).sum() / sxx if sxx > 0 else 0.
    return rbar - slope * xbar, slope


@dataclass
class SlopeTerm(Term):
    """A linear function of the column: the term `linear(column)`."""

    _label = 'linear'

    def setup(self, x):
        self.x_ = np.asarray(x, float)
        self.center_ = self.x_.mean()
        return self

    def smooth(self, r):
        _, slope = _weighted_ls(self.x_, r, self.w_)
        return slope * (self.x_ - self.center_)

    def linear_columns(self):
        return (self.x_ - self.center_)[:, None]

    def basis(self, x=None):
        return (self.x_ if x is None else np.asarray(x, float))[:, None]

    nonlinear = False

    def edf(self):
        return 1.

    def finalize(self, f, r, coef=None):
        _, self.coef_ = _weighted_ls(self.x_, f, np.ones_like(f))
        return self

    def predict(self, x):
        return self.coef_ * (np.asarray(x, float) - self.center_)


@dataclass
class CategoricalTerm(Term):
    """
    A categorical column, one effect per level: the term `categorical(column)`
    (R's `factor`).
    Levels not seen in fitting predict 0. The levels are in the order of a
    pandas categorical's categories (those seen in fitting), otherwise
    sorted. In modified backfitting its
    indicators are fitted jointly with the other terms' linear columns when it
    has at most `max_joint_levels` levels; otherwise it is backfitted.
    """

    _label = 'categorical'
    max_joint_levels = 50

    categories = None        # the categories of a pandas categorical column (set by GAM)

    def setup(self, x):
        x = np.asarray(x)
        if self.categories is None:
            self.levels_, self.codes_ = np.unique(x, return_inverse=True)
        else:
            seen = set(pd.unique(x))
            self.levels_ = np.array([c for c in self.categories if c in seen], dtype=object)
            self.codes_ = pd.Index(self.levels_).get_indexer(x)
        return self

    def _codes(self, x):
        """The positions of the values `x` among the levels (-1: not seen in fitting)."""
        return pd.Index(self.levels_).get_indexer(np.asarray(x).reshape(-1))

    def linear_columns(self):
        nl = len(self.levels_)
        if not 1 < nl <= self.max_joint_levels:
            return None
        return (self.codes_[:, None] == np.arange(1, nl)[None, :]).astype(float)

    @property
    def nonlinear(self):
        return self.linear_columns() is None

    def basis(self, x=None):
        # the indicators of all the levels (unseen levels: 0)
        codes = self.codes_ if x is None else self._codes(x)
        return (codes[:, None] == np.arange(len(self.levels_))[None, :]).astype(float)

    def n_coef(self):
        return len(self.levels_)

    def smooth(self, r):
        nl = len(self.levels_)
        sw = np.bincount(self.codes_, weights=self.w_, minlength=nl)
        swr = np.bincount(self.codes_, weights=self.w_ * r, minlength=nl)
        means = np.divide(swr, sw, out=np.zeros(nl), where=sw > 0)
        return means[self.codes_]

    def edf(self):
        nl = len(self.levels_)
        return float(np.sum(np.bincount(self.codes_, weights=self.w_, minlength=nl) > 0) - 1)

    def finalize(self, f, r, coef=None):
        nl = len(self.levels_)
        counts = np.bincount(self.codes_, minlength=nl)
        sums = np.bincount(self.codes_, weights=f, minlength=nl)
        self.effects_ = pd.Series(np.divide(sums, counts, out=np.zeros(nl), where=counts > 0),
                                  index=self.levels_)
        return self

    def predict(self, x):
        return self.effects_.reindex(np.asarray(x)).fillna(0.).to_numpy()


@dataclass
class SplineTerm(Term):
    """
    A cubic smoothing spline in the column: the term `s(column, df=4)` (or
    `s(column, lam=...)`), the natural cubic spline minimizing the weighted
    residual sum of squares plus `lam` times the integral of its squared
    second derivative. Its knots are those of R's `gam::s` (and
    `smooth.spline`): all the distinct values of the column when there are
    fewer than 50, otherwise `.nknots.smspl` of them (about 200), evenly
    spaced among the sorted distinct values.

    With `lam` the smoothing parameter is fixed. With `df` (the default, 4)
    it is chosen, at every working problem's weights, so that the smoother's
    trace less 1 (for the constant) is `df`, as R's `gam::s`; the term then has
    no fixed objective. `lam` is on the scale of the GAM's objective, the
    deviance over twice the sum of the weights, plus `lam / 2` times the
    integral. (R's `mgcv`, with `bs='cr'` and the knots at the distinct
    values, fits the same spline with `sp = lam * sum(weights) *
    smooth$S.scale`, as it rescales its penalties.)
    """
    df: Optional[float] = None
    lam: Optional[float] = None

    _label = 's'

    def __post_init__(self):
        if self.df is not None and self.lam is not None:
            raise ValueError('give one of df and lam')

    @property
    def has_objective(self):
        return self.lam is not None

    def setup(self, x):
        ss = _scatter_smooth()
        self.x_ = np.asarray(x, float)
        if len(np.unique(self.x_)) < 4:
            raise ValueError(f'{self.name}: a smoothing spline needs at least 4 distinct values')
        self._smoother = ss.SplineSmoother(self.x_, lamval=self.lam or 0.)
        self.knots_ = np.asarray(self._smoother.knots)
        if self.df is not None and not 1 < self.df < len(self.knots_) - 1:
            raise ValueError(f'{self.name}: df should be in (1, {len(self.knots_) - 1})')
        self.center_ = self.x_.mean()
        # the coefficients of the function x: the spline reproduces it
        self._smoother.smooth(self.x_)
        self._x_coef = self._smoother.spline_coef_.copy()
        self._coef = self._nl_coef = None
        self._basis = self._penalty = self._at_knots = None
        return self

    def set_weights(self, w):
        super().set_weights(w)
        self._smoother.update_weights(self.w_)
        if self.df is not None:
            self.lam_ = self._smoother._find_lamval_for_df(self.df + 1)
        else:
            self.lam_ = self.lam
        self._smoother.lamval = self.lam_

    def smooth(self, r):
        f = self._smoother.smooth(r)
        center = f.mean()
        # a constant added to the coefficients adds it to the spline
        self._coef = self._smoother.spline_coef_ - center
        return f - center

    def coef(self):
        return self._coef

    def linear_columns(self):
        return (self.x_ - self.center_)[:, None]

    def smooth_nonlinear(self, r):
        S = self._smoother
        f = S.smooth(r)
        # the weighted least squares line through the fit
        self._nl_coef = S.spline_coef_ - S.coef_ * self._x_coef - S.intercept_
        return f - (S.coef_ * self.x_ + S.intercept_)

    def combine_coef(self, beta, center):
        return beta[0] * (self._x_coef - self.center_) + self._nl_coef - center

    def roughness(self, coef):
        """
        The integral of the squared second derivative of the spline with
        coefficients `coef`: from its second derivatives `g` at the knots
        (linear between them), `sum h (g_i^2 + g_i g_{i+1} + g_{i+1}^2) / 3`,
        a sum of non-negative terms (accurate for a nearly linear spline, of
        which the quadratic form loses all digits to cancellation).
        """
        _, V2, _ = self._knot_values()
        g = V2 @ np.asarray(coef, float)
        lo, hi = g[:-1], g[1:]
        return float(np.sum(np.diff(self.knots_) * (lo**2 + lo * hi + hi**2)) / 3)

    def _unit_values(self, x, deriv=0):
        # the splines whose coefficients are the unit vectors (the natural
        # cubic spline basis of the coefficients `spline_coef_`), at x
        S = self._smoother
        saved = getattr(S, 'spline_coef_', None)
        K = len(self.knots_)
        values = np.empty((x.shape[0], K))
        for k in range(K):
            e = np.zeros(K)
            e[k] = 1.
            S.set_spline_coef(e)
            values[:, k] = S.predict(x, deriv=deriv)
        if saved is not None:
            S.set_spline_coef(saved)
        return values

    def _knot_values(self):
        # the basis' values and second derivatives at the knots, and its
        # slopes at the boundary knots: they determine it everywhere. The
        # second derivatives of a natural cubic spline are those of its
        # values at the knots (Green & Silverman, 1994, section 2.1)
        if getattr(self, '_at_knots', None) is None:
            t = self.knots_
            V = self._unit_values(t)
            h = np.diff(t)
            QtV = V[:-2] / h[:-1, None] - V[1:-1] * (1 / h[:-1] + 1 / h[1:])[:, None] + V[2:] / h[1:, None]
            R = np.zeros((3, len(t) - 2))
            R[1] = (h[:-1] + h[1:]) / 3
            R[0, 1:] = R[2, :-1] = h[1:-1] / 6
            V2 = np.zeros_like(V)
            V2[1:-1] = solve_banded((1, 1), R, QtV)
            V1 = np.array([(V[1] - V[0]) / h[0] - h[0] * V2[1] / 6,
                           (V[-1] - V[-2]) / h[-1] + h[-1] * V2[-2] / 6])
            self._at_knots = (V, V2, V1)
        return self._at_knots

    def _basis_values(self, x):
        # a cubic between the knots, from its values and second derivatives
        # at the ends of the interval; linear beyond the boundary knots
        t = self.knots_
        V, V2, V1 = self._knot_values()
        i = np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 2)
        h = t[i + 1] - t[i]
        b = (x - t[i]) / h
        a = 1 - b
        c = (a**3 - a) * h**2 / 6
        d = (b**3 - b) * h**2 / 6
        values = (a[:, None] * V[i] + b[:, None] * V[i + 1] +
                  c[:, None] * V2[i] + d[:, None] * V2[i + 1])
        for end, side in [(0, x < t[0]), (-1, x > t[-1])]:
            if np.any(side):
                values[side] = V[end] + (x[side] - t[end])[:, None] * V1[end]
        return values

    def basis(self, x=None):
        if x is None:
            if getattr(self, '_basis', None) is None:
                self._basis = self._basis_values(self.x_)
            return self._basis
        return self._basis_values(np.asarray(x, float))

    def n_coef(self):
        # (without forming the basis: a column per knot)
        return len(self.knots_)

    def penalty_matrix(self):
        """
        The integral of the products of the basis' second derivatives: they
        are linear between the knots (and 0 beyond), so it is exact from
        their values at the knots.
        """
        if getattr(self, '_penalty', None) is None:
            _, V2, _ = self._knot_values()
            h = np.diff(self.knots_)[:, None] / 6
            lo, hi = V2[:-1], V2[1:]
            cross = lo.T @ (h * hi)
            self._penalty = 2 * (lo.T @ (h * lo) + hi.T @ (h * hi)) + cross + cross.T
        return self._penalty

    def penalty_lam(self):
        return self.lam_

    def penalty(self, f, coef=None):
        if self.lam is None or coef is None:     # no coefficients: the zero function
            return 0.
        return self.lam * self.roughness(coef) / 2

    def edf(self):
        return self._smoother.compute_df(self.lam_) - 1

    def finalize(self, f, r, coef=None):
        if coef is None:
            coef = np.zeros_like(self._smoother.spline_coef_)
        self._smoother.set_spline_coef(coef)
        return self

    def predict(self, x):
        return self._smoother.predict(np.asarray(x, float))


@dataclass
class LoessTerm(Term):
    """
    A loess smooth of the column (local polynomial regression, from
    `scatter_smooth`): the term `lo(column, span=0.5, degree=1)`. It minimizes
    no objective. Its degrees of freedom are the trace of the smoother less 1.
    """
    span: float = 0.5
    degree: int = 1

    has_objective = False
    _label = 'lo'

    def setup(self, x):
        ss = _scatter_smooth()
        self.x_ = np.asarray(x, float)
        # the smooth is evaluated once per distinct value
        self._x_unique, self._x_inverse = np.unique(self.x_, return_inverse=True)
        self._smoother = ss.LoessSmoother(self.x_, span=self.span, degree=self.degree)
        return self

    def _smooth_at_train(self, r):
        self._smoother.smooth(r)
        return self._smoother.predict(self._x_unique)[self._x_inverse]

    def set_weights(self, w):
        super().set_weights(w)
        self._smoother.update_weights(self.w_)

    def smooth(self, r):
        return self._smooth_at_train(r)

    # no linear_columns: loess is not a symmetric smoother, so splitting off
    # its linear part would change the fit (a single lo() term would no
    # longer give the loess smooth); it is backfitted whole

    def edf(self):
        # the trace of the smoother (its leverages, at the current weights)
        return self._smoother.compute_df() - 1

    def finalize(self, f, r, coef=None):
        # the fit is the smooth of r plus a function the smoother reproduces:
        # a constant (backfitting), or a line (modified backfitting)
        d = f - self._smooth_at_train(r)
        if self.degree < 1:
            self.line_ = (d.mean(), 0.)
        else:
            self.line_ = _weighted_ls(self.x_, d, np.ones_like(d))
        return self

    def predict(self, x):
        x = np.asarray(x, float)
        return self._smoother.predict(x) + self.line_[0] + self.line_[1] * x


@dataclass
class ColumnsTerm(Term):
    """
    A block of columns of `X` (`column` a list of them) entering linearly: a
    parametric term, as the columns ModelSpec builds for a contrast, a
    numeric column or a basis (`ns()`, `poly()`). Its fit is the weighted
    least squares fit on the (centered) columns, and in modified backfitting
    they are fitted jointly with the other terms' linear parts.
    """
    label: Optional[str] = None

    _label = 'columns'

    def setup(self, x):
        x = np.asarray(x, float)
        self.X_ = x.reshape((x.shape[0], -1))
        self.center_ = self.X_.mean(0)
        return self

    def linear_columns(self):
        return self.X_ - self.center_

    def basis(self, x=None):
        if x is None:
            return self.X_
        x = np.asarray(x, float)
        return x.reshape((x.shape[0], -1))

    nonlinear = False

    def smooth(self, r):
        return _weighted_fit(self.linear_columns(), r, self.w_) - (self.w_ * r).sum() / self.w_.sum()

    def edf(self):
        Xc = self.linear_columns()[self.w_ > 0]
        return float(np.linalg.matrix_rank(Xc))

    def finalize(self, f, r, coef=None):
        self.coef_ = np.linalg.lstsq(self.linear_columns(), f - f.mean(), rcond=None)[0]
        self.offset_ = f.mean()
        return self

    def predict(self, x):
        x = np.asarray(x, float).reshape((np.asarray(x).shape[0], -1))
        return (x - self.center_) @ self.coef_ + self.offset_


@dataclass
class PenalizedColumnsTerm(Term):
    """
    A block of columns of `X` (`column` a list of them) entering linearly,
    with quadratic penalties `sum_m lam[m] c' S_m c / 2` on their
    coefficients `c` (`penalties` the matrices `S_m`, on the scale of the
    GAM's objective), as mgcv's `paraPen`. With `lam` None the smoothing
    parameters are estimated (`method='REML'` or `'ML'`).

    The term is not centered: the penalties, or the columns (without the
    constant in their span), identify it. Its smoother is the penalized
    weighted least squares fit on the columns.
    """
    penalties: list = field(default_factory=list)
    lam: Optional[list] = None
    label: Optional[str] = None

    centered = False
    _label = 'penalized_columns'

    @property
    def has_objective(self):
        return self.lam is not None

    def setup(self, x):
        x = np.asarray(x, float)
        self.X_ = x.reshape((x.shape[0], -1))
        K = self.X_.shape[1]
        self.penalties = [np.asarray(S, float) for S in self.penalties]
        if not self.penalties or any(S.shape != (K, K) for S in self.penalties):
            raise ValueError(f'{self.name}: give one or more penalties of shape ({K}, {K})')
        if self.lam is not None and len(self.lam) != len(self.penalties):
            raise ValueError(f'{self.name}: give one lam per penalty')
        self._coef = None
        return self

    def set_weights(self, w):
        super().set_weights(w)
        if self.lam is None:
            raise ValueError(f'{self.name}: give lam (or estimate it, method="REML")')
        self.lam_ = [float(l) for l in self.lam]
        self._xwx = self.X_.T @ (self.w_[:, None] * self.X_)
        self._A = self._xwx + sum(l * S for l, S in zip(self.lam_, self.penalties))

    def smooth(self, r):
        self._coef = np.linalg.lstsq(self._A, self.X_.T @ (self.w_ * r), rcond=None)[0]
        return self.X_ @ self._coef

    # the smoother reproduces only the penalties' null space, which the
    # constant need not be in: modified backfitting smooths it whole
    def smooth_nonlinear(self, r):
        return self.smooth(r)

    def combine_coef(self, beta, center):
        return self._coef

    def coef(self):
        return self._coef

    def penalty(self, f, coef=None):
        if self.lam is None or coef is None:
            return 0.
        return sum(l * coef @ S @ coef for l, S in zip(self.lam, self.penalties)) / 2

    def edf(self):
        return float(np.trace(np.linalg.lstsq(self._A, self._xwx, rcond=None)[0]))

    def basis(self, x=None):
        if x is None:
            return self.X_
        x = np.asarray(x, float)
        return x.reshape((x.shape[0], -1))

    def penalty_matrices(self):
        return list(self.penalties)

    def penalty_matrix(self):
        return sum(self.penalties)

    def penalty_lams(self):
        return list(self.lam_)

    def penalty_lam(self):
        if len(self.penalties) > 1:
            raise ValueError(f'{self.name} has several penalties: use penalty_lams()')
        return self.lam_[0]

    def set_penalty_lams(self, lams):
        self.lam = [float(l) for l in lams]

    def finalize(self, f, r, coef=None):
        self.coef_ = np.zeros(self.X_.shape[1]) if coef is None else np.asarray(coef, float)
        return self

    def predict(self, x):
        return self.basis(x) @ self.coef_


@dataclass
class RandomEffectTerm(Term):
    """
    A random effect of the groups of the column `column`: the term
    `re(column)`, one effect per group (a random intercept), or with
    `by=`, one coefficient of the column `by` per group (a random slope),
    `f(g, v) = b_g v`. With `column` a list of columns, the groups are
    their combinations (their interaction, labelled `'a:b'`): nested
    effects, as lme4's `(1 | batch/cask)`, are `re('batch') +
    re(['batch', 'cask'])`. The effects are penalized by `lam / 2` times their sum
    of squares (on the scale of the GAM's objective, the deviance over twice
    the sum of the weights), the Gaussian prior of a mixed model; or, with
    `df`, `lam` is chosen at each working problem's weights so that the trace
    of the term's smoother (the effects' shrinkage factors' sum) is `df`.

    The penalty identifies the effects, so the term is not centered (as
    mgcv's `bs="re"` and lme4). Groups not seen in fitting predict 0.
    """
    by: Optional[Union[str, int]] = None
    lam: Optional[float] = None
    df: Optional[float] = None

    centered = False
    _label = 're'
    level_names = None       # the names of the groups, when they are codes (ModelSpec)

    def __post_init__(self):
        if self.df is not None and self.lam is not None:
            raise ValueError('give one of df and lam')

    @property
    def name(self):
        label = getattr(self, 'label', None)
        if label:
            return label
        group = _group_label(self.column)
        return f're({group}, by={self.by})' if self.by is not None else f're({group})'

    @property
    def columns(self):
        group = list(self.column) if _is_interaction(self.column) else [self.column]
        if self.by is None:
            return group if _is_interaction(self.column) else self.column
        return group + [self.by]

    @property
    def has_objective(self):
        return self.lam is not None

    def _split(self, x):
        x = np.asarray(x)
        if not _is_interaction(self.column) and self.by is None:
            return x.reshape(-1), None
        x = x.reshape((x.shape[0], -1))
        k = len(self.column) if _is_interaction(self.column) else 1
        g = _interaction(x[:, :k]) if _is_interaction(self.column) else x[:, 0]
        return g, (None if self.by is None else x[:, k].astype(float))

    def setup(self, x):
        g, v = self._split(x)
        self.levels_, self.codes_ = np.unique(g, return_inverse=True)
        self.v_ = np.ones(g.shape[0]) if v is None else v
        self._coef = None
        return self

    def set_weights(self, w):
        super().set_weights(w)
        self._s = np.bincount(self.codes_, weights=self.w_ * self.v_**2, minlength=len(self.levels_))
        if self.df is not None:
            self.lam_ = self._lam_for_df(self.df)
        elif self.lam is not None:
            self.lam_ = self.lam
        else:
            raise ValueError(f'{self.name}: give lam or df (or estimate it, method="REML")')

    def _lam_for_df(self, df):
        s = self._s[self._s > 0]
        if not 0 < df < len(s):
            raise ValueError(f'{self.name}: df should be in (0, {len(s)})')
        return np.exp(brentq(lambda t: (s / (s + np.exp(t))).sum() - df, -50, 50, xtol=1e-12))

    def smooth(self, r):
        num = np.bincount(self.codes_, weights=self.w_ * self.v_ * r, minlength=len(self.levels_))
        self._coef = num / (self._s + self.lam_)
        return self._coef[self.codes_] * self.v_

    # the smoother reproduces no function (its eigenvalues are below 1):
    # modified backfitting smooths it whole
    def smooth_nonlinear(self, r):
        return self.smooth(r)

    def combine_coef(self, beta, center):
        return self._coef

    def coef(self):
        return self._coef

    def penalty(self, f, coef=None):
        if self.lam is None or coef is None:
            return 0.
        return self.lam * np.sum(coef**2) / 2

    def edf(self):
        return float(np.sum(self._s / (self._s + self.lam_)))

    def basis(self, x=None):
        if x is None:
            codes, v = self.codes_, self.v_
        else:
            g, v = self._split(x)
            codes = _level_codes(self.levels_, g)
            v = np.ones(g.shape[0]) if v is None else v
        return (codes[:, None] == np.arange(len(self.levels_))[None, :]) * v[:, None]

    def penalty_matrix(self):
        return np.identity(len(self.levels_))

    penalized = True

    def n_coef(self):
        return len(self.levels_)

    def n_penalties(self):
        return 1

    def penalty_lam(self):
        return self.lam_

    def finalize(self, f, r, coef=None):
        index = (self.levels_ if self.level_names is None else
                 np.asarray(self.level_names)[self.levels_.astype(int)])
        self.effects_ = pd.Series(np.zeros(len(self.levels_)) if coef is None else np.asarray(coef),
                                  index=index, name=self.name)
        return self

    def predict(self, x):
        g, v = self._split(x)
        codes = _level_codes(self.levels_, g)
        effects = np.where(codes >= 0, self.effects_.to_numpy()[np.maximum(codes, 0)], 0.)
        return effects * (1. if v is None else v)


def _name(label, column, **kwargs):
    # the term's name shows the arguments given (that differ from the defaults)
    args = ', '.join(f'{k}={v}' for k, v in kwargs.items() if v is not None)
    return f'{label}({column}{", " + args if args else ""})'


@dataclass
class CorrelatedRandomEffectTerm(Term):
    """
    Correlated random effects of the groups of `column` (a column, or a
    list of them: their combinations): an intercept (`intercept=True`) and
    a slope in each of the columns `slopes` per group, `f(g, v) = b_g0 +
    sum_m b_gm v_m`, the vectors `b_g` of the groups independent
    `N(0, Sigma)`. `covariance` is `Sigma`'s structure:

    - `'unstructured'` (the default), lme4's `(1 + v | g)` and nlme's
      `pdSymm`, by the log-Cholesky factor of the relative covariance
      (`glmnet.gam.penalties.CholeskyPenalty`);
    - `'compound_symmetry'`, nlme's `pdCompSymm`: a common variance and
      correlation (`CompoundSymmetryPenalty`);
    - `'diagonal'`, nlme's `pdDiag`: independent effects with their own
      variances (`DiagonalPenalty`).

    The penalty is `sum_g b_g' B b_g / 2`, `B` the precision on the scale
    of the GAM's objective (`precision`, a `q x q` matrix, `q` the number of
    effects per group; `Sigma = dispersion (sum(weights) B)^{-1}`). With
    `precision` None it is estimated by `method='REML'` or `'ML'` (a given
    precision is used as it is, whatever the structure).

    The penalty identifies the effects: the term is not centered. Groups
    not seen in fitting predict 0.
    """
    slopes: list = field(default_factory=list)
    intercept: bool = True
    precision: Optional[np.ndarray] = None
    covariance: str = 'unstructured'

    label: Optional[str] = None

    centered = False
    penalty_varies = True        # its penalty matrix changes with B (REML, ML)
    _label = 'correlated_random_effect'

    @property
    def has_objective(self):
        return self.precision is not None

    @property
    def columns(self):
        group = list(self.column) if _is_interaction(self.column) else [self.column]
        return group + list(self.slopes)

    @property
    def effect_names(self):
        return (['(Intercept)'] if self.intercept else []) + [str(v) for v in self.slopes]

    def _split(self, x):
        x = np.asarray(x)
        x = x.reshape((x.shape[0], -1))
        k = len(self.column) if _is_interaction(self.column) else 1
        g = _interaction(x[:, :k]) if _is_interaction(self.column) else x[:, 0]
        V = x[:, k:].astype(float)
        if self.intercept:
            V = np.column_stack([np.ones(x.shape[0]), V])
        return g, V

    def setup(self, x):
        g, self.V_ = self._split(x)
        self.q_ = self.V_.shape[1]
        if self.q_ == 0:
            raise ValueError(f'{self.name}: give an intercept or slopes')
        self.levels_, self.codes_ = np.unique(g, return_inverse=True)
        if self.precision is not None:
            self.precision = np.asarray(self.precision, float).reshape((self.q_, self.q_))
        self._coef = None
        return self

    def set_precision(self, B):
        """Fix the precision `B` (on the scale of the GAM's objective)."""
        self.precision = np.asarray(B, float)

    def set_weights(self, w):
        super().set_weights(w)
        if self.precision is None:
            raise ValueError(f'{self.name}: give precision (or estimate it, method="REML")')
        G, q = len(self.levels_), self.q_
        # each group's V' W V
        outer = self.w_[:, None, None] * self.V_[:, :, None] * self.V_[:, None, :]
        self._vwv = np.zeros((G, q, q))
        np.add.at(self._vwv, self.codes_, outer)
        self._A = self._vwv + self.precision[None]

    def smooth(self, r):
        G, q = len(self.levels_), self.q_
        rhs = np.zeros((G, q))
        np.add.at(rhs, self.codes_, (self.w_ * r)[:, None] * self.V_)
        b = np.linalg.solve(self._A, rhs[:, :, None])[:, :, 0]
        self._coef = b.reshape(-1)
        return np.sum(b[self.codes_] * self.V_, axis=1)

    # the smoother reproduces no function (the penalty is of full rank):
    # modified backfitting smooths it whole
    def smooth_nonlinear(self, r):
        return self.smooth(r)

    def combine_coef(self, beta, center):
        return self._coef

    def coef(self):
        return self._coef

    def penalty(self, f, coef=None):
        if self.precision is None or coef is None:
            return 0.
        b = np.asarray(coef).reshape((-1, self.q_))
        return float(np.einsum('gi,ij,gj->', b, self.precision, b)) / 2

    def edf(self):
        return float(np.trace(np.linalg.solve(self._A, self._vwv), axis1=1, axis2=2).sum())

    def basis(self, x=None):
        if x is None:
            codes, V = self.codes_, self.V_
        else:
            g, V = self._split(x)
            codes = _level_codes(self.levels_, g)
        G, q = len(self.levels_), self.q_
        B = np.zeros((V.shape[0], G * q))
        seen = codes >= 0
        rows = np.nonzero(seen)[0]
        for m in range(q):
            B[rows, codes[seen] * q + m] = V[seen, m]
        return B

    def penalty_matrices(self):
        return [np.kron(np.identity(len(self.levels_)), self.precision)]

    penalized = True

    def n_coef(self):
        return len(self.levels_) * self.q_

    def penalty_matrix(self):
        if self.precision is None:
            # (its structure: of full rank)
            return np.identity(len(self.levels_) * self.q_)
        return self.penalty_matrices()[0]

    def penalty_lams(self):
        return [1.]

    def penalty_lam(self):
        return 1.

    def set_penalty_lams(self, lams):
        raise ValueError(f'{self.name}: its penalty is a precision matrix (set_precision)')

    def finalize(self, f, r, coef=None):
        G, q = len(self.levels_), self.q_
        b = np.zeros((G, q)) if coef is None else np.asarray(coef, float).reshape((G, q))
        self.effects_ = pd.DataFrame(b, index=self.levels_, columns=self.effect_names)
        return self

    def predict(self, x):
        g, V = self._split(x)
        codes = _level_codes(self.levels_, g)
        b = self.effects_.to_numpy()
        return np.where(codes >= 0, np.sum(b[np.maximum(codes, 0)] * V, axis=1), 0.)


def correlated_random_effect(column, slopes=(), intercept=True, precision=None,
                             covariance='unstructured', name=None):
    """
    Correlated random effects per group of `column` (see
    `CorrelatedRandomEffectTerm`): an intercept and slopes in the columns
    `slopes`, with an unstructured covariance, lme4's `(1 + v | g)`; its
    precision estimated by `method='REML'` or `'ML'` unless given.
    """
    if _is_interaction(column):
        column = list(column)
    if covariance not in ('unstructured', 'compound_symmetry', 'diagonal'):
        raise ValueError("covariance should be one of 'unstructured', 'compound_symmetry' "
                         "or 'diagonal'")
    term = CorrelatedRandomEffectTerm(column, slopes=list(slopes), intercept=intercept,
                                      precision=precision, covariance=covariance)
    effects = (['1'] if intercept else []) + [str(v) for v in slopes]
    structure = '' if covariance == 'unstructured' else f', covariance={covariance!r}'
    term.label = name or (f'correlated_random_effect({" + ".join(effects)} | '
                          f'{_group_label(column)}{structure})')
    return term


def re(column, by=None, lam=None, df=None, name=None):
    """
    A random effect term (see `RandomEffectTerm`): a random intercept per
    group of `column`, or with `by`, a random slope of the column `by` per
    group. `column` may be a list of columns: the groups are then their
    combinations, so that `re('batch') + re(['batch', 'cask'])` are the
    nested effects of lme4's `(1 | batch/cask)`.
    """
    if _is_interaction(column):
        column = list(column)
    term = RandomEffectTerm(column, by=by, lam=lam, df=df)
    term.label = name or _name('re', _group_label(column), by=by, lam=lam, df=df)
    return term


def penalized_columns(columns, penalties, lam=None, name=None):
    """
    A block of columns with quadratic penalties on their coefficients (see
    `PenalizedColumnsTerm`), as mgcv's `paraPen`: `columns` a list of
    columns, `penalties` a list of matrices, `lam` one smoothing parameter
    per penalty (None: estimated, with `method='REML'` or `'ML'`).
    """
    columns = list(columns) if isinstance(columns, (list, tuple)) else [columns]
    term = PenalizedColumnsTerm(columns, penalties=list(penalties),
                                lam=None if lam is None else list(np.atleast_1d(lam)))
    term.label = name or f'penalized_columns({", ".join(str(c) for c in columns)})'
    return term


def s(column, df=None, lam=None, name=None):
    """
    A smoothing spline term (see `SplineTerm`): `df=4` by default (with
    `method='fixed'`; estimated with `'REML'` or `'ML'`).
    """
    term = SplineTerm(column, df=df, lam=lam)
    term.label = name or _name('s', column, df=df, lam=lam)
    return term


def lo(column, span=0.5, degree=1, name=None):
    """A loess term (see `LoessTerm`)."""
    term = LoessTerm(column, span=span, degree=degree)
    term.label = name or _name('lo', column, span=None if span == 0.5 else span,
                               degree=None if degree == 1 else degree)
    return term


def linear(column):
    """A linear term (see `SlopeTerm`)."""
    return SlopeTerm(column)


def categorical(column):
    """A categorical term, one effect per level (see `CategoricalTerm`; R's `factor`)."""
    return CategoricalTerm(column)


# the former names, deprecated

_FACTOR_DEPRECATED = ('factor() is deprecated, and will be removed: use categorical() '
                      '(CategoricalTerm)')


@dataclass
class FactorTerm(CategoricalTerm):
    """Deprecated: `CategoricalTerm`."""

    _label = 'factor'

    def __post_init__(self):
        warnings.warn(_FACTOR_DEPRECATED, FutureWarning, stacklevel=3)


def factor(column):
    """Deprecated: `categorical()`. (Its term is named `factor(column)`, as before.)"""
    warnings.warn(_FACTOR_DEPRECATED, FutureWarning, stacklevel=2)
    term = CategoricalTerm(column)
    term.label = f'factor({column})'
    return term
