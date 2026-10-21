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
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd


def _scatter_smooth():
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
class FactorTerm(Term):
    """
    A categorical column, one effect per level: the term `factor(column)`.
    Levels not seen in fitting predict 0. In modified backfitting its
    indicators are fitted jointly with the other terms' linear columns when it
    has at most `max_joint_levels` levels; otherwise it is backfitted.
    """

    _label = 'factor'
    max_joint_levels = 50

    def setup(self, x):
        self.levels_, self.codes_ = np.unique(np.asarray(x), return_inverse=True)
        return self

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
        codes = self.codes_ if x is None else _level_codes(self.levels_, x)
        return (codes[:, None] == np.arange(len(self.levels_))[None, :]).astype(float)

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
        if self.df is None and self.lam is None:
            self.df = 4.

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
        """The integral of the squared second derivative of the spline with coefficients `coef`."""
        return self._smoother.roughness(coef)

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
            from scipy.linalg import solve_banded
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
class RandomEffectTerm(Term):
    """
    A random effect of the groups of the column `column`: the term
    `re(column)`, one effect per group (a random intercept), or with
    `by=`, one coefficient of the column `by` per group (a random slope),
    `f(g, v) = b_g v`. The effects are penalized by `lam / 2` times their sum
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
        return f're({self.column}, by={self.by})' if self.by is not None else f're({self.column})'

    @property
    def columns(self):
        return self.column if self.by is None else [self.column, self.by]

    @property
    def has_objective(self):
        return self.lam is not None

    def _split(self, x):
        x = np.asarray(x)
        if self.by is None:
            return x.reshape(-1), None
        x = x.reshape((x.shape[0], -1))
        return x[:, 0], x[:, 1].astype(float)

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
        from scipy.optimize import brentq
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


def re(column, by=None, lam=None, df=None, name=None):
    """
    A random effect term (see `RandomEffectTerm`): a random intercept per
    group of `column`, or with `by`, a random slope of the column `by` per
    group.
    """
    term = RandomEffectTerm(column, by=by, lam=lam, df=df)
    term.label = name or _name('re', column, by=by, lam=lam, df=df)
    return term


def s(column, df=None, lam=None, name=None):
    """A smoothing spline term (see `SplineTerm`); `df=4` by default."""
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


def factor(column):
    """A categorical term, one effect per level (see `FactorTerm`)."""
    return FactorTerm(column)
