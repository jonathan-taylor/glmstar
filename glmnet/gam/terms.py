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
        raise ImportError('the smoothing spline and loess terms of ISLP.gam use the '
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
    no objective. Its degrees of freedom, the trace of the smoother less 1,
    are computed by smoothing the unit vectors, one smooth per observation.
    """
    span: float = 0.5
    degree: int = 1

    has_objective = False
    _label = 'lo'

    def setup(self, x):
        ss = _scatter_smooth()
        self.x_ = np.asarray(x, float)
        self._smoother = ss.LoessSmoother(self.x_, span=self.span, degree=self.degree)
        return self

    def set_weights(self, w):
        super().set_weights(w)
        self._smoother.update_weights(self.w_)

    def smooth(self, r):
        self._smoother.smooth(r)
        return self._smoother.predict(self.x_)

    # no linear_columns: loess is not a symmetric smoother, so splitting off
    # its linear part would change the fit (a single lo() term would no
    # longer give the loess smooth); it is backfitted whole

    def edf(self):
        # the trace of the smoother, by smoothing the unit vectors
        n = self.x_.shape[0]
        trace = 0.
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.
            trace += self.smooth(e)[i]
        return trace - 1

    def finalize(self, f, r, coef=None):
        # the fit is the smooth of r plus a function the smoother reproduces:
        # a constant (backfitting), or a line (modified backfitting)
        self._smoother.smooth(r)
        d = f - self._smoother.predict(self.x_)
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


def s(column, df=None, lam=None):
    """A smoothing spline term (see `SplineTerm`); `df=4` by default."""
    return SplineTerm(column, df=df, lam=lam)


def lo(column, span=0.5, degree=1):
    """A loess term (see `LoessTerm`)."""
    return LoessTerm(column, span=span, degree=degree)


def linear(column):
    """A linear term (see `SlopeTerm`)."""
    return SlopeTerm(column)


def factor(column):
    """A categorical term, one effect per level (see `FactorTerm`)."""
    return FactorTerm(column)
