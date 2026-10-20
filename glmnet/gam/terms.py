"""
Terms of a generalized additive model: each is a weighted smoother of one
column of `X`, as in Hastie & Tibshirani's local scoring.

A term is set up on its column of the training data (`setup`), told the
weights of the current working (IRLS) problem (`set_weights`), and then asked
to smooth partial residuals at those weights (`smooth`), as often as
backfitting needs. Its fits are centered by the backfitter, to mean 0 over the
training rows (unweighted, as `mgcv`), the intercept taking up the rest. After
the fit, `finalize` gives it its final fit, from which it predicts at new
values (`predict`).

`penalty(f)` is the term's contribution to the objective, so that local
scoring can check that its steps decrease it. A term whose smoothing does not
minimize a fixed objective (loess, or a spline fitted to a fixed `df` whose
smoothing parameter changes with the weights) has `has_objective = False`.
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.linalg import solveh_banded


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
        return f'{self._label}({self.column})'

    def setup(self, x):
        """Set the term up on its column `x` of the training data."""
        raise NotImplementedError

    def set_weights(self, w):
        """The weights of the next smooths, those of the working problem."""
        self.w_ = np.asarray(w, float)

    def smooth(self, r):
        """The term's weighted fit to `r`, at the training rows."""
        raise NotImplementedError

    def penalty(self, f):
        """The term's penalty at its fit `f` (at the training rows)."""
        return 0.

    def edf(self):
        """The degrees of freedom of the term, at the current weights, without the constant."""
        raise NotImplementedError

    def finalize(self, f, r):
        """
        Fix the fitted function from its (centered) fit `f` at the training
        rows and the partial residuals `r` it was fitted to.
        """
        raise NotImplementedError

    def predict(self, x):
        """The fitted function at `x`."""
        raise NotImplementedError


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

    def edf(self):
        return 1.

    def finalize(self, f, r):
        _, self.coef_ = _weighted_ls(self.x_, f, np.ones_like(f))
        return self

    def predict(self, x):
        return self.coef_ * (np.asarray(x, float) - self.center_)


@dataclass
class FactorTerm(Term):
    """
    A categorical column, one effect per level: the term `factor(column)`.
    Levels not seen in fitting predict 0.
    """

    _label = 'factor'

    def setup(self, x):
        self.levels_, self.codes_ = np.unique(np.asarray(x), return_inverse=True)
        return self

    def smooth(self, r):
        nl = len(self.levels_)
        sw = np.bincount(self.codes_, weights=self.w_, minlength=nl)
        swr = np.bincount(self.codes_, weights=self.w_ * r, minlength=nl)
        means = np.divide(swr, sw, out=np.zeros(nl), where=sw > 0)
        return means[self.codes_]

    def edf(self):
        nl = len(self.levels_)
        return float(np.sum(np.bincount(self.codes_, weights=self.w_, minlength=nl) > 0) - 1)

    def finalize(self, f, r):
        nl = len(self.levels_)
        counts = np.bincount(self.codes_, minlength=nl)
        sums = np.bincount(self.codes_, weights=f, minlength=nl)
        self.effects_ = pd.Series(np.divide(sums, counts, out=np.zeros(nl), where=counts > 0),
                                  index=self.levels_)
        return self

    def predict(self, x):
        return self.effects_.reindex(np.asarray(x)).fillna(0.).to_numpy()


def _reinsch_QR(t):
    """
    Green & Silverman's banded Q (as its three diagonals) and R (upper banded,
    for `solveh_banded`) for the natural cubic spline with knots `t`: the
    penalty of the spline with values `g` at the knots is
    `g^T Q R^{-1} Q^T g`, the integral of its squared second derivative.
    """
    h = np.diff(t)
    Qd = np.vstack([1 / h[:-1], -1 / h[:-1] - 1 / h[1:], 1 / h[1:]])
    R = np.zeros((2, len(t) - 2))
    R[1] = (h[:-1] + h[1:]) / 3
    R[0, 1:] = h[1:-1] / 6
    return Qd, R


@dataclass
class SplineTerm(Term):
    """
    A cubic smoothing spline in the column: the term `s(column, df=4)` (or
    `s(column, lam=...)`). Its knots are the distinct values of the column, so
    it is the natural cubic spline minimizing the weighted residual sum of
    squares plus `lam` times the integral of its squared second derivative.

    With `lam` the smoothing parameter is fixed. With `df` (the default, 4)
    it is chosen, at every working problem's weights, so that the smoother's
    trace less 1 (for the constant) is `df`, as R's `gam::s`; the term then has
    no fixed objective. `lam` is on the scale of the GAM's objective, the
    deviance over twice the sum of the weights, plus `lam / 2` times the
    integral. (R's `mgcv`, with `bs='cr'` and these knots, fits the same
    spline with `sp = lam * sum(weights) * smooth$S.scale`, as it rescales
    its penalties.)
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
        self.knots_, self._inverse = np.unique(self.x_, return_inverse=True)
        if len(self.knots_) < 4:
            raise ValueError(f'{self.name}: a smoothing spline needs at least 4 distinct values')
        if self.df is not None and not 1 < self.df < len(self.knots_) - 1:
            raise ValueError(f'{self.name}: df should be in (1, {len(self.knots_) - 1})')
        self._smoother = ss.SplineSmoother(self.x_, lamval=self.lam or 0.,
                                           engine='reinsch')
        self._Q, self._R = _reinsch_QR(self.knots_)
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
        self._smoother.smooth(r)
        return self._smoother.predict(self.x_)

    def _values_at_knots(self, f):
        g = np.empty(len(self.knots_))
        g[self._inverse] = f
        return g

    def roughness(self, f):
        """The integral of the squared second derivative of the spline with values `f`."""
        g = self._values_at_knots(f)
        Q = self._Q
        Qg = Q[0] * g[:-2] + Q[1] * g[1:-1] + Q[2] * g[2:]
        return float(Qg @ solveh_banded(self._R, Qg))

    def penalty(self, f):
        if self.lam is None:
            return 0.
        return self.lam * self.roughness(f) / 2

    def edf(self):
        return self._smoother.compute_df(self.lam_) - 1

    def finalize(self, f, r):
        # the fit is a natural cubic spline with these knots: interpolating its
        # values at the knots recovers it
        ss = _scatter_smooth()
        self._interpolant = ss.SplineSmoother(self.knots_, lamval=0., engine='reinsch')
        self._interpolant.smooth(self._values_at_knots(f))
        return self

    def predict(self, x):
        return self._interpolant.predict(np.asarray(x, float))


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

    def edf(self):
        # the trace of the smoother, by smoothing the unit vectors
        n = self.x_.shape[0]
        trace = 0.
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.
            trace += self.smooth(e)[i]
        return trace - 1

    def finalize(self, f, r):
        self._smoother.smooth(r)
        self.center_ = self._smoother.predict(self.x_).mean()
        return self

    def predict(self, x):
        return self._smoother.predict(np.asarray(x, float)) - self.center_


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
