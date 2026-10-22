"""
Cyclic smooths: `cyclic_spline(x)`, as mgcv's `s(x, bs="cc")` (Wood, 2017,
section 5.3.2), for a covariate whose ends meet, such as the hour of the
day or the day of the year. A cyclic cubic regression spline is
parametrized by its values `b` at `k - 1` knots of `k` (the last knot is
the first again): between knots it is the cubic with those values and the
second derivatives `d = B^{-1} D b`, `B` and `D` cyclic tridiagonal, so
that the function and its first two derivatives are continuous around the
cycle. Its penalty, the integral of the squared second derivative, is
`b' D' B^{-1} D b`, whose null space is the constant (a line is not
periodic).

References
----------
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .terms import Term
from .tensor import place_knots


class CyclicCubicSpline(object):
    """
    The cyclic cubic regression spline basis with knots `knots` (`k` of
    them; the cycle from the first to the last): its values (`basis(x)`, `x`
    taken modulo the cycle) and penalty (`penalty`), on the `k - 1`
    coefficients, the values at the first `k - 1` knots.
    """

    def __init__(self, knots):
        t = self.knots = np.asarray(knots, float)
        if t.shape[0] < 4:
            raise ValueError('a cyclic cubic spline needs at least 4 knots')
        if np.any(np.diff(t) <= 0):
            raise ValueError('the knots should be increasing')
        m = t.shape[0] - 1                       # the number of coefficients
        h = np.diff(t)                           # h[j] = t[j+1] - t[j], j = 0..m-1
        B = np.zeros((m, m))
        D = np.zeros((m, m))
        for i in range(m):
            prev, nxt = (i - 1) % m, (i + 1) % m
            # (h[i - 1] is the spacing before knot i, cyclically)
            B[i, i] += (h[i - 1] + h[i]) / 3
            B[i, prev] += h[i - 1] / 6
            B[i, nxt] += h[i] / 6
            D[i, prev] += 1 / h[i - 1]
            D[i, i] += -1 / h[i - 1] - 1 / h[i]
            D[i, nxt] += 1 / h[i]
        self._F = np.linalg.solve(B, D)          # the second derivatives at the knots
        self.penalty = D.T @ self._F
        self.penalty = (self.penalty + self.penalty.T) / 2
        self.period = t[-1] - t[0]
        self._m, self._h = m, h

    def basis(self, x):
        t, m, h = self.knots, self._m, self._h
        x = t[0] + np.mod(np.asarray(x, float) - t[0], self.period)
        j = np.clip(np.searchsorted(t, x, side='right') - 1, 0, m - 1)
        hj = h[j]
        am = (t[j + 1] - x) / hj
        ap = (x - t[j]) / hj
        cm = ((t[j + 1] - x)**3 / hj - hj * (t[j + 1] - x)) / 6
        cp = ((x - t[j])**3 / hj - hj * (x - t[j])) / 6
        nxt = (j + 1) % m
        rows = np.arange(x.shape[0])
        B = np.zeros((x.shape[0], m))
        B[rows, j] += am
        B[rows, nxt] += ap
        B += cm[:, None] * self._F[j] + cp[:, None] * self._F[nxt]
        return B


@dataclass
class CyclicSplineTerm(Term):
    """
    A cyclic cubic regression spline of the column: the term
    `cyclic_spline(column)`, mgcv's `s(column, bs="cc")`; see
    `glmnet.gam.cyclic`.

    Parameters
    ----------
    k : int
        The number of knots, 10 by default as in mgcv; the basis has `k - 1`
        functions.
    period : tuple, optional
        The ends of the cycle, as `(0, 24)` for the hour of the day; the knots
        are evenly spaced between them. By default the cycle is the range of
        the data, and the knots are mgcv's `place.knots`.
    knots : array, optional
        The knots, the first and last the ends of the cycle.
    lam : float, optional
        The smoothing parameter, on the scale of the GAM's objective; None:
        estimated (`method='REML'` or `'ML'`).
    """
    k: int = 10
    period: Optional[tuple] = None
    knots: Optional[np.ndarray] = None
    lam: Optional[float] = None

    _label = 'cyclic_spline'

    @property
    def has_objective(self):
        return self.lam is not None

    def setup(self, x):
        x = np.asarray(x, float).reshape(-1)
        if self.knots is not None:
            knots = np.asarray(self.knots, float)
        elif self.period is not None:
            a, b = self.period
            if np.any((x < a) | (x > b)):
                raise ValueError(f'{self.name}: values outside the period {self.period}')
            knots = np.linspace(a, b, self.k)
        else:
            knots = place_knots(x, self.k)
        self.spline_ = CyclicCubicSpline(knots)
        self.x_ = x
        self._basis = self.spline_.basis(x)
        self._coef = self._nl_coef = None
        return self

    def set_weights(self, w):
        super().set_weights(w)
        if self.lam is None:
            raise ValueError(f'{self.name}: give lam (or estimate it, method="REML")')
        self.lam_ = float(self.lam)
        B = self._basis
        self._xwx = B.T @ (self.w_[:, None] * B)
        self._A = self._xwx + self.lam_ * self.spline_.penalty

    def _solve(self, r):
        return np.linalg.lstsq(self._A, self._basis.T @ (self.w_ * r), rcond=None)[0]

    def smooth(self, r):
        coef = self._solve(r)
        f = self._basis @ coef
        center = f.mean()
        # the basis sums to 1: a constant added to the coefficients adds it
        self._coef = coef - center
        return f - center

    def coef(self):
        return self._coef

    # the smoother reproduces only the constant: modified backfitting smooths
    # it whole, less its weighted mean
    def smooth_nonlinear(self, r):
        coef = self._solve(r)
        f = self._basis @ coef
        mean = np.sum(self.w_ * f) / np.sum(self.w_)
        self._nl_coef = coef - mean
        return f - mean

    def combine_coef(self, beta, center):
        return self._nl_coef - center

    def penalty(self, f, coef=None):
        if self.lam is None or coef is None:
            return 0.
        return self.lam * float(coef @ self.spline_.penalty @ coef) / 2

    def edf(self):
        return float(np.trace(np.linalg.lstsq(self._A, self._xwx, rcond=None)[0])) - 1

    def basis(self, x=None):
        if x is None:
            return self._basis
        return self.spline_.basis(np.asarray(x, float).reshape(-1))

    def penalty_matrix(self):
        return self.spline_.penalty

    def penalty_lam(self):
        return self.lam_

    def finalize(self, f, r, coef=None):
        self.coef_ = np.zeros(self._basis.shape[1]) if coef is None else np.asarray(coef, float)
        return self

    def predict(self, x):
        return self.basis(x) @ self.coef_


def cyclic_spline(column, k=10, period=None, knots=None, lam=None, name=None):
    """
    A cyclic cubic regression spline of the column (see `CyclicSplineTerm`):
    mgcv's `s(column, bs="cc", k=k)`. With `lam` None its smoothing parameter
    is estimated (`GAM(method='REML')` or `'ML'`).

    >>> cyclic_spline('hour', period=(0, 24))
    """
    term = CyclicSplineTerm(column, k=k, period=None if period is None else tuple(period),
                            knots=knots, lam=lam)
    args = ''.join(f', {a}={v}' for a, v in [('k', None if k == 10 else k),
                                               ('period', period), ('lam', lam)] if v is not None)
    term.label = name or f'cyclic_spline({column}{args})'
    return term
