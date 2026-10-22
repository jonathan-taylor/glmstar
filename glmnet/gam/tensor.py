"""
Tensor product smooths: `tensor_spline(x, z)`, as mgcv's `te(x, z, bs="cr")` (Wood,
2006; Wood, 2017, section 5.6). The function of several columns is
`f(x, z) = sum_ij c_ij a_i(x) b_j(z)`, with `a_i` and `b_j` the marginal
bases: cubic regression splines, natural cubic splines parametrized by
their values at `k` knots (mgcv's `bs="cr"`, knots placed by mgcv's
`place.knots`). Each margin keeps its own penalty, the integral of the
squared second derivative `S_m`, as `S_1 (x) I` and `I (x) S_2` on the
coefficients `c`: the term has one smoothing parameter per margin, and is
invariant to the scales of the columns.

The penalties' null space is spanned by the products of the margins' linear
functions, `1, x, z, x z` (in two columns); modified backfitting fits
`x, z, x z` jointly with the other terms' linear parts.

References
----------
- Wood, S. N. (2006). Low-rank scale-invariant tensor product smooths for
  generalized additive mixed models. *Biometrics* 62(4), 1025–1036.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
"""
from dataclasses import dataclass
from functools import reduce
from typing import Optional

import numpy as np
from scipy.linalg import solve_banded

from .terms import Term


def place_knots(x, k):
    """
    `k` knots spread evenly through the sorted distinct values of `x`
    (interpolating between them), as mgcv's `place.knots`.
    """
    x = np.unique(np.asarray(x, float))
    n = x.shape[0]
    if k > n:
        raise ValueError(f'{k} knots, but only {n} distinct values')
    if k < 3:
        raise ValueError('a cubic regression spline needs at least 3 knots')
    delta = (n - 1) / (k - 1)
    i = np.arange(1, k - 1)
    lower = np.floor(delta * i).astype(int)          # (0-based)
    frac = delta * i - lower
    knots = np.empty(k)
    knots[0], knots[-1] = x[0], x[-1]
    knots[1:-1] = x[lower] * (1 - frac) + x[lower + 1] * frac
    return knots


class CubicRegressionSpline(object):
    """
    The natural cubic spline basis parametrized by its values at the knots
    `knots` (mgcv's `bs="cr"`): its values (`basis(x)`; linear beyond the
    boundary knots) and the integral of the products of its second
    derivatives (`penalty`), exact: they are linear between the knots.
    """

    def __init__(self, knots):
        t = self.knots = np.asarray(knots, float)
        k = t.shape[0]
        h = np.diff(t)
        V = np.identity(k)
        # the second derivatives at the knots (Green & Silverman, 1994,
        # section 2.1), 0 at the boundary
        QtV = V[:-2] / h[:-1, None] - V[1:-1] * (1 / h[:-1] + 1 / h[1:])[:, None] + V[2:] / h[1:, None]
        R = np.zeros((3, k - 2))
        R[1] = (h[:-1] + h[1:]) / 3
        R[0, 1:] = R[2, :-1] = h[1:-1] / 6
        V2 = np.zeros((k, k))
        V2[1:-1] = solve_banded((1, 1), R, QtV)
        # the slopes at the boundary knots
        V1 = np.array([(V[1] - V[0]) / h[0] - h[0] * V2[1] / 6,
                       (V[-1] - V[-2]) / h[-1] + h[-1] * V2[-2] / 6])
        self._V2, self._V1 = V2, V1
        lo, hi = V2[:-1], V2[1:]
        w = h[:, None] / 6
        cross = lo.T @ (w * hi)
        self.penalty = 2 * (lo.T @ (w * lo) + hi.T @ (w * hi)) + cross + cross.T

    def basis(self, x):
        t, V2, V1 = self.knots, self._V2, self._V1
        x = np.asarray(x, float)
        k = t.shape[0]
        i = np.clip(np.searchsorted(t, x, side='right') - 1, 0, k - 2)
        h = t[i + 1] - t[i]
        b = (x - t[i]) / h
        a = 1 - b
        B = np.zeros((x.shape[0], k))
        rows = np.arange(x.shape[0])
        B[rows, i] += a
        B[rows, i + 1] += b
        B += ((a**3 - a) * h**2 / 6)[:, None] * V2[i] + ((b**3 - b) * h**2 / 6)[:, None] * V2[i + 1]
        for end, side in [(0, x < t[0]), (-1, x > t[-1])]:
            if np.any(side):
                B[side] = (np.identity(k)[end] + (x[side] - t[end])[:, None] * V1[end])
        return B


def _row_kron(A, B):
    """The row-wise Kronecker product: row `i` is `A[i] (x) B[i]`."""
    return (A[:, :, None] * B[:, None, :]).reshape((A.shape[0], -1))


@dataclass
class TensorTerm(Term):
    """
    A tensor product smooth of several columns (`column`, a list of them):
    the term `tensor_spline(x, z)`, mgcv's `te(x, z, bs="cr")`; see
    `glmnet.gam.tensor`.

    Parameters
    ----------
    k : tuple
        The number of knots of each margin (mgcv's `k`; 5 by default, as
        mgcv's).
    knots : list, optional
        The knots of each margin (by default mgcv's `place.knots`).
    lam : list, optional
        One smoothing parameter per margin (on the scale of the GAM's
        objective); None: estimated (`method='REML'` or `'ML'`).
    """
    k: Optional[tuple] = None
    knots: Optional[list] = None
    lam: Optional[list] = None

    _label = 'tensor_spline'

    @property
    def has_objective(self):
        return self.lam is not None

    def setup(self, x):
        X = np.asarray(x, float)
        X = X.reshape((X.shape[0], -1))
        d = X.shape[1]
        if d < 2:
            raise ValueError(f'{self.name}: a tensor product needs at least two columns')
        k = tuple(np.broadcast_to(np.asarray(5 if self.k is None else self.k, int), (d,)))
        knots = (self.knots if self.knots is not None else
                 [place_knots(X[:, m], k[m]) for m in range(d)])
        if len(knots) != d:
            raise ValueError(f'{self.name}: give the knots of each of the {d} margins')
        self.margins_ = [CubicRegressionSpline(t) for t in knots]
        self.X_ = X
        self._basis = self._basis_values(X)
        dims = [len(m.knots) for m in self.margins_]
        eyes = [np.identity(q) for q in dims]
        self.penalties_ = [reduce(np.kron, [m.penalty if j == i else eyes[j] for j in range(d)])
                           for i, m in enumerate(self.margins_)]
        if self.lam is not None and len(np.atleast_1d(self.lam)) != d:
            raise ValueError(f'{self.name}: give one lam per margin')
        # the functions the smoother reproduces beyond the constant, the
        # products of the margins' (1, x): their columns, centered, and
        # their coefficients
        self.center_ = X.mean(0)
        lin_coef, lin_cols = [], []
        for subset in range(1, 2**d):
            factors_c, factors_x = [], []
            for m, margin in enumerate(self.margins_):
                on = (subset >> m) & 1
                factors_c.append(margin.knots - self.center_[m] if on else np.ones(dims[m]))
                factors_x.append(X[:, m] - self.center_[m] if on else np.ones(X.shape[0]))
            lin_coef.append(reduce(np.kron, factors_c))
            lin_cols.append(np.prod(factors_x, axis=0))
        self._lin_coef = np.array(lin_coef).T             # (K, 2^d - 1)
        self._lin_cols = np.column_stack(lin_cols)
        self._coef = self._nl_coef = None
        return self

    def _basis_values(self, X):
        return reduce(_row_kron, [m.basis(X[:, i]) for i, m in enumerate(self.margins_)])

    def set_weights(self, w):
        super().set_weights(w)
        if self.lam is None:
            raise ValueError(f'{self.name}: give lam (or estimate it, method="REML")')
        self.lam_ = [float(l) for l in np.atleast_1d(self.lam)]
        B = self._basis
        self._xwx = B.T @ (self.w_[:, None] * B)
        self._A = self._xwx + sum(l * S for l, S in zip(self.lam_, self.penalties_))

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

    def linear_columns(self):
        return self._lin_cols

    def smooth_nonlinear(self, r):
        coef = self._solve(r)
        f = self._basis @ coef
        # less its weighted least squares fit on the constant and the
        # reproduced functions
        sq = np.sqrt(self.w_)
        L = np.column_stack([np.ones(f.shape[0]), self._lin_cols])
        a = np.linalg.lstsq(sq[:, None] * L, sq * f, rcond=None)[0]
        self._nl_coef = coef - a[0] - self._lin_coef @ a[1:]
        return f - L @ a

    def combine_coef(self, beta, center):
        return self._lin_coef @ beta + self._nl_coef - center

    def penalty(self, f, coef=None):
        if self.lam is None or coef is None:
            return 0.
        return sum(l * coef @ S @ coef for l, S in zip(np.atleast_1d(self.lam), self.penalties_)) / 2

    def edf(self):
        return float(np.trace(np.linalg.lstsq(self._A, self._xwx, rcond=None)[0])) - 1

    def basis(self, x=None):
        if x is None:
            return self._basis
        X = np.asarray(x, float)
        return self._basis_values(X.reshape((X.shape[0], -1)))

    def penalty_matrices(self):
        return list(self.penalties_)

    def penalty_matrix(self):
        return sum(self.penalties_)

    def penalty_lams(self):
        return list(self.lam_)

    def penalty_lam(self):
        raise ValueError(f'{self.name} has several penalties: use penalty_lams()')

    def set_penalty_lams(self, lams):
        self.lam = [float(l) for l in lams]

    def finalize(self, f, r, coef=None):
        self.coef_ = np.zeros(self._basis.shape[1]) if coef is None else np.asarray(coef, float)
        return self

    def predict(self, x):
        return self.basis(x) @ self.coef_


def tensor_spline(*columns, k=None, knots=None, lam=None, name=None):
    """
    A tensor product smooth of the columns (see `TensorTerm`): mgcv's
    `te(*columns, bs="cr", k=k)`. With `lam` None its smoothing parameters,
    one per column, are estimated (`GAM(method='REML')` or `'ML'`).

    >>> tensor_spline('x', 'z')
    >>> tensor_spline('lon', 'lat', k=(8, 6))
    """
    if len(columns) < 2:
        raise ValueError('tensor_spline needs at least two columns')
    term = TensorTerm(list(columns), k=k, knots=knots,
                      lam=None if lam is None else list(np.atleast_1d(lam)))
    args = ''.join(f', {a}={v}' for a, v in [('k', k), ('lam', lam)] if v is not None)
    term.label = name or f'tensor_spline({", ".join(str(c) for c in columns)}{args})'
    return term
