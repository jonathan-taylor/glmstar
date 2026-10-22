"""
Factor-by smooths: `by_factor_spline(x, by=f)`, as mgcv's `s(x, by=f,
bs="cr")` (Wood, 2017, section 7.4.1): a curve of `x` for each level of the
factor `f`, `f_l(x)` on the rows of level `l` and 0 on the others. Each is a
cubic regression spline (mgcv's `bs="cr"`) on knots placed on all of `x`,
constrained to sum to 0 over its level's rows (the constraint absorbed in
its basis, as mgcv). The levels' means are not in the term: the model
should include `factor(f)` too, as mgcv's `y ~ f + s(x, by=f)`.

Each level has its own penalty, the integral of its curve's squared second
derivative, and so its own smoothing parameter; with `pooled_lam=True` the
levels share one (mgcv's `id`), as when the curves are thought equally
smooth. Either way the smoothing parameters are given or estimated by REML
or ML: the term has one penalty per level, or their sum.

References
----------
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
"""
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .terms import Term
from .tensor import place_knots, CubicRegressionSpline


@dataclass
class FactorSplineTerm(Term):
    """
    A cubic regression spline of the column for each level of the factor
    `by`: the term `by_factor_spline(column, by=by)`, mgcv's `s(column,
    by=by, bs="cr")`; see `glmnet.gam.by_factor`.

    Parameters
    ----------
    by : str or int
        The factor's column. Its levels are a pandas categorical's categories
        (those seen in fitting), in their order, otherwise sorted. Levels not
        seen in fitting predict 0.
    k : int
        The number of knots, 10 by default as in mgcv; each level's curve has
        `k - 1` coefficients (after its constraint).
    knots : array, optional
        The knots, shared by the levels; by default mgcv's `place.knots` on
        all of the column.
    lam : float or list, optional
        The smoothing parameter(s), on the scale of the GAM's objective: one
        for all the levels, or one per level. None: estimated (`method='REML'`
        or `'ML'`).
    pooled_lam : bool
        One smoothing parameter for all the levels (mgcv's `id`), estimated
        or given; by default each level has its own.
    """
    by: Union[str, int] = None
    k: int = 10
    knots: Optional[np.ndarray] = None
    lam: Optional[Union[float, list]] = None
    pooled_lam: bool = False

    centered = False         # (each level's curve is constrained in its basis)
    categories = None        # the categories of a pandas categorical `by` (set by GAM)
    _label = 'by_factor_spline'

    @property
    def columns(self):
        return [self.column, self.by]

    @property
    def factor_column(self):
        return self.by

    @property
    def has_objective(self):
        return self.lam is not None

    def _split(self, x):
        x = np.asarray(x)
        x = x.reshape((x.shape[0], -1))
        return x[:, 0].astype(float), x[:, 1]

    def _codes(self, f):
        """The positions of the values `f` among the levels (-1: not seen in fitting)."""
        return pd.Index(self.levels_).get_indexer(np.asarray(f).reshape(-1))

    def setup(self, x):
        x, f = self._split(x)
        if self.categories is None:
            self.levels_ = np.unique(f)
        else:
            seen = set(pd.unique(f))
            self.levels_ = np.array([c for c in self.categories if c in seen], dtype=object)
        self.codes_ = self._codes(f)
        L = len(self.levels_)
        knots = (np.asarray(self.knots, float) if self.knots is not None
                 else place_knots(x, self.k))
        self.spline_ = CubicRegressionSpline(knots)
        B = self.spline_.basis(x)
        k = B.shape[1]
        # each level's constraint, its curve summing to 0 over its rows,
        # absorbed: its coefficients Z_l g
        self.Z_ = []
        for l in range(L):
            sums = B[self.codes_ == l].sum(0)
            self.Z_.append(np.linalg.qr(sums[:, None], mode='complete')[0][:, 1:])
        S = self.spline_.penalty
        m = k - 1
        self._level_penalties = []
        for l, Z in enumerate(self.Z_):
            P = np.zeros((L * m, L * m))
            P[l * m:(l + 1) * m, l * m:(l + 1) * m] = Z.T @ S @ Z
            self._level_penalties.append((P + P.T) / 2)
        self._m = m
        if self.lam is not None:
            n_lam = 1 if self.pooled_lam else L
            lam = np.atleast_1d(np.asarray(self.lam, float))
            if lam.shape[0] == 1:
                lam = np.repeat(lam, n_lam)
            if lam.shape[0] != n_lam:
                raise ValueError(f'{self.name}: give one lam, or one per level ({L})')
            self.lam = [float(l) for l in lam]
        self.x_ = x
        self._basis = self._basis_values(x, self.codes_)
        self._coef = None
        return self

    def _basis_values(self, x, codes):
        B = self.spline_.basis(x)
        m, L = self._m, len(self.levels_)
        X = np.zeros((x.shape[0], L * m))
        for l, Z in enumerate(self.Z_):
            rows = codes == l
            X[rows, l * m:(l + 1) * m] = B[rows] @ Z
        return X

    def set_weights(self, w):
        super().set_weights(w)
        if self.lam is None:
            raise ValueError(f'{self.name}: give lam (or estimate it, method="REML")')
        self.lam_ = [float(l) for l in self.lam]
        self._xwx = self._basis.T @ (self.w_[:, None] * self._basis)
        self._A = self._xwx + sum(l * S for l, S in zip(self.lam_, self.penalty_matrices()))

    def smooth(self, r):
        self._coef = np.linalg.lstsq(self._A, self._basis.T @ (self.w_ * r), rcond=None)[0]
        return self._basis @ self._coef

    # the smoother does not reproduce the constant (each curve sums to 0 on
    # its level): modified backfitting smooths it whole
    def smooth_nonlinear(self, r):
        return self.smooth(r)

    def combine_coef(self, beta, center):
        return self._coef

    def coef(self):
        return self._coef

    def penalty(self, f, coef=None):
        if self.lam is None or coef is None:
            return 0.
        return sum(l * coef @ S @ coef for l, S in zip(self.lam, self.penalty_matrices())) / 2

    def edf(self):
        return float(np.trace(np.linalg.lstsq(self._A, self._xwx, rcond=None)[0]))

    def level_edf(self):
        """The degrees of freedom of each level's curve, at the current weights (a Series)."""
        F = np.linalg.lstsq(self._A, self._xwx, rcond=None)[0]
        d = np.diag(F)
        m = self._m
        return pd.Series([d[l * m:(l + 1) * m].sum() for l in range(len(self.levels_))],
                         index=self.levels_)

    def basis(self, x=None):
        if x is None:
            return self._basis
        x, f = self._split(x)
        return self._basis_values(x, self._codes(f))

    def penalty_matrices(self):
        if self.pooled_lam:
            return [sum(self._level_penalties)]
        return list(self._level_penalties)

    def penalty_matrix(self):
        return sum(self._level_penalties)

    def penalty_lams(self):
        return list(self.lam_)

    def penalty_lam(self):
        if len(self.penalty_matrices()) > 1:
            raise ValueError(f'{self.name} has a penalty per level: use penalty_lams()')
        return self.lam_[0]

    def set_penalty_lams(self, lams):
        self.lam = [float(l) for l in lams]

    def finalize(self, f, r, coef=None):
        self.coef_ = np.zeros(self._basis.shape[1]) if coef is None else np.asarray(coef, float)
        return self

    def predict(self, x):
        return self.basis(x) @ self.coef_


def by_factor_spline(column, by, k=10, knots=None, lam=None, pooled_lam=False, name=None):
    """
    A cubic regression spline of the column for each level of the factor
    `by` (see `FactorSplineTerm`): mgcv's `s(column, by=by, bs="cr", k=k)`,
    or with `pooled_lam=True` one smoothing parameter for all the levels
    (mgcv's `id`). Include `factor(by)` too, for the levels' means. With
    `lam` None the smoothing parameters are estimated (`GAM(method='REML')`
    or `'ML'`).

    >>> by_factor_spline('age', by='sex', pooled_lam=True)
    """
    term = FactorSplineTerm(column, by=by, k=k, knots=knots, lam=lam, pooled_lam=pooled_lam)
    args = ''.join(f', {a}={v}' for a, v in [('k', None if k == 10 else k), ('lam', lam),
                                               ('pooled_lam', pooled_lam or None)]
                   if v is not None)
    term.label = name or f'by_factor_spline({column}, by={by}{args})'
    return term
