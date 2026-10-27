"""
Varying-coefficient terms (Hastie and Tibshirani, 1993): `varying(term,
by=...)` lets any smooth term vary with another column, as mgcv's `by=`.

- A categorical `by` (`varying(s('x'), by='f')`, mgcv's `s(x, by=f)`): a
  curve of the term's columns for each level, `sum_l 1[f_i = l] f_l(x_i)`,
  zero on the other levels' rows. When the term is centered (a smooth)
  each level's curve is constrained as the term alone, as mgcv's: it sums
  to 0 over all the training rows' values of its columns. The levels' means
  are then the model's: it includes `categorical(f)`, as mgcv's
  `y ~ f + s(x, by=f)`.
- A numeric `by` (`varying(s('x'), by='z', as_categorical=False)`, mgcv's
  `s(x, by=z)`): the varying coefficient `z_i f(x_i)`, not centered (`z
  f(x)` is not confounded with the intercept). `by` is categorical
  unless `as_categorical=False`, whatever its values' type (a categorical
  coded 0 and 1 is categorical).

A term with a basis and penalties (`s()`, `cubic_regression_spline()`,
`cyclic_spline()`, `tensor_spline()`, `gaussian_process()`, `random_effect()`,
`penalized_columns()`) gives the varying term's basis: its basis masked to
each level (with the constraint absorbed, `B Z`), or multiplied
by `z`, and its penalties on each level's coefficients. Each level then has
its own smoothing parameters (one per penalty of the term), or with
`pooled_lam=True` all the levels share them (mgcv's `id`); they are given
(`lam`) or estimated by REML or ML. Backfitting, the joint solve and REML/ML
work as for any term with a basis.

A term without a basis (`lo()`) is backfitted only: a copy of its smoother
for each level, fitted to the level's partial residuals (centered on the
level, unlike the terms with a basis), or for a numeric `by`, Hastie and
Tibshirani's update, the smooth of
`r / z` with weights `w z^2`, times `z`.

`cubic_regression_spline(x, k=10)` is mgcv's `s(x, bs="cr")`, a low-rank
natural cubic spline parametrized by its values at `k` knots;
`varying(cubic_regression_spline(x), by=f)` is mgcv's `s(x, by=f,
bs="cr")`.

References
----------
- Hastie, T. and Tibshirani, R. (1993). Varying-coefficient models. *Journal
  of the Royal Statistical Society, Series B* 55(4), 757–796.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .terms import Smoother, _column_label, _levels
from .tensor import place_knots, CubicRegressionSpline


@dataclass
class CubicRegressionSplineSmoother(Smoother):
    """
    A cubic regression spline of the column, mgcv's `s(column, bs="cr",
    k=k)`: the natural cubic spline parametrized by its values at `k` knots
    (mgcv's `place.knots` on the column, or `knots`), with the penalty the
    integral of its squared second derivative. A low-rank alternative to
    `s()`, whose knots are (nearly) all the distinct values.

    Parameters
    ----------
    k : int
        The number of knots (10, as mgcv).
    knots : array, optional
        The knots.
    lam : float, optional
        The smoothing parameter, on the scale of the GAM's objective; None:
        estimated by a `GLMM`.
    """
    k: int = 10
    knots: Optional[np.ndarray] = None
    lam: Optional[float] = None

    _label = 'cubic_regression_spline'

    @property
    def has_objective(self):
        return self.lam is not None

    def setup(self, x):
        x = np.asarray(x, float).reshape(-1)
        knots = np.asarray(self.knots, float) if self.knots is not None else place_knots(x, self.k)
        self.spline_ = CubicRegressionSpline(knots)
        self.x_ = x
        self._basis = self.spline_.basis(x)
        self.center_ = x.mean()
        # the line x: the spline's values at the knots are the knots
        self._x_coef = self.spline_.knots.copy()
        self._coef = self._nl_coef = None
        return self

    def set_weights(self, w):
        super().set_weights(w)
        if self.lam is None:
            raise ValueError(f'{self.name}: give lam (or estimate it with a GLMM)')
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

    def linear_columns(self):
        return (self.x_ - self.center_)[:, None]

    def smooth_nonlinear(self, r):
        coef = self._solve(r)
        f = self._basis @ coef
        # less its weighted least squares line
        sw = np.sqrt(self.w_)
        L = np.column_stack([np.ones_like(self.x_), self.x_])
        a, b = np.linalg.lstsq(sw[:, None] * L, sw * f, rcond=None)[0]
        self._nl_coef = coef - a - b * self._x_coef
        return f - a - b * self.x_

    def combine_coef(self, beta, center):
        return beta[0] * (self._x_coef - self.center_) + self._nl_coef - center

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


def cubic_regression_spline(column, k=10, knots=None, lam=None, name=None):
    """
    A cubic regression spline of the column (see `CubicRegressionSplineSmoother`):
    mgcv's `s(column, bs="cr", k=k)`. With `lam` None its smoothing parameter
    is estimated (`GLMM(method='REML')` or `'ML'`).
    """
    term = CubicRegressionSplineSmoother(column, k=k, knots=knots, lam=lam)
    args = ''.join(f', {a}={v}' for a, v in [('k', None if k == 10 else k), ('lam', lam)]
                   if v is not None)
    term.label = name or f'cubic_regression_spline({column}{args})'
    return term


def _as_float(x):
    try:
        return np.asarray(x, float)
    except (TypeError, ValueError):
        return np.asarray(x)


@dataclass
class VaryingSmoother(Smoother):
    """
    The term `inner` varying with the column `by`: the term
    `varying(inner, by=by)`; see `glmnet.gam.varying`.

    Parameters
    ----------
    inner : Smoother
        A smooth term (`s()`, `cyclic_spline()`, `tensor_spline()`, `lo()`,
        ...); its own `lam` (if given) is that of every level, and it should
        not be given `df`.
    by : str or int
        The modifying column: categorical (by default), or with
        `as_categorical=False` a number.
    lam : float or list, optional
        The smoothing parameters, on the scale of the GAM's objective: one
        for all, or one for each penalty of each level (the level's
        penalties in order, level by level), or with `pooled_lam`, one per
        penalty of `inner`. None: `inner`'s, else estimated by a
        `GLMM`.
    pooled_lam : bool
        A categorical `by`: the levels share the smoothing parameters (mgcv's
        `id`); by default each level has its own.
    as_categorical : bool
        Whether `by` is categorical (a curve per level, its values the levels,
        whatever their type), or a number (`z f(x)`).
    """
    inner: Smoother = None
    by: Union[str, int] = None
    lam: Optional[Union[float, list]] = None
    pooled_lam: bool = False
    as_categorical: bool = True

    centered = False          # (the constraints are absorbed in the basis)
    categories = None         # the categories of a pandas categorical `by` (set by GAM)
    _label = 'varying'

    def __post_init__(self):
        if getattr(self.inner, 'df', None) is not None:
            raise ValueError(f'{self.name}: give the inner term lam, not df '
                             '(or estimate it with a GLMM)')

    # the columns: the inner term's, then `by`

    @property
    def _inner_columns(self):
        c = self.inner.columns
        return list(c) if isinstance(c, (list, tuple)) else [c]

    @property
    def columns(self):
        return self._inner_columns + [self.by]

    @property
    def categorical_column(self):
        """The categorical column `by` (None if numeric), whose categories `GAM` records."""
        return self.by if self.as_categorical else None

    @property
    def has_objective(self):
        return self.lam is not None and getattr(self, 'has_basis_', True)

    @property
    def has_basis(self):
        # (its inner term's, per level: none for lo(), which is backfitted)
        return getattr(self, 'has_basis_', self.inner.has_basis)

    def _split(self, x):
        x = np.asarray(x)
        x = x.reshape((x.shape[0], -1))
        inner = _as_float(x[:, :-1])
        if not isinstance(self.inner.columns, (list, tuple)):
            inner = inner[:, 0]
        return inner, x[:, -1]

    def _codes(self, f):
        """The positions of the values `f` among the levels (-1: not seen in fitting)."""
        return pd.Index(self.levels_).get_indexer(np.asarray(f).reshape(-1))

    # setting up

    def setup(self, x):
        xi, by = self._split(x)
        self.is_categorical_ = bool(self.as_categorical)
        if self.is_categorical_:
            self.levels_, self.codes_ = _levels(by, self.categories)
        else:
            self.z_ = np.asarray(by, float)
        self.xi_ = xi
        self.inner_ = deepcopy(self.inner)
        self.inner_.setup(xi)
        B = self.inner_.basis()
        self.has_basis_ = B is not None
        if self.has_basis_:
            self._setup_basis(np.asarray(B, float))
        else:
            self._setup_copies(xi)
        self._coef = None
        return self

    def _setup_basis(self, B):
        Ss = [np.asarray(S, float) for S in self.inner_.penalty_matrices()]
        self._n_inner_penalties = len(Ss)
        self.Z_, self.means_, self._offsets = [], [], [0]
        if self.is_categorical_:
            # each level's curve is constrained as the term alone would be,
            # over all the training rows (as mgcv's): it sums to 0 over the
            # rows' values of the columns, 1' B Z = 0, as if every row were of
            # its level. (The fits are those of any other constraint, as the
            # constraint on its level's rows; this one is mgcv's, and its ML
            # criterion's parametrization of the penalties' null space.)
            Z, mean = np.identity(B.shape[1]), np.zeros(B.shape[1])
            if self.inner_.centered:
                ones = np.ones(B.shape[0])
                c = np.linalg.lstsq(B, ones, rcond=None)[0]
                if np.linalg.norm(B @ c - ones) < 1e-8 * np.sqrt(B.shape[0]):
                    # the constant in the span
                    Z = np.linalg.qr(B.sum(0)[:, None], mode='complete')[0][:, 1:]
                else:
                    # (a linear term: its columns centered)
                    mean = B.mean(0)
            for l in range(len(self.levels_)):
                self.Z_.append(Z)
                self.means_.append(mean)
                self._offsets.append(self._offsets[-1] + Z.shape[1])
        else:
            self.Z_, self.means_ = [np.identity(B.shape[1])], [np.zeros(B.shape[1])]
            self._offsets.append(B.shape[1])
        P = self._offsets[-1]
        # the penalties: each inner penalty on each level's coefficients
        self._level_penalties = []                   # [level][m]
        for l, Z in enumerate(self.Z_):
            sl = slice(self._offsets[l], self._offsets[l + 1])
            level = []
            for S in Ss:
                E = np.zeros((P, P))
                E[sl, sl] = Z.T @ S @ Z
                level.append((E + E.T) / 2)
            self._level_penalties.append(level)
        n_lam = len(self.penalty_matrices())
        lam = self.lam if self.lam is not None else getattr(self.inner, 'lam', None)
        if lam is not None:
            lam = np.atleast_1d(np.asarray(lam, float))
            if lam.shape[0] == 1:
                lam = np.repeat(lam, n_lam)
            elif (not self.pooled_lam and self.is_categorical_ and
                  lam.shape[0] == self._n_inner_penalties):
                # (the inner term's lams, for every level)
                lam = np.tile(lam, len(self.Z_))
            if lam.shape[0] != n_lam:
                raise ValueError(f'{self.name}: give one lam, or {n_lam} (one per penalty '
                                 'of each level)')
            self.lam = [float(l) for l in lam]
        self._basis = self._basis_values(self.inner_.basis(), self._by_rows())

    def _by_rows(self, by=None):
        """
        The levels' codes (categorical) or the values of `by` (a number), at
        the training rows or at `by`.
        """
        if by is None:
            return self.codes_ if self.is_categorical_ else self.z_
        return self._codes(by) if self.is_categorical_ else np.asarray(by, float)

    def _basis_values(self, B, by):
        B = np.asarray(B, float)
        if not self.is_categorical_:
            return by[:, None] * B
        X = np.zeros((B.shape[0], self._offsets[-1]))
        for l, (Z, mean) in enumerate(zip(self.Z_, self.means_)):
            rows = by == l
            X[rows, self._offsets[l]:self._offsets[l + 1]] = (B[rows] - mean) @ Z
        return X

    def _setup_copies(self, xi):
        # a term without a basis: a copy of its smoother for each level, or
        # for a number, one on the rows where it is not 0
        if self.lam is not None:
            raise ValueError(f'{self.name}: {self.inner.name} has no basis: give its own '
                             'smoothing (as lo(span=...))')
        if self.is_categorical_:
            self._rows = [self.codes_ == l for l in range(len(self.levels_))]
        else:
            self._rows = [self.z_ != 0]
        self.copies_ = []
        for rows in self._rows:
            copy = deepcopy(self.inner)
            copy.setup(xi[rows])
            self.copies_.append(copy)

    # the working problems

    def set_weights(self, w):
        super().set_weights(w)
        if self.has_basis_:
            if self.lam is None:
                raise ValueError(f'{self.name}: give lam (or estimate it with a GLMM)')
            self.lam_ = [float(l) for l in self.lam]
            self._xwx = self._basis.T @ (self.w_[:, None] * self._basis)
            self._A = self._xwx + sum(l * S for l, S in zip(self.lam_, self.penalty_matrices()))
        else:
            weight = self.w_ if self.is_categorical_ else self.w_ * self.z_**2
            for copy, rows in zip(self.copies_, self._rows):
                copy.set_weights(weight[rows])

    def smooth(self, r):
        if self.has_basis_:
            self._coef = np.linalg.lstsq(self._A, self._basis.T @ (self.w_ * r), rcond=None)[0]
            return self._basis @ self._coef
        f = np.zeros_like(r)
        for copy, rows in zip(self.copies_, self._rows):
            if self.is_categorical_:
                g = copy.smooth(r[rows])
                # (each level's curve sums to 0 on its rows)
                f[rows] = g - g.mean()
            else:
                z = self.z_[rows]
                f[rows] = z * copy.smooth(r[rows] / z)
        return f

    # the smoother does not reproduce the constant: modified backfitting
    # smooths it whole
    def smooth_nonlinear(self, r):
        return self.smooth(r)

    def combine_coef(self, beta, center):
        return self._coef

    def coef(self):
        return self._coef if self.has_basis_ else None

    def penalty(self, f, coef=None):
        if not self.has_basis_ or self.lam is None or coef is None:
            return 0.
        return sum(l * coef @ S @ coef for l, S in zip(self.lam, self.penalty_matrices())) / 2

    def edf(self):
        if self.has_basis_:
            return float(np.trace(np.linalg.lstsq(self._A, self._xwx, rcond=None)[0]))
        # (a copy's edf is without its constant: kept for a number)
        return float(sum(copy.edf() for copy in self.copies_) + (not self.is_categorical_))

    def level_edf(self):
        """
        A categorical `by`: the degrees of freedom of each level's curve, at
        the current weights (a Series).
        """
        if not self.is_categorical_:
            raise ValueError(f'{self.name}: by is not categorical')
        if not self.has_basis_:
            return pd.Series([copy.edf() for copy in self.copies_], index=self.levels_)
        d = np.diag(np.linalg.lstsq(self._A, self._xwx, rcond=None)[0])
        return pd.Series([d[self._offsets[l]:self._offsets[l + 1]].sum()
                          for l in range(len(self.levels_))], index=self.levels_)

    # the basis and the penalties

    def basis(self, x=None):
        if not self.has_basis_:
            return None
        if x is None:
            return self._basis
        xi, by = self._split(x)
        return self._basis_values(self.inner_.basis(xi), self._by_rows(by))

    def penalty_matrices(self):
        if not self.has_basis_:
            return []
        if self.pooled_lam and self.is_categorical_:
            return [sum(level[m] for level in self._level_penalties)
                    for m in range(self._n_inner_penalties)]
        return [S for level in self._level_penalties for S in level]

    def penalty_matrix(self):
        Ss = self.penalty_matrices()
        return sum(Ss) if Ss else None

    def penalty_lams(self):
        return list(self.lam_)

    def penalty_lam(self):
        if len(self.penalty_matrices()) > 1:
            raise ValueError(f'{self.name} has several penalties: use penalty_lams()')
        return self.lam_[0]

    def set_penalty_lams(self, lams):
        self.lam = [float(l) for l in lams]

    # the fitted function

    def finalize(self, f, r, coef=None):
        if self.has_basis_:
            self.coef_ = (np.zeros(self._basis.shape[1]) if coef is None
                          else np.asarray(coef, float))
            return self
        for copy, rows in zip(self.copies_, self._rows):
            if self.is_categorical_:
                copy.finalize(f[rows], r[rows])
            else:
                z = self.z_[rows]
                copy.finalize(f[rows] / z, r[rows] / z)
        return self

    def predict(self, x):
        if self.has_basis_:
            return self.basis(x) @ self.coef_
        xi, by = self._split(x)
        if not self.is_categorical_:
            return np.asarray(by, float) * self.copies_[0].predict(xi)
        codes = self._codes(by)
        out = np.zeros(codes.shape[0])
        for l, copy in enumerate(self.copies_):
            rows = codes == l
            if np.any(rows):
                out[rows] = copy.predict(xi[rows])
        return out


def varying(term, by, lam=None, pooled_lam=False, as_categorical=True, name=None):
    """
    The term `term` varying with the column `by` (see `VaryingSmoother`): a curve
    per level of the categorical `by` (mgcv's `s(x, by=f)`; include `categorical(by)`
    too), or with `as_categorical=False`, `z f(x)` for a number `by` (mgcv's
    `s(x, by=z)`). With
    `pooled_lam=True` the levels share the smoothing parameters (mgcv's
    `id`). With `lam` None (and none in `term`) they are estimated
    (`GLMM(method='REML')` or `'ML'`).

    >>> varying(s('age'), by='sex', pooled_lam=True)
    """
    vterm = VaryingSmoother(term.column, inner=term, by=by, lam=lam, pooled_lam=pooled_lam,
                        as_categorical=as_categorical)
    options = [('lam', lam), ('pooled_lam', pooled_lam or None),
               ('as_categorical', None if as_categorical else False)]
    args = ''.join(f', {a}={v}' for a, v in options if v is not None)
    vterm.label = name or f'varying({term.name}, by={_column_label(by)}{args})'
    return vterm
