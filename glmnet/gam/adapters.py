"""
Smoothers from scikit-learn estimators.

- `TransformerSmoother`: a `PenalizedSmoother` whose basis is a
  scikit-learn transformer's features (`fit` on the training rows,
  `transform` at any), with quadratic penalties on their coefficients:
  `SplineTransformer` with a difference penalty is a P-spline (Eilers &
  Marx, 1996). With `sum_to_zero=True` (the default) the basis is
  constrained, as mgcv's smooths, so that the fit sums to 0 over the
  training rows: a basis spanning the constant (B-splines sum to 1) is then
  identified apart from the intercept. As any `PenalizedSmoother`, it is
  fitted by backfitting or jointly, has standard errors, and a `GLMM`
  estimates its smoothing parameters.
- `RegressorSmoother`: any regressor that takes `sample_weight` (a tree,
  nearest neighbours), backfitted (Hastie & Tibshirani, 1990, chapter 9).
  It has no basis: a `GAM` only, without standard errors. Its degrees of
  freedom are Ye's (1998) generalized degrees of freedom, by Monte Carlo,
  unless given. Backfitting a smoother that is not linear need not
  converge: check `converged_`.

References
----------
- Eilers, P. H. C. and Marx, B. D. (1996). Flexible smoothing with
  B-splines and penalties. *Statistical Science* 11(2), 89–121.
- Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman & Hall.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.), section 5.4.1 (identifiability constraints). Chapman &
  Hall/CRC.
- Ye, J. (1998). On measuring and correcting the effects of data mining and
  model selection. *Journal of the American Statistical Association*
  93(441), 120–131.
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from sklearn.base import clone

from .terms import Smoother, PenalizedSmoother


def difference_penalty(k, order=2):
    """The penalty `D' D` of the `order`-th differences of `k` coefficients (a P-spline's)."""
    D = np.diff(np.identity(k), n=order, axis=0)
    return D.T @ D


def sum_to_zero(B):
    """
    `Z` (`k x (k - 1)`, orthonormal columns) with `1' B Z = 0`: the basis
    `B Z`, its fits summing to 0 over the rows of `B`, and the penalties
    `Z' S Z`, as mgcv's absorption of the constraint (by the QR
    decomposition of the column sums).
    """
    c = B.sum(0)[:, None]
    Q, _ = np.linalg.qr(c, mode='complete')
    return Q[:, 1:]


def _columns(x):
    x = np.asarray(x)
    return x.reshape((x.shape[0], -1))


@dataclass
class TransformerSmoother(PenalizedSmoother):
    """
    A penalized smoother on the features of a scikit-learn transformer of
    the column(s) (`transformer`, cloned and fitted on the training rows),
    with the penalties `penalties`: a list of matrices on the features'
    coefficients, or `'difference'` (`difference_penalty` of order
    `order`, the features in order, as a `SplineTransformer`'s B-splines),
    or `'ridge'` (the identity). With `sum_to_zero` the basis is
    constrained to fits summing to 0 over the training rows (`sum_to_zero()`),
    for a basis spanning the constant. `lam` as for `PenalizedSmoother`.

    >>> from sklearn.preprocessing import SplineTransformer
    >>> TransformerSmoother('age', SplineTransformer(n_knots=20), lam=1.)
    """
    transformer: object = None
    penalties: Union[str, list] = 'difference'
    order: int = 2
    sum_to_zero: bool = True
    label: Optional[str] = None

    _label = 'transformer'

    def setup(self, x):
        if self.transformer is None:
            raise ValueError(f'{self.name}: give a transformer')
        X = _columns(x)
        self.transformer_ = clone(self.transformer).fit(X)
        F = self._features(X)
        self.Z_ = sum_to_zero(F) if self.sum_to_zero else np.identity(F.shape[1])
        k = F.shape[1]
        if isinstance(self.penalties, str):
            if self.penalties == 'difference':
                Ss = [difference_penalty(k, self.order)]
            elif self.penalties == 'ridge':
                Ss = [np.identity(k)]
            else:
                raise ValueError(f"{self.name}: penalties should be 'difference', 'ridge' "
                                 'or a list of matrices')
        else:
            Ss = [np.asarray(S, float) for S in self.penalties]
            if not Ss or any(S.shape != (k, k) for S in Ss):
                raise ValueError(f'{self.name}: give penalties of shape ({k}, {k}), the '
                                 'transformer\'s features')
        self.penalties_ = [self.Z_.T @ S @ self.Z_ for S in Ss]
        return super().setup(X)

    def _features(self, X):
        F = self.transformer_.transform(X)
        F = F.toarray() if hasattr(F, 'toarray') else F
        return np.asarray(F, float)

    def basis(self, x=None):
        X = self.x_ if x is None else _columns(x)
        return self._features(X) @ self.Z_

    def penalty_matrices(self):
        return list(self.penalties_)


@dataclass
class RegressorSmoother(Smoother):
    """
    A scikit-learn regressor of the column(s) as a smoother: each smooth
    fits a clone of `estimator` to the partial residuals with the working
    weights (its `fit` must take `sample_weight`). Backfitting only (no
    basis: a `GAM`, without standard errors). Its degrees of freedom are
    `df` if given, else Ye's generalized degrees of freedom at the final
    weights, `tr(dF/dr)`, by Monte Carlo (`n_df_samples` fits to noise at the
    weights, Hutchinson's estimator of the trace for a linear smoother, with
    a standard error of about `sqrt(2 (df + 1) / n_df_samples)`), less 1 for
    the constant.

    >>> from sklearn.tree import DecisionTreeRegressor
    >>> RegressorSmoother('age', DecisionTreeRegressor(max_depth=3))
    """
    estimator: object = None
    df: Optional[float] = None
    n_df_samples: int = 20
    random_state: int = 0
    label: Optional[str] = None

    has_objective = False
    _label = 'regressor'

    def setup(self, x):
        if self.estimator is None:
            raise ValueError(f'{self.name}: give an estimator')
        self.x_ = np.asarray(x)
        self.X_ = _columns(x)
        return self

    def _fit(self, r):
        return clone(self.estimator).fit(self.X_, r, sample_weight=self.w_)

    def smooth(self, r):
        self.estimator_ = self._fit(r)
        return self.estimator_.predict(self.X_)

    def edf(self):
        if self.df is not None:
            return float(self.df)
        # z with independent entries of variance 1 / w (Rademacher, scaled):
        # E[z' W S z] = tr(S) for a linear smoother S (Hutchinson); its
        # standard error is about sqrt(2 (edf + 1) / n_df_samples)
        rng = np.random.default_rng(self.random_state)
        w = self.w_
        sd = np.where(w > 0, 1 / np.sqrt(np.where(w > 0, w, 1)), 0.)
        traces = []
        for _ in range(self.n_df_samples):
            z = rng.choice([-1., 1.], w.shape[0]) * sd
            traces.append(np.sum(w * z * self._fit(z).predict(self.X_)))
        return float(np.mean(traces) - 1)

    def finalize(self, f, r, coef=None):
        # the last smooth, shifted to the (centered) final fit
        if getattr(self, 'estimator_', None) is None:
            self.estimator_ = self._fit(r)
        self.shift_ = float(np.mean(self.estimator_.predict(self.X_) - f))
        return self

    def predict(self, x):
        return self.estimator_.predict(_columns(x)) - self.shift_
