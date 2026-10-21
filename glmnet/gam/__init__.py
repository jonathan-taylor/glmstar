"""
Generalized additive models: local scoring with pluggable smoothers, on
glmstar's IRLS.

A list of terms, each naming a column of `X`:

>>> from glmnet.gam import GAM, s, lo, factor, linear
>>> GAM(terms=[s('age', df=4), s('year', lam=1e-3), factor('education')]).fit(X, y)

(ISLP's `ISLP.gam` builds the terms from a ModelSpec, smooth features next
to ordinary columns.)

The smoothing spline and loess terms use the `scatter_smooth` package.
"""
from .terms import (Term,
                    SplineTerm,
                    LoessTerm,
                    SlopeTerm,
                    FactorTerm,
                    ColumnsTerm,
                    s,
                    lo,
                    linear,
                    factor)
from .backfit import GAMState, GAMRegularizer
from .estimator import GAM, GAMControl
