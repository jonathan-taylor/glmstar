"""
Generalized additive models: local scoring with pluggable smoothers.

>>> from ISLP.gam import GAM, s, lo, linear, factor
>>> GAM(terms=[s('age', df=4), s('year', lam=1e-3), factor('education')]).fit(X, y)

The smoothing spline and loess terms use the `scatter_smooth` package
(`pip install scatter-smooth`).
"""
from .terms import (Term,
                    SplineTerm,
                    LoessTerm,
                    SlopeTerm,
                    FactorTerm,
                    s,
                    lo,
                    linear,
                    factor)
from .backfit import GAMState, GAMRegularizer
from .estimator import GAM, GAMControl
