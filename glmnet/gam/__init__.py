"""
Generalized additive models: local scoring with pluggable smoothers, on
glmstar's IRLS.

With a ModelSpec, smooth features next to ordinary columns:

>>> from ISLP.models import ModelSpec as MS
>>> from ISLP.gam import GAM, s, lo
>>> GAM(model_spec=MS([s('year', df=4), s('age', df=5), 'education'])).fit(Wage, Wage['wage'])

or with a list of terms, each naming a column:

>>> from ISLP.gam import GAM, s, factor
>>> GAM(terms=[s('age', df=4), s('year', lam=1e-3), factor('education')]).fit(X, y)

The smoothing spline and loess terms use the `scatter_smooth` package.
"""
from .terms import (Term,
                    SplineTerm,
                    LoessTerm,
                    SlopeTerm,
                    FactorTerm,
                    ColumnsTerm,
                    linear,
                    factor)
from .model_spec import s, lo, SmoothColumn
from .backfit import GAMState, GAMRegularizer
from .estimator import GAM, GAMControl
