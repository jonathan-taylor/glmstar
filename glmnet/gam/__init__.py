"""
Generalized additive models: local scoring with pluggable smoothers, on
glmstar's IRLS.

A list of terms, each naming a column of `X`:

>>> from glmnet.gam import GAM, s, lo, factor, linear
>>> GAM(terms=[s('age', df=4), s('year', lam=1e-3), factor('education')]).fit(X, y)

Random effects (`re(group)`, `re(group, by=x)`) are terms like the smooths,
penalized by their sum of squares, so mixed models are GAMs:

>>> from glmnet.gam import GAM, s, re
>>> GAM(terms=[s('Days', df=3), re('Subject', lam=0.1), re('Subject', by='Days', lam=1)])

(ISLP's `ISLP.gam` builds the terms from a ModelSpec, smooth features next
to ordinary columns.)

Each working problem of local scoring is solved by backfitting the terms (as
R's `gam`, the default) or jointly on their bases (`GAMControl(solver='joint')`,
as mgcv): the same fit. When every term has a basis (all but loess) the fit
has a Bayesian covariance (`posterior_covariance_`, as mgcv's `Vp`), with
standard errors for predictions and the terms' curves.

Smoothing parameters (and random effects' variances) can also be estimated,
by REML or ML as mgcv's (`GAM(method='REML')`, with JAX: `pip install jax`).

The smoothing spline and loess terms use the `scatter_smooth` package.
"""
from .terms import (Term,
                    SplineTerm,
                    LoessTerm,
                    SlopeTerm,
                    FactorTerm,
                    ColumnsTerm,
                    RandomEffectTerm,
                    s,
                    lo,
                    re,
                    linear,
                    factor)
from .backfit import GAMState, GAMRegularizer
from .joint import JointDesign, JointRegularizer
from .reml import LaplaceCriterion, SmoothingSelection
from .estimator import GAM, GAMControl
