"""
How a GAM's penalty parameters (smoothing parameters, random effects'
variances and covariances) are chosen: `GAM(method=...)` is a string
(`'fixed'`, `'REML'`, `'ML'`) or a `SmoothingMethod`, an object whose
`select(gam, dispersion)` sets the estimated terms' parameters and returns
a `SmoothingSelection`. `GAM` then fits with them.

A method can use:
- `estimated_terms(gam)`, the terms whose parameters it is to choose (penalized,
  with neither `lam` nor `df`, nor a given precision), each with its
  penalty (`term.penalty_matrices()`, `term.set_penalty_lams()`, or
  `term.set_precision()` for correlated random effects);
- `gam.inner_fit()`, the fit at the terms' current parameters (local
  scoring, by backfitting unless the method asks for the joint solve);
- the REML and ML criteria (`glmnet.gam.reml.LaplaceCriterion`), whose
  linear algebra (`glmnet.gam.linalg`) and penalties
  (`glmnet.gam.penalties`) are its own objects.

Built in: `Fixed` (the terms' `lam` or `df`, as R's `gam`) and `Laplace`
(REML or ML, as mgcv's). The documentation's page "Crossed random effects
at scale" writes one: the method of moments for the variances of crossed
random effects, in time linear in the number of observations
(`glmnet.gam.moments`; Gao & Owen, 2017; Ghosh, Hastie & Owen, 2022).
"""
import numpy as np

# (estimated_terms is defined with the REML criterion, which uses it, and
# shared here: the two modules would otherwise import each other)
from .reml import select_smoothing, estimated_terms


class SmoothingMethod(object):
    """
    A way of choosing a GAM's penalty parameters. Subclasses set:

    - `name`: shown in `smoothing_.method`;
    - `estimates`: whether it chooses any (False: the terms' own);
    - `joint`: whether `GAM` fits with the joint solve (the REML criterion
      needs its design; a method for large data should backfit);
    - `covariance_edf`: whether `edf_` is mgcv's, the block traces of the
      influence of the joint fit (it forms the dense `P x P` covariance;
      False: the terms' smoothers' traces, as R's `gam`);
    - `experimental`: whether `GAM` warns (`ExperimentalWarning`).

    and implement `select(gam, dispersion)`.
    """
    name = None
    estimates = True
    joint = False
    covariance_edf = False
    experimental = True

    def select(self, gam, dispersion=None):
        """Set the estimated terms' parameters of the set-up `gam`; return a `SmoothingSelection`."""
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}()'


class Fixed(SmoothingMethod):
    """The terms' own smoothing (`lam` or `df`; `s()` with neither has `df=4`), as R's `gam`."""
    name = 'fixed'
    estimates = False
    experimental = False

    def select(self, gam, dispersion=None):
        return None


class Laplace(SmoothingMethod):
    """REML or ML (`method`), the Laplace approximate marginal likelihood, as mgcv's (`glmnet.gam.reml`)."""
    joint = True
    covariance_edf = True

    def __init__(self, method='REML'):
        if method not in ('REML', 'ML'):
            raise ValueError("method should be one of 'REML' or 'ML'")
        self.name = method

    def select(self, gam, dispersion=None):
        return select_smoothing(gam, dispersion)

    def __repr__(self):
        return f"Laplace('{self.name}')"


def resolve_method(method):
    """The `SmoothingMethod` of `GAM(method=...)`: a string or a `SmoothingMethod`."""
    if isinstance(method, SmoothingMethod):
        return method
    if method == 'fixed':
        return Fixed()
    if method in ('REML', 'ML'):
        return Laplace(method)
    raise ValueError("method should be one of 'fixed', 'REML', 'ML' or a "
                     "glmnet.gam.selection.SmoothingMethod")
