"""
Generalized additive models: local scoring with pluggable smoothers, on
glmstar's IRLS.

A `GAM` is a list of terms, each a smoother of columns of `X`; its
smoothing is fixed (each term's `lam` or `df`, as R's `gam`):

>>> from glmnet.gam import GAM, s, lo, categorical, linear
>>> GAM(terms=[s('age', df=4), s('year', lam=1e-3), categorical('education')]).fit(X, y)

A `GLMM` has the same terms, its penalized terms random effects whose
variances (the smoothing parameters) it estimates: by REML or ML as
mgcv's and lme4's (experimental: see `glmnet.gam.reml`; with JAX, `pip
install glmstar[gam]`, not needed for the Cox family), or by a
`SmoothingMethod` of one's own. Some parameters can be held fixed: a term
given `lam`, or a list of them with None for those to estimate.

>>> from glmnet.gam import GLMM, s, random_effect as re
>>> GLMM(terms=[s('Days'), re('Subject'), re('Subject', by='Days')])

The terms are smoothers, `Smoother` (backfitting needs only their
`smooth`), most of them `PenalizedSmoother`: a basis and quadratic
penalties, what the joint solve, the covariance and a GLMM need. A new term is a subclass of
one of them (the documentation's page on GAMs, "Writing a smoother").

(ISLP's `ISLP.gam` builds the terms from a ModelSpec, smooth features next
to ordinary columns.)

Each working problem of local scoring is solved by backfitting the terms (as
R's `gam`, the default) or jointly on their bases (`GAMControl(solver='joint')`,
as mgcv): the same fit. When every term has a basis (all but loess) the fit
has a Bayesian covariance (`posterior_covariance_`, as mgcv's `Vp`), with
standard errors for predictions and the terms' curves.

Gaussian process smooths (`gaussian_process(x)`, `gaussian_process(lon, lat)`:
mgcv's `s(..., bs="gp")`,
the low rank kriging of Kammann & Wand, 2003) are terms in one or more
columns.

The smoothing spline and loess terms use the `scatter_smooth` package.

References
----------
- Hastie, T. and Tibshirani, R. (1986). Generalized additive models.
  *Statistical Science* 1(3), 297–318.
- Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman & Hall.
- Buja, A., Hastie, T. and Tibshirani, R. (1989). Linear smoothers and
  additive models. *The Annals of Statistics* 17(2), 453–510.
- Green, P. J. and Silverman, B. W. (1994). *Nonparametric Regression and
  Generalized Linear Models: A Roughness Penalty Approach*. Chapman & Hall.
- Cleveland, W. S., Grosse, E. and Shyu, W. M. (1992). Local regression
  models. In J. M. Chambers and T. J. Hastie (eds.), *Statistical Models in
  S*, chapter 8. Wadsworth & Brooks/Cole.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and
  marginal likelihood estimation of semiparametric generalized linear
  models. *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N., Pya, N. and Säfken, B. (2016). Smoothing parameter and
  model selection for general smooth models. *Journal of the American
  Statistical Association* 111(516), 1548–1563.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
- Kammann, E. E. and Wand, M. P. (2003). Geoadditive models. *Journal of
  the Royal Statistical Society, Series C* 52(1), 1–18.
- Wahba, G. (1983). Bayesian "confidence intervals" for the cross-validated
  smoothing spline. *Journal of the Royal Statistical Society, Series B*
  45(1), 133–150.
- Nychka, D. (1988). Bayesian confidence intervals for smoothing splines.
  *Journal of the American Statistical Association* 83(404), 1134–1143.
- Marra, G. and Wood, S. N. (2012). Coverage properties of confidence
  intervals for generalized additive model components. *Scandinavian
  Journal of Statistics* 39(1), 53–74.
"""
from .terms import (Smoother,
                    PenalizedSmoother,
                    SplineSmoother,
                    LoessSmoother,
                    LinearSmoother,
                    CategoricalSmoother,
                    _FactorTerm,
                    ColumnsSmoother,
                    RandomEffectSmoother,
                    CorrelatedRandomEffectSmoother,
                    PenalizedColumnsSmoother,
                    s,
                    lo,
                    random_effect,
                    re,
                    correlated_random_effect,
                    penalized_columns,
                    linear,
                    categorical,
                    factor)
from .gp import GaussianProcessSmoother, gaussian_process
from .tensor import TensorSplineSmoother, tensor_spline
from .cyclic import CyclicSplineSmoother, cyclic_spline
from .families import QuasiPoisson, QuasiBinomial, NegativeBinomial
from .varying import (VaryingSmoother, CubicRegressionSplineSmoother, varying,
                      cubic_regression_spline)
from .adapters import TransformerSmoother, RegressorSmoother, difference_penalty, sum_to_zero
from .backfit import GAMState, GAMRegularizer
from .joint import JointDesign, JointRegularizer
from .reml import LaplaceCriterion, CoxLaplaceCriterion, SmoothingSelection, ExperimentalWarning
from .estimator import GAM, GLMM, GAMControl
from .selection import SmoothingMethod, Fixed, Laplace
from .moments import moment_equations
from .plot import plot_term, plot_terms

from .terms import _deprecated_names, _RENAMED
# the former names of the smoothers (glmstar 0.5.0), deprecated
__getattr__ = _deprecated_names(dict(_RENAMED, GPTerm='GaussianProcessSmoother',
                                     TensorTerm='TensorSplineSmoother',
                                     CyclicSplineTerm='CyclicSplineSmoother'),
                                globals(), __name__)
del _deprecated_names, _RENAMED
