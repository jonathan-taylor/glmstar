"""
Families with a parameter that `GLMM(method='REML')` or `'ML'` estimates
with the smoothing parameters, beyond statsmodels':

- `QuasiPoisson` and `QuasiBinomial` (R's `quasipoisson`, `quasibinomial`):
  the Poisson's and binomial's mean and variance functions, with a free
  dispersion `phi` (variance `phi mu`, `phi mu (1 - mu)`). Under REML and ML
  the dispersion is estimated with the smoothing parameters, as mgcv's: the
  quasi-likelihood's "saturated log-likelihood" is that of a Gaussian,
  `-(n/2) log phi + sum(log w)/2`. With fixed smoothing it is Pearson's
  estimate, as R's `glm`. They have no likelihood: no log-likelihood or AIC.
- `NegativeBinomial(theta=None)` (mgcv's `nb()`; `theta` given: its
  `negbin(theta)`, or statsmodels' `NegativeBinomial(alpha=1/theta)`): the
  variance `mu + mu^2 / theta`; with `theta` None it is estimated by REML or
  ML with the smoothing parameters, its log a parameter of the criterion.
"""
import warnings

from statsmodels.genmod.families import family as sm_family, varfuncs
from statsmodels.tools.sm_exceptions import ValueWarning


class QuasiPoisson(sm_family.Poisson):
    """The Poisson's mean and variance, with a free dispersion (R's `quasipoisson`)."""
    quasi = True


class QuasiBinomial(sm_family.Binomial):
    """The binomial's mean and variance, with a free dispersion (R's `quasibinomial`)."""
    quasi = True


class NegativeBinomial(sm_family.NegativeBinomial):
    """
    The negative binomial, variance `mu + mu^2 / theta` (statsmodels'
    `alpha = 1 / theta`), log link by default. With `theta` None it is
    estimated by `GLMM(method='REML')` or `'ML'` (as mgcv's `nb()`; the fitted
    value is the GAM's `theta_`); otherwise fixed (mgcv's `negbin(theta)`).
    """

    def __init__(self, theta=None, link=None, check_link=True):
        self.estimate_theta = theta is None
        with warnings.catch_warnings():
            # (statsmodels warns of its default alpha)
            warnings.simplefilter('ignore', ValueWarning)
            super().__init__(link=link, alpha=1. if theta is None else 1 / float(theta),
                             check_link=check_link)

    @property
    def theta(self):
        return 1 / self.alpha

    def set_theta(self, theta):
        """Set `theta` (the variance function's too)."""
        self.alpha = 1 / float(theta)
        self.variance = varfuncs.NegativeBinomial(alpha=self.alpha)

    def __repr__(self):
        theta = 'estimated' if self.estimate_theta else f'{self.theta:g}'
        return f'NegativeBinomial(theta={theta})'
