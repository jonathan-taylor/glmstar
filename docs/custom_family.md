---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Custom GLM families

`GLMNet` fits the elastic net for any GLM given as a
[statsmodels](https://www.statsmodels.org/stable/glm.html) family (see
[GLM families](glmnet_family.md)). A family is made of three parts, and
writing them yourself lets you fit a GLM that statsmodels does not provide:

- a **link** $g$, with $\eta = g(\mu)$, its inverse $\mu = g^{-1}(\eta)$
  and the derivative $d\mu/d\eta$;
- a **variance function** $V(\mu)$;
- the **deviance**, through the unit deviance $d(y, \mu)$, with total
  deviance $\sum_i w_i\, d(y_i, \mu_i)$.

`GLMNet` fits by IRLS, so these are all it needs. At each step it solves a
penalized weighted least squares problem with working response
$z = \eta + (y - \mu)/(d\mu/d\eta)$ and weights
$w\,(d\mu/d\eta)^2 / V(\mu)$. It uses the deviance to check that each step
decreases the objective and to decide when IRLS has converged.

As an example, we write probit regression by hand and compare it with
statsmodels' own probit link.

```{code-cell} ipython3
import numpy as np
import statsmodels.api as sm
from scipy.special import xlogy
from scipy.stats import norm
from statsmodels.genmod.families import Family
from statsmodels.genmod.families.links import Link
from statsmodels.genmod.families.varfuncs import VarianceFunction

from glmnet import GLMNet
```

## The link

The probit link is the standard normal quantile function,
$g(\mu) = \Phi^{-1}(\mu)$, so $g^{-1}(\eta) = \Phi(\eta)$ and
$d\mu/d\eta = \phi(\eta)$.

```{code-cell} ipython3
class ProbitLink(Link):

    def __call__(self, p):
        return norm.ppf(p)

    def inverse(self, z):
        return norm.cdf(z)

    def deriv(self, p):
        return 1 / norm.pdf(norm.ppf(p))

    def inverse_deriv(self, z):
        return norm.pdf(z)
```

## The variance function

For a binary response $V(\mu) = \mu(1-\mu)$:

```{code-cell} ipython3
class BernoulliVariance(VarianceFunction):

    def __call__(self, mu):
        return mu * (1 - mu)

    def deriv(self, mu):
        return 1 - 2 * mu
```

## The family

The family combines the link and the variance function, and defines the
unit deviance in `_resid_dev`. statsmodels checks that the link is one of
the family's `links`. `loglike_obs` is not used by `GLMNet`, but it lets
the family be used with statsmodels' own `GLM` as well.

```{code-cell} ipython3
class ProbitFamily(Family):

    links = [ProbitLink]
    safe_links = [ProbitLink]

    def __init__(self):
        super().__init__(ProbitLink(), BernoulliVariance())

    def _resid_dev(self, endog, mu):
        # unit deviance 2 [y log(y / mu) + (1 - y) log((1 - y) / (1 - mu))]
        return 2 * (xlogy(endog, endog / mu) +
                    xlogy(1 - endog, (1 - endog) / (1 - mu)))

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        return var_weights * (xlogy(endog, mu) + xlogy(1 - endog, 1 - mu))
```

## Fitting the path

We simulate a probit model with three nonzero coefficients:

```{code-cell} ipython3
rng = np.random.default_rng(0)
n, p = 300, 10
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:3] = [0.8, -0.6, 0.4]
y = rng.binomial(1, norm.cdf(X @ beta)).astype(float)
```

and fit the path with our family and with statsmodels' binomial family
with its probit link:

```{code-cell} ipython3
by_hand = GLMNet(family=ProbitFamily()).fit(X, y)
builtin = GLMNet(family=sm.families.Binomial(link=sm.families.links.Probit())).fit(X, y)
```

The two paths are the same:

```{code-cell} ipython3
print(by_hand.coefs_.shape, builtin.coefs_.shape)
np.abs(by_hand.coefs_ - builtin.coefs_).max(), np.abs(by_hand.intercepts_ - builtin.intercepts_).max()
```

```{code-cell} ipython3
ax = by_hand.coef_path_.plot()
```

The family also works with statsmodels' (unpenalized) `GLM`. There the
fit agrees with the built-in probit to statsmodels' convergence
tolerance:

```{code-cell} ipython3
X1 = sm.add_constant(X)
glm_by_hand = sm.GLM(y, X1, family=ProbitFamily()).fit()
glm_builtin = sm.GLM(y, X1, family=sm.families.Binomial(link=sm.families.links.Probit())).fit()
np.abs(glm_by_hand.params - glm_builtin.params).max()
```

## Cross-validation

Cross-validation works as for any family:

```{code-cell} ipython3
_, cvpath = by_hand.cross_validation_path(X, y, cv=5)
ax = cvpath.plot(score='ProbitFamily Deviance')
```

`GLMNet` recognizes the binomial family by its class. A family written by
hand therefore gets the generic default scores: its own deviance, named
after the class, and the squared and absolute errors of the fitted
probabilities. It does not get the binomial ones (AUC, accuracy, and the
binomial deviance with probabilities clamped as in R's `cv.glmnet`):

```{code-cell} ipython3
[c for c in cvpath.scores.columns if not c.startswith('SD(')]
```

Other scores can be passed to `cross_validation_path` with `scorers=`.
