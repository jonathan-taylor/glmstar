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

# GLM families: `GLMNet` with a `family`

This document parallels the R vignette [The `family` Argument for `glmnet`](https://glmnet.stanford.edu/articles/glmnetFamily.html).

## Introduction

The `glmnet` package fits a generalized linear model (GLM) via
penalized maximum likelihood. Concretely, it solves the problem

$$ \min_{\beta_0, \beta} \frac{1}{N}\sum_{i=1}^N w_i l_i(y_i, \beta_0 + \beta^T x_i) + \lambda \left[\frac{1 - \alpha}{2}\|\beta\|_2^2 + \alpha \|\beta\|_1 \right] $$

over a grid of values of $\lambda$ covering the entire range. In the equation above, $l_i(y_i, \eta_i)$ is the negative log-likelihood contribution for observation $i$. $\alpha \in [0,1]$ is a tuning parameter which bridges the gap between the lasso ($\alpha = 1$, the default) and ridge regression ($\alpha = 0$), while $\lambda$ controls the overall strength of the penalty.

`glmnet` solves the minimization problem above very efficiently for a limited number of built-in families, each with its own estimator whose whole path algorithm is implemented in C++: `GaussNet` (Gaussian), `LogNet` (binomial), `FishNet` (Poisson), `MultiClassNet` (multinomial), `MultiGaussNet` (multi-response Gaussian) and `CoxNet` (Cox). These correspond to R's `family="gaussian"`, `"binomial"`, `"poisson"`, `"multinomial"`, `"mgaussian"` and `"cox"`; see [Quick Start](quick_start.md).

Apart from these built-in families, the `GLMNet` estimator fits a penalized regression model for *any* GLM, given as a [statsmodels](https://www.statsmodels.org/stable/glm.html) family object -- the analogue of passing a `family()` object to R's `glmnet`.

### Using family objects

All the functionality of the path estimators applies to `GLMNet` with a family, and hence it expands the scope of the package considerably. In particular,

* methods such as `predict`, `interpolate_coefs` and the coefficient path plots work as before;
* `cross_validation_path` can be used for selecting the tuning parameters;
* upper and lower bound constraints, penalty factors, `exclude`, standardization, weights and offsets work as before.

`GLMNet` fits the model for each value of $\lambda$ with a proximal Newton algorithm, also known as iteratively reweighted least squares (IRLS). The outer IRLS loop is written in Python, while the inner loop solves the weighted least squares problem with the elastic net penalty in C++. It uses warm starts as it moves down the path, and so is reasonably efficient.

### More on GLM families

A GLM is a linear model for a response variable whose conditional distribution belongs to a one-dimensional exponential family. Apart from the Gaussian, Poisson and binomial families, there are other interesting members of this family, e.g. Gamma, inverse Gaussian and negative binomial. A GLM consists of 3 parts:

1. A linear predictor: $\eta_i = \beta_0 + \beta^T x_i$,
2. A link function: $\eta_i = g(\mu_i)$, and
3. A random component: $y_i \sim f(y \mid \mu_i)$.

The user specifies the link function $g$ and the family of response distributions $f(\cdot \mid \mu)$, and fitting a GLM amounts to estimating $\beta$ by maximum likelihood.

In statsmodels, these parts are encapsulated in a family object, which carries its link function along with the variance and deviance functions used by IRLS. For example, the binomial family used for logistic regression:

```{code-cell} ipython3
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod.families import family as sm_family, links as sm_links

from glmnet import GaussNet, LogNet, FishNet, GLMNet
from glmnet.glmnet import GLMNetControl
from glmnet.paths.fastnet import FastNetControl

fam = sm_family.Binomial()
print(type(fam).__name__, '| link:', type(fam.link).__name__)
print([name for name in ['link', 'variance', 'deviance', 'starting_mu', 'weights'] if hasattr(fam, name)])
```

`GLMNet` can fit penalized GLMs for any family that can be expressed as such an object, including families users write themselves.

Generally this option should be used only if the desired family is not one of the built-in estimators, because for those the entire path algorithm is in C++ and so is faster.

## Fitting Gaussian, binomial and Poisson GLMs

First, we fit ordinary least squares with the elastic net penalty both ways. We set up some fake data:

```{code-cell} ipython3
rng = np.random.default_rng(1)
x = rng.standard_normal((100, 5))
y = x[:, :2].sum(1) + rng.standard_normal(100)
```

To fit a linear regression by least squares we use the Gaussian family. The built-in estimator is `GaussNet`; we can also use `GLMNet` with `family=sm_family.Gaussian()` to fit the same model.

```{code-cell} ipython3
oldfit = GaussNet().fit(x, y)
newfit = GLMNet(family=sm_family.Gaussian()).fit(x, y)
```

Of course if we really wanted to fit this model we would use `GaussNet`, because it is faster. Here we want to show that the two are equivalent. The algorithms differ slightly, so the solutions agree up to the convergence thresholds; tightening the coordinate descent threshold `thresh` (and, for the IRLS loop of `GLMNet`, the Newton tolerance `epsnr`) makes them agree to near machine precision:

```{code-cell} ipython3
def compare(fast, general):
    m = min(len(fast.lambda_values_), len(general.lambda_values_))
    diff = lambda a, b: f'{np.max(np.abs(a[:m] - b[:m])):.1e}'
    return pd.Series({'lambdas (built-in / GLMNet)': f'{len(fast.lambda_values_)} / {len(general.lambda_values_)}',
                      'max |lambda diff|': diff(fast.lambda_values_, general.lambda_values_),
                      'max |coef diff|': diff(fast.coefs_, general.coefs_),
                      'max |intercept diff|': diff(fast.intercepts_, general.intercepts_)})

thresh, epsnr = 1e-18, 1e-14
oldfit = GaussNet(control=FastNetControl(thresh=thresh)).fit(x, y)
newfit = GLMNet(family=sm_family.Gaussian(),
                control=GLMNetControl(thresh=thresh, epsnr=epsnr, mxitnr=100)).fit(x, y)
compare(oldfit, newfit)
```

Next, the binomial and Poisson families:

```{code-cell} ipython3
biny = (y > 0).astype(float)    # binary data
cnty = np.ceil(np.exp(y))       # count data

pd.DataFrame({
    'binomial': compare(LogNet(control=FastNetControl(thresh=thresh)).fit(x, biny),
                        GLMNet(family=sm_family.Binomial(),
                               control=GLMNetControl(thresh=thresh, epsnr=epsnr, mxitnr=100)).fit(x, biny)),
    'poisson': compare(FishNet(control=FastNetControl(thresh=thresh)).fit(x, cnty),
                       GLMNet(family=sm_family.Poisson(),
                              control=GLMNetControl(thresh=thresh, epsnr=epsnr, mxitnr=100)).fit(x, cnty))})
```

The coefficients agree along the common part of the path. The two estimators decide when to stop the path slightly differently (based on the change in fraction of deviance explained), so one path may be a lambda or two longer.

### Timing comparisons

In the examples above, `GLMNet` with a family simply replicates the built-in estimators. For these GLMs we recommend the built-in estimators for computational efficiency. The table below compares the time to fit the whole path:

```{code-cell} ipython3
def time_fit(make, X, Y, reps=3):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        make().fit(X, Y)
        times.append(time.perf_counter() - start)
    return np.median(times)

rows = []
for n, p in [(100, 10), (500, 50), (1000, 100)]:
    X = rng.standard_normal((n, p))
    Y = X[:, :2].sum(1) + rng.standard_normal(n)
    Ybin = (Y > 0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for family, fast, general, response in [
                ('gaussian', GaussNet, sm_family.Gaussian, Y),
                ('binomial', LogNet, sm_family.Binomial, Ybin)]:
            t_fast = time_fit(fast, X, response)
            t_general = time_fit(lambda: GLMNet(family=general()), X, response)
            rows.append({'family': family, 'n': n, 'p': p,
                         'built-in (s)': t_fast, 'GLMNet (s)': t_general,
                         'ratio': t_general / t_fast})
pd.DataFrame(rows).round(4)
```

## Fitting other GLMs

The real power of `GLMNet` with a family object is in fitting GLMs other than the built-in ones. For example, performing probit regression with the elastic net penalty is as simple as:

```{code-cell} ipython3
newfit = GLMNet(family=sm_family.Binomial(link=sm_links.Probit())).fit(x, biny)
newfit.coefs_[-1]
```

For the *complementary log-log* link we would specify `family=sm_family.Binomial(link=sm_links.CLogLog())`.

We can fit nonlinear least-squares models by using a different link with the Gaussian family, for example `family=sm_family.Gaussian(link=sm_links.Log())`:

```{code-cell} ipython3
ypos = np.exp(y / 3)    # a positive response
newfit = GLMNet(family=sm_family.Gaussian(link=sm_links.Log())).fit(x, ypos)
newfit.coefs_[-1]
```

R's `quasipoisson()` family allows for overdispersion in count data. Its estimating equations, and hence its penalized coefficient path, are the same as for the Poisson family -- only the dispersion estimate differs -- so `family=sm_family.Poisson()` gives the same path.

The negative binomial is often used to model overdispersed count data (instead of Poisson regression), and is also easy. statsmodels parametrizes it by $\alpha = 1/\theta$, so R's `negative.binomial(theta = 5)` is:

```{code-cell} ipython3
newfit = GLMNet(family=sm_family.NegativeBinomial(alpha=1/5)).fit(x, cnty)
newfit.coefs_[-1]
```

Other statsmodels families, such as `Gamma`, `InverseGaussian` and `Tweedie`, work the same way:

```{code-cell} ipython3
newfit = GLMNet(family=sm_family.Gamma(link=sm_links.Log())).fit(x, ypos)
ax = newfit.coef_path_.plot()
ax.set_title('Gamma regression with log link');
```

A family that statsmodels does not provide can be written by specifying
its link, variance function and deviance; see
[Custom GLM families](custom_family.md).

## Fitted `GLMNet` objects

`GLMNet` and the built-in estimators share their interface: the fitted path is in `coefs_`, `intercepts_` and `lambda_values_`, a summary of degrees of freedom and fraction of deviance explained is in `summary_`, and `predict`, `interpolate_coefs`, `cross_validation_path` and `coef_path_.plot()` work the same way. For example, cross-validation for the probit model:

```{code-cell} ipython3
probit = GLMNet(family=sm_family.Binomial(link=sm_links.Probit())).fit(x, biny)
_, cvpath = probit.cross_validation_path(x, biny, cv=5)
ax = cvpath.plot(score='Binomial Deviance')
```

## Step size halving within IRLS

For the built-in non-Gaussian families, the C++ path algorithm solves the optimization problem via IRLS, taking a unit Newton step in each iteration. Because it is forced to take a unit step, this can result in non-convergence in some cases.

Here is an example of non-convergence for Poisson data. The statsmodels GLM fit converges and gives coefficients that are reasonably close to the truth (each is 0.25):

```{code-cell} ipython3
rng = np.random.default_rng(2020)
n, p = 100, 4
x = rng.uniform(5, 10, (n, p))
y = rng.poisson(np.exp(x.mean(1))).astype(float)

glmfit = sm.GLM(y, x, family=sm_family.Poisson()).fit()
glmfit.params
```

Fitting with `lambda_values=[0]` is equivalent to fitting an unregularized GLM. With the built-in `FishNet` the unit Newton steps diverge: the C++ code reports that the first (and only) lambda did not converge, and an empty model is returned, just as in R:

```{code-cell} ipython3
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    oldfit = FishNet(standardize=False, fit_intercept=False, lambda_values=[0.]).fit(x, y)
for w in caught:
    print(w.message)
oldfit.coefs_
```

`GLMNet` with a family object performs step size halving. After computing the Newton step, it checks whether the new solution has an infinite (or astronomically large) objective value or results in invalid $\eta$ or $\mu$; if so, it halves the step size repeatedly until these conditions no longer hold. (NumPy may warn about overflow while it evaluates the rejected steps.)

```{code-cell} ipython3
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    newfit = GLMNet(family=sm_family.Poisson(), standardize=False, fit_intercept=False,
                    lambda_values=[0.], control=GLMNetControl(mxitnr=50)).fit(x, y)
newfit.coefs_[0]
```

The coefficients are close to those from statsmodels, and are numerically indistinguishable once the convergence criteria are tightened in both fits:

```{code-cell} ipython3
thresh = 1e-15
glmfit = sm.GLM(y, x, family=sm_family.Poisson()).fit(tol=thresh, maxiter=100)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    newfit = GLMNet(family=sm_family.Poisson(), standardize=False, fit_intercept=False,
                    lambda_values=[0.],
                    control=GLMNetControl(mxitnr=100, thresh=thresh, epsnr=thresh)).fit(x, y)
np.max(np.abs(newfit.coefs_[0] - glmfit.params))
```
