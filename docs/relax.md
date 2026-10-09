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

# Relaxed fits

The lasso shrinks the coefficients of the variables it selects towards
zero. The *relaxed lasso* refits each selected model without the
penalty and blends the two fits:

$$\hat\beta_\gamma(\lambda) = \gamma\,\hat\beta(\lambda) + (1-\gamma)\,\hat\beta^{\rm relax}(\lambda), \qquad 0 \le \gamma \le 1,$$

where $\hat\beta^{\rm relax}(\lambda)$ is the unpenalized fit on the
variables active at $\lambda$. With $\gamma=1$ this is the lasso, and with
$\gamma=0$ it is the unpenalized refit. As in R's
`glmnet(..., relax=TRUE)`, every estimator takes `relax=True`.

```{code-cell} ipython3
import numpy as np
from glmnet import GaussNet

rng = np.random.default_rng(0)
n, p = 200, 30
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:5] = [2, -1.5, 1, 0.75, -0.5]
y = X @ beta + 2 * rng.standard_normal(n)

fit = GaussNet(relax=True).fit(X, y)
```

The fit holds both paths. `predict` and `interpolate_coefs` take `gamma`:

```{code-cell} ipython3
lam = fit.lambda_values_[20]
coef_lasso, _ = fit.interpolate_coefs(lam, gamma=1)
coef_relaxed, _ = fit.interpolate_coefs(lam, gamma=0)
np.round(np.c_[beta, coef_lasso, coef_relaxed][:8], 2)
```

The relaxed coefficients of the selected variables are not shrunk.

Each distinct active set along the path is refit once. Active sets with
more than `relax_maxp` variables (by default $n-3$, as in R) are not
refit.

## Choosing $\lambda$ and $\gamma$ by cross-validation

`cross_validation_path` then cross-validates $\gamma$ as well as
$\lambda$, as `cv.glmnet(..., relax=TRUE)` does. By default it uses
$\gamma \in \{0, 0.25, 0.5, 0.75, 1\}$.

```{code-cell} ipython3
_, cvpath = fit.cross_validation_path(X, y, cv=10)
ax = cvpath.plot(score='Mean Squared Error')
```

There is a curve for each value of $\gamma$. The chosen pairs are in
`index_best` (R's `lambda.min` and `gamma.min`) and `index_1se`:

```{code-cell} ipython3
cvpath.index_best
```

```{code-cell} ipython3
cvpath.index_1se
```

`cv_coefs` and `cv_predict` use the chosen pair directly, as R's
`coef(cvfit, s="lambda.min")` and `predict(cvfit, newx, s="lambda.min")`
do. Their default is the one standard error choice (`which='1se'`), as
in R:

```{code-cell} ipython3
coef_best, intercept_best = fit.cv_coefs(which='best')
np.nonzero(coef_best)[0]
```

```{code-cell} ipython3
fit.cv_predict(X[:5], which='best')
```

The coefficient path of a blend can be plotted with `relaxed_coef_path`:

```{code-cell} ipython3
ax = fit.relaxed_coef_path(gamma=0).plot()
```

The lasso's cross-validation results ($\gamma=1$) are still in
`fit.score_path_`, as they would be without `relax`.
