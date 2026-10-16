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

# Unpenalized GLMs, as in the ISLP labs

The labs of
[An Introduction to Statistical Learning with Python](https://github.com/intro-stat-learning/ISLP_labs)
fit linear, logistic and Poisson regressions with statsmodels (`sm.OLS`,
`sm.GLM`), on designs built by ISLP's `ModelSpec`. glmstar's unpenalized `GLM`
gives the same analysis. Each item below works the same way for every family,
Cox's included:

| statsmodels | `GLM` |
|---|---|
| `results.params` | `params_` |
| `summarize(results)`, `results.summary()` | `summarize(fit)`, `fit.summary()`; also `summary_` |
| `results.conf_int()` | `conf_int()` |
| `results.scale` | `dispersion_` |
| `results.llf`, `.aic` | `loglike_`, `aic_`, `bic_` |
| `results.fittedvalues`, `.resid` | `fittedvalues_`, `resid_`, `residuals(kind)` |
| `results.predict(newX)` | `predict(newX)` |
| `get_prediction(newX).conf_int()` | `predict_conf_int(newX)`, `predict_se(newX)` |
| `get_influence().hat_matrix_diag` | `leverage(X)` |
| `anova_lm(results1, results2)` | `anova(fit1, fit2)` |

`GLM` takes the `ModelSpec` design as it is. Its constant `intercept` column
duplicates the model's own intercept, so `GLM` drops it with a warning, and
the coefficients carry the same names as in statsmodels.

```{code-cell} ipython3
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from sklearn.model_selection import KFold, cross_validate
from sklearn.base import clone

from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize, poly

from glmnet import GLM, anova
from glmnet.glm import GaussianGLM

warnings.filterwarnings('ignore', message="dropping the constant column 'intercept'")
```

## Linear regression (Chapter 3)

```{code-cell} ipython3
Boston = load_data('Boston')
y = Boston['medv']
X = MS(['lstat', 'age']).fit_transform(Boston)

fit = GLM(summarize=True).fit(X, y)
results = sm.OLS(y, X).fit()
pd.concat({'GLM': summarize(fit), 'OLS': summarize(results)}, axis=1)
```

The full summary, with the intervals of `conf_int`:

```{code-cell} ipython3
fit.summary()
```

Predictions at new points, with confidence intervals for the mean:

```{code-cell} ipython3
design = MS(['lstat', 'age']).fit(Boston)
new = design.transform(pd.DataFrame({'lstat': [5, 10, 15], 'age': [50, 50, 50]}))
pd.concat({'GLM': fit.predict_conf_int(new),
           'OLS': pd.DataFrame(results.get_prediction(new).conf_int(),
                               columns=['lower', 'upper'])}, axis=1)
```

Leverage, and the residuals against the fitted values:

```{code-cell} ipython3
np.abs(fit.leverage(X) - results.get_influence().hat_matrix_diag).max(), \
    np.argmax(fit.leverage(X))
```

```{code-cell} ipython3
np.abs(fit.resid_ - results.resid).max(), np.abs(fit.fittedvalues_ - results.fittedvalues).max()
```

### Comparing nested models

`anova` is the analysis of deviance, the same for every family: the drop in
deviance divided by the dispersion of the larger model, against $\chi^2$ on
the drop in residual degrees of freedom. For a linear model that statistic is
$\mathrm{df}$ times the $F$ statistic of `anova_lm`, and the two p-values agree
closely when the residual degrees of freedom are large.

```{code-cell} ipython3
X3 = MS([poly('lstat', degree=2), 'age']).fit_transform(Boston)
fit3 = GLM(summarize=True).fit(X3, y)
display(anova(fit, fit3))
anova_lm(results, sm.OLS(y, X3).fit())
```

## Logistic and Poisson regression (Chapter 4)

```{code-cell} ipython3
Smarket = load_data('Smarket')
allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
X = MS(allvars).fit_transform(Smarket)
y = (Smarket.Direction == 'Up').astype(float)

logit = GLM(family=sm.families.Binomial(), summarize=True).fit(X, y)
pd.concat({'GLM': summarize(logit),
           'sm.GLM': summarize(sm.GLM(y, X, family=sm.families.Binomial()).fit())}, axis=1)
```

```{code-cell} ipython3
probs = logit.predict(X)
labels = np.where(probs > 0.5, 'Up', 'Down')
pd.crosstab(labels, Smarket.Direction)
```

The Poisson regression of the bike-share counts, with sum-to-zero contrasts
for month and hour:

```{code-cell} ipython3
from ISLP.models import contrast
Bike = load_data('Bikeshare')
X2 = MS([contrast('mnth', 'sum'), contrast('hr', 'sum'),
         'workingday', 'temp', 'weathersit']).fit_transform(Bike)
pois = GLM(family=sm.families.Poisson(), summarize=True).fit(X2, Bike['bikers'])
pois_sm = sm.GLM(Bike['bikers'], X2, family=sm.families.Poisson()).fit()
np.abs(pois.params_ - pois_sm.params).max(), pois.loglike_ - pois_sm.llf, pois.aic_ - pois_sm.aic
```

## Cross-validation and the bootstrap (Chapter 5)

`GaussianGLM` is a scikit-learn regressor, so it goes straight into
`cross_validate`, where the lab uses ISLP's `sklearn_sm` wrapper of `sm.OLS`.

```{code-cell} ipython3
Auto = load_data('Auto')
cv = KFold(n_splits=10, shuffle=True, random_state=0)
for degree in range(1, 4):
    X = MS([poly('horsepower', degree)]).fit_transform(Auto)
    scores = cross_validate(GaussianGLM(), X, Auto['mpg'], cv=cv,
                            scoring='neg_mean_squared_error')
    print(degree, -scores['test_score'].mean())
```

The bootstrap standard errors of the coefficients, against `summary_`:

```{code-cell} ipython3
X = MS(['horsepower']).fit_transform(Auto)
y = Auto['mpg']
rng = np.random.default_rng(0)
boot = []
for _ in range(1000):
    idx = rng.choice(len(y), len(y))
    boot.append(clone(GaussianGLM()).fit(X.iloc[idx], y.iloc[idx]).params_)
pd.DataFrame({'bootstrap': pd.DataFrame(boot).std(),
              'model': GaussianGLM(summarize=True).fit(X, y).summary_['std err']})
```
