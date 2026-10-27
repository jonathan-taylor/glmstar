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

# Linear hypotheses and ANOVA tables

For an unpenalized `GLM` or `CoxLM`, `glmnet.hypothesis` has:

- `linear_hypothesis(fit, L, rhs)`: the Wald test of `L beta = rhs`, as R's
  `car::linearHypothesis`;
- `anova_table(estimator, spec, X, y, type)`: the ANOVA table, by term, of
  the design of a modelmatrix `ModelSpec`, sequential (type I, R's
  `anova(lm)`) or each term last (type III);
- `sandwich_covariance(fit, meat=..., groups=...)`: the sandwich
  covariance with a meat of one's own, built from the observations' scores
  (`score_contributions`), or clustered.

Each takes the covariance of the coefficients as `covariance=`: the fit's
(the default), `'model'`, `'sandwich'` (HC0), or an array such as a
`sandwich_covariance`. The statistics are t and F, on the residual degrees
of freedom, for least squares with a model-based covariance; otherwise z
and chi-squared.

We use the `Carseats` data of ISLR: the sales of child car seats by their
price and the quality of their shelving location.

```{code-cell} ipython3
import numpy as np
from ISLP import load_data
from modelmatrix import ModelSpec as MS, contrast

from glmnet import GLM
from glmnet.glm import summarize, leverage
from glmnet.hypothesis import (linear_hypothesis, anova_table, sandwich_covariance,
                               score_contributions)

Carseats = load_data('Carseats')
spec = MS(['Price', 'ShelveLoc', ('Price', 'ShelveLoc')]).fit(Carseats)
X = spec.transform(Carseats)
fit = GLM(summarize=True).fit(X, Carseats['Sales'])
summarize(fit)
```

## A linear hypothesis

`L` is given by the names of the coefficients, those of `summarize`: a
dict is a row, here that good and medium shelving locations sell the same
(at a price of 0, as there is an interaction):

```{code-cell} ipython3
linear_hypothesis(fit, {'ShelveLoc[Good]': 1, 'ShelveLoc[Medium]': -1})
```

A list of rows is tested jointly, here that the slope in price is the same
in every location (the F test of the interaction):

```{code-cell} ipython3
linear_hypothesis(fit, [{'Price:ShelveLoc[Good]': 1}, {'Price:ShelveLoc[Medium]': 1}])
```

A coefficient left out is 0: the intercept enters only when it is named.
The mean sales at a good location at a price of 120:

```{code-cell} ipython3
linear_hypothesis(fit, {'intercept': 1, 'Price': 120,
                        'ShelveLoc[Good]': 1, 'Price:ShelveLoc[Good]': 120}).estimates
```

`L` can also be an array, with a column for each coefficient, the
intercept first. An array without the intercept's column is an error,
rather than a guess.

## ANOVA tables

The sequential table (type I): each term after those before it, as R's
`anova(lm(Sales ~ Price * ShelveLoc))`:

```{code-cell} ipython3
anova_table(GLM(), spec, Carseats, Carseats['Sales'])
```

Each term after all the others (type III) is the Wald test that its
coefficients are 0. The test of a main effect in an interaction depends on
the contrasts of the factor: with sum contrasts it is the usual one (with
treatment contrasts `anova_table` warns):

```{code-cell} ipython3
spec_sum = MS(['Price', contrast('ShelveLoc', 'sum'), ('Price', contrast('ShelveLoc', 'sum'))])
anova_table(GLM(), spec_sum, Carseats, Carseats['Sales'], type='III')
```

For a GLM the tables are the analysis of deviance (chi-squared, or F for
least squares), and for a Cox model of the partial likelihood.

## A meat of one's own

`score_contributions(fit)` are the observations' scores of the
coefficients; the sum of their outer products is HC0's meat, the default of
`sandwich_covariance`. Another meat is a few lines, e.g. HC3's, which
scales each score by `1 / (1 - h)`, `h` its leverage:

```{code-cell} ipython3
S = score_contributions(fit).to_numpy()
h = np.asarray(leverage(fit, X))
meat = (S / (1 - h)[:, None]).T @ (S / (1 - h)[:, None])
HC3 = sandwich_covariance(fit, meat=meat)
summarize(fit, covariance=HC3)
```

`sandwich_covariance(fit, groups=g)` is the cluster-robust covariance
(without a small-sample correction). Any of these can be given to
`linear_hypothesis` or to `anova_table(type='III')` as `covariance=`.
