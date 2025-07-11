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

# Regularized Cox Regression

This vignette describes how one can use the `glmnet` package to fit regularized Cox models.

The Cox proportional hazards model is commonly used for the study of the relationship between predictor variables and survival time. In the usual survival analysis framework, we have data of the form $(y_1, x_1, \delta_1), \ldots, (y_n, x_n, \delta_n)$ where $y_i$, the observed time, is a time of failure if $\delta_i$ is 1 or a right-censored time if $\delta_i$ is 0. We also let $t_1 < t_2 < \ldots < t_m$ be the increasing list of unique failure times, and let $j(i)$ denote the index of the observation failing at time $t_i$.

The Cox model assumes a semi-parametric form for the hazard

$$
h_i(t) = h_0(t) e^{x_i^T \beta},
$$

where $h_i(t)$ is the hazard for patient $i$ at time $t$, $h_0(t)$ is a shared baseline hazard, and $\beta$ is a fixed, length $p$ vector. In the classic setting $n \geq p$, inference is made via the partial likelihood

$$
L(\beta) = \prod_{i=1}^m \frac{e^{x_{j(i)}^T \beta}}{\sum_{j \in R_i} e^{x_j^T \beta}},
$$

where $R_i$ is the set of indices $j$ with $y_j \geq t_i$ (those at risk at time $t_i$).

Note there is no intercept in the Cox model as it is built into the baseline hazard, and like it, would cancel in the partial likelihood.

In `glmnet`, we penalize the negative log of the partial likelihood with an elastic net penalty.

## Basic usage for right-censored data

We use synthetic data for illustration. `X` must be an $n\times p$ matrix of covariate values --- each row corresponds to a patient and each column a covariate. `y` is an $n \times 2$ matrix, with a column `"time"` of failure/censoring times, and `"status"` a 0/1 indicator, with 1 meaning the time is a failure time, and 0 a censoring time.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glmnet import CoxNet
from glmnet.cox import CoxFamilySpec
from glmnet.data import make_survival

# Generate synthetic survival data
X, y, coef = make_survival(n_samples=100, n_features=20, 
                          n_informative=5, snr=3.0, 
                          random_state=42)
print("First 5 rows of survival data:")
print(y.head())
```

We apply the `CoxNet` function to compute the solution path under default settings:

```{code-cell} ipython3
# Create Cox family specification
family = CoxFamilySpec(y, event_id='event', status_id='status', tie_breaking='efron')
fit = CoxNet(family=family).fit(X, y)
```

All the standard options such as `alpha`, `weights`, `nlambda` and `standardize` apply, and their usage is similar as in the Gaussian case.

We can plot the coefficients with the `plot` method:

```{code-cell} ipython3
ax = fit.coef_path_.plot()
ax.set_title('Coefficient Paths for Cox Regression')
```

As before, we can extract the coefficients at certain values of $\lambda$:

```{code-cell} ipython3
coefs, intercept = fit.interpolate_coefs(0.05)
coefs
```

Since the Cox Model is not commonly used for prediction, we do not give an illustrative example on prediction. If needed, users can refer to the help file by typing `help(CoxNet.predict)`.

### Cross-validation

The `cross_validation_path` method can be used to compute $K$-fold cross-validation (CV) for the Cox model. The usage is similar to that for other families except for two main differences.

First, `type_measure` only supports `"deviance"` (also default) which gives the partial-likelihood, and `"C"`, which gives the Harrell *C index*. This is like the area under the curve (AUC) measure of concordance for survival data, but only considers comparable pairs. Pure concordance would record the fraction of pairs for which the order of the death times agree with the order of the predicted risk. However, with survival data, if an observation is right censored at a time *before* another observation's death time, they are not comparable.

The code below illustrates how one can perform cross-validation using the Harrell C index. Note that unlike most error measures, a higher C index means better prediction performance.

```{code-cell} ipython3
cvfit = CoxNet(family=family).fit(X, y)
_, cvpath = cvfit.cross_validation_path(X, y, cv=5)
```

Once fit, we can view the optimal $\lambda$ value and a cross validated error plot to help evaluate our model.

```{code-cell} ipython3
score = 'Cox Deviance'
ax = cvpath.plot(score=score) # C index
ax.set_title('Cross-validation Results')
```

As with other families, the left vertical line in our plot shows us where the CV-error curve hits its minimum. The right vertical line shows us the most regularized model with CV-error within 1 standard deviation of the minimum. We also extract such optimal $\lambda$'s:

```{code-cell} ipython3
lambda_min = cvpath.index_best[score]
lambda_1se = cvpath.index_1se[score]
lambda_min, lambda_1se
```

Second, the option `grouped = True` (default) obtains the CV partial likelihood for the Kth fold by subtraction, i.e. by subtracting the log partial likelihood evaluated on the full dataset from that evaluated on the $(K-1)/K$ dataset. This makes more efficient use of risk sets. With `grouped = False` the log partial likelihood is computed only on the $K$th fold, which is only reasonable if each fold has a large number of observations.

### Handling of ties

`glmnet` handles ties in survival time using either the Breslow or the Efron approximation. The choice of tie-breaking method can affect the coefficient estimates when there are tied event times.

```{code-cell} ipython3
# Generate data with ties using make_survival
X, y, coef = make_survival(n_samples=500, n_features=15, 
                           n_informative=5, snr=3.0,
                           random_state=42, discretize=True)


# Fit with Breslow approximation
family_breslow = CoxFamilySpec(y, event_id='event', status_id='status', tie_breaking='breslow')
fit_breslow = CoxNet(family=family_breslow).fit(X, y)

# Fit with Efron approximation
family_efron = CoxFamilySpec(y, event_id='event', status_id='status', tie_breaking='efron')
fit_efron = CoxNet(family=family_efron).fit(X, y)

# Compare coefficients at lambda=0
coefs_breslow, _ = fit_breslow.interpolate_coefs(0)
coefs_efron, _ = fit_efron.interpolate_coefs(0)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(coefs_breslow, coefs_efron);
ax.set_title('Comparison of Efron vs. Breslow tie-breaking methods')
```

The Breslow approximation is generally faster but may be less accurate when there are many ties. The Efron approximation provides a more accurate estimate of the partial likelihood when ties are present, but is computationally more intensive.

## Cox models for start-stop data

Since version 4.1 `glmnet` can fit models where the response is a (start, stop] time interval. As explained in Therneau & Grambsch (2000), the ability to work with start-stop responses opens the door to fitting regularized Cox models with

* time-dependent covariates,
* time-dependent strata,
* left truncation,
* multiple time scales,
* multiple events per subject,
* independent increment, marginal, and conditional models for correlated data, and
* various forms of case-cohort models.

The code below shows how to create a response of this type and how to fit such a model with `glmnet`.

```{code-cell} ipython3
X, yss, coef = make_survival(n_samples=200, n_features=15, 
                            n_informative=5, snr=3.0, 
                            start_id=True, discretize=True, 
                            random_state=42)
```

The start-stop data looks as follows:

```{code-cell} ipython3
yss.head()
```

Let's fit a regularized Cox model with start-stop data:

```{code-cell} ipython3
family = CoxFamilySpec(yss, event_id='event', status_id='status', start_id='start', tie_breaking='efron')
fit = CoxNet(family=family).fit(X, yss)
```

`cross_validation_path` works with start-stop data too:

```{code-cell} ipython3
_, cvpath = fit.cross_validation_path(X, yss, cv=5)
ax = cvpath.plot(score=score)
ax.set_title('Cross-validation Results for Start-Stop Data');
```

## Stratified Cox models

One extension of the Cox regression model is to allow for strata that divide the observations into disjoint groups. Each group has its own baseline hazard function, but the groups share the same coefficient vector for the covariates provided by the design matrix `X`.

`glmnet` can fit stratified Cox models with the elastic net penalty. Since `glmnet` does not use a model formula, we achieve this by adding a strata column to the response DataFrame.

```{code-cell} ipython3
rng = np.random.default_rng(0)
strata = rng.choice(range(1, 6), size=(yss.shape[0],))
y2 = yss.copy()
y2['strata'] = strata
print("First 6 rows of stratified data:")
print(y2.head(6))

# Fit stratified Cox model (commented out until implementation)
family = CoxFamilySpec(y2, event_id='event', status_id='status', strata_id='strata', tie_breaking='efron')
fit = CoxNet(family=family).fit(X, y2)

# Cross-validation with stratified data (commented out until implementation)
cv_fit = CoxNet(family=family).fit(X, y2)
_, cvpath = cv_fit.cross_validation_path(X, y2, cv=5)
ax = cvpath.plot(score='Cox Deviance')
ax.set_title('Cross-validation Results for Stratified Cox Model')
```

## Plotting survival curves

Fitting a regularized Cox model using `CoxNet` returns an object that can be used for prediction and survival curve plotting. The `predict` method allows the user to get survival predictions from the model.

```{code-cell} ipython3
# Get survival predictions for specific lambda value
lambda_val = 0.05
predictions = fit.predict(X[:5], interpolation_grid=lambda_val)
print("Survival predictions for first 5 individuals:")
print(predictions)
```

To be consistent with other methods in `glmnet`, if the `interpolation_grid` parameter is not specified, predictions are returned for the entire `lambda` sequence.

```{code-cell} ipython3
# Get predictions for all lambda values
all_predictions = fit.predict(X[:3])
print(f"Shape of predictions: {all_predictions.shape}")
print(f"Number of lambda values: {len(fit.lambda_values_)}")
```

The `predict` method is available for cross-validation objects as well. By default, the lambda value chosen is the "lambda.1se" value stored in the CV object.

```{code-cell} ipython3
# Predictions using cross-validation object
cv_predictions = fit.predict(X[:5])
cv_predictions.shape
```

## References

1. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

2. Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2011). Regularization paths for Cox's proportional hazards model via coordinate descent. *Journal of Statistical Software*, 39(5), 1-13.

3. Therneau, T. M., & Grambsch, P. M. (2000). *Modeling survival data: extending the Cox model*. Springer Science & Business Media.

---

*This document adapts the R glmnet vignette for the Python glmnet package. The original R vignette was written by Kenneth Tay, Noah Simon, Jerome Friedman, Trevor Hastie, Rob Tibshirani, and Balasubramanian Narasimhan.*

```{code-cell} ipython3

```
