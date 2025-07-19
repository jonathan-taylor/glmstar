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

# Introduction

`glmnet` is a Python package that fits generalized linear models via penalized maximum likelihood. The regularization path is computed for the lasso or elastic net penalty at a grid of values (on the log scale) for the regularization parameter lambda. The algorithm is extremely fast and can exploit sparsity in the input matrix `X`. It fits linear, logistic and multinomial, poisson, and Cox regression models. It can also fit multi-response linear regression, generalized linear models for custom families, and relaxed lasso regression models. The package includes methods for prediction and plotting, and functions for cross-validation.

The original authors of glmnet are Jerome Friedman, Trevor Hastie, Rob Tibshirani, Balasubramanian Narasimhan, Kenneth Tay and Noah Simon, with contribution from Junyang Qian. This document adapts the R vignette for the Python `glmnet` package.

# Installation

You can install the Python `glmnet` package using pip:

```{code-cell} ipython3
%pip install glmstar
```

---

# Quick Start

Below is a quick demonstration of the main functions and outputs using the Python `glmnet` API. We will use synthetic data for illustration.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from glmnet.data import make_dataset
```

The following are the `GLMNet` estimators:

```{code-cell} ipython3
from glmnet import (GaussNet, 
                    LogNet, 
                    MultiClassNet, 
                    FishNet, 
                    MultiGaussNet)
```

Let's generate some synthetic regression data:

```{code-cell} ipython3
X_gaussian, y_gaussian, coef_gaussian, intercept_gaussian = make_dataset(GaussNet, n_samples=100, 
                                                                         n_features=20, 
                                                                         n_informative=5, 
                                                                         snr=5.0, 
                                                                         random_state=42)
```

All of the `GLMNet` objects are `sklearn` estimators
which use the `fit / predict` methods:

```{code-cell} ipython3
fit_gaussian = GaussNet().fit(X_gaussian, y_gaussian)
```

It is common to plot LASSO coefficient paths. The `GLMNet` objects
have a `coef_path_` object after fitting that can be plotted:

```{code-cell} ipython3
# Plot the coefficient paths
ax = fit_gaussian.coef_path_.plot();
ax.set_title('Coefficient Paths for Linear Regression');
```

## Caveat: using weights

For several `GLMNet` estimators we may have to pass several columns to specify
the response (e.g. for `CoxNet`). We also may want to specify offset
and possibly sample weights. For this reason, weights are specified
as a column of the `y` argument to fit, typically a `pd.DataFrame`.

Using this approach allows construction of several columns from `y` that are not simply the
response. This potentially use other uses beyond weights and offsets.

```{warning}
Since weights are found through the response, the `sample_weights` argument to `fit` is ignored!
```

Here's an example of using weight with this approach. The `X` argument will still typically
be an `np.ndarray`.

```{code-cell} ipython3
# Create offset and weights
weights = np.random.uniform(0.5, 2.0, size=X_gaussian.shape[0])
Df = pd.DataFrame({'response':y_gaussian,
                   'weight': weights})
```

Now fit with `response_id`, and `weight_id`:

```{code-cell} ipython3
fit_gaussian = GaussNet(response_id="response", weight_id="weight").fit(X_gaussian, Df)
```

# Linear Regression: GaussNet

The `R` package `glmnet` uses a single function to fit all versions of `GLMNet`. This package
follows the `sklearn` model more closely, using data-type specific estimators instead
of a single function to dispatch based on the data.

Perhaps the most model used in the package is the Gaussian linear model or "least squares". 
We'll just reimport the object here for emphasis.

```{code-cell} ipython3
from glmnet import GaussNet
```

We'll demonstrate this using synthetic data for illustration:

```{code-cell} ipython3
X_gaussian, y_gaussian, coef_gaussian, intercept_gaussian = make_dataset(GaussNet, n_samples=100, 
                                                                         n_features=20, 
                                                                         n_informative=5, 
                                                                         snr=5.0, 
                                                                         random_state=42)
```

Fit the model using the basic call:

```{code-cell} ipython3
fit_gaussian = GaussNet().fit(X_gaussian, y_gaussian)
```

The fitted object contains all the relevant information of the fitted model for further use. We can visualize the coefficients by plotting the regularization paths:

```{code-cell} ipython3
ax = fit_gaussian.coef_path_.plot();
ax.set_title('Coefficient Paths for Linear Regression');
```

Each curve corresponds to a variable. It shows the path of its coefficient against the $\ell_1$-norm of the whole coefficient vector as $\lambda$ varies.

We can obtain the model coefficients at specific $\lambda$ values by interpolation.
Let's get coefficients at $\lambda=0.1$:

```{code-cell} ipython3
coefs, intercept = fit_gaussian.interpolate_coefs(0.1)
coefs
```

Users can also make predictions at specific $\lambda$ values with new input data:

```{code-cell} ipython3
np.random.seed(0)
new_X = np.random.standard_normal((5, X_gaussian.shape[1]))
```

We can of course make predictions at specific $\lambda$ values:

```{code-cell} ipython3
lambda_values = [0.1, 0.05]
predictions = fit_gaussian.predict(new_X, interpolation_grid=lambda_values)
predictions.shape
```

## Cross-validation

Cross-validation is perhaps the simplest and most widely used method for selecting the optimal $\lambda$. We can perform cross-validation using the `cross_validation_path` method:

```{code-cell} ipython3
# Perform cross-validation
cvfit_gaussian = GaussNet().fit(X_gaussian, y_gaussian)
_, cvpath = cvfit_gaussian.cross_validation_path(X_gaussian, y_gaussian, cv=5)
```

```{code-cell} ipython3
ax = cvpath.plot(score='Mean Squared Error');
ax.set_title('Cross-validation Results');
```

The cross-validation curve shows the mean squared error (red dotted line) along with upper and lower standard deviation curves. Two special values are indicated:

- `lambda.min`: the value of $\lambda$ that gives minimum mean cross-validated error
- `lambda.1se`: the value of $\lambda$ that gives the most regularized model such that the cross-validated error is within one standard error of the minimum

We can get the model coefficients at these optimal values:

```{code-cell} ipython3
lambda_min = cvpath.index_best['Mean Squared Error']
coef_min, intercept_min = cvfit_gaussian.interpolate_coefs(lambda_min)
coef_min
```

## Predicting with fitted objects

We can extract the coefficients and make predictions for a fitted object at certain values of $\lambda$. Here is a simple example:

```{code-cell} ipython3
coef, intercept = fit_gaussian.interpolate_coefs(0.5)
coef, intercept
```

Users can make predictions from the fitted object. The `predict` method allows users to choose the type of prediction returned:

- "link" returns the fitted values (i.e. $\hat\beta_0 + x_i^T\hat\beta$)
- "response" gives the same output as "link" for "gaussian" family

For example, the following code gives the fitted values for the first 5 observations at $\lambda = 0.05$:

```{code-cell} ipython3
linpred = fit_gaussian.predict(X_gaussian[:5], 
                               interpolation_grid=0.05)
response = fit_gaussian.predict(X_gaussian[:5], 
                                interpolation_grid=0.05,
                                prediction_type='response')
assert np.allclose(linpred, response)
```

For classification problems, one can also ask for `class`.

# Commonly used function arguments

In this section we briefly describe some other useful arguments when calling `GaussNet`: `upper_limits`, `lower_limits`, `penalty_factor`, and `fit_intercept`.


`GaussNet` and other `GLMNet` objects provide various arguments for users to customize the fit:

- `alpha`: the elastic net mixing parameter $\alpha$, with range $\alpha \in [0,1]$. $\alpha = 1$ is lasso regression (default) and $\alpha = 0$ is ridge regression.
- `nlambda`: the number of $\lambda$ values in the sequence (default is 100).
- `lambda_values`: can be provided if the user wants to specify the lambda sequence.
- `standardize`: logical flag for $x$ variable standardization prior to fitting the model sequence.

## Setting the ElasticNet parameter $\alpha$

As an example, we set $\alpha = 0.2$ (more like a ridge regression), and give double weight to the latter half of the observations:

+++

Create the weights with double weight for latter half:

```{code-cell} ipython3
weights = np.ones(len(X_gaussian))
weights[len(X_gaussian)//2:] = 2
Df['weight'] = weights
```

Fit with $\alpha=0.2$ and custom weights:

```{code-cell} ipython3
fit_ridge_gaussian = GaussNet(alpha=0.2, nlambda=20, weight_id='weight').fit(X_gaussian, Df)
ax = fit_ridge_gaussian.coef_path_.plot();
ax.set_title('Ridge Regression (α=0.2) with Custom Weights');
```

## Upper and lower limits 

Suppose we want to fit our model but limit the coefficients to be bigger than -0.1 and less than 1.2:

```{code-cell} ipython3
# Fit with coefficient limits
lower, upper = -1, 0.5
fit_limited_gaussian = GaussNet(lower_limits=lower, upper_limits=upper).fit(X_gaussian, y_gaussian)
fit_limited_gaussian.coef_path_.plot();
plt.title(f'Coefficient Paths with Limits [{lower:.1f}, {upper:.1f}]');
```

Often we want the coefficients to be positive: to do so, we just need to specify `lower_limits = 0`:

```{code-cell} ipython3
# Fit with non-negative coefficients
fit_positive_gaussian = GaussNet(lower_limits=0).fit(X_gaussian, y_gaussian)
fit_positive_gaussian.coef_path_.plot();
plt.title('Non-negative Coefficient Paths');
```

The `penalty_factor` argument allows users to apply separate penalty factors to each coefficient. This is very useful when we have prior knowledge or preference over the variables:

```{code-cell} ipython3
# Set penalty factors for variables 1, 3, and 5 to be zero (no penalty)
penalty_factor = np.ones(X_gaussian.shape[1])
penalty_factor[[0, 2, 4]] = 0  # Variables 1, 3, 5 (0-indexed)

fit_penalty_gaussian = GaussNet(penalty_factor=penalty_factor).fit(X_gaussian, y_gaussian)
fit_penalty_gaussian.coef_path_.plot();
plt.title('Coefficient Paths with Custom Penalty Factors');
```

We see from the plot that the three variables with zero penalty factors always stay in the model, while the others follow typical regularization paths and are shrunk to zero eventually.

## Fitting with or without an intercept

The `fit_intercept` argument allows the user to decide if an intercept should be included in the model or not (it is never penalized). The default is `fit_intercept = True`:

```{code-cell} ipython3
# Fit without intercept
fit_no_intercept_gaussian = GaussNet(fit_intercept=False).fit(X_gaussian, y_gaussian)

# Compare with intercept
fit_with_intercept_gaussian = GaussNet(fit_intercept=True).fit(X_gaussian, y_gaussian)

print("Intercept values:")
print(f"With intercept: {fit_with_intercept_gaussian.intercepts_[0]:.4f}")
print(f"Without intercept: {fit_no_intercept_gaussian.intercepts_[0]:.4f}")
```

# Logistic Regression: LogNet

Logistic regression is a widely-used model when the response is binary. Suppose the response variable takes values in $\mathcal{G}=\{1,2\}$. Denote $y_i = I(g_i=1)$. We model

$$\mbox{Pr}(G=2|X=x)=\frac{e^{\beta_0+\beta^Tx}}{1+e^{\beta_0+\beta^Tx}},$$

which can be written in the following form:

$$\log\frac{\mbox{Pr}(G=2|X=x)}{\mbox{Pr}(G=1|X=x)}=\beta_0+\beta^Tx,$$

the so-called "logistic" or log-odds transformation.

The objective function for logistic regression is the penalized negative binomial log-likelihood:

$$
\min_{(\beta_0, \beta) \in \mathbb{R}^{p+1}} -\left[\frac{1}{N} \sum_{i=1}^N y_i \cdot (\beta_0 + x_i^T \beta) - \log (1+e^{(\beta_0+x_i^T \beta)})\right] + \lambda \big[ (1-\alpha)\|\beta\|_2^2/2 + \alpha\|\beta\|_1\big].
$$

```{code-cell} ipython3
from glmnet import LogNet
```

For illustration purposes, we generate synthetic data:

```{code-cell} ipython3
X_binomial, y_binomial, coef_binomial, intercept_binomial = make_dataset(LogNet, n_samples=200, 
                                                                         n_features=20, 
                                                                         n_informative=5, 
                                                                         snr=3.0, 
                                                                         random_state=42)
```

Use the existing binomial data for testing

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(X_binomial, y_binomial, test_size=0.5, random_state=42)
fit_logistic = LogNet().fit(X_binomial, y_binomial)
```

As before, we can print and plot the fitted object, extract the coefficients at specific $\lambda$'s and also make predictions:

```{code-cell} ipython3
ax = fit_logistic.coef_path_.plot();
ax.set_title('Logistic Regression Coefficient Paths');
```

## Predicting classes for `LogNet`

Prediction is a little different for logistic regression, mainly in the function argument `prediction_type`:

- "link" gives the linear predictors
- "response" gives the fitted probabilities
- "class" produces the class label corresponding to the maximum probability

First, we'll produce the usual linear predictors:

```{code-cell} ipython3
fit_logistic.predict(X_test[:5], interpolation_grid=0.05, prediction_type="link")
```

Next, the probabilities:

```{code-cell} ipython3
fit_logistic.predict(X_test[:5], interpolation_grid=0.05, prediction_type="response")
```

And finally the class predictions

```{code-cell} ipython3
fit_logistic.predict(X_test[:5], interpolation_grid=0.05, prediction_type="class")
```

For logistic regression, cross-validation has similar arguments and usage as Gaussian:

```{code-cell} ipython3
cvfit_logistic = LogNet().fit(X_binomial, y_binomial)
_, cvpath = cvfit_logistic.cross_validation_path(X_binomial, y_binomial, cv=5)
ax = cvpath.plot(score='Binomial Deviance');
ax.set_title('Cross-validation Results for Logistic Regression');
```

# Multinomial Regression: MultiClassNet

The multinomial model extends the binomial when the number of classes is more than two. Suppose the response variable has $K$ levels ${\cal G}=\{1,2,\ldots,K\}$. Here we model

$$\mbox{Pr}(G=k|X=x)=\frac{e^{\beta_{0k}+\beta_k^Tx}}{\sum_{\ell=1}^Ke^{\beta_{0\ell}+\beta_\ell^Tx}}.$$

There is a linear predictor for each class!

Let ${Y}$ be the $N \times K$ indicator response matrix, with elements $y_{i\ell} = I(g_i=\ell)$. Then the elastic net penalized negative log-likelihood function becomes

$$
\ell(\{\beta_{0k},\beta_{k}\}_1^K) = -\left[\frac{1}{N} \sum_{i=1}^N \Big(\sum_{k=1}^Ky_{il} (\beta_{0k} + x_i^T \beta_k)- \log \big(\sum_{\ell=1}^K e^{\beta_{0\ell}+x_i^T \beta_\ell}\big)\Big)\right] +\lambda \left[ (1-\alpha)\|\beta\|_F^2/2 + \alpha\sum_{j=1}^p\|\beta_j\|_q\right].
$$

We support two options for $q$: $q\in \{1,2\}$. When $q=1$, this is a lasso penalty on each of the parameters. When $q=2$, this is a grouped-lasso penalty on all the $K$ coefficients for a particular variable, which makes them all be zero or nonzero together.

```{code-cell} ipython3
from glmnet import MultiClassNet
```

For the `family = "multinomial"` case, usage is similar to that for `family = "binomial"`. We generate synthetic data:

```{code-cell} ipython3
X_multinomial, y_multinomial, coef_multinomial, intercept_multinomial = make_dataset(MultiClassNet, n_samples=100, 
                                                                                     n_features=20, 
                                                                                     n_informative=5, 
                                                                                     n_classes=3, 
                                                                                     snr=3.0, 
                                                                                     random_state=42)
```

Fitting is similar to all other `GLMNet` estimators:

```{code-cell} ipython3
fit_multinomial = MultiClassNet().fit(X_multinomial, y_multinomial)
```

For the plot, it will produce a figure showing the $\ell_2$-norm in one figure:

```{code-cell} ipython3
ax = fit_multinomial.coef_path_.plot();
ax.set_title('Multinomial Regression Coefficient Paths');
```

We can also do cross-validation:

```{code-cell} ipython3
cvfit_multinomial = MultiClassNet().fit(X_multinomial, y_multinomial)
_, cvpath = cvfit_multinomial.cross_validation_path(X_multinomial, y_multinomial, cv=10);
```

There are several scores available:

```{code-cell} ipython3
cvpath.scores.columns
```

Let's plot the deviance:

```{code-cell} ipython3
cvpath.plot(score='Multinomial Deviance');
ax.set_title('Cross-validation Results for Multinomial Regression');
```

Users may wish to predict at the optimally selected $\lambda$:

```{code-cell} ipython3
best_lambda = cvpath.index_best['Multinomial Deviance']
predictions = fit_multinomial.predict(X_multinomial[:10], 
                          interpolation_grid=best_lambda, 
                          prediction_type="class")
print("Class predictions at lambda_min:")
print(predictions)
```

# Multi-Response Regression: MultiGaussNet

The multi-response Gaussian family is useful when there are a number of (correlated) responses, also known as the "multi-task learning" problem. Here, a variable is either included in the model for all the responses, or excluded for all the responses.

As the name suggests, the response $y$ is not a vector but a matrix of quantitative responses. As a result, the coefficients at each value of lambda are also a matrix.

`glmnet` solves the problem

$$
\min_{(\beta_0, \beta) \in \mathbb{R}^{(p+1)\times K}}\frac{1}{2N} \sum_{i=1}^N \|y_i -\beta_0-\beta^T x_i\|^2_F+\lambda \left[ (1-\alpha)\|\beta\|_F^2/2 + \alpha\sum_{j=1}^p\|\beta_j\|_2\right].
$$

Here $\beta_j$ is the $j$th row of the $p\times K$ coefficient matrix $\beta$, and we replace the absolute penalty on each single coefficient by a group-lasso penalty on each coefficient $K$-vector $\beta_j$ for a single predictor.

```{code-cell} ipython3
from glmnet import MultiGaussNet
```

We use synthetic data for illustration:

```{code-cell} ipython3
X, y, coef_multigauss, intercept_multigauss = make_dataset(MultiGaussNet, n_samples=100, 
                                                           n_features=20, 
                                                           n_informative=5, 
                                                           n_targets=3, 
                                                           snr=5.0, 
                                                           random_state=42)
```

We now fit a regularized multi-response Gaussian model:

```{code-cell} ipython3
fit_multigaussian = MultiGaussNet().fit(X, y)
```

We can visualize the coefficients by plotting the $\ell_2$ norm of each variable's coefficient vector:

```{code-cell} ipython3
ax = fit_multigaussian.coef_path_.plot();
ax.set_title('Multi-response Linear Regression Coefficient Paths');
```

We can similarly extract the coefficients and make predictions at requested values of $\lambda$:

```{code-cell} ipython3
lambda_vals = [0.1, 0.01]
predictions = fit_multigaussian.predict(X[:5], interpolation_grid=lambda_vals)
predictions.shape
```

# Poisson Regression: FishNet

Poisson regression is used to model count data under the assumption of Poisson error, or otherwise non-negative data where the mean and variance are proportional. Like the Gaussian and binomial models, the Poisson distribution is a member of the exponential family of distributions. We usually model its positive mean on the log scale: $\log \mu(x) = \beta_0+\beta' x$.

The log-likelihood for observations $\{x_i,y_i\}_1^N$ is given by

$$
l(\beta|X, Y) = \sum_{i=1}^N \left(y_i (\beta_0+\beta^T x_i) - e^{\beta_0+\beta^Tx_i}\right).
$$

As before, we optimize the penalized log-likelihood:

$$
\min_{\beta_0,\beta} -\frac1N l(\beta|X, Y)  + \lambda \left((1-\alpha) \sum_{i=1}^N \beta_i^2/2 +\alpha \sum_{i=1}^N |\beta_i|\right).
$$

```{code-cell} ipython3
from glmnet import FishNet
```

We generate Poisson data:

```{code-cell} ipython3
X_poisson, y_poisson, coef_poisson, intercept_poisson = make_dataset(FishNet, n_samples=100, 
                                                                     n_features=20, 
                                                                     n_informative=5, 
                                                                     snr=3.0, 
                                                                     random_state=42)
```

Let's fit our model:

```{code-cell} ipython3
fit_poisson = FishNet().fit(X_poisson, y_poisson)
```

We plot the coefficients to have a first sense of the result:

```{code-cell} ipython3
ax = fit_poisson.coef_path_.plot();
ax.set_title('Poisson Regression Coefficient Paths');
```

As before, we can extract the coefficients and make predictions at certain $\lambda$'s:

```{code-cell} ipython3
fit_poisson.interpolate_coefs(interpolation_grid=1)
predictions = fit_poisson.predict(X_poisson[:5], interpolation_grid=[1, 0.1], prediction_type="response")
```

We may also use cross-validation to find the optimal $\lambda$'s:

```{code-cell} ipython3
# Perform cross-validation
cvfit_poisson = FishNet().fit(X_poisson, y_poisson)
_, cvpath = cvfit_poisson.cross_validation_path(X_poisson, y_poisson, cv=10)
ax = cvpath.plot(score='Poisson Deviance');
ax.set_title('Cross-validation Results for Poisson Regression');
```

# Cox Regression: CoxNet

The Cox proportional hazards model is commonly used for the study of the relationship between predictor variables and survival time. We have a separate vignette dedicated solely to fitting regularized Cox models with the `glmnet` package; please consult that vignette for details.

# Assessing models on test data

Once we have fit a series of models using `glmnet`, we often assess their performance on a set of evaluation or test data. We usually go through the process of building a prediction matrix, deciding on the performance measure, and computing these measures for a series of values for `lambda`.

## Performance measures

We can compute performance measures on a validation or test dataset. Here's an example for logistic regression:

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(X_binomial, y_binomial, test_size=0.5, random_state=42)
```

We'll first fit on the training data:

```{code-cell} ipython3
fit_logistic = LogNet().fit(X_train, y_train)
```

Next, we compute the path of scores
with the test data:

```{code-cell} ipython3
score_path = fit_logistic.score_path(X_test, y_test)
ax = score_path.plot(score='Binomial Deviance');
ax.set_title('Test Set Performance');
```

## ROC curves for binomial data

In the special case of binomial models, users often would like to see the ROC curve for validation or test data:

```{code-cell} ipython3
best_lambda = score_path.index_best['Binomial Deviance']
pred_probs = fit_logistic.predict(X_test, 
                                  interpolation_grid=best_lambda,
                                  prediction_type="response")
```

Let's compute an ROC curve

```{code-cell} ipython3
fpr, tpr, _ = roc_curve(y_test, pred_probs)
roc_auc = auc(fpr, tpr)
```

Finally, we plot the curve:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')
```

For classification problems, one can also ask for `class`.

+++

# Excluding features

Sometimes we want to filter variables before fitting the model. This can be done using the `exclude` argument. Below, we exclude the first, third and sixth variable

```{code-cell} ipython3
exclude_vars = [0,2,5] # 0-indexed
fit_excluded = GaussNet(exclude=exclude_vars).fit(X_gaussian, y_gaussian)
fit_excluded.coef_path_.plot()
assert np.allclose(fit_excluded.coefs_[:,exclude_vars], 0)
```

## Dynamically prefiltering features

You can also subclass any `GLMNet` class and override the `prefilter` method to dynamically exclude features.  
Below, we exclude all features `j` for which $X_j'y < 0$:

```{code-cell} ipython3
class PrefilterGaussNet(GaussNet):
    def prefilter(self, X, y):
        # Exclude features where the sum of X[:, j] * y is negative
        return np.nonzero(X.T @ y < 0)[0]
```

Let's now fit the model with prefiltering

```{code-cell} ipython3
model = PrefilterGaussNet()
model.fit(X_gaussian, y_gaussian)
print("Excluded features:", model.excluded_)
```

Let's check that the prefiltered features are indeed zeroed out:

```{code-cell} ipython3
np.allclose(model.coefs_[:,model.excluded_], 0)
```

The prefiltered features are combined with any specified by the `exclude` argument. 

```{code-cell} ipython3
model = PrefilterGaussNet(exclude=[0])
model.fit(X_gaussian, y_gaussian)
print("Excluded features:", model.excluded_)
np.allclose(model.coefs_[:,model.excluded_], 0)
```

# Other Package Features

## Offsets

Like other generalized linear models, `glmnet` allows for an "offset". This is a fixed vector of $N$ numbers that is added into the linear predictor. For example, you may have fitted some other logistic regression using other variables (and data), and now you want to see if the present variables can add further predictive power. To do this, you can use the predicted logit from the other model as an offset in the `glmnet` call.

```{code-cell} ipython3
offset = np.random.randn(X_gaussian.shape[0])
Df = pd.DataFrame({'response':y_gaussian,
                   'offset': offset})

# Fit with response_id, offset_id, and weight_id
fit_with_offset = GaussNet(response_id="response", 
                           offset_id='offset').fit(X_gaussian, Df)


# Compare with fit without offset
fit_no_offset = GaussNet(response_id='response').fit(X_gaussian, Df)

print("Intercept with offset:", fit_with_offset.intercepts_[0])
print("Intercept without offset:", fit_no_offset.intercepts_[0])
```

Of course, weights can also be added as above with a `weight_id` argument
to the `GLMNet` object.

+++

# Appendix: Convergence Criteria and Internal Parameters

The `glmnet` algorithm uses several internal parameters to control convergence and numerical stability:

- **Tolerance**: The algorithm stops when the relative change in the objective function is less than a specified tolerance.
- **Maximum iterations**: The maximum number of iterations for the coordinate descent algorithm.
- **Warm starts**: The algorithm uses warm starts to efficiently compute the entire regularization path.

These parameters can be adjusted if needed, though the defaults work well for most applications.

# References

1. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

2. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical learning with sparsity: the lasso and generalizations*. CRC press.

3. Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2011). Regularization paths for Cox's proportional hazards model via coordinate descent. *Journal of Statistical Software*, 39(5), 1-13.

---

*This document adapts the R glmnet vignette for the Python glmnet package. The original R vignette was written by Trevor Hastie, Junyang Qian, and Kenneth Tay.*
