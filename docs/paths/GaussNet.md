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

# GaussNet

The `GaussNet` class implements a linear regression model that uses
**elastic net regularization**. It's designed for modeling continuous
response variables (i.e., data that follows a Gaussian distribution).

---

## Data Requirements

The model expects the following inputs:

* **`X`**: A 2D array-like object (like a NumPy array or pandas DataFrame) with the shape `(n_samples, n_features)`. This contains your continuous predictor variables.
* **`y`**: A 1D array-like object (like a NumPy array or pandas Series) with the shape `(n_samples,)`. This contains your continuous response variable.

---

## Lasso Regression (`alpha=1.0`)

Here's a step-by-step example of how to use `GaussNet`. By default, it performs Lasso regression (`alpha=1.0`).

First, we import the necessary libraries.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from glmnet.data import make_dataset
from glmnet.paths.gaussnet import GaussNet
```

### Generate a synthetic dataset

Next, we'll create a dataset suitable for linear regression using `make_dataset`.

```{code-cell} ipython3
X, y, _, _ = make_dataset(GaussNet, n_samples=100, n_features=10, snr=5)
```

### Initialize and fit the model

Now, we initialize the `GaussNet` model and fit it to our data. The model is automatically fit over an entire path of regularization strengths (lambda values).

```{code-cell} ipython3
model = GaussNet()
model.fit(X, y)
```

### Plot the coefficient paths

Finally, we can visualize how each feature's coefficient changes as regularization increases. This plot is essential for understanding which variables are most important at different levels of penalization.

```{code-cell} ipython3
ax = model.coef_path_.plot()
ax.set_title("Coefficient Path for Lasso (alpha=1.0)");
```

## Elastic Net Regression (`alpha=0.4`)

The `alpha` parameter controls the mix between L1 (Lasso) and L2
(Ridge) regularization. Setting **`alpha=0.4`** creates an elastic net
model that is a blend of both penalties (specifically, 40% L1 and 60%
L2).


+++

### Initialize and fit the elastic net model

Here, we specify `alpha=0.4` when creating the `GaussNet` instance to build an elastic net model. Then, we fit it to the data.

```{code-cell} ipython3
model_elastic = GaussNet(alpha=0.4)
model_elastic.fit(X, y)
```

### Plot the coefficient paths

The resulting plot shows the coefficient paths for the elastic net model. You can compare these paths to the pure Lasso example to see the effect of the L2 penalty.

```{code-cell} ipython3
ax = model_elastic.coef_path_.plot()
ax.set_title("Coefficient Path for Elastic Net (alpha=0.4)")
```

-----

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.gaussnet.GaussNet
    :members:
    :inherited-members:
    :show-inheritance:
```
