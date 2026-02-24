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

# LogNet

The `LogNet` class fits a logistic regression model with elastic net regularization (binomial family). It is suitable for binary classification problems.

**Data requirements:**
> The `LogNet` class fits logistic regression models (binomial family) with elastic net regularization. It expects:
> - `X`: a 2D NumPy array or pandas DataFrame of shape `(n_samples, n_features)` (predictors).
> - `y`: a 1D array, Series, or DataFrame column of shape `(n_samples,)` containing binary labels. Labels can be 0/1, True/False, or two unique string values (e.g., 'A'/'B'). If string labels are provided, they will be automatically encoded.

## Example Usage

### Basic Example

```{code-cell} ipython3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from glmnet.data import make_dataset
from glmnet.paths.lognet import LogNet

X, y, coef, intercept = make_dataset(LogNet, n_samples=100, n_features=10, snr=5)
model = LogNet()
model.fit(X, y)
print(model.coefs_.shape)
```

### Example with String Labels ('A' and 'B')

```{code-cell} ipython3
# Generate synthetic data with string labels
np.random.seed(42)
n_samples, n_features = 200, 15

# Create feature matrix
X = np.random.randn(n_samples, n_features)

# Create binary labels as strings 'A' and 'B'
# Class 'A' for first half, 'B' for second half
y = np.array(['A'] * (n_samples // 2) + ['B'] * (n_samples // 2))

# Add some signal to make classification meaningful
# Make features 0-4 predictive of class 'A'
X[:n_samples//2, :5] += 0.5

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Fit the model
model = LogNet(alpha=0.5, lambda_values=np.logspace(-3, 0, 50))
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test, prediction_type='class')
y_pred_proba = model.predict_proba(X_test)

print(f"Predicted classes: {y_pred[:5]}")
print(f"Prediction probabilities shape: {y_pred_proba.shape}")
print(f"First 5 predictions: {y_pred[:5].shape}")
print(f"First 5 probability scores: {y_pred_proba[:5].shape}")
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.lognet.LogNet
    :members:
    :inherited-members:
    :show-inheritance:
```
