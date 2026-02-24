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

# MultiClassNet

The `MultiClassNet` class fits a multinomial logistic regression model with elastic net regularization (multinomial family). It is suitable for multi-class classification problems.

**Data requirements:**
> The `MultiClassNet` class fits multinomial logistic regression models for multi-class classification. It expects:
> - `X`: a 2D NumPy array or pandas DataFrame of shape `(n_samples, n_features)` (predictors).
> - `y`: a 1D array, Series, or DataFrame column of shape `(n_samples,)` containing integer or string class labels (with at least 2 unique values).

## Example Usage

### Basic Example

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.multiclassnet import MultiClassNet

X, y, coef, intercept = make_dataset(MultiClassNet, n_samples=100, n_features=10, 
                                    n_classes=3, snr=5)
model = MultiClassNet()
model.fit(X, y)
print(model.coefs_.shape)
```

### Example with String Labels ('cat', 'dog', 'mouse')

```{code-cell} ipython3
import numpy as np
import pandas as pd
from glmnet.paths.multiclassnet import MultiClassNet
from sklearn.model_selection import train_test_split

# Generate synthetic data with string labels
np.random.seed(42)
n_samples, n_features = 300, 20

# Create feature matrix
X = np.random.randn(n_samples, n_features)

# Create multi-class labels as strings
# 100 samples each for 'cat', 'dog', 'mouse'
y = np.array(['cat'] * 100 + ['dog'] * 100 + ['mouse'] * 100)

# Add some signal to make classification meaningful
# Make features 0-4 predictive of 'cat'
X[:100, :5] += 0.8
# Make features 5-9 predictive of 'dog'  
X[100:200, 5:10] += 0.8
# Make features 10-14 predictive of 'mouse'
X[200:, 10:15] += 0.8

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Fit the model
model = MultiClassNet(alpha=0.5, lambda_values=np.logspace(-3, 0, 50))
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test, prediction_type='class')
y_pred_proba = model.predict_proba(X_test)

print(f"Classes: {model.categories_}")
print(f"Predicted classes: {np.unique(y_pred)}")
print(f"Prediction probabilities shape: {y_pred_proba.shape}")
print(f"First 5 predictions: {y_pred[:5]}")
print(f"First 5 probability scores (shape): {y_pred_proba[:5].shape}")
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.multiclassnet.MultiClassNet
    :members:
    :inherited-members:
    :show-inheritance:
```

```{code-cell} ipython3

```
