---
jupytext:
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

```{code-cell} ipython3
import rpy2
%load_ext rpy2.ipython

import numpy as np
from glmnet.glmnet import RegGLM
rng = np.random.default_rng(0)
```

```{code-cell} ipython3
n, p = 1000, 50
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:2] = [1,2]
y = rng.standard_normal(n) + X @ beta
W = np.ones(n) # rng.uniform(1, 2, size=(n,))
```

```{code-cell} ipython3
%%R -i X,y,W,n,p -o coef_,intercept_
library(glmnet)
G = glmnet(X, y, standardize=FALSE, intercept=TRUE, weights=W)
B = predict(G, s=2 / sqrt(n), type='coef', exact=TRUE, x=X, y=y, weights=W)
coef_ = B[2:(p+1)]
intercept_ = B[1]
```

```{code-cell} ipython3
coef_
```

```{code-cell} ipython3
G = RegGLM(lambda_val=2 / np.sqrt(n), standardize=False, fit_intercept=True)
G.fit(X, y, sample_weight=W)
G.coef_
```

```{code-cell} ipython3
np.testing.assert_allclose(G.intercept_, intercept_)
```

```{code-cell} ipython3
np.testing.assert_allclose(G.coef_, coef_, rtol=1e-5, atol=1e-5)
```

```{code-cell} ipython3
from sklearn.model_selection import cross_validate
cross_validate(G, X, y, cv=5)
```

```{code-cell} ipython3

```
