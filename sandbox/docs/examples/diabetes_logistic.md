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
import numpy as np
import pandas as pd
import statsmodels.api as sm

from glmnet import GLMNet, GLM
from sklearn.base import clone
import statsmodels.api as sm
from ISLP.models import summarize
import logging
logging.basicConfig(filename='log.txt', level=logging.INFO)
import rpy2
%load_ext rpy2.ipython
```

```{code-cell} ipython3
%%R -o X,Y,N
#install.packages('lars', repo='http://cloud.r-project.org')
library(lars)
library(glmnet)
data(diabetes)
X = model.matrix(lm(y ~ x, data=diabetes))[,-1]
N = colnames(diabetes$x)
Y = diabetes$y
system.time(glmnet:::glmnet.path(X, Y>140, family=binomial,alpha=0.4))
```

```{code-cell} ipython3
X = pd.DataFrame(X, columns=N)
```

```{code-cell} ipython3
yb = Y > 140
```

```{code-cell} ipython3
%%timeit 
G4 = GLMNet(family=sm.families.Binomial(), alpha=0.4 )
G4.fit(X, yb)
```

```{code-cell} ipython3
G4 = GLMNet(family=sm.families.Binomial(), alpha=0.4 )
G4.fit(X, yb)
G4.plot_coefficients()
G4.coefs_.shape
```

```{code-cell} ipython3
G = GLM(family=sm.families.Binomial(), summarize=True)
G.fit(X, yb)
G.summary_
```

```{code-cell} ipython3
X_ = G.design_ @ np.identity(G.design_.shape[1])
X_ = pd.DataFrame(X_, columns=G.summary_.index)
from ISLP.models import summarize
summarize(sm.GLM(yb, X_, 
                 family=sm.families.Binomial()).fit())
```

```{code-cell} ipython3
X_.shape, len(G.summary_.index)
```

```{code-cell} ipython3

```
