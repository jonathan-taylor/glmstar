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
from glmnet import GLMNet, RegGLM, GLM
from sklearn.base import clone
import statsmodels.api as sm
from ISLP.models import summarize
import logging
logging.basicConfig(filename='log.txt', level=logging.DEBUG)
from sklearn.metrics import accuracy_score

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
plot(glmnet(X, Y, family=gaussian(), alpha=0.4), xvar='dev')
system.time(cv.glmnet(X, Y, family=gaussian(), 
                      alpha=0.4, nfolds=10))
```

```{code-cell} ipython3
X = pd.DataFrame(X, columns=N)
G = GLMNet(alpha=.4)
G.fit(X, Y)
```

```{code-cell} ipython3
G.plot_coefficients(legend=True, xvar='dev')
```

```{code-cell} ipython3
#%%prun 
cv_scores, lambda_best, lambda_1se = G.cross_validation_path(X, Y, cv=10, alignment='lambda')
cv_scores
```

```{code-cell} ipython3
#%%timeit
G.control.logging = True
G.cross_validation_path(X, Y, cv=50, alignment='lambda')
```

```{code-cell} ipython3
G.cv_scores_
```

```{code-cell} ipython3
G.plot_coefficients(xvar='norm');
```

```{code-cell} ipython3
G.plot_coefficients(xvar='dev');
```

```{code-cell} ipython3
G.plot_cross_validation()
```

```{code-cell} ipython3
G2 = clone(G)
G2.fit(X, Y)
G2.cross_validation_path(X, Y, cv=10, alignment='fraction')
G2.plot_cross_validation(c='green', label='My label', legend=True, col_1se='blue', score='Mean Absolute Error', xvar='dev')
```

```{code-cell} ipython3
yb = Y > 140
X.insert(0, 'intercept', np.ones(Y.shape[0]))
glm = sm.GLM(yb,X,family=sm.families.Binomial())
results=glm.fit()
summarize(results)
```

```{code-cell} ipython3
from glmnet import GLM, RegGLM

#G3 = GLMNetPath(alpha=.4,family=sm.families.Binomial())
G3 = GLM(family=sm.families.Binomial(), summarize=True)
G3.fit(X.drop(columns=['intercept']), yb)
G3.summary_
```

```{code-cell} ipython3
np.isnan(yb).sum()
```

```{code-cell} ipython3
G3 = RegGLM(family=sm.families.Binomial(), alpha=0.4, lambda_val=0.02081227758950013)
G3.fit(X.drop(columns=['intercept']), yb)
G3.coef_
G3.null_deviance_
```

```{code-cell} ipython3
G4 = GLMNet(family=sm.families.Binomial(), alpha=0.4 )
G4.fit(X.drop(columns=['intercept']), yb)
G4.cross_validation_path(X.drop(columns=['intercept']), yb, cv=10, alignment='lambda')
```

```{code-cell} ipython3
G4.summary_
```

```{code-cell} ipython3
%%R
print(glmnet(X, Y>140, family='binomial', alpha=0.4))
```

```{code-cell} ipython3
G4.plot_cross_validation()
```

```{code-cell} ipython3
%%R
plot(cv.glmnet(X, Y>140, family='binomial', alpha=0.4, type.measure='auc'))
```

```{code-cell} ipython3
G4.plot_cross_validation(score='Accuracy')
```

```{code-cell} ipython3
G4.plot_cross_validation(score='AUC')
```

```{code-cell} ipython3
def misclass_score(y, yhat, sample_weight):
    # y is assumed binary
    # yhat will be on response scale
    label = yhat > 0.5
    return 1 - accuracy_score(y, 
                              label, sample_weight=sample_weight, normalize=True)
    
G5 = GLMNet(family=sm.families.Binomial(), alpha=0.4 )
G5.fit(X.drop(columns=['intercept']), yb)
G5.cross_validation_path(X.drop(columns=['intercept']), yb, cv=10, alignment='lambda',
                         scorers=[('Misclassification Error', misclass_score, 'min')])
G5.plot_cross_validation(score='Misclassification Error')
```

```{code-cell} ipython3
def my_score(y, yhat, sample_weight):
    # y is assumed binary
    # yhat will be on response scale
    label = yhat > 0.6
    return 1 - accuracy_score(y, 
                              label, sample_weight=sample_weight, normalize=True)
    
G6 = GLMNet(family=sm.families.Binomial(), alpha=0.4 )
G6.fit(X.drop(columns=['intercept']), yb)
G6.cross_validation_path(X.drop(columns=['intercept']), yb, cv=10, alignment='lambda',
                         scorers=[('My Error', my_score, 'min')])
G6.plot_cross_validation(score='My Error')
G6.cv_scores_
```

```{code-cell} ipython3

```
