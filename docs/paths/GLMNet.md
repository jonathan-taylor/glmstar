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

# GLMNet

The `GLMNet` class is the base estimator for generalized linear models with elastic net regularization. It is not typically used directly, but forms the foundation for specialized classes like `GaussNet`, `LogNet`, `FishNet`, and `CoxNet`.

**Data requirements:**

The `GLMNet` class is a general-purpose estimator for penalized generalized linear models. It expects a feature matrix `X` (NumPy array or pandas DataFrame) and a response vector or DataFrame `y`. The type and structure of `y` depend on the specific family (e.g., Gaussian, binomial, Poisson) you use with `GLMNet`. For most use cases, you should use a more specific subclass (like `GaussNet` or `LogNet`) that sets the family automatically. These specific subclasses are faster than the flexible `GLMNet` method.

## Example Usage

```{code-cell} ipython3
from copy import copy
import numpy as np
from glmnet.data import make_dataset
from glmnet.glmnet import GLMNet
from glmnet.family import BinomFamilySpec
```

 We're also going to implement our own link function so we need some additional imports from `scipy` and `statsmodels`

```{code-cell} ipython3
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.genmod.families.links as L
```

```{code-cell} ipython3
X, y, coef, intercept = make_dataset(GLMNet(), n_samples=100, n_features=10, snr=5, random_state=3)
model = GLMNet()
model.fit(X, y)
print(model.coefs_.shape)
```

## Binary Data Example

For binary classification problems, you can use `GLMNet` with a binomial family:

```{code-cell} ipython3
model = GLMNet(family=BinomFamilySpec())
X, y, coef, intercept = make_dataset(model, n_samples=100, n_features=10, snr=5, random_state=3)
model.fit(X, y)
print(f"Coefficient shape: {model.coefs_.shape}")
print(f"Coefficient shape: {model.intercepts_.shape}")
```

## Custom link example

We can also create our own GLM family from `statsmodels`. Here is an implementation of a link function:

```{code-cell} ipython3
class TLink(L.Link):
    """
    A custom link function for a t-distribution.

    The inverse link is the cumulative distribution function (CDF) of a
    t-distribution.

    Parameters
    ----------
    df : float
        The degrees of freedom for the t-distribution.
    """
    def __init__(self, df=10):
        super(TLink, self).__init__()
        self.df = df

    def __call__(self, p):
        """
        The link function. This is the Percent Point Function (PPF) or
        quantile function of the t-distribution.

        g(p) = t.ppf(p, df)
        """
        # Clip probabilities to avoid inf values at 0 or 1
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return stats.t.ppf(p, self.df)

    def inverse(self, z):
        """
        The inverse link function. This is the Cumulative Distribution
        Function (CDF) of the t-distribution.

        g^{-1}(z) = t.cdf(z, df)
        """
        return stats.t.cdf(z, self.df)

    def inverse_deriv(self, z):
        """
        The derivative of the inverse link function. This is the
        Probability Density Function (PDF) of the t-distribution.

        d/dz g^{-1}(z) = t.pdf(z, df)
        """
        return stats.t.pdf(z, self.df)
```

The `Binomial` family in `statsmodels` has a fixed set of links so we must subclass it to use our custom link:

```{code-cell} ipython3
class BinomialTFamily(sm.families.Binomial):
    """
    A Binomial family that allows for the custom TLink.
    """
    def __init__(self, link=None, **kwargs):
        # Add our custom link to the list of safe links
        self.links = copy(self.links)
        self.links.append(TLink)
        # Initialize the parent Binomial class
        super(BinomialTFamily, self).__init__(link=link, **kwargs)
        
        
        
```

Now let's use it to fit `GLMNet`. 

```{code-cell} ipython3
t_link_10_df = TLink(df=10)
binomial_t_family = BinomialTFamily(link=t_link_10_df)
```

```{code-cell} ipython3
model = GLMNet(family=BinomFamilySpec(base=binomial_t_family))
X, y, coef, intercept = make_dataset(model, n_samples=100, n_features=10, snr=5, random_state=3)
model.fit(X, y)
print(f"Coefficient shape: {model.coefs_.shape}")
print(f"Coefficient shape: {model.intercepts_.shape}")
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.glmnet.GLMNet
    :members:
    :inherited-members:
    :show-inheritance:
```
