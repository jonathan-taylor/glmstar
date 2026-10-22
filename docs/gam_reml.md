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

# REML and ML for GAMs: the design

This page explains how `GAM(method='REML')` and `GAM(method='ML')` choose
the smoothing parameters (the code is `glmnet.gam.reml`). It writes the
criterion out in a few lines of JAX for a small example, minimizes it
"by hand", and checks that `GAM` gets the same answer. The same check is
a test, `tests/gam/test_gam_reml_byhand.py`.

The method is that of mgcv (Wood, 2011), and the notation follows Wood
(2017, chapter 6).

## The model

The linear predictor is $\eta = X\beta$. The columns of $X$ are the
intercept, then one block for each term: a spline's basis, a factor's
indicators, a random effect's group indicators. Term $j$ has a penalty
$\lambda_j \beta_j^T S_j \beta_j$, and $S_\lambda = \sum_j \lambda_j S_j$
puts the terms' penalties together, block diagonally. The penalized fit
minimizes

$$D_p(\beta) = D(\beta) + \beta^T S_\lambda \beta,$$

$D$ the deviance (with prior weights $a_i$).

The empirical Bayes view: the penalty is a Gaussian prior on the
coefficients,

$$\beta \sim N\big(0,\ \phi\, S_\lambda^{-}\big),$$

improper (flat) along the null space of $S_\lambda$. That null space is
spanned by the intercept, the unpenalized terms, and the functions a
penalty does not see, such as the linear part of a cubic spline. Call its
dimension $M$, and the rank of $S_j$, $r_j$. $\phi$ is the dispersion,
which is 1 for the binomial and Poisson. For a random effect, $S_j = I$
and $\lambda_j = \phi / \sigma_j^2$, the usual variance ratio of a mixed
model.

The log-likelihood is $\ell(\beta) = -D(\beta)/(2\phi) + \ell_s(\phi)$,
where $\ell_s$ is the saturated log-likelihood. The smoothing parameters
$\rho_j = \log\lambda_j$ (and $\log\phi$) maximize the marginal likelihood
of $y$, with $\beta$ integrated out. Here is that integral, by Laplace's
method at the penalized fit $\hat\beta$. $H$ is the Hessian of $D_p/2$ at
$\hat\beta$:

$$H = X^T W X + S_\lambda, \qquad W = \operatorname{diag}\Big(\frac{a_i}{2}\,\frac{\partial^2 d(y_i, \mu_i)}{\partial \eta_i^2}\Big),$$

with $d$ the unit deviance. This is the *observed* information. It equals
Fisher's for canonical links, and for the Gaussian it does not depend on
$\beta$, so the Laplace approximation is then exact. The negative log
marginal likelihood is, up to a constant,

$$
V(\rho, \phi) = \frac{D_p(\hat\beta)}{2\phi} - \ell_s(\phi)
  + \tfrac12 \log|H| - \tfrac12 \log|S_\lambda|_+ - \tfrac{M}{2}\log(2\pi\phi).
$$

Here is where the terms come from:

- the prior's normalizing constant gives $|S_\lambda/\phi|_+^{1/2}$ over
  its $P - M$ dimensions, where $|\cdot|_+$ is the product of the nonzero
  eigenvalues;
- the Gaussian integral gives $(2\pi)^{P/2}|H/\phi|^{-1/2}$;
- the powers of $2\pi$ and $\phi$ leave $M$.

This is **REML**. It integrates the unpenalized coefficients too, with a
flat prior, as REML does for the fixed effects of a mixed model. **ML**
integrates only the penalized coefficients. Its $H$ is restricted to them,
and it has no last term.

With one penalty per term, the blocks of $S_\lambda$ do not overlap, so

$$\log|S_\lambda|_+ = \sum_j \big(r_j \log\lambda_j + \log|S_j|_+\big).$$

## By hand: a Gaussian additive model

Here are two smooths of simulated data, fitted by REML.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy

jax.config.update('jax_enable_x64', True)

from glmnet.gam import GAM, s

rng = np.random.default_rng(0)
n = 300
data = pd.DataFrame({'x1': rng.uniform(0, 1, n), 'x2': rng.uniform(0, 1, n)})
eta = np.sin(2 * np.pi * data['x1']) + 3 * (data['x2'] - 0.5)**2
data['y'] = eta + 0.4 * rng.standard_normal(n)
data['count'] = rng.poisson(np.exp(0.5 + 0.7 * eta))

G = GAM(terms=[s('x1'), s('x2')], method='REML').fit(data, data['y'])
G.summary()
```

We take the design and the penalties from the fit. `design_` is the
intercept, then each spline's basis with its sum-to-zero constraint
absorbed. `penalties` are the $S_j$ on those columns. Everything else we
write ourselves.

```{code-cell} ipython3
D = G.design_
X = jnp.asarray(D.X)
y = jnp.asarray(data['y'], float)
P = X.shape[1]

def embed(sl, S_j):                     # S_j as a P x P matrix
    E = np.zeros((P, P))
    E[sl, sl] = S_j
    return jnp.asarray(E)

S = [embed(sl, S_j) for sl, S_j in zip(D.slices, D.penalties)]
ev = [np.linalg.eigvalsh(S_j) for S_j in D.penalties]
ev = [e[e > e.max() * 1e-10] for e in ev]
r = [len(e) for e in ev]                    # the ranks r_j
logdet_S1 = [np.log(e).sum() for e in ev]   # log|S_j|_+
M = P - sum(r)                              # the null space's dimension
P, r, M
```

Each spline has $r_j = K_j - 1$: the linear function is the null space of
its penalty, and the constant has been removed. With the intercept, that
gives $M = 3$.

For the Gaussian, $\hat\beta$ is a ridge regression, $H = X^TX +
S_\lambda$, and $\ell_s(\phi) = -\tfrac n2\log(2\pi\phi)$. The parameters
are $(\rho_1, \rho_2, \log\phi)$:

```{code-cell} ipython3
def S_lam(rho):
    return sum(jnp.exp(rho_j) * S_j for rho_j, S_j in zip(rho, S))

def V_gaussian(params):
    rho, phi = params[:2], jnp.exp(params[2])
    H = X.T @ X + S_lam(rho)
    beta = jnp.linalg.solve(H, X.T @ y)
    Dp = jnp.sum((y - X @ beta)**2) + beta @ S_lam(rho) @ beta
    ls = -n / 2 * jnp.log(2 * np.pi * phi)
    logdet_S = sum(r_j * rho_j + l_j for r_j, rho_j, l_j in zip(r, rho, logdet_S1))
    return (Dp / (2 * phi) - ls + jnp.linalg.slogdet(H)[1] / 2 - logdet_S / 2
            - M * jnp.log(2 * np.pi * phi) / 2)

value_and_grad = jax.jit(jax.value_and_grad(V_gaussian))
fit = minimize(lambda p: [np.asarray(v, float) for v in value_and_grad(p)],
               np.zeros(3), jac=True, method='BFGS', options=dict(gtol=1e-9))
fit.x, fit.fun
```

`GAM`'s smoothing parameters (on the scale of the prior weights,
`lam_raw`), dispersion and criterion are the same:

```{code-cell} ipython3
sel = G.smoothing_
pd.DataFrame({'by hand': np.r_[np.exp(fit.x), fit.fun],
              'GAM': np.r_[sel.lam_raw, sel.dispersion, sel.score]},
             index=['lam_1', 'lam_2', 'phi', 'criterion'])
```

### What the optimum says, in terms of residuals

Two identities hold at the optimum. Setting $\partial V/\partial\phi = 0$
gives REML's estimate of the dispersion,

$$\hat\phi = \frac{\|y - X\hat\beta\|^2 + \hat\beta^T S_\lambda \hat\beta}{n - M},$$

the penalized residual sum of squares over $n$ less the number of
unpenalized coefficients. ML would divide by $n$.

```{code-cell} ipython3
lam = sel.lam_raw
S_hat = D.penalty(lam)
H = D.X.T @ D.X + S_hat
beta = np.linalg.solve(H, D.X.T @ data['y'].values)
resid = data['y'].values - D.X @ beta
phi = (resid @ resid + beta @ S_hat @ beta) / (n - M)
phi, sel.dispersion, np.allclose(D.X @ beta, G.predict(data))
```

Setting $\partial V/\partial\rho_j = 0$ (for the Gaussian, $H$ does not
depend on $\beta$) gives

$$\frac{\lambda_j\, \hat\beta_j^T S_j \hat\beta_j}{\phi} = r_j - \lambda_j \operatorname{tr}\big(H^{-1}S_j\big) = \mathrm{edf}_j - (K_j - r_j).$$

On the left is the size of the fitted term as the prior measures it,
which has expectation $r_j$ under the prior. On the right are the degrees
of freedom of the term's penalized part. REML matches the two, as in
Harville's (1977) and Schall's (1991) estimating equations for variance
components. Here $K_j - r_j = 1$, the linear part:

```{code-cell} ipython3
Hinv = np.linalg.inv(H)
pd.DataFrame({'lam b\'S b / phi': [lam[j] * beta[sl] @ S_j @ beta[sl] / phi
                                   for j, (sl, S_j) in enumerate(zip(D.slices, D.penalties))],
              'r - lam tr(H^-1 S)': [r[j] - lam[j] * np.trace(Hinv[sl, sl] @ S_j)
                                     for j, (sl, S_j) in enumerate(zip(D.slices, D.penalties))],
              'edf - 1': G.edf_.values - 1}, index=G.edf_.index)
```

## By hand: a Poisson additive model

For a Poisson response, $\hat\beta(\rho)$ is not linear in $y$, and $W$
depends on $\hat\beta$. We find $\hat\beta(\rho)$ by Newton's method on
$D_p/2$ and let JAX differentiate through the iterations:

```{code-cell} ipython3
yc = jnp.asarray(data['count'], float)

def deviance(beta):
    mu = jnp.exp(X @ beta)
    return 2 * jnp.sum(xlogy(yc, yc) - xlogy(yc, mu) - (yc - mu))

def hessian(beta, rho):                   # X' W X + S_lam, W = mu for the log link
    return X.T @ (jnp.exp(X @ beta)[:, None] * X) + S_lam(rho)

def beta_hat(rho, iterations=30):
    beta = jnp.zeros(P).at[0].set(jnp.log(yc.mean()))
    grad = jax.grad(lambda b: deviance(b) / 2 + b @ S_lam(rho) @ b / 2)
    for _ in range(iterations):
        beta = beta - jnp.linalg.solve(hessian(beta, rho), grad(beta))
    return beta

def V_poisson_at(rho, beta):
    Dp = deviance(beta) + beta @ S_lam(rho) @ beta
    ls = jnp.sum(xlogy(yc, yc) - yc - gammaln(yc + 1))
    logdet_S = sum(r_j * rho_j + l_j for r_j, rho_j, l_j in zip(r, rho, logdet_S1))
    return (Dp / 2 - ls + jnp.linalg.slogdet(hessian(beta, rho))[1] / 2 - logdet_S / 2
            - M * jnp.log(2 * np.pi) / 2)

def V_poisson(rho):
    return V_poisson_at(rho, beta_hat(rho))

value_and_grad = jax.jit(jax.value_and_grad(V_poisson))
fit = minimize(lambda p: [np.asarray(v, float) for v in value_and_grad(p)],
               np.zeros(2), jac=True, method='BFGS', options=dict(gtol=1e-9))

G_pois = GAM(terms=[s('x1'), s('x2')], family=sm.families.Poisson(),
             method='REML').fit(data, data['count'])
sel = G_pois.smoothing_
pd.DataFrame({'by hand': np.r_[np.exp(fit.x), fit.fun],
              'GAM': np.r_[sel.lam_raw, sel.score]},
             index=['lam_1', 'lam_2', 'criterion'])
```

### The gradient, and the implicit derivative

Differentiating through 30 Newton steps works, but it is wasteful. `GAM`
instead uses the implicit function theorem. $\hat\beta$ solves
$\partial D_p/\partial\beta = 0$, so

$$\frac{d\hat\beta}{d\rho_k} = -H^{-1}\lambda_k S_k\hat\beta,$$

and

$$\frac{dV}{d\rho_k} = \frac{\partial V}{\partial\rho_k}\Big|_{\hat\beta}
+ \frac{\partial V}{\partial\beta}\Big|_{\hat\beta}\,\frac{d\hat\beta}{d\rho_k}.$$

Because $\hat\beta$ minimizes $D_p$, only $\tfrac12\log|H|$ contributes to
$\partial V/\partial\beta$, through $W(\hat\beta)$. That is the third
derivative of the log-likelihood. For the Gaussian it is zero. Both
partial derivatives are JAX gradients of $V$ with $\beta$ held as an
argument. Away from the optimum, the two ways agree:

```{code-cell} ipython3
rho = jnp.array([-3., -1.])
b = beta_hat(rho)
g_rho, g_beta = jax.grad(V_poisson_at, argnums=(0, 1))(rho, b)
H = hessian(b, rho)
dbeta = -jnp.linalg.solve(H, jnp.stack([jnp.exp(rho_k) * S_k @ b
                                        for rho_k, S_k in zip(rho, S)], axis=1))
implicit = g_rho + g_beta @ dbeta
pd.DataFrame({'through Newton': np.asarray(jax.grad(V_poisson)(rho)),
              'implicit': np.asarray(implicit),
              'of which via W(beta)': np.asarray(g_beta @ dbeta)})
```

## The design of `glmnet.gam.reml`

The code computes the same $V$ and gradient as above. These are the
choices it makes.

**The inner fit is `GAM`'s own.** At each $\rho$, $\hat\beta$ comes from
the same local scoring (IRLS) that `method='fixed'` uses, with the joint
penalized least squares solve (`GAMControl(solver='joint')`). It is in
numpy, warm started from the last fit, with a tighter tolerance. So the
REML fit at its optimum *is* the fixed-$\lambda$ fit at those $\lambda$s,
and the final `GAM` has all of the usual attributes. JAX never sees the
iterations. It evaluates $V(\rho, \beta)$ with $\beta$ given
(`LaplaceCriterion.criterion`), and its two partial derivatives, combined
by the implicit derivative as above (`value_and_grad`).

**Coordinates.** Each penalized block is rotated to the eigenvectors of
its $S_j$ (the matrix `T`, and the criterion's `X` is $XT$). Then:

- $S_\lambda$ is diagonal, `lam @ M`, where row $j$ of `M` holds the
  eigenvalues of $S_j$ in its block's coordinates;
- $\log|S_\lambda|_+$ is the closed form above (`rank`, `log_ev`);
- the null space is a set of coordinates (`penalized` is the rest), so
  ML's restriction of $H$ is a submatrix.

The fit and the criterion are invariant to this rotation, and $\beta$ is
rotated back when needed.

**The family** is a small table in JAX (`_jax_family`). It holds the
inverse link, the unit deviance $d$, the saturated log-likelihood
$\ell_s(\phi)$, and whether $\phi$ is fixed. $W$ comes from `jax.grad`
twice of $d(y, g^{-1}(\eta))$, so it is the observed information for any
link in the table. Families are Gaussian, binomial, Poisson, Gamma and
inverse Gaussian. Links are identity, log, logit, probit, cloglog,
inverse and inverse squared.

**Parameters and scale.** The parameters are $\rho_j = \log\lambda_j$ for
the estimated terms, then $\log\phi$ when the dispersion is free.

- Terms given `lam` keep it.
- Terms given `df` have their $\lambda$ matched to the df at the prior
  weights, then held fixed, as mgcv's `sp` argument.
- `GAM`'s objective is the deviance over twice the *sum of the prior
  weights* plus $\sum_j \lambda_j\beta_j^TS_j\beta_j/2$. The criterion's
  $\lambda$ is the term's `lam` times that sum. `smoothing_.lam` is on
  the objective's scale, and `smoothing_.lam_raw` on the criterion's,
  which is mgcv's `sp` apart from mgcv's rescaling of each $S_j$.

**The search.**

1. Starting values follow mgcv. Each $\lambda_j$ makes the mean diagonal
   of $\lambda_jS_j$ match that of its block of $X^TWX$, at the
   intercept-only fit. $\phi$ starts at that fit's deviance over $n$.
2. L-BFGS-B runs with the analytic gradient, in a box of $\pm 25$ around
   the start.
3. A few steps of Newton's method follow (`_newton`). The Hessian is a
   central difference of the gradient. The steps are halved as needed, and
   parameters at the box's edge (a $\lambda$ at 0 or $\infty$, in effect)
   are held there.

   Near the optimum a step can lower $V$ by less than its rounding
   error. Such a step is still taken if it halves the gradient.

   The criterion is converged when the free part of the gradient is
   below `grad_tol`.
4. If the inner fit does not converge, $V$ is meaningless there. This
   happens at a $\lambda$ near 0 with nearly as many coefficients as
   observations. The search then gets a value larger than any seen, with
   a gradient towards larger $\lambda$ (`_unconverged`), so it turns back.

**Outputs.**

- `smoothing_`: a `SmoothingSelection` with the method, the terms, `lam`,
  `lam_raw`, `dispersion`, `score` ($V$ at the optimum, mgcv's
  `gcv.ubre`), `gradient`, `converged` and the number of evaluations.
- `reml_score_`.
- `dispersion_`, which is $\hat\phi$.
- `posterior_covariance_`, which is $\phi H^{-1}$ (mgcv's `Vp`), and
  `frequentist_covariance_` (`Ve`).
- `edf_`, the block traces of $H^{-1}X^TWX$ (mgcv's edf).
- `variance_components_`, which is $\phi/\lambda_j$ for each penalized
  term and $\phi$ (as `gam.vcomp`).

The covariances are at the last IRLS step's working weights. They ignore
the uncertainty in $\hat\rho$ (mgcv's `Vc` correction is not
implemented).

**Cox.** For `family='cox'` the likelihood is the partial likelihood (as
mgcv's `cox.ph`), with no intercept and no $\phi$. Its $W$ is not
diagonal, and coxdev gives it as an operator. `CoxLaplaceCriterion` is in
numpy. The derivative of $\log|H|$ along $d\hat\beta/d\rho_k$ needs the
partial likelihood's third derivatives, and the code takes it as a central
difference of the information along $X\,d\hat\beta/d\rho_k$. So its
tolerances are looser.

**Not yet.**

- Several penalties per term (tensor products, correlated random
  effects). With several penalties per term, $\log|S_\lambda|_+$ no
  longer separates; this is Wood's (2011) stable transform.
- The `Vc` correction.
- `lo()` terms, which have no basis.

## References

- Harville, D. A. (1977). Maximum likelihood approaches to variance
  component estimation and to related problems. *Journal of the American
  Statistical Association* 72(358), 320–338.
- Schall, R. (1991). Estimation in generalized linear models with random
  effects. *Biometrika* 78(4), 719–727.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and
  marginal likelihood estimation of semiparametric generalized linear
  models. *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
- Wood, S. N., Pya, N. and Säfken, B. (2016). Smoothing parameter and model
  selection for general smooth models. *Journal of the American Statistical
  Association* 111(516), 1548–1563.
