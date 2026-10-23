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

This page explains how `GLMM(method='REML')` and `GLMM(method='ML')` choose
the smoothing parameters (the code is `glmnet.gam.reml`). It writes the
criterion out in a few lines of JAX for a small example, minimizes it
"by hand", and checks that `GAM` gets the same answer. The same check is
a test, `tests/gam/test_gam_reml_byhand.py`.

The method is that of mgcv (Wood, 2011), and the notation follows Wood
(2017, chapter 6).

## The model

The linear predictor is $\eta = X\beta$. The columns of $X$ are the
intercept, then one block for each term: a spline's basis, a categorical's
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
integrates only the penalized coefficients, and treats the unpenalized
ones as parameters, at the same joint mode $\hat\beta$. Its $H$ is
$H_{PP}$, the block of the penalized coordinates, and it has no last term.
The difference between the two is the subject of the section "ML, and
REML versus ML" below.

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

from glmnet.gam import GAM, GLMM, s

rng = np.random.default_rng(0)
n = 300
data = pd.DataFrame({'x1': rng.uniform(0, 1, n), 'x2': rng.uniform(0, 1, n)})
eta = np.sin(2 * np.pi * data['x1']) + 3 * (data['x2'] - 0.5)**2
data['y'] = eta + 0.4 * rng.standard_normal(n)
data['count'] = rng.poisson(np.exp(0.5 + 0.7 * eta))

G = GLMM(terms=[s('x1'), s('x2')], method='REML').fit(data, data['y'])
G.summary()
```

The model, as `GAM` displays it:

```{code-cell} ipython3
G
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

### ML, and REML versus ML

Split the coordinates into the penalized ones, $P$, and the $M$
unpenalized ones, $F$: the intercept, the linear terms, and the
penalties' null spaces (here each spline's linear part). Then

$$\log|H| = \log|H_{PP}| + \log|C_F|, \qquad
C_F = H_{FF} - H_{FP}H_{PP}^{-1}H_{PF} = X_F^TW(I - \mathcal{S})X_F,$$

with $\mathcal{S}$ the smoother of the penalized terms (their joint
penalized fit), so that

$$V_{\text{REML}} - V_{\text{ML}} = \tfrac12\log|C_F| - \tfrac{M}{2}\log(2\pi\phi).$$

$C_F$ is the information about the unpenalized coefficients once the
penalized ones are integrated out: for the Gaussian, the GLS information.
$\tfrac12\log|C_F|$ is Harville's (1974) REML correction. At the same
parameters the two criteria differ by it:

```{code-cell} ipython3
from glmnet.gam.reml import LaplaceCriterion

C_reml = LaplaceCriterion(G, [0, 1], method='REML')
C_ml = LaplaceCriterion(G, [0, 1], method='ML')
params = np.r_[np.log(G.smoothing_.lam_raw), np.log(G.smoothing_.dispersion)]
b_hat = C_reml.coef(params)
ev = C_reml._evaluation(params, b_hat)       # (the criterion's coordinates)
H_c = ev['X'].T @ ev['X'] + ev['S']
P_c = C_reml.penalized
F_c = np.setdiff1d(np.arange(H_c.shape[0]), P_c)
C_F = H_c[np.ix_(F_c, F_c)] - H_c[np.ix_(F_c, P_c)] @ np.linalg.solve(
    H_c[np.ix_(P_c, P_c)], H_c[np.ix_(P_c, F_c)])
phi_hat = G.smoothing_.dispersion
(C_reml.criterion(params, b_hat) - C_ml.criterion(params, b_hat),
 np.linalg.slogdet(C_F)[1] / 2 - len(F_c) * np.log(2 * np.pi * phi_hat) / 2)
```

What this means in practice:

- **The dispersion.** REML's $\hat\phi = D_p/(n - M)$ accounts for the $M$
  degrees of freedom of the unpenalized coefficients; ML's is $D_p/n$.
  Likewise REML's smoothing parameters and variances account for them,
  which is why REML is the usual choice for estimating them (Wood, 2011).
- **Comparing models.** $\log|C_F|$ changes with the parametrization of
  the unpenalized coefficients, and with which of them are in the model.
  So REML criteria can be compared only between models with the same
  unpenalized part (and agree with mgcv's or lme4's only up to a constant
  when a categorical's effects are parametrized differently). ML criteria are
  log-likelihoods of the models, comparable across them, as for a
  likelihood ratio or AIC between models with different fixed effects.
- **Non-Gaussian families.** ML is the Laplace approximation with the
  unpenalized coefficients at the mode, glmer's `nAGQ=0`. glmer's default,
  `nAGQ=1`, also optimizes the approximation in them, and differs a
  little.

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

G_pois = GLMM(terms=[s('x1'), s('x2')], family=sm.families.Poisson(),
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
the same local scoring (IRLS) that a `GAM` uses, with the joint
penalized least squares solve (`GAMControl(solver='joint')`). It is in
numpy, warm started from the last fit, with a tighter tolerance. So the
REML fit at its optimum *is* the fixed-$\lambda$ fit at those $\lambda$s,
and the final `GAM` has all of the usual attributes. The criterion then
takes one Newton step from the inner fit's $\hat\beta$ with its own
factorization of $H$. That gives $\hat\beta$ the criterion's precision,
independent of the inner fit's warm start, so the inner fit can stop at
a looser tolerance.

**The four operations.** $V$ and its gradient (the formula in the next
section, through the implicit derivative as above) need only $\log|H|$,
solves, the traces $\operatorname{tr}(H^{-1}S_k)$ and the leverages
$x_i^TH^{-1}x_i$, from `glmnet.gam.linalg.DenseHessian`. ML needs the
same quantities for $H_{PP}$ (`DenseHessian.restrict`), and solves with
the full $H$ for the implicit derivative. A sparse or matrix-free
implementation can replace it (see "What ML needs" below). JAX gives only
the derivatives of the family's unit deviance.

**Penalties as objects** (`glmnet.gam.penalties`). Each term's penalty is
a function $S(\theta)$ of its parameters. At each evaluation it gives its
matrix, its derivatives in $\theta$, $\log|S|_+$ and its derivatives, and
possibly a change of its coordinates. There are three kinds:

- `ScaledPenalty`, one penalty $\lambda S$ ($\theta = \log\lambda$),
  diagonal in the eigenvectors of $S$. Each penalized block is rotated to
  them once (the matrix `T`), so the null space is a set of coordinates
  (`penalized` is the rest) and ML's restriction of $H$ is a submatrix.
- `SumPenalty`, several penalties $\sum_k\lambda_kS_k$ (a tensor product,
  `penalized_columns`), in Wood's (2011) stable reparametrization at each
  evaluation. The dominant penalty's range is split off, with its null
  space set exactly to 0, then the next's in that null space, and so on.
  A $\lambda$ going to infinity then multiplies only its own block.
- `CholeskyPenalty`, the precision of correlated random effects,
  $I_G \otimes \Sigma^{-1}$, with $\Sigma = LL^T$ in log-Cholesky form
  (lme4's relative covariance factor). It is evaluated in the coordinates
  of $b = Lu$ in each group, where the penalty is the identity, so a
  singular $\Sigma$ is a diagonal entry of $L$ going to 0.

The fit and the criterion are invariant to these changes of coordinates.
For a non-orthogonal one, $\log|H|$ is corrected by its log determinant.

**The family** is a small table in JAX (`_jax_family`). It holds the
inverse link, the unit deviance $d$, the saturated log-likelihood
$\ell_s(\phi)$, and whether $\phi$ is fixed. $W$ comes from `jax.grad`
twice of $d(y, g^{-1}(\eta))$, so it is the observed information for any
link in the table. Families are Gaussian, binomial, Poisson, Gamma,
inverse Gaussian and negative binomial, and the quasi-Poisson and
quasi-binomial (`glmnet.gam.families`). Links are identity, log, logit,
probit, cloglog, inverse and inverse squared.

- **A quasi family** has a free $\phi$ and mgcv's "saturated
  log-likelihood" $\ell_s(\phi) = -\tfrac{n}{2}\log\phi + \tfrac12\sum_i
  \log w_i$, a Gaussian's in $\phi$.
- **The negative binomial's** $d$ and $\ell_s$ also depend on $\theta$ (the
  variance $\mu + \mu^2/\theta$). When $\theta$ is estimated (mgcv's
  `nb()`), $\log\theta$ is one more parameter. Its derivative has the
  same three parts as a smoothing parameter's: in $d$ and $\ell_s$ with
  $\beta$ held; in $\log|H|$ through $W$ (`jax.grad` of $d$'s second
  derivative in $\eta$, weighting the leverages); and through $\beta$, with
  $d\hat\beta/d\log\theta = -H^{-1} X^T (w\, \partial_{\log\theta}
  d')/2$, $d'$ the derivative of $d$ in $\eta$.

**Parameters and scale.** The parameters are the penalties' $\theta$ for
the estimated terms ($\rho_j = \log\lambda_j$, or a log-Cholesky
factor), then $\log\phi$ when the dispersion is free, then the negative
binomial's $\log\theta$ when it is estimated.

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
3. Newton's method follows (`_newton`, up to 100 steps: towards an optimum
   at infinity, a singular covariance, each moves a parameter by about 1). The Hessian is a
   central difference of the gradient. The steps are halved as needed, and
   parameters at the box's edge (a $\lambda$ at 0 or $\infty$, in effect)
   are held there.

   Near the optimum a step can lower $V$ by less than its rounding
   error. Such a step is still taken if it halves the gradient.

   The criterion is converged when the free part of the gradient is
   below `grad_tol`.
4. If the inner fit does not converge, $V$ is meaningless there. This
   happens at a $\lambda$ near 0 with nearly as many coefficients as
   observations, and at a separated binomial fit, whose deviance's
   derivatives are not finite. The search then gets a value larger than
   any seen, with a gradient towards larger $\lambda$ (`_unconverged`),
   so it turns back.

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
  term, the variances of correlated random effects (their covariance in
  the term's `covariance_`), and $\phi$ (as `gam.vcomp`).

The covariances are at the last IRLS step's working weights. They ignore
the uncertainty in $\hat\rho$ (mgcv's `Vc` correction is not
implemented).

**Cox.** For `family='cox'` the likelihood is the partial likelihood (as
mgcv's `cox.ph`), with no intercept and no $\phi$. Its $W$ is not
diagonal, and coxdev gives it as an operator. `CoxLaplaceCriterion` is in
numpy, on the same penalty objects and `DenseHessian`. The derivative of
$\log|H|$ along $d\hat\beta/d\rho_k$ needs the partial likelihood's third
derivatives, and the code takes it as a central difference of the
information along $X\,d\hat\beta/d\rho_k$. So its tolerances are looser,
and at a singular covariance of correlated random effects its gradient
is not precise enough to converge.

**Not yet.**

- Sparse and matrix-free implementations of the four operations (below).
- Other structured covariances of correlated random effects (banded,
  AR(1)); compound symmetry and diagonal are `CompoundSymmetryPenalty` and
  `DiagonalPenalty`.
- The `Vc` correction.
- `lo()` terms, which have no basis.

## Toward large problems: lessons from mixed-model software

Everything above forms $H$, a dense $P \times P$ matrix. That is fine for
a few smooths, but not for a random effect with $10^5$ levels, or for the
crossed random effects of Ghosh, Hastie and Owen (2022), where $N$ is in
the millions. Mixed-model software has solved this problem several times.
These are the lessons we are building into the design.

### The criterion needs only four operations on $H$

Write $h_i = W_{ii}$ and $h_i' = \partial h_i/\partial\eta_i$. At
$\hat\beta$, the derivative of $V$ in $\rho_k$ is

$$
\frac{dV}{d\rho_k} = \frac{\lambda_k\hat\beta^TS_k\hat\beta}{2\phi}
 + \frac{\lambda_k}{2}\operatorname{tr}\big(H^{-1}S_k\big)
 - \frac{\lambda_k}{2}\operatorname{tr}\big(S_\lambda^{+}S_k\big)
 + \frac12\sum_i h_i'\,\big(x_i^TH^{-1}x_i\big)\,x_i^T\frac{d\hat\beta}{d\rho_k},
\qquad \frac{d\hat\beta}{d\rho_k} = -H^{-1}\lambda_kS_k\hat\beta .
$$

The last term is the third-derivative term of the Poisson example. So the
value and the gradient need only:

1. $\log|H|$;
2. solves $H^{-1}b$;
3. the traces $\operatorname{tr}(H^{-1}S_k)$;
4. the leverages $x_i^TH^{-1}x_i$.

$S_\lambda$ is block diagonal over the terms and small within each term, so
its own log-determinant and traces are cheap. None of the four needs $H$
or $H^{-1}$ as a dense matrix. The criterion is therefore written against
an interface with these four operations, and its first implementation is
the dense one used now. Two others can sit behind the same interface:

- **Sparse Cholesky.** $H$ is sparse when the bases are: indicators of
  groups, or B-splines.
  - lme4 (Bates et al., 2015) factors the mixed-model equations with a
    sparse Cholesky. The fill-reducing ordering and the symbolic
    factorization are computed once and reused at every value of the
    variance parameters.
  - MixedModels.jl goes further, with a *blocked* Cholesky whose blocks
    have the structure of the model: diagonal for the first grouping
    factor, small dense blocks for random slopes, sparse blocks for
    crossed factors.
  - With a sparse factor, the traces and leverages come from Takahashi's
    selected inversion (Takahashi, Fagan and Chen, 1973). It computes the
    entries of $H^{-1}$ on the sparsity pattern of the factor, never all of
    $H^{-1}$. That is how average-information REML (Gilmour, Thompson and
    Cullis, 1995, in ASReml and the animal-breeding codes) fits models with
    millions of levels.
- **Matrix-free.** The solves are done by conjugate gradients, or by
  backfitting, which is itself an iterative solver of these equations
  (Ghosh, Hastie and Owen, 2022).
  - $\log|H|$ comes from stochastic Lanczos quadrature (Ubaru, Chen and
    Saad, 2017).
  - The traces and leverages come from Hutchinson's (1989) estimator,
    $\operatorname{tr}(A) \approx \frac1m\sum_l z_l^TAz_l$ for random
    signs $z_l$.
  - This is how GPyTorch (Gardner et al., 2018) fits Gaussian processes,
    and how BOLT-LMM (Loh et al., 2015) fits mixed models to genetic data,
    with a preconditioner in both.

  The estimates are noisy, so the outer optimizer must tolerate noise in
  the gradient.

### What ML needs

ML's criterion and gradient are REML's with $H_{PP}$ in place of $H$ in
the log determinant, the traces and the leverages. The implicit
derivative still solves with the full $H$, because $\hat\beta$ is the
joint mode of all the coefficients. `DenseHessian` factors $H_{PP}$ a
second time (`restrict`). A sparse or matrix-free backend need not. With
the $M$ columns $H^{-1}e_f$ of the unpenalized coordinates, $M$ more
solves, and $B = (H^{-1})_{FF} = C_F^{-1}$:

$$\log|H_{PP}| = \log|H| + \log|B|, \qquad
H_{PP}^{-1} = (H^{-1})_{PP} - (H^{-1})_{PF}\,B^{-1}(H^{-1})_{FP}.$$

The second is a rank-$M$ correction. So ML's traces and leverages are
REML's less small corrections built from those columns. **REML needs the
four operations; ML needs the four and $M$ solves.** $M$ is usually small:
the intercept, the linear terms, and one or two per spline.

In the backfitting picture of large problems, those $M$ solves are
backfitting the penalized terms on the columns of $X_F$. That is also how
`GAM.fixed_effects_` computes the GLS covariance $C_F^{-1}$ without the
dense design. So ML, REML and the fixed effects' standard errors share
one computation.

### The outer optimizer can avoid the Hessian of $V$

Newton's method needs the Hessian of $V$ in $\rho$, which takes many more
solves. mgcv's extended Fellner–Schall update (Wood and Fasiolo, 2017),

$$\lambda_k \leftarrow \lambda_k\,\frac{\operatorname{tr}(S_\lambda^{+}S_k) - \operatorname{tr}(H^{-1}S_k)}{\hat\beta^TS_k\hat\beta/\phi},$$

needs only operations 2 and 3. Its fixed point is the identity checked by
hand above, in the residuals of the Gaussian example. Average-information
REML similarly replaces the Hessian by a matrix of solves. Both are the
optimizers that scale. The current quasi-Newton and Newton search stays
for small problems, where it converges fastest.

### Penalties as parametrized objects

A term's prior precision is $S(\theta)/\phi$ for a few parameters $\theta$:

- one $\lambda$ times a fixed matrix (a spline, `re()`);
- a sum $\sum_m\lambda_mS_m$ (`tensor_spline()`, `penalized_columns()`);
- the inverse of an unstructured covariance, $I\otimes\Sigma^{-1}$, for
  correlated random effects (`correlated_random_effect()`);
- a structured covariance, compound symmetry or diagonal
  (`correlated_random_effect(..., covariance=...)`); later, AR(1).

nlme's `pdMat` classes (Pinheiro and Bates, 2000) are the pattern, and
`glmnet.gam.penalties` implements it. Each penalty gives $S(\theta)$, its
derivatives in $\theta$, and $\log|S(\theta)|_+$, so the criterion does
not care which kind it is.

For an unstructured $\Sigma$ we use the log-Cholesky parametrization
(Pinheiro and Bates, 1996) of lme4's relative covariance factor $L$, the
*covariance's* Cholesky factor, not the precision's. A singular $\Sigma$
(lme4's boundary fit, a correlation of $\pm1$) is then a diagonal entry
of $L$ going to 0, with the other parameters finite. A factor of the
precision would have to go to infinity along a direction that is not a
coordinate, and the search stops short: we tried it, and against `glmer`
it stopped at a correlation of 0.73, with a worse likelihood, where the
optimum is 1. As in lme4, the evaluation is in the coordinates of
$b = Lu$, where the penalty is the identity.

### Smaller lessons

- **Profiling.** For the Gaussian, lme4 profiles $\beta$ and $\phi$ out of
  the criterion in closed form. We already profile $\beta$, through the
  inner fit. $\phi$ is still a parameter of the search, but the identity
  $\hat\phi = (\|y - X\hat\beta\|^2 + \hat\beta^TS_\lambda\hat\beta)/(n - M)$
  above profiles it out too, leaving only $\rho$ to search over.
- **Automatic differentiation of the Laplace approximation.** TMB and
  glmmTMB (Kristensen et al., 2016; Brooks et al., 2017) do what our JAX
  criterion does, the inner fit with the implicit derivative, but with
  sparse Hessians. JAX has no sparse Cholesky. A sparse implementation will
  wrap a scipy or CHOLMOD factorization with its own derivative
  (`jax.custom_vjp`), or use the matrix-free route, which needs no
  factorization.
- **Big data GAMs.** mgcv's `bam` builds $X^TWX$ from blocks of rows,
  never the $N \times P$ design. With `discrete=TRUE` it discretizes the
  covariates, so that the cross products are computed on the few distinct
  values (Wood et al., 2017).
- **Sparse precisions.** INLA (Rue, Martino and Chopin, 2009) works with
  sparse precision matrices of latent Gaussian fields. Our penalties
  already are precisions. Its SPDE representation of a Matérn field
  (Lindgren, Rue and Lindström, 2011) is a sparse alternative to the dense
  basis of a Gaussian process term.

## References

- Bates, D., Mächler, M., Bolker, B. and Walker, S. (2015). Fitting linear
  mixed-effects models using lme4. *Journal of Statistical Software*
  67(1), 1–48.
- Brooks, M. E., Kristensen, K., van Benthem, K. J., Magnusson, A., Berg,
  C. W., Nielsen, A., Skaug, H. J., Mächler, M. and Bolker, B. M. (2017).
  glmmTMB balances speed and flexibility among packages for zero-inflated
  generalized linear mixed modeling. *The R Journal* 9(2), 378–400.
- Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q. and Wilson,
  A. G. (2018). GPyTorch: Blackbox matrix-matrix Gaussian process inference
  with GPU acceleration. *Advances in Neural Information Processing
  Systems* 31.
- Ghosh, S., Hastie, T. and Owen, A. B. (2022). Backfitting for large scale
  crossed random effects regressions. *The Annals of Statistics* 50(1).
  arXiv:2007.10612.
- Gilmour, A. R., Thompson, R. and Cullis, B. R. (1995). Average
  information REML: an efficient algorithm for variance parameter
  estimation in linear mixed models. *Biometrics* 51(4), 1440–1450.
- Harville, D. A. (1974). Bayesian inference for variance components
  using only error contrasts. *Biometrika* 61(2), 383–385.
- Harville, D. A. (1977). Maximum likelihood approaches to variance
  component estimation and to related problems. *Journal of the American
  Statistical Association* 72(358), 320–338.
- Hutchinson, M. F. (1989). A stochastic estimator of the trace of the
  influence matrix for Laplacian smoothing splines. *Communications in
  Statistics — Simulation and Computation* 18(3), 1059–1076.
- Kristensen, K., Nielsen, A., Berg, C. W., Skaug, H. and Bell, B. M.
  (2016). TMB: Automatic differentiation and Laplace approximation.
  *Journal of Statistical Software* 70(5), 1–21.
- Lindgren, F., Rue, H. and Lindström, J. (2011). An explicit link between
  Gaussian fields and Gaussian Markov random fields: the stochastic partial
  differential equation approach. *Journal of the Royal Statistical
  Society, Series B* 73(4), 423–498.
- Loh, P.-R. et al. (2015). Efficient Bayesian mixed-model analysis
  increases association power in large cohorts. *Nature Genetics* 47(3),
  284–290.
- Pinheiro, J. C. and Bates, D. M. (1996). Unconstrained parametrizations
  for variance-covariance matrices. *Statistics and Computing* 6(3),
  289–296.
- Pinheiro, J. C. and Bates, D. M. (2000). *Mixed-Effects Models in S and
  S-PLUS*. Springer.
- Rue, H., Martino, S. and Chopin, N. (2009). Approximate Bayesian
  inference for latent Gaussian models by using integrated nested Laplace
  approximations. *Journal of the Royal Statistical Society, Series B*
  71(2), 319–392.
- Schall, R. (1991). Estimation in generalized linear models with random
  effects. *Biometrika* 78(4), 719–727.
- Takahashi, K., Fagan, J. and Chen, M.-S. (1973). Formation of a sparse
  bus impedance matrix and its application to short circuit study. In
  *Proceedings of the 8th PICA Conference*, 63–69.
- Ubaru, S., Chen, J. and Saad, Y. (2017). Fast estimation of tr(f(A)) via
  stochastic Lanczos quadrature. *SIAM Journal on Matrix Analysis and
  Applications* 38(4), 1075–1099.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and
  marginal likelihood estimation of semiparametric generalized linear
  models. *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
- Wood, S. N. and Fasiolo, M. (2017). A generalized Fellner-Schall method
  for smoothing parameter optimization with application to Tweedie
  location, scale and shape models. *Biometrics* 73(4), 1071–1081.
- Wood, S. N., Li, Z., Shaddick, G. and Augustin, N. H. (2017).
  Generalized additive models for gigadata: modeling the U.K. black smoke
  network daily data. *Journal of the American Statistical Association*
  112(519), 1199–1210.
- Wood, S. N., Pya, N. and Säfken, B. (2016). Smoothing parameter and model
  selection for general smooth models. *Journal of the American Statistical
  Association* 111(516), 1548–1563.
