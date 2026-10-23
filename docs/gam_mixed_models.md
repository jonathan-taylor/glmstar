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

# Mixed models as GAMs

A penalized term of a GAM is a random effect in disguise: its quadratic
penalty is the log density of a Gaussian prior on its coefficients, and
choosing its smoothing parameter by REML is estimating the prior's variance
as the REML of a mixed model (Wood, 2017, chapters 2 and 6). Conversely, a
classical random effect is a GAM term with an identity penalty.
`random_effect(g)` (imported as `re`: `from glmnet.gam import
random_effect as re`) is
that term in `glmnet.gam`: one coefficient $b_k$ per group $k$ of the column
`g`, with

$$b_k \sim N(0, \sigma_b^2) \ \text{independently}, \qquad \sigma_b^2 = \phi/\lambda,$$

$\phi$ the dispersion and $\lambda$ the term's smoothing parameter (on the
scale of the prior weights); `re(g, by=x)` is a random slope, a coefficient
$b_k x$ per group. With `method='REML'` (or `'ML'`) the variances are
estimated by maximizing the restricted (or full) marginal likelihood: for
the Gaussian this is exactly the REML (or ML) of the linear mixed model, as
lme4's `lmer` (Bates et al., 2015) and nlme's `lme` (Pinheiro and Bates,
2000); for other families it is its Laplace approximation, as mgcv's
`gam(..., method="REML")` with `bs="re"` terms and lme4's `glmer`.

This page fits the classical designs: a one-way layout, crossed and nested
random effects, random intercepts and slopes, a binomial GLMM, an
observation-level random effect for overdispersed counts, and a smooth with
a random effect. The data are simulated with the designs of classical
examples (lme4's `Penicillin`, `Pastes`, `sleepstudy` and `cbpp`). Each fit
is checked against lme4, nlme and mgcv in glmstar's tests
(`tests/gam/test_gam_mixed_r.py`): the variance components, fixed effects,
fitted values, predicted random effects and log-likelihoods agree to the
optimizers' precision.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from glmnet.gam import GAM, s, random_effect as re, linear, categorical, correlated_random_effect


def effects(rng, labels, sd):
    """Normal effects, one per label."""
    labels = np.unique(labels)
    return dict(zip(labels, rng.normal(0, sd, len(labels))))
```

## Random intercepts

The one-way random effects model: 30 schools, of 2 to 40 students each,
with a score

$$y_{ij} = \mu + b_i + \epsilon_{ij}, \qquad b_i \sim N(0, 3^2), \quad \epsilon_{ij} \sim N(0, 8^2).$$

```{code-cell} ipython3
rng = np.random.default_rng(1)
sizes = rng.integers(2, 41, 30)
School = pd.DataFrame({'school': np.repeat([f's{i:02d}' for i in range(30)], sizes)})
School['score'] = (50 + School['school'].map(effects(rng, School['school'], 3))
                   + rng.normal(0, 8, School.shape[0]))

oneway = GLMM(terms=[re('school')], method='REML').fit(School, School['score'])
oneway.variance_components_
```

`variance_components_` has the variance of each random effect and of the
residual (`scale`, the dispersion), and their standard deviations, as
lme4's `VarCorr` (and mgcv's `gam.vcomp`). The intercept is the estimate
of $\mu$, a generalized least squares estimate:

```{code-cell} ipython3
oneway.intercept_
```

### Predicted random effects

The fitted coefficients of `re()`, `effects_`, are the predicted random
effects (the BLUPs, lme4's `ranef`): the posterior means of the $b_i$. For
a school of $n_i$ students its effect is its mean residual shrunk towards 0
by the factor $n_i / (n_i + \sigma^2/\sigma_b^2)$, so a small school's
effect is shrunk the most. Fitting the schools as fixed effects instead
(`categorical('school')`) shrinks none of them:

```{code-cell} ipython3
fixed = GAM(terms=[categorical('school')]).fit(School, School['score'])
blup = oneway.terms_[0].effects_
mean = School.groupby('school')['score'].mean() - oneway.intercept_
n = School['school'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(mean, blup, s=3 * n, alpha=0.7)
lim = np.abs(mean).max() * 1.05
ax.plot([-lim, lim], [-lim, lim], 'k:', lw=1)
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('school mean less the intercept (fixed effect)')
ax.set_ylabel('predicted random effect (BLUP)')
ax.set_title('shrinkage: area of each point is the school size');
```

The predicted effects with their standard errors are what `GAM.plot` draws
for an `re()` term, a point per group with bars of 2 standard errors. They
are from the posterior covariance, `posterior_covariance_` (mgcv's `Vp`),
and so include the uncertainty in the intercept: they are larger than
lme4's conditional standard deviations (`ranef(fit, condVar=TRUE)`), which
take the fixed effects as known, much larger when there are few groups.

```{code-cell} ipython3
ax = oneway.plot()[0]
ax.tick_params(axis='x', labelrotation=90, labelsize=7)
```

A group not seen in fitting has the random effect's prior mean 0, so its
prediction is the intercept:

```{code-cell} ipython3
oneway.predict(pd.DataFrame({'school': ['s00', 'new school']}))
```

### REML and ML

ML estimates the variances with the fixed effects (here only $\mu$) as
known, and so underestimates them, as the maximum likelihood estimate of a
variance divides by $n$ rather than $n - 1$; REML corrects for the fixed
effects. `reml_score_` is the minimized criterion, the negative log REML
(or ML) likelihood: for this Gaussian fit, minus lme4's `logLik` of
`lmer(score ~ 1 + (1|school), REML=TRUE)` (or `FALSE`).

```{code-cell} ipython3
oneway_ml = GLMM(terms=[re('school')], method='ML').fit(School, School['score'])
pd.DataFrame({'REML': oneway.variance_components_['sd'],
              'ML': oneway_ml.variance_components_['sd']})
```

```{code-cell} ipython3
oneway.reml_score_, oneway_ml.reml_score_
```

Likelihood ratio tests of fixed effects use the ML fits: REML likelihoods
of models with different fixed effects are not comparable.

## Crossed random effects

Each of 24 plates is inoculated with each of 6 samples of an antibiotic
(the design of lme4's `Penicillin`), and the diameter of the zone of
inhibited growth is measured. Plates and samples are both random: two
`re()` terms, the effects crossed.

```{code-cell} ipython3
rng = np.random.default_rng(2)
Plates = pd.DataFrame({'plate': np.repeat([f'p{i:02d}' for i in range(24)], 6),
                       'sample': np.tile(list('ABCDEF'), 24)})
Plates['diameter'] = (23 + Plates['plate'].map(effects(rng, Plates['plate'], 0.85))
                      + Plates['sample'].map(effects(rng, Plates['sample'], 1.9))
                      + rng.normal(0, 0.55, Plates.shape[0]))

crossed = GLMM(terms=[re('plate'), re('sample')], method='REML').fit(Plates, Plates['diameter'])
crossed.variance_components_
```

This is `lmer(diameter ~ 1 + (1|plate) + (1|sample))`. With only 6
samples their variance is poorly determined; the 24 plates' better. The
samples' bars are wide for the same reason: each sample is measured on all
24 plates, but their mean is confounded with the intercept (lme4's
conditional standard deviation of each sample's effect is 0.18, ours 1.2).

```{code-cell} ipython3
axes = crossed.plot()
axes[0].tick_params(axis='x', labelrotation=90, labelsize=7)
axes[0].figure.tight_layout()
```

## Nested random effects

Ten batches of a chemical paste, three casks from each batch and two
measurements of strength from each cask (the design of lme4's `Pastes`).
The casks are nested in the batches: cask `a` of one batch has nothing to
do with cask `a` of another. A random effect of the casks is one per
batch and cask, so its groups are their combinations: `re(['batch',
'cask'])`, whose groups are labelled `'B00:a'`, and so on.

```{code-cell} ipython3
rng = np.random.default_rng(3)
Pastes = pd.DataFrame({'batch': np.repeat([f'B{i:02d}' for i in range(10)], 6),
                       'cask': np.tile(np.repeat(list('abc'), 2), 10)})
Pastes['sample'] = Pastes['batch'] + ':' + Pastes['cask']
Pastes['strength'] = (60 + Pastes['batch'].map(effects(rng, Pastes['batch'], 1.3))
                      + Pastes['sample'].map(effects(rng, Pastes['sample'], 2.9))
                      + rng.normal(0, 0.8, Pastes.shape[0]))

nested = GLMM(terms=[re('batch'), re(['batch', 'cask'])],
             method='REML').fit(Pastes, Pastes['strength'])
nested.variance_components_
```

This is lme4's `(1|batch/cask)`, that is `(1|batch) + (1|batch:cask)`, and
nlme's `lme(strength ~ 1, random=~1|batch/cask)`. `re('cask')` instead
would be a random effect of 3 casks crossed with the batches, a different
model. Nested or crossed, the terms are the same kind; only the labels of
the groups differ. (`re('sample')`, on a column of the combined labels, is
the same term.)

```{code-cell} ipython3
nested.terms_[1].effects_.head()
```

## Random intercepts and slopes

Eighteen subjects' reaction times over 10 days of sleep deprivation (the
design of lme4's `sleepstudy`): each subject has its own intercept and its
own slope in `days`, about the population's line.

```{code-cell} ipython3
rng = np.random.default_rng(4)
Sleep = pd.DataFrame({'subject': np.repeat([f'S{i:02d}' for i in range(18)], 10),
                      'days': np.tile(np.arange(10.), 18)})
Sleep['reaction'] = (250 + 10 * Sleep['days'] + Sleep['subject'].map(effects(rng, Sleep['subject'], 25))
                     + Sleep['days'] * Sleep['subject'].map(effects(rng, Sleep['subject'], 6))
                     + rng.normal(0, 25, Sleep.shape[0]))

slopes = GLMM(terms=[linear('days'), re('subject'), re('subject', by='days')], method='REML')
slopes.fit(Sleep, Sleep['reaction'])
slopes.variance_components_
```

The model, as `GAM` displays it: two independent random effects per subject.

```{code-cell} ipython3
slopes
```

The two random effects are independent: this is lme4's
`reaction ~ days + (days||subject)`, i.e. `(1|subject) + (0 + days|subject)`,
and mgcv's `s(subject, bs="re") + s(subject, days, bs="re")`. (Without a
correlation between a subject's intercept and slope the model depends on
the origin of `days`: here day 0, the start of the study. The correlated
version, lme4's `(days|subject)`, is below.)

The fixed slope, with its standard error from the posterior covariance
(for the Gaussian, the generalized least squares standard error, lme4's
`vcov`):

```{code-cell} ipython3
block = slopes.design_.slices[0]
slopes.terms_[0].coef_.item(), np.sqrt(slopes.posterior_covariance_[block, block]).item()
```

Each subject's fitted line is the population's plus its two effects:

```{code-cell} ipython3
fig, axes = plt.subplots(3, 6, figsize=(11, 5.5), sharex=True, sharey=True)
fitted = slopes.predict(Sleep)
population = slopes.intercept_ + slopes.partial_effect(0, Sleep['days'])
for ax, (subject, rows) in zip(axes.ravel(), Sleep.groupby('subject')):
    ax.plot(rows['days'], rows['reaction'], 'o', ms=3)
    ax.plot(rows['days'], fitted[rows.index], 'C1')
    ax.plot(rows['days'], population[rows.index], 'k:', lw=1)
    ax.set_title(subject, fontsize=8)
fig.supxlabel('days'); fig.supylabel('reaction')
fig.tight_layout()
```

### Correlated intercepts and slopes

`correlated_random_effect('subject', slopes=['days'])` gives each subject
a vector of effects, an intercept and a slope, drawn from $N(0, \Sigma)$
with $\Sigma$ unstructured: lme4's `reaction ~ days + (days|subject)`.
REML estimates $\Sigma$ through its Cholesky factor (lme4's relative
covariance factor), so a singular $\Sigma$, a correlation of $\pm 1$, is
reached as one of its diagonal entries goes to 0.

```{code-cell} ipython3
correlated = GLMM(terms=[linear('days'), correlated_random_effect('subject', slopes=['days'])],
                 method='REML').fit(Sleep, Sleep['reaction'])
correlated.variance_components_
```

Its display shows the intercept and slope drawn together, with an unstructured covariance:

```{code-cell} ipython3
correlated
```

The covariance, and the correlation of a subject's intercept and slope:

```{code-cell} ipython3
Sigma = correlated.terms_[1].covariance_
Sigma, Sigma.iloc[0, 1] / np.sqrt(Sigma.iloc[0, 0] * Sigma.iloc[1, 1])
```

The data were simulated with independent intercepts and slopes. The
estimated correlation, about $-0.25$, is poorly determined by 18 subjects.
Unlike the independent model, this one does not depend on the origin of
`days`: shifting it changes $\Sigma$ but not the fit.

```{code-cell} ipython3
shifted = Sleep.assign(days=Sleep['days'] - 4.5)
refit = GLMM(terms=[linear('days'), correlated_random_effect('subject', slopes=['days'])],
            method='REML').fit(shifted, shifted['reaction'])
np.abs(refit.predict(shifted) - correlated.predict(Sleep)).max()
```

The predicted effects, one row per subject:

```{code-cell} ipython3
correlated.terms_[1].effects_.head()
```

## A generalized linear mixed model

Binomial counts: the number of cases of a disease among the animals of 15
herds in each of 4 periods (the design of lme4's `cbpp`), with a fixed
effect of the period and a random effect of the herd on the log odds. The
response is the proportion, with the herd sizes as weights.

```{code-cell} ipython3
rng = np.random.default_rng(5)
Herds = pd.DataFrame({'herd': np.repeat([f'h{i:02d}' for i in range(15)], 4),
                      'period': np.tile(list('1234'), 15)})
Herds['size'] = rng.integers(5, 35, Herds.shape[0])
eta = (-1.4 + Herds['period'].map({'1': 0, '2': -1, '3': -1.1, '4': -1.6})
       + Herds['herd'].map(effects(rng, Herds['herd'], 0.65)))
Herds['incidence'] = rng.binomial(Herds['size'], 1 / (1 + np.exp(-eta)))
Herds['p'] = Herds['incidence'] / Herds['size']

glmm = {}
for method in ['ML', 'REML']:
    glmm[method] = GLMM(terms=[categorical('period'), re('herd')], family=sm.families.Binomial(),
                       method=method, response_id='p', weight_id='size')
    glmm[method].fit(Herds, Herds[['p', 'size']])
pd.DataFrame({method: glmm[method].variance_components_['sd'] for method in glmm})
```

The binomial model, as `GAM` displays it:

```{code-cell} ipython3
glmm['ML']
```

The marginal likelihood of a GLMM has no closed form. Both criteria are
Laplace approximations (Wood, 2011): the random effects are integrated out
about the mode of the penalized likelihood, the fixed effects at that
mode. ML is glmer's `nAGQ=0` exactly; glmer's default, `nAGQ=1`, also
optimizes the approximation over the fixed effects, a slightly different
estimate (here the standard deviations differ by about 0.1%, the fitted
log odds by up to 0.04). Both differ from penalized quasi-likelihood
(PQL: MASS's `glmmPQL`, mgcv's `gamm`), which estimates the variances
from the working linear model of each IRLS step (Breslow and Clayton,
1993), and is biased for binary or small binomial counts. The dispersion
is 1: `scale` is fixed.

The herd effects are on the log odds scale:

```{code-cell} ipython3
ax = glmm['ML'].plot(terms=['random_effect(herd)'])[0]
ax.tick_params(axis='x', labelrotation=90)
```

`predict` gives the probabilities, for a herd seen in fitting (with its
effect) or a new one (the effect 0, the median herd):

```{code-cell} ipython3
new = pd.DataFrame({'herd': ['h00', 'h00', 'new', 'new'], 'period': ['1', '4', '1', '4']})
prob, std = glmm['ML'].predict(new, return_std=True)
new.assign(probability=prob, std_err=std)
```

## Overdispersed counts: an observation-level random effect

Counts more variable than Poisson can be modeled with a random effect per
observation, $y_i \sim \text{Poisson}(\exp(\eta_i + b_i))$ with
$b_i \sim N(0, \sigma_b^2)$: a Poisson-lognormal model (Elston et al.,
2001). In a GAM this is `re()` of a column with a different label in each
row.

```{code-cell} ipython3
rng = np.random.default_rng(6)
Counts = pd.DataFrame({'x': np.round(rng.uniform(0, 1, 200), 3),
                       'obs': [f'o{i:03d}' for i in range(200)]})
Counts['y'] = rng.poisson(np.exp(1 + Counts['x'] + rng.normal(0, 0.5, 200)))

poisson = GAM(terms=[linear('x')], family=sm.families.Poisson()).fit(Counts, Counts['y'])
pearson = np.sum((Counts['y'] - poisson.predict(Counts))**2 / poisson.predict(Counts))
print(f'Poisson fit: Pearson chi-squared over its df = {pearson / poisson.df_resid_:.2f}')

olre = GLMM(terms=[linear('x'), re('obs')], family=sm.families.Poisson(), method='ML')
olre.fit(Counts, Counts['y'])
olre.variance_components_
```

The Poisson fit has more than twice the variance it assumes. The
observation-level effect's standard deviation, about 0.5 as simulated,
accounts for it; the slope is close in both fits, and its standard error
is now larger:

```{code-cell} ipython3
def slope(fit):
    block = fit.design_.slices[0]
    return fit.terms_[0].coef_.item(), np.sqrt(fit.posterior_covariance_[block, block]).item()

pd.DataFrame({'Poisson': slope(poisson), 'Poisson-lognormal': slope(olre)},
             index=['slope', 'std err'])
```

This is `glmer(y ~ x + (1|obs), family=poisson)` (with `nAGQ=0`). Note
that `predict` at a new observation sets its effect to 0, so it predicts
the median count, $\exp(\eta)$, not the mean, $\exp(\eta + \sigma_b^2/2)$.

## A smooth with a random effect

Smooths and random effects combine freely, a *generalized additive mixed
model*: here 30 subjects followed over a year, each with its own level,
and a common nonlinear trend in time.

```{code-cell} ipython3
rng = np.random.default_rng(7)
Growth = pd.DataFrame({'subject': np.repeat([f'S{i:02d}' for i in range(30)], 12),
                       'time': np.tile(np.linspace(0, 1, 12), 30)})
Growth['y'] = (np.sin(2 * np.pi * Growth['time']) + Growth['subject'].map(effects(rng, Growth['subject'], 0.7))
               + rng.normal(0, 0.3, Growth.shape[0]))

gamm = GLMM(terms=[s('time'), re('subject')], method='REML').fit(Growth, Growth['y'])
gamm.summary()
```

The model:

```{code-cell} ipython3
gamm
```

```{code-cell} ipython3
gamm.variance_components_.loc[['random_effect(subject)', 'scale']]
```

Both smoothing parameters are estimated together, the spline's as the
variance of its coefficients' prior: this is mgcv's
`gam(y ~ s(time) + s(subject, bs="re"), method="REML")`, the GAMM fitted
as a GAM (mgcv's `gamm` and the `gamm4` package instead fit it as a mixed
model, with nlme or lme4).

```{code-cell} ipython3
axes = gamm.plot()
times = np.linspace(0, 1, 100)
axes[0].plot(times, np.sin(2 * np.pi * times) - np.sin(2 * np.pi * Growth['time']).mean(),
             'k:', label='truth (centered)')
axes[0].legend()
axes[1].tick_params(axis='x', labelrotation=90, labelsize=7)
axes[0].figure.tight_layout()
```

## What is not supported yet

- **Other structured covariances of correlated random effects.**
  `correlated_random_effect` has an unstructured covariance (nlme's
  `pdSymm`, lme4's `(x|g)`) by default, or `covariance='compound_symmetry'`
  (`pdCompSymm`) or `'diagonal'` (`pdDiag`). Banded, AR(1) or factor-analytic
  covariances of the effects are not available.
- **Different variances by group,** e.g. nlme's `pdDiag` with one
  variance per level of another categorical, or a random effect whose variance
  differs between strata (a nested sum of `re()` terms with different
  `by=` per stratum).
- **Correlated or heteroscedastic residuals,** nlme's `correlation=`
  (e.g. AR(1) within subject) and `weights=` variance functions: the
  residuals are independent, with variance the dispersion over the (known)
  prior weights.
- **Inference for fixed effects** beyond the posterior standard errors: no
  Kenward–Roger or Satterthwaite degrees of freedom, and no test of a
  variance component on the boundary.

## References

- Bates, D., Mächler, M., Bolker, B. and Walker, S. (2015). Fitting linear
  mixed-effects models using lme4. *Journal of Statistical Software*
  67(1), 1–48.
- Breslow, N. E. and Clayton, D. G. (1993). Approximate inference in
  generalized linear mixed models. *Journal of the American Statistical
  Association* 88(421), 9–25.
- Elston, D. A., Moss, R., Boulinier, T., Arrowsmith, C. and Lambin, X.
  (2001). Analysis of aggregation, a worked example: numbers of ticks on
  red grouse chicks. *Parasitology* 122(5), 563–569.
- Pinheiro, J. C. and Bates, D. M. (2000). *Mixed-Effects Models in S and
  S-PLUS*. Springer.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
