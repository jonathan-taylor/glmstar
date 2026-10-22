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

# A Poisson GAM: bike sharing

The `Bikeshare` data of the `ISLP` package are the hourly counts of rentals
of Washington's Capital Bikeshare system in 2011, with the weather and the
calendar (Fanaee-T and Gama, 2014). The question here: how does hourly
ridership depend on the hour of the day, the temperature, the weather and
whether the day is a working day?

The ISLR lab on generalized linear models (James et al., 2023, section 4.7)
fits a Poisson regression of the count `bikers` on factors for the hour and
the month, the temperature (linearly), the weather and the working day.
This page replaces the hour's 23 coefficients and the linear temperature by
smooths, estimates their smoothing by REML, and then asks what the additive
model misses: interactions, and variance beyond the Poisson's.

```{code-cell} ipython3
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP import load_data

from glmnet.gam import GAM, s, re, factor, tensor_spline, by_factor_spline, ExperimentalWarning

Bike = load_data('Bikeshare')
Bike[['bikers', 'hr', 'mnth', 'weekday', 'workingday', 'weathersit', 'temp']].dtypes
```

`hr` and `mnth` are categorical. A smooth needs the hour as a number,
0 to 23. `factor()` keeps a categorical's order of its levels, so the
months stay in calendar order. The temperature is normalized to lie
between 0 and 1.

```{code-cell} ipython3
Bike['hour'] = Bike['hr'].astype(int)
Bike['month'] = Bike['mnth']
Bike['weathersit'].value_counts()
```

One hour has heavy rain or snow: its level's effect would be estimated
from one count. It is merged with light rain or snow.

```{code-cell} ipython3
Bike['weather'] = Bike['weathersit'].astype(str).replace({'heavy rain/snow': 'light rain/snow'})
Bike.shape, Bike['day'].nunique()
```

The data are the 8645 hours of 365 days (`day` is the day of the year).

## An additive Poisson GAM

:::{warning}
REML and ML are **experimental** in glmstar 0.4: the fits, smoothing
parameters and variance components agree with mgcv, lme4 and nlme in the
tests, but the interface may change and the optimization is less tested than
the rest of `glmnet.gam`. `GAM(method='REML')` warns
(`glmnet.gam.ExperimentalWarning`); check `smoothing_.converged`, and
compare with fixed smoothing parameters or with R.
:::

Smooths of the hour and the temperature, and factors for the weather, the
working day and the month, as in the lab, with the smoothing parameters
estimated by REML:

```{code-cell} ipython3
poisson = sm.families.Poisson()
terms = [s('hour'), s('temp'), factor('weather'), factor('workingday'), factor('month')]
additive = GAM(terms=terms, family=poisson, method='REML').fit(Bike, Bike['bikers'])
additive.summary()
```

The warning is shown once; the other fits below are also REML or ML.

```{code-cell} ipython3
warnings.simplefilter('ignore', ExperimentalWarning)
additive.smoothing_.converged, dict(zip(additive.smoothing_.terms, additive.smoothing_.lam))
```

The hour has 24 distinct values, so its spline has at most 23 degrees of
freedom beyond the constant: REML gives it 22.9, nearly a factor, as in
the lab. The temperature has 48 distinct values (it is recorded to 0.02),
and REML gives its spline 44.7 degrees of freedom: a wiggly curve.

```{code-cell} ipython3
axes = additive.plot()
axes[0].figure.set_size_inches(11, 6.5)
axes[0].figure.tight_layout()
```

On the log scale, ridership is lowest at 4am and peaks at 8am and 5pm; it
rises with the temperature to a maximum and falls beyond it. The
temperature of the maximum, and the ratios of the mean count in each
weather to that of a clear hour:

```{code-cell} ipython3
temps = np.linspace(Bike['temp'].min(), Bike['temp'].max(), 200)
weather = additive.terms_[2].effects_
temps[np.argmax(additive.partial_effect('s(temp)', temps))], np.exp(weather - weather['clear'])
```

The curve is highest at 0.74; clouds or mist lower the mean count by 9%,
rain or snow by 44%. The working day's effect
is small (its two levels' effects differ by about 0.013 on the log scale):
working days have commuters, the others more riders in the middle of the
day, and the additive model gives both the same hourly profile. The bands
of the temperature's curve are narrow despite its wiggles; both are signs
of the overdispersion below.

## Interactions

### Hour and temperature

`tensor_spline('hour', 'temp')` is a smooth function of both, mgcv's
`te(hour, temp, bs="cr")`, with a smoothing parameter for each margin. Its
basis has $k_1 k_2$ columns; with the hour's 24 knots, one per hour, and 10
for the temperature, it has 240. The models are compared by their ML
criteria (`reml_score_` with `method='ML'`, the negative log marginal
likelihood, comparable between models with different unpenalized terms),
their AIC and the deviance explained.

```{code-cell} ipython3
def compare(fits):
    """ML criterion, AIC, deviance explained and degrees of freedom of fits."""
    return pd.DataFrame({'ML criterion': [f.reml_score_ for f in fits.values()],
                         'AIC': [f.aic_ for f in fits.values()],
                         'deviance explained': [1 - f.deviance_ / f.null_deviance_ for f in fits.values()],
                         'edf': [f.edf_.sum() for f in fits.values()]},
                        index=list(fits))

t0 = time.time()
additive_ml = GAM(terms=terms, family=poisson, method='ML').fit(Bike, Bike['bikers'])
tensor = GAM(terms=[tensor_spline('hour', 'temp', k=(24, 10)), factor('weather'),
                    factor('workingday'), factor('month')],
             family=poisson, method='ML').fit(Bike, Bike['bikers'])
print(f'{time.time() - t0:.1f}s')
compare({'additive': additive_ml, 'tensor': tensor})
```

The tensor product model, as `GAM` displays it:

```{code-cell} ipython3
tensor
```

```{code-cell} ipython3
ax = tensor.plot(terms=[0])[0]
ax.set_xlabel('hour'); ax.set_ylabel('temp')
ax.figure.tight_layout()
```

The ridges at 8am and 5pm are there at every temperature. How much a warm
hour (0.7) adds to a cool one (0.3), on the log scale, at 8am, 1pm and 5pm:

```{code-cell} ipython3
f = {temp: tensor.partial_effect(0, pd.DataFrame({'hour': [8, 13, 17], 'temp': temp}))
     for temp in [0.3, 0.7]}
pd.Series(f[0.7] - f[0.3], index=['8am', '1pm', '5pm'])
```

Warmth matters least to the morning commute (0.27 on the log scale, a
factor of 1.3) and most at 5pm (0.57, a factor of 1.8), 1pm between: an
interaction the additive model cannot have.

### Working days and the others

The additive model's hourly profile is one compromise between two
different days. The mean count in each hour, on working days and on the
others, and the additive model's mean fitted count:

```{code-cell} ipython3
Bike['fitted'] = additive.predict(Bike)
means = Bike.groupby(['workingday', 'hour'])[['bikers', 'fitted']].mean()
fig, ax = plt.subplots(figsize=(6, 4))
for wd, label in [(1, 'working days'), (0, 'other days')]:
    ax.plot(means.loc[wd].index, means.loc[wd, 'bikers'], 'o', ms=4, label=f'{label}: observed')
    ax.plot(means.loc[wd].index, means.loc[wd, 'fitted'], '-', color=ax.lines[-1].get_color(),
            label=f'{label}: additive fit')
ax.set_xlabel('hour'); ax.set_ylabel('mean bikers per hour')
ax.legend(fontsize=8);
```

Working days have the commute's peaks at 8am and 5pm; the other days one
broad peak in the early afternoon. A curve of the hour for each level of
the working day, `by_factor_spline('hour', by='workingday')` (mgcv's
`s(hour, by=workingday)`, with `workingday` a factor), lets the hour's
profile differ between them, the other terms shared; each curve has its own
smoothing parameter. Further, the model can be fitted separately to the
working days and the others: every term then depends on the working day.
The two parts of the data are independent with their own parameters, so
the log-likelihoods, the ML criteria and the AIC of the two fits add, and
compare with the additive model's.

```{code-cell} ipython3
by_hour = GAM(terms=[by_factor_spline('hour', by='workingday', k=24), s('temp'),
                     factor('weather'), factor('workingday'), factor('month')],
              family=poisson, method='ML').fit(Bike, Bike['bikers'])
by_hour.terms_[0].level_edf()
```

```{code-cell} ipython3
days = {1: 'working days', 0: 'other days'}
split_terms = [s('hour'), s('temp'), factor('weather'), factor('month')]
split = {wd: GAM(terms=split_terms, family=poisson, method='ML')
             .fit(Bike[Bike['workingday'] == wd], Bike.loc[Bike['workingday'] == wd, 'bikers'])
         for wd in days}
table = compare({'additive': additive_ml, 'tensor': tensor, 'hour by working day': by_hour,
                 **{f'split: {days[wd]}': split[wd] for wd in days}})
table.loc['split: total'] = table.iloc[3:].sum()
table.loc['split: total', 'deviance explained'] = 1 - (sum(f.deviance_ for f in split.values())
                                                       / additive_ml.null_deviance_)
print('converged:', all(f.smoothing_.converged for f in [additive_ml, tensor, by_hour, *split.values()]))
table
```

Under the Poisson every one of these differences is large. The tensor
product lowers the additive model's ML criterion from 135850 to 131542,
with 142 more degrees of freedom. A curve of the hour for each kind of day
lowers it to 72446, with 18 more, and raises the deviance explained from
0.793 to 0.913: its gain is about 15 times the tensor product's.
Splitting by the working day, every term interacting with it, lowers the
criterion only a little further, to 70206, with 50 more degrees of freedom
than the curves by the working day. The working day's interaction with the
hour is the one that matters. (These are
Poisson likelihoods: with the overdispersion below, all the differences are
overstated.)

## Overdispersion

A Poisson count's variance is its mean. Pearson's $\chi^2$ over the
residual degrees of freedom estimates the ratio of the variance to the mean,
if the model for the mean is right:

```{code-cell} ipython3
def pearson(fit, data):
    """Pearson's chi-squared of a fit over its residual degrees of freedom."""
    mu = fit.predict(data)
    return np.sum((data['bikers'] - mu)**2 / mu) / fit.df_resid_

pd.Series({'additive': pearson(additive_ml, Bike),
           **{f'split: {days[wd]}': pearson(split[wd], Bike[Bike['workingday'] == wd])
              for wd in days}})
```

Splitting by the working day removes much of it, not all: the counts of
the working days still vary about 9 times as much as a
Poisson's, those of the other days about 12 times.
Under the Poisson, every count is taken as that precise. That is why REML
gave the temperature a wiggly curve with narrow bands, and why every
difference in the table above is so large.

From here on, the working days, with the smoothing parameters again by REML.
There are two remedies.

- **A dispersion** $\phi$: the variance $\phi\mu$, a quasi-likelihood.
  `fit(X, y, dispersion=phi)` fixes it, here at the Pearson estimate of the
  Poisson fit; it is used by REML (the criterion is that of the Poisson
  fit with prior weights $1/\phi$) and in the standard errors. (mgcv's
  `quasipoisson` instead estimates $\phi$ with the smoothing parameters;
  `glmnet.gam` has no quasi families. Its AIC, `aic_`, is not a
  likelihood's.)
- **A random effect of the day**, `re('day')`: the hours of a day share
  its unmeasured conditions (events, the weather between the records), so
  their counts are correlated. This is a Poisson GLMM, with the day's effect
  on the log scale $N(0, \sigma_d^2)$, its likelihood by Laplace's method.
  An observation-level random effect (`re()` of a row label, as on the
  [mixed models page](gam_mixed_models.md); Elston et al., 2001) would
  model the hour's own extra variation, but it has a coefficient per row,
  5911 here, and the REML criterion's dense Hessian, $P \times P$ with $P$
  the number of coefficients, makes it slow at this size.

The day's effect leaves overdispersion within the day, so the last fit has
both: the random effect of the day, and a dispersion at the Pearson
estimate of the Poisson GLMM.

```{code-cell} ipython3
work = Bike[Bike['workingday'] == 1]
t0 = time.time()
fits = {'Poisson': GAM(terms=split_terms, family=poisson, method='REML').fit(work, work['bikers'])}
phi = pearson(fits['Poisson'], work)
fits['dispersion'] = GAM(terms=split_terms, family=poisson, method='REML').fit(
    work, work['bikers'], dispersion=phi)
fits['day'] = GAM(terms=split_terms + [re('day')], family=poisson, method='REML').fit(
    work, work['bikers'])
phi_day = pearson(fits['day'], work)
fits['day + dispersion'] = GAM(terms=split_terms + [re('day')], family=poisson,
                               method='REML').fit(work, work['bikers'], dispersion=phi_day)
print(f'{time.time() - t0:.1f}s')
pd.DataFrame({'converged': [f.smoothing_.converged for f in fits.values()],
              'dispersion': [f.dispersion_ for f in fits.values()],
              'edf s(hour)': [f.edf_['s(hour)'] for f in fits.values()],
              'edf s(temp)': [f.edf_['s(temp)'] for f in fits.values()],
              'sd re(day)': [f.variance_components_['sd'].get('re(day)', np.nan) for f in fits.values()]},
             index=list(fits))
```

The hour keeps nearly all its degrees of freedom: its pattern is strong
enough. The temperature's spline is the one that changes: it has
39.6 degrees of freedom under the Poisson, 29.9 with the day's random
effect, and 10.8 with a dispersion. The days' effects have a standard
deviation of 0.19 on the log scale (0.18 with the dispersion): a day one
standard deviation above the median has about 1.2 times its riders. The
random effect lowers Pearson's statistic from 9.2 to 6.9, the dispersion of
the last fit.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
temps = np.linspace(work['temp'].min(), work['temp'].max(), 200)
for name in ['Poisson', 'day + dispersion']:
    curve, std = fits[name].partial_effect('s(temp)', temps, return_std=True)
    line, = ax.plot(temps, curve, label=name)
    ax.fill_between(temps, curve - 2 * std, curve + 2 * std, color=line.get_color(), alpha=0.25)
ax.set_xlabel('temp'); ax.set_ylabel('s(temp)')
ax.set_title('working days: the temperature, with 2 standard errors')
ax.legend();
```

The fitted mean count, with its standard error, at 8am, 1pm and 5pm on a
clear day in April at a temperature of 0.5, and the standard error of the
difference between July's and January's effects (on the log scale), under
each model:

```{code-cell} ipython3
new = pd.DataFrame({'hour': [8, 13, 17], 'temp': 0.5, 'weather': 'clear', 'month': 'April', 'day': -1})
rows = {}
for name, fit in fits.items():
    mean, std = fit.predict(new, return_std=True)
    # the month's block: its 12 effects are Z times its 11 coefficients
    block, Z = fit.design_.slices[3], fit.design_.Z[3]
    contrast = Z.T @ np.eye(12)[6] - Z.T @ np.eye(12)[0]      # July less January
    V = fit.posterior_covariance_[block, block]
    rows[name] = [*mean, *std, np.sqrt(contrast @ V @ contrast)]
pd.DataFrame(rows, index=['mean 8am', 'mean 1pm', 'mean 5pm', 'std err 8am',
                          'std err 1pm', 'std err 5pm', 'std err July - January']).T
```

The dispersion multiplies the standard errors of the predictions by about
2.3 to 2.5, and that of the months' difference by 3.0, against
$\sqrt{\phi} = 3.0$: for the predictions, the smoother curve of the
temperature offsets part of it. The day's random effect does more, a factor of 4
to 6 on the Poisson's: the months' effects and the predictions are
averages over days, about 20 working days a month, not hundreds of
independent hours, and their standard errors grow accordingly. A new day
(`day=-1`, not seen in fitting) has the random effect's mean 0, so with it
the prediction is that of a median day: lower than the fits without the
random effect, which estimate the mean over the days.

## Predictions

The last model, the random effect of the day with a dispersion, fitted to
working days and to the other days: the mean count by hour on a clear day
in April at temperatures 0.3 and 0.7, with bands of 2
standard errors (the delta method, from the posterior covariance).

```{code-cell} ipython3
final = {1: fits['day + dispersion']}
other = Bike[Bike['workingday'] == 0]
first = GAM(terms=split_terms + [re('day')], family=poisson, method='REML').fit(other, other['bikers'])
final[0] = GAM(terms=split_terms + [re('day')], family=poisson, method='REML').fit(
    other, other['bikers'], dispersion=pearson(first, other))

hours = np.arange(24)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, wd in zip(axes, days):
    for temp in [0.3, 0.7]:
        grid = pd.DataFrame({'hour': hours, 'temp': temp, 'weather': 'clear', 'month': 'April', 'day': -1})
        mean, std = final[wd].predict(grid, return_std=True)
        line, = ax.plot(hours, mean, marker='o', ms=3, label=f'temp {temp}')
        ax.fill_between(hours, mean - 2 * std, mean + 2 * std, color=line.get_color(), alpha=0.25)
    ax.set_title(days[wd]); ax.set_xlabel('hour')
axes[0].set_ylabel('bikers per hour (a median day)')
axes[0].legend();
```

```{code-cell} ipython3
pd.DataFrame({days[wd]: [final[wd].dispersion_, final[wd].variance_components_.loc['re(day)', 'sd'],
                         final[wd].smoothing_.converged] for wd in days},
             index=['dispersion', 'sd re(day)', 'converged'])
```

The working days have the commute's two peaks, the other days one broad
peak from about 11am to 4pm. The other days also vary more from one to the
next: their random effect's standard deviation is 0.30, against 0.18. Each fit is additive on the log scale, so
the two temperatures' curves differ by a constant factor, $\exp\{f(0.7) -
f(0.3)\}$:

```{code-cell} ipython3
pd.Series({days[wd]: np.exp(np.diff(final[wd].partial_effect('s(temp)', [0.3, 0.7])).item())
           for wd in days})
```

A warm day draws more riders than a cool one, and more so when it is not a
working day: commuters ride anyway.

## What `glmnet.gam` does not have

- **Cyclic splines,** mgcv's `bs="cc"`: the hour's curve should join at
  midnight, and here 11pm and midnight are not tied. `cyclic_spline('hour',
  period=(0, 24))` ties them; this page keeps `s('hour')`.
- **Quasi families and the negative binomial under REML:** the dispersion
  of a Poisson fit can be fixed (`dispersion=`), not estimated with the
  smoothing parameters.
- **Correlated residuals:** the hours of a day are a time series. The
  random effect of the day gives them a common correlation, not one that
  decays with the time between them.

## References

- Elston, D. A., Moss, R., Boulinier, T., Arrowsmith, C. and Lambin, X.
  (2001). Analysis of aggregation, a worked example: numbers of ticks on
  red grouse chicks. *Parasitology* 122(5), 563–569.
- Fanaee-T, H. and Gama, J. (2014). Event labeling combining ensemble
  detectors and background knowledge. *Progress in Artificial
  Intelligence* 2(2–3), 113–127.
- James, G., Witten, D., Hastie, T., Tibshirani, R. and Taylor, J. (2023).
  *An Introduction to Statistical Learning, with Applications in Python*.
  Springer.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
