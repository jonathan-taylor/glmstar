# Changelog

## Unreleased

### New features

- **Gaussian process smooths in `glmnet.gam`:** `gp(x)`, `gp(lon, lat)` (any
  number of columns), the low rank Gaussian process (kriging) smoother of
  Kammann & Wand (2003), as mgcv's `s(..., bs="gp")`: its knots, linear
  trend (or none, `trend=False`), Matern, power exponential and spherical
  correlation functions, default range and basis dimension, and mgcv's `m`
  codes. It takes `lam`, `df`, or neither under REML/ML, with both solvers,
  every family, weights and offsets; its basis spans mgcv's, so fits at
  fixed `sp`, edf, standard errors and REML/ML estimates agree with mgcv's;
  at full rank without a trend it is scikit-learn's
  `GaussianProcessRegressor` at fixed hyperparameters. `plot` draws a
  two-column term as filled contours. A docs page, "Gaussian processes as
  GAM terms".

## 0.4.0 (2026-10-21)

### Packaging

- **Windows wheels** (x64, built with MSVC, the C++ runtime linked
  statically). glmstar had none: pip built it from the sdist, and with a
  MinGW gcc on the PATH the extensions needed DLLs that Python doesn't find
  ("DLL load failed while importing _elnet_point"). Each wheel is tested to
  import and fit, and the tests run on Windows. Requires coxdev 0.3.1, whose
  Windows wheels import (#152).

### Fixes

- **Log-likelihood with zero weights:** `GLM.loglike_` (and the AIC and BIC),
  and `glmnet.gam`'s, are over the observations of positive weight: those of
  the fit without the others. A weight of 0 gave the Gaussian's, Gamma's and
  inverse Gaussian's `-inf` (as R's `logLik`), with a divide-by-zero warning (#149).
- **`glmnet.gam` REML/ML for the Cox family converged short** at times: near
  the optimum a Newton step lowers the criterion by less than its precision.
  Such a step is now taken when it halves the gradient, and the differences
  of the gradient have a step suited to its precision (#149).
- **`glmnet.gam` REML/ML: unconverged inner fits.** Near a smoothing
  parameter of 0 the penalized likelihood can have its maximum far out (a
  Cox spline with nearly as many knots as events), the inner fits do not
  converge, and the criterion at them, meaningless and very negative, drew
  the search (BrainCancer's `s(gtv)` got 57 degrees of freedom; mgcv gives
  1). There the criterion is now larger than any seen, with its gradient
  towards larger smoothing parameters (#149).
- **The C++ solvers accept read-only arrays** for the inputs they only read:
  the data, indices and pointers of a sparse (CSC) `X`; the dense `X`,
  weights, limits, centers and scales of the point solver behind `GLMNet` and
  `RegGLM`; and the penalty factors, excluded variables, lambda values and
  survival data of the path solvers. (The path solvers standardize a dense
  `X`, `y` and weights in place, so those stay writeable; the estimators pass
  copies.) A sparse matrix with read-only buffers, such as a `dgCMatrix`
  passed from R by reticulate or a memory-mapped array, raised `ValueError:
  array is not writeable` in `GLMNet`, `RegGLM` and the path estimators
  (`LogNet`, `GaussNet`, `FishNet`, `MultiGaussNet`, `CoxNet`). It is still
  mapped without a copy (#145).

### Changed

- **`anova` is the F-test for least squares:** for Gaussian fits with the
  identity link, an estimated dispersion and the model-based covariance (where
  the coefficients' statistics are t), the table has `F` and `P(>F)` columns:
  the drop in deviance (residual sum of squares) over the drop in residual
  degrees of freedom, divided by the largest model's dispersion, against F on
  those degrees of freedom and the largest model's residual degrees of
  freedom, as R's `anova.lm` and statsmodels' `anova_lm`. Other fits keep
  `chi2` and `P(>chi2)` (R's `anova(test="Chisq")`) (#143).

### New

- **Documentation: a page on GAMs** (`docs/gam.md`), after the GAM part of
  ISLR's lab on nonlinear modeling: the Wage GAM and its plot, smoothing by
  `df` or `lam`, loess, backfitting and the joint solve, standard errors, a
  logistic GAM, REML, random effects and a Cox GAM (BrainCancer); and
  references (Hastie and Tibshirani; Buja, Hastie and Tibshirani; Wood; ...)
  in `glmnet.gam`'s docstrings (#151).
- **`GAM.plot()`** (and `glmnet.gam.plot_term`): the fits of a GAM's terms, one
  panel each, as R's `plot.Gam`: a curve over the range of the column with
  dashed bands of 2 pointwise standard errors (from the posterior covariance;
  not for loess) and a rug of the data, or one point per level with error
  bars for factors and random effects (#150).
- **`glmnet.gam`: generalized additive models**, fitted by local scoring on
  glmstar's IRLS, each working problem solved by (modified) backfitting, as
  R's `gam::gam`. Terms: smoothing splines (`s()`, by `df` or `lam`, R's
  knots, from `scatter_smooth`), loess (`lo()`), linear terms, factors and
  blocks of columns, each naming a column of `X`. Families: those of `GLM`
  (statsmodels'), with weights and offsets, and the Cox model (`CoxFamily`:
  Efron or Breslow ties, start times, strata). Checked against `gam::gam`,
  `mgcv` and `coxph`. New dependency: `scatter-smooth` (the smoothers).
  (Developed in ISLP; ISLP's `ISLP.gam` will build the terms from a
  ModelSpec.) (#146)
- **`glmnet.gam`: the joint (mgcv-style) solve, random effects and
  standard errors.** `GAMControl(solver='joint')` solves each working problem
  as one penalized least squares fit on the terms' bases (the same fit as
  backfitting; for Cox, Newton steps with the full Hessian of the partial
  likelihood). `re(group)` and `re(group, by=x)` are random intercepts and
  slopes, so mixed models are GAMs. With a basis for every term, the fit has
  mgcv's Bayesian covariance (`posterior_covariance_`, `frequentist_covariance_`,
  `edf_influence_`) and standard errors: `predict(..., return_std=True)`,
  `partial_effect(..., return_std=True)`. Checked against mgcv (`sp` fixed)
  and `coxph` (#147).
- **`glmnet.gam`: smoothing parameters by REML or ML**, as mgcv's
  `gam(method=...)`: `GAM(method='REML')` or `'ML'` estimates the smoothing
  parameters of the `s()` and `re()` terms given neither `lam` nor `df` (and
  the dispersion), by Laplace's method, with exact gradients (JAX: `pip
  install glmstar[gam]`; for the Cox family, mgcv's `cox.ph`, numpy).
  Random effects' variances are `variance_components_`. Checked against
  mgcv, lme4 (`lmer`, `glmer`) and nlme (`lme`) (#148).
- **`CoefPath.df_`:** the coefficient path as a data frame, indexed by
  -log(lambda), with the feature names as columns (a column per feature and
  response for multi-response paths) (#144).

## 0.3.2 (2026-10-20)

### New

- **`GLMControl(convergence='eta')`:** stop the IRLS iterations on the
  relative change in the linear predictor (in norm) instead of the objective
  (`'objective'`, the default, unchanged) (#138).
- **`n_iter_`** of the estimators fitted by IRLS (`GLM`, `CoxLM`, `RegGLM`, ...):
  the number of IRLS iterations (#138).
- **`glmnet.irls.IRLS`** documents what it needs of the family, regularizer and
  state, and takes `info`, a dict filled with the number of iterations and of
  halved steps. A regularizer with `has_objective = False` (for smoothers that
  do not minimize an objective) skips the descent check, and its iterations
  stop on the change in the linear predictor (#138).

### Changed

- **`GLM.summary()` is replaced by the function `glmnet.glm.summarize(fit,
  conf_int=False, level=None, full=False)`**, for `GLM` and `CoxLM`: the
  coefficient table (`coef`, `std err`, `t` or `z`, `P>|t|` or `P>|z|`, and
  with `conf_int` or a `level` the confidence intervals), unrounded, as a
  DataFrame. `full=True` also prints a description of the fit above it (the
  family, observations, degrees of freedom of the model and residuals,
  deviance and null deviance, dispersion, log-likelihood, AIC, BIC and the
  covariance); by default only the table is returned. ISLP's `summarize`
  parsed the statsmodels-style tables of `summary()`; it can call this
  directly (#141).
- **`GLM.residuals(kind)` and `GLM.leverage(X)` are now the functions
  `glmnet.glm.residuals(fit, kind)` and `glmnet.glm.leverage(fit, X)`**, as
  `confidence_interval`, for `GLM` and `CoxLM`; the methods are removed. The
  default `kind` is still `'response'` for a `GLM` and `'martingale'` for a
  `CoxLM`, and the values are unchanged (#139).
- **`glmnet.irls` cleanup:** dead code removed, and "ISLR" corrected to "IRLS"
  in the log messages. The `boundary` returned by `IRLS` is True if any step
  was halved at a boundary, not only the last; an error from step halving
  names the check that failed. Fits are unchanged (#138).

### Fixes

- **`predict` docstrings of `GLM`, `BinomialGLM` and `BinomialRegGLM`:** the
  prediction is an array even if `X` is a DataFrame, as for sklearn's
  estimators (since #107; before, pandas' `@` made it a Series on the index
  of `X`), now tested (#137).

## 0.3.1 (2026-10-19)

### New

- **`glmnet.glm.prediction_interval(fit, X_new, alpha, weights)`:** prediction
  intervals for an unpenalized Gaussian `GLM`, as R's
  `predict.lm(interval="prediction")` (#119).
- **`feature_names_`** of `GLM` / `CoxLM`: the names of the entries of `params_`
  and of the rows and columns of `covariance_`, `np.asarray(params_.index)`
  (with `'intercept'` if fitted, without the excluded variables), unlike
  sklearn's `feature_names_in_` (the columns of the `X` of `fit`) (#128).

### Changed

- **A ModelSpec `intercept` column is dropped without a warning:** it is
  ISLP's convention, and the warning was raised at every fold of a
  cross-validation and every candidate of a stepwise selection. Other
  constant columns are still excluded with a warning (#136).
- **`GLM.score` docstring:** the weights default to ones, so the score is the
  log-likelihood summed over the observations (as `CoxLM.score`), not an
  average; the code is unchanged (#136).
- **`GLM.conf_int(X_new)` is now the function
  `glmnet.glm.confidence_interval(fit, X_new, alpha)`**, next to
  `prediction_interval`; the method, added in 0.3.0, is removed. For the
  short name, `from glmnet.glm import confidence_interval as conf_int` (#119).
- **`anova` is no longer imported into `glmnet`:** use
  `from glmnet.glm import anova`, as for the other GLM-only functions (#119).
- **Labelled outputs of the unpenalized `GLM` / `CoxLM` analysis** (the values
  are unchanged):
  - indexed by the coefficients: `params_` (Series) and `summary_` (DataFrame)
    as before; `covariance_` stays an array, labelled by the new
    `feature_names_` (above);
  - at the rows of `X_new`: `predict_se` is a Series on its index if it is a
    DataFrame, else an array; `confidence_interval` and
    `prediction_interval` are always DataFrames (columns `estimate`, `lower`,
    `upper`), now on the index of a DataFrame `X_new` (a `RangeIndex` otherwise, as before);
  - at the observations of the fit: `residuals` and `leverage` are Series on
    the index of `X` if it was a DataFrame, else arrays (#128).
- **The feature names are checked at predict, as sklearn does:** if `fit` saw
  a DataFrame, `predict`, `predict_se`, `confidence_interval`,
  `prediction_interval` and `leverage` raise a `ValueError` for a DataFrame
  whose columns are missing, extra or reordered, and warn for an array (or for a DataFrame when `fit`
  saw an array). A ModelSpec `intercept` column, dropped at `fit`, is dropped
  before the check (#128).

### Fixes

- **`loglike_`, `aic_` and `bic_` of a binomial `GLM` with counts** (trials,
  successes) or proportions with weights: they took one trial per row, so they
  lacked R's log(choose(trials, successes)) (AIC 648.4 for R's 352.1 in the
  tests). They now match R's `logLik` / `AIC` for grouped data; 0/1 responses
  are unchanged (#133).
- **The cross-validated deviance of a stratified Cox model**
  (`cross_validation_path`'s "Cox Deviance" and "Cox Deviance (Difference)"):
  `CoxScorer` / `CoxDiffScorer` dropped `strata_id` when building each fold's
  family, so the folds were scored without the strata (off by up to 0.5 from
  `cv.glmnet` in the tests). It now matches R's `cv.glmnet(grouped=FALSE)` (#134).
- **lambda_max of a `GLMNet` path with unpenalized variables** (e.g. a zero
  penalty factor) on a non-canonical link: their fit used the path's
  tolerance (`epsnr=1e-6`), which stops early for linear convergence, so
  every default lambda was off by about 1e-5 relative (cloglog). It now uses
  `GLMControl`'s tolerance and matches R's `glmnet` (#135).
- **The dispersion of `GLM` for the families with a free dispersion** (Gamma,
  inverse Gaussian, Tweedie, Gaussian with a non-identity link): it was fixed at
  1, so their standard errors, intervals and tests were wrong (e.g. 1.6 times too
  large for a Gamma fit). It is now estimated by Pearson's chi-squared over the
  residual degrees of freedom, as R's `summary.glm` and statsmodels do; it stays 1
  for the binomial, Poisson, negative binomial and Cox (#119).
- **t or z:** `summary_`, `summary()` and the confidence intervals use t on `df_resid_` only for
  the Gaussian with identity link, an estimated dispersion and `sandwich=False`,
  and z otherwise; a given `dispersion=` or the sandwich used to keep t (#119).
- **The Gaussian's log-likelihood** (`loglike_`, `aic_`, `bic_`) is at the maximum
  likelihood estimate of its variance for every link, as R's `logLik.glm` (#119).
- **`glmnet.__version__`** was `"unknown"` in an installed glmstar: it
  looked up the distribution `glmnet`, not `glmstar` (#116).
- **Unpenalized `GLM` fits:** `GLMControl.epsnr` defaults to `1e-10` (it
  stopped early, e.g. near separation); the fit starts at the mean of the
  response and halves steps leaving the family's domain, so the Gamma
  (inverse link), identity and sqrt links fit; constant columns of `X`
  other than `intercept` are excluded with a warning instead of giving a
  singular fit (any constant column, for Cox); `CoxLM` and `RegCoxLM` have a
  `score` on new data. The penalized path fits are unchanged (#124).
- **`GaussNet.get_fixed_lambda` for `alpha < 1`:** `GaussNet`, as R's
  `glmnet(family="gaussian")`, fits `y / s_y` at `lambda / s_y`, dividing
  the ridge part of the penalty by `s_y` (the weighted sd of `y`, now
  `response_scale_`); `RegGLM` and the IRLS `GLMNet`, as R's
  `family=gaussian()`, do not. Both match R and are unchanged, but
  `get_fixed_lambda` now returns the `RegGLM` solving the path's problem
  (before, it differed for `alpha < 1`). The other families agree (#123).
- **`CoxLM` converges at the default control:** its Newton steps use the
  full information of the partial likelihood instead of its diagonal, which
  converged slowly and stopped far from the maximum (BrainCancer:
  coefficients off by 0.56 from `coxph`). `GLM` and `CoxLM` warn when
  they stop at `mxitnr` without converging (#122).
- **`RegGLM` is an sklearn estimator:** `lambda_val` defaults to 0, so
  `RegGLM`, `GaussianRegGLM`, `BinomialRegGLM` and `RegCoxLM` construct
  without arguments, clone and tune in `GridSearchCV` / `Pipeline`.
  `GaussianRegGLM` no longer fails at construction (its family is checked
  at `fit`), and `BinomialRegGLM.predict` works (and takes a DataFrame) (#121).
- **`predict` takes a DataFrame `X`** for `GLMNet`, the C++ paths
  (`GaussNet`, `LogNet`, `FishNet`, `CoxNet`, ...), `CoxNetIRLS` and the
  multi-response paths, as `fit` does, so `cross_validation_path`,
  `cv_predict` and `score_path` work with a DataFrame. The multi-response
  `predict` takes a sparse `X` again (#120).

### Removed

- **`Design.add_constant(X)`**, added in 0.3.0: use statsmodels'
  `sm.add_constant(X)` instead, or fit the intercept with
  `fit_intercept=True` (#118).

### Tests

- **An estimator sweep** (`tests/test_estimator_sweep.py`): every estimator
  is checked for `clone`, DataFrame and sparse `X`, `GridSearchCV`, the
  lambda scale and convergence (#117).
- **More families compared with R's `glmnet(family=<family object>)`** in
  `tests/test_irls_r_parity.py`: binomial with the cloglog and cauchit links,
  the negative binomial with known theta (`MASS::negative.binomial`), the
  inverse Gaussian (log and canonical links), Gamma with the identity and
  inverse links, Gaussian with the log link, and Tweedie (`statmod::tweedie`,
  skipped without statmod); the options (weights, offset, penalty factors,
  limits...) also for cloglog and the negative binomial. One known
  discrepancy is an `xfail`: with an unpenalized variable and a non-canonical
  link, the null fit for `lambda_max` uses the path's `epsnr=1e-6`, so the
  default lambda values are about 1e-5 (relative) off R's (#132).
- **The unpenalized `GLM` analysis against R's `glm`:** `summary_` (names,
  estimates, standard errors, statistics, p-values, intervals),
  `dispersion_`, `covariance_`, `df_resid_`, `deviance_`, residuals,
  `loglike_` / `aic_` / `bic_`, `predict_se`, `confidence_interval` and
  `prediction_interval` against `summary.glm`, `vcov`, `logLik`, `predict.glm`
  and `predict.lm`, for the Gaussian (identity and log), binomial (logit and
  probit), Poisson, Gamma (log and inverse), inverse Gaussian and negative
  binomial families, with prior weights, an offset and without an intercept.
  The tests pin down where the two differ by design: z rather than t for the
  families with an estimated dispersion other than the Gaussian with identity
  link, the negative binomial's dispersion fixed at 1, the dispersion not
  counted in AIC / BIC, and the Gamma / inverse Gaussian log-likelihood at the
  Pearson dispersion (#131).
- **Cox against R on combined features**
  (`tests/compare_R/test_cox_edges_r_comparison.py`): (start, stop] data
  with strata, weights, an offset and tied times, dense and sparse, for the
  `CoxNet` path and `predict(newoffset=)` (vs `glmnet`), the deviance (vs
  `coxnet.deviance`, both tie methods, `pred=` and `x=, beta=`), the C index
  of a stratified response (vs `Cindex`), the ungrouped cross-validated
  deviance (vs `cv.glmnet(grouped=FALSE)`) and `CoxLM` (vs `coxph` with
  `strata()` and `offset()`: coefficients, standard errors, log-likelihood).
  The cross-validated deviance of a stratified model is a strict xfail:
  `CoxScorer` drops the strata (#130).
- **Binomial counts and proportions against R** (`tests/compare_R/test_binomial_counts_r_comparison.py`):
  `LogNet` with (trials, successes) or proportions + weights, and `GLMNet(family=Binomial())`
  with proportions + weights, against R's `glmnet` with `cbind(failures, successes)`
  (lambda values, coefficients, intercepts, dev.ratio, link / response predictions, with and
  without an offset) and `cv.glmnet`; the same path for counts, proportions + weights and the
  0/1 data with one row per trial; the unpenalized `GLM` / `BinomialGLM` against R's
  `glm(cbind(s, f) ~ X, binomial)` (coefficients, standard errors, p-values, deviance, null
  deviance, predictions and their standard errors). The AIC / log-likelihood of grouped data
  differs from R's (strict xfail) (#129).

### Documentation and maintenance

- **The inference vignettes fit with the estimators:** the unpenalized fits
  in the Cox, score bootstrap, overdispersed Poisson, probit and
  autocorrelated-errors vignettes use `GLM` / `CoxLM` (with `summary_`, and
  `sandwich=True` for the robust Cox standard errors) instead of statsmodels,
  `lstsq` or a hand-written solver; the Cox pages' one-step Newton refits are
  replaced by the relaxed fit of `CoxNetIRLS` and a `CoxLM` fit on the
  selected features, and their score residuals come from `coxdev` (#127).
- **Smaller wheels:** `glmnet/test_data`, 144 unused pickles (5.6 MB) from
  the first packaging commit, is removed. Nothing read them, but they were
  installed with every wheel and sdist (#115).

## 0.3.0 (2026-10-16)

### Deprecated

- **Renamed, for the lasso / elastic net:** the modules
  `glmnet.inference.glm_problem` and `glm_inference` are now `glmnet_problem`
  and `glmnet_inference`, and the function `glmnet_problem` is now
  `problem_from_solution`. The old names still work, with a
  `DeprecationWarning`, and will be removed in a later release (#109).

### New

- **The standard analysis of an unpenalized `GLM` fit**, the same for
  every family, Cox included (`CoxLM`), so that ISLP's labs can use `GLM`
  where they use `sm.OLS` / `sm.GLM` (#107):
  - `summary_`: coefficients, standard errors, t or z, p-values and 95%
    intervals; `summary(alpha)` for ISLP's `summarize`;
  - `sandwich=True`: the robust (HC0) covariance, with the expected
    information as the bread as in R's `sandwich(glm)`; for Cox, from
    coxdev's score residuals, as `coxph(robust=TRUE)`;
  - `conf_int(X_new)` and `predict_se(X)` for the linear predictor,
    `residuals(kind)`, `leverage(X)` (R's `hatvalues`), and `loglike_`,
    `aic_` and `bic_`;
  - `anova(*fits)`, the analysis of deviance, as R's
    `anova(test="Chisq")`;
  - ISLP `ModelSpec` designs: the constant `intercept` column is dropped
    with a warning; intercept-only fits work.
- **`Design.add_constant(X)`:** `[1, X]` as a `Design` on the original scale
  of `X`, like statsmodels' `add_constant` but without copying a dense `X`
  (#108).
- **`warm_state` works**, on the original scale of the features, for
  `GLMNet`, `RegGLM` and `GLM`; `GLMNet.fit` starts its first lambda from
  it (#101).
- **Requirements:** coxdev >= 0.3.0. `c_index` now uses coxdev's C-index,
  and glmstar's own concordance extension is removed (#103).

### Fixes

- **Unpenalized `GLM`** (#107):
  - the Gaussian dispersion was fixed at 1, so the standard errors were
    wrong; it is now deviance / `df_resid_`;
  - `exclude` was ignored by the Newton step;
  - `BinomialGLM.predict('class')` raised an error.
- **Refitting:** each `fit` starts afresh. Refitting an estimator on new
  data used to reuse the design and family of the earlier fit: `GLM`,
  `BinomialGLM`, `CoxLM`, `RegGLM`, `ElNet` and the C++ path estimators
  gave wrong fits or failed, and a refit `CoxNetIRLS` used the old data's
  null deviance and lambda path (#111).
- **`GLM` on a sparse `X` with `standardize=True`** returned coefficients
  on the wrong scale (#110).
- **Elastic net inference:** `Sigma` no longer includes the ridge term of
  the Hessian, which overstated the score's variance for `alpha < 1` (#90).
- **Binomial means** are clipped to `[1e-9, 1 - 1e-9]` in `glm_moments`, as
  in `GLMNet`'s fit; the variance used to underflow to 0 for large `|eta|`
  (#108).
- **Weights:** the family sees only the normalized weights, so a
  `CoxNetIRLS` fit no longer rebuilds its `CoxDeviance` several times
  (#104).

### Documentation and maintenance

- The Cox inference vignettes fit with `CoxNetIRLS` at lambda = 0 (#101).
- The workflows' actions run on Node 24 (#112).
- **Tests:** the warnings that R's glmnet gives too (`pmax` exceeded,
  relaxed refits that do not converge) are marked as expected, so the
  test suite's warnings summary is empty. The Cox case of
  `test_big_glm_exclude_limits_matches_R` compared two empty models: glmnet's
  Cox fit at lambda = 0 does not converge when a limit binds, in R as in
  glmstar. It is now a strict expected failure (#114).

## 0.2.0 (2026-10-14)

### Breaking changes

- **Selective inference has moved to
  [lassoinf](https://github.com/jonathan-taylor/lassoinf).** The selective
  inference module is removed from glmstar; the docs have a lassoinf quick
  start (#58). `glmnet.inference` still extracts the problem a fit solved,
  for lassoinf to use (#61).
- **Cox models:**
  - `predict` gives the link and the response as R's glmnet does (#83).
  - Ties default to Breslow's method, as in R's glmnet (#83).
- **Exclusions and penalty factors:** one hook, `get_penalty_factor`, now
  covers R's `exclude` and `penalty.factor`. Prefilter subclassing is
  deprecated (#66, #87).
- **Requirements:** Python >= 3.10 (#93), coxdev >= 0.2.0 (#98),
  pandas >= 1.4, and matplotlib, which is now a dependency (#92).

### New

- **The relaxed lasso**, as R's `relax=TRUE` (#77).
  `cv_coefs` / `cv_predict` give results at the cross-validated choice,
  with extras for relaxed fits (#80).
- **`big_glm`:** unpenalized fits with the glmnet machinery, as R's
  `bigGlm` (#72).
- **`make_x`:** builds a design matrix from a data frame, as R's
  `makeX` (#67).
- **Assessment:** `assess`, `confusion` and `roc` helpers; `score_path`
  includes the offset (#62).
- **`nonzero()`:** the indices of the nonzero coefficients at each lambda,
  as R's `type="nonzero"` (#64).
- **`predict` takes an offset**, as R's `newoffset` (#68).
- **Path options:**
  - `pmax` on the C++ path estimators (#71);
  - `get_fixed_lambda` for the multi-response paths (#69);
  - a progress bar only when `control.itrace` is set, as R's
    `trace.it` (#65).
- **`CoefPath.plot(label=...)`**, as R's `plot(fit, label=TRUE)` (#63).
- **A linear term `G'beta`** for `GLMNet` (#91).
- **`sample_score`** for GLM and Cox families (#99).
- **Faster solves:**
  - the C++ solves, including Cox, release the GIL, so they can run in
    threads (#46, #74);
  - the coefficient path is built in place in the solver's buffer (#47).
- **Documentation:**
  - migrated to Jupyter Book 2, with a Sphinx API reference (#89);
  - new vignettes: R-style `exclude` (#60), custom GLM families (#86),
    the score bootstrap and carving (#91), and resampling the score in
    Cox models (#99);
  - the inference calibrations are run locally, with their saved results
    and provenance (#95).

### Fixes

- **IRLS `GLMNet`:**
  - matches R's `glmnet.path` in its penalty factors, start and stopping
    rules (#85);
  - `exclude` is no longer off by one, and works at lambda 0 (#75);
  - works on scipy sparse matrix input, as do `GLM` and `RegGLM` (#81).
- **Limits:** the user's limits are no longer overwritten by glmnet's
  placeholder (#84).
- **Binomial links:** non-logit links are no longer treated as logit (#94).
- **Other fixes:**
  - `get_fixed_lambda` for the C++ path estimators (#59);
  - `MultiClassNet.predict` with a scalar `interpolation_grid` (#70);
  - `glmstar_problem` uses the fitted `penalty_factor_` when present (#73).
- **Tests:** Efron Cox paths compared with R's glmnet are strict expected
  failures. R's glmnet 5.1 compiles an older coxdev, whose Efron
  `diag_hessian` is wrong (#100).
