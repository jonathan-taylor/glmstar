# Changelog

## Unreleased

### New features

- **REML and ML with one large random effect, factored by blocks**
  (`glmnet.gam.blocked`, `GAMControl(reml_backend='auto' | 'dense' |
  'blocked')`). The random effect with the most levels is kept implicit
  (its basis one code and value per row, its penalty diagonal), and the
  Hessian of the coefficients is factored by its Schur complement on the
  others: the criterion, its gradient and mgcv's edf exactly (to 1e-12 of
  the dense backend), without the `P x P` matrix. `'auto'` chooses it above
  `max_dense_coef` coefficients when one random effect has all but at most
  that many. The inner fit is then by backfitting, polished. A random
  intercept of 5 x 10^4 levels in 1.5 x 10^5 rows: REML in under 5 minutes
  and about 3 GB, where the dense Hessian alone would be 18 GB (#190).
- **`GLMM`: a GAM whose smoothing parameters are estimated.** It has a
  `GAM`'s terms, its penalized terms random effects, and `method='REML'`
  (the default), `'ML'` or a `SmoothingMethod`; `smoothing_`,
  `variance_components_` and `reml_score_` are its own. A `GAM`'s smoothing
  is fixed (#193).
- **Some smoothing parameters held fixed in a GLMM:** besides a term given
  `lam`, a term with several penalties takes a list with `None` for those
  to estimate (`tensor_spline('x', 'z', lam=[1e-2, None])`, a varying
  term's levels, `penalized_columns`) (#193).
- **`Smoother` and `PenalizedSmoother`**, the terms' base classes (the
  former `Term`). A `Smoother` is what backfitting needs (`setup`,
  `smooth`, `edf`, `finalize`, `predict`); a `PenalizedSmoother` is given
  only a basis and its penalties (`basis()`, `penalty_matrices()`, `lam`)
  and inherits the smoothing, degrees of freedom and prediction, and with
  them the joint solve, standard errors and a GLMM's estimates. The GAM
  page shows how to write both (#194).
- **Varying-coefficient terms, `varying(term, by=...)`** (`VaryingSmoother`,
  mgcv's `by=`; Hastie and Tibshirani, 1993). Any smooth varies with a
  categorical, a curve per level (`varying(s('x'), by='f')`, mgcv's `s(x,
  by=f)`; include `categorical('f')`), or with a number, `z f(x)`
  (`varying(s('x'), by='z', as_categorical=False)`, mgcv's `s(x, by=z)`, not
  centered). `by` is categorical by default, whatever its values (one coded
  0 and 1 too). Each level's curve is constrained as mgcv's: it sums to 0
  over all the rows' values, so the ML criterion is exactly mgcv's. A term
  with a basis (`s()`, `cyclic_spline()`, `tensor_spline()`,
  `gaussian_process()`, `re()`, ...) gives its basis per level, with its
  penalties per level or, with `pooled_lam=True`, shared (mgcv's `id`); its
  smoothing parameters are given or estimated by REML/ML. `lo()` varies by
  backfitting: a loess per level, or Hastie and Tibshirani's update for a
  number. Plots draw a curve per level, or the varying coefficient; the
  model display shows it. Against mgcv (categorical and numeric `by`; `s()`,
  cyclic splines and tensor products; per level and pooled; REML and ML):
  the ML criterion to 1e-7 (tensor products to 1e-9), Gaussian REML fits to
  1e-7 (#188).
- **`cubic_regression_spline(x, k=10)`** (`CubicRegressionSplineSmoother`),
  mgcv's `s(x, bs="cr")`: the low-rank natural cubic spline with values at
  `k` knots (mgcv's `place.knots`) (#188).

### Deprecated

- **`GAM(method=...)`:** `'fixed'` is a GAM's only smoothing (drop it);
  `'REML'`, `'ML'` and a `SmoothingMethod` are a `GLMM`'s
  (`GLMM(method=...)`). Both still work, with a FutureWarning, as does
  `GLMM(method='fixed')` (#193).
- **`re()` is now `random_effect()`**, an explicit name (the docs import
  it as `re`: `from glmnet.gam import random_effect as re`); `re()` still
  works, with a FutureWarning. Its terms are named `random_effect(...)`, as
  are their rows of `variance_components_` and `smoothing_.terms` (#194).
- **The smoothers' former names:** `Term`, `SplineTerm`, `LoessTerm`,
  `SlopeTerm` (now `LinearSmoother`), `ColumnsTerm`, `RandomEffectTerm`,
  `CorrelatedRandomEffectTerm`, `PenalizedColumnsTerm`, `GPTerm`
  (`GaussianProcessSmoother`), `TensorTerm` (`TensorSplineSmoother`) and
  `CyclicSplineTerm` are now `...Smoother`; the old names still work, with
  a FutureWarning, as does `FactorTerm` (`CategoricalSmoother`) (#194).

### Removed

- **`by_factor_spline` and `FactorSplineTerm`** (new in 0.5.0): use
  `varying(cubic_regression_spline(x, k=k), by=f, pooled_lam=...)`, the
  same model. Its curves are now constrained as mgcv's (summing to 0 over
  all the rows' `x`, not over the level's rows): the fits are unchanged,
  the curves shift by constants that move to `categorical(f)`'s effects,
  and the ML criterion is exactly mgcv's (it depends on the
  parametrization of the penalties' null space; REML does not) (#188).

### Changed

- **`categorical()` (`CategoricalSmoother`) is the categorical term's name**,
  as pandas': `factor()` and `FactorTerm` (R's name) still work, with a
  `FutureWarning`, and their terms keep the names `factor(...)`; they will
  be removed. The docs use `categorical()` (#188).
- **REML/ML:** a term's penalties are grouped by the coefficients they
  share; each group (a level of a varying term) is evaluated on its own
  block, so a tensor product by a categorical needs no dense log
  determinant over all the levels (#188).

### Documentation

- **The GAM page is about fixed smoothing**, with a section on the
  smoothers' structure and on writing one (a step function, a Fourier
  basis, a P-spline from `SplineTransformer`, a tree); **a new page,
  Generalized linear mixed models,** has REML and ML, holding some
  parameters fixed, other methods and large random effects (#194).
- The GAM page's section on factor-by smooths is now on varying
  coefficients (per level, pooled, and a numeric `by`); the R translation
  page lists `varying()` and `cubic_regression_spline()`. The Bikeshare
  page uses `varying()` (its import of `by_factor_spline` had been lost in
  a merge, so the page did not run in 0.5.0) (#188).

### Fixes

- **`varying(lo(x), by=...)` under REML/ML** raises that REML and ML need a
  basis, as before #189: a varying term has a basis when its inner term does
  (`has_basis`) (#191).
- **Large random effects without dense matrices.** Fitting a GAM with a
  random effect of many levels no longer forms its `R x R` penalty (each
  term says whether it is `penalized`, has a basis, `has_basis`, and how
  many coefficients, `n_coef()`, without forming them), and the dense
  covariance is formed only for at most `GAMControl(max_dense_coef=20000)`
  coefficients: above, `posterior_covariance_`, `edf_influence_` and
  `design_` are None, standard errors raise saying so, plots have no bands,
  and REML and ML (whose Hessian is dense) raise. The fit,
  `fixed_effects_`, `summary()` and predictions do not need them. On
  simulated ratings of the size of Ghosh, Hastie and Owen's (5 million
  ratings, 750,000 customers, 4,000 items; the docs' `ratings()`
  generator, not their data, which is not public), the crossed-effects fit
  of the docs takes 56 s and 2.9 GB (it would have taken about 14 GB); at
  1 million ratings 0.8 GB, down from 2.9 GB (#189).

### Packaging

- **Fewer CI runs.** The tests run on pull requests (Linux only) and on
  pushes to main (macOS, Linux and Windows), not on every push to every
  branch; by hand too (`workflow_dispatch`). A new push cancels the run in
  progress for its branch, and changes only to the docs or Markdown files
  do not run them (#187).

## 0.5.0 (2026-10-22)

### New features

- **Overdispersion under REML and ML** (`glmnet.gam.families`):
  `QuasiPoisson()` and `QuasiBinomial()` (R's `quasipoisson`,
  `quasibinomial`) have a free dispersion, estimated with the smoothing
  parameters as mgcv's (Pearson's with fixed smoothing, as R's `glm`; no
  log-likelihood or AIC). `NegativeBinomial(theta=None)` estimates the
  negative binomial's theta by REML or ML with the smoothing parameters,
  as mgcv's `nb()` (`fit.negative_binomial_theta_`, `smoothing_`'s too;
  the null deviance at it, and the AIC counting it); `theta=` fixes it, as
  `negbin(theta)`. The criterion's gradient in log theta is exact
  (implicit in the coefficients). Against mgcv (REML and ML): theta and
  the dispersions to 1e-4, ML criteria to 1e-7, fits to 1e-4. The
  Bikeshare page fits both (#183).
- **Factor-by smooths, `by_factor_spline(x, by=f, k=10, lam=None,
  pooled_lam=False)`** (`FactorSplineTerm`, mgcv's `s(x, by=f, bs="cr")`):
  a cubic regression spline of `x` for each level of the factor `f`, zero
  off its level and summing to 0 on it (include `factor(f)` for the
  levels' means, as mgcv's `y ~ f + s(x, by=f)`). Each level has its own
  smoothing parameter, or with `pooled_lam=True` one shared by all of them
  (mgcv's `id`); either is given or estimated by REML/ML. A categorical's
  levels keep their order; `level_edf()` gives each curve's degrees of
  freedom, and the plot is a curve per level. Under REML and ML, penalties
  of a term on disjoint coefficients are now single penalties (exact
  log-determinants, no stable basis). Against mgcv, Gaussian and Poisson,
  REML and ML, per level and pooled: our criterion is at most its value at
  mgcv's smoothing parameters, Gaussian REML fits to 1e-8. The Bikeshare
  page fits the hour by the working day, and the R translation page lists
  it (#182).
- **A GAM shows the model it specifies.** In a notebook, `GAM` displays
  its model in LaTeX (`text/latex`, from `GAM.model_latex()`): the
  response's distribution and link, the linear predictor as the sum of
  the terms, and a line per term saying what it is (a smooth and its
  penalty, a random effect and its distribution, a covariance's
  structure) and how its smoothing parameter or variance is chosen. It
  needs only the specification, so it works before fitting. This replaces
  scikit-learn's HTML diagram for `GAM`. The GAM pages of the docs display
  their models (#181).
- **Cyclic smooths, `cyclic_spline(x, k=10, period=None)`**
  (`CyclicSplineTerm`, mgcv's `s(x, bs="cc")`), for a covariate whose ends
  meet, such as the hour of the day. It is a cyclic cubic regression
  spline: values at `k - 1` knots, with the function and its first two
  derivatives continuous around the cycle (Wood, 2017, section 5.3.2).
  `period=(0, 24)` sets the cycle's ends; by default it is the data's
  range, with mgcv's knots. Its smoothing parameter is fixed (`lam`) or
  REML/ML. Against mgcv `s(hour, bs="cc")` with the same knots, REML and
  ML for the Gaussian and Poisson: fits to 1e-8, edf to 1e-7, criteria to
  1e-9 (#179).
- **Structured covariances of correlated random effects:**
  `correlated_random_effect(..., covariance='compound_symmetry')` (nlme's
  `pdCompSymm`: a common variance and correlation) and `'diagonal'`
  (`pdDiag`), next to the default `'unstructured'`. They are new penalty
  objects on one general `CovariancePenalty`, which takes `Sigma(theta)`
  and its derivatives and evaluates in lme4's coordinates `b = L u`. Its
  parameters for compound symmetry are nlme's, valid for any value. With
  a positive correlation, compound symmetry is exactly `re(g) + re([g,
  condition])`, and diagonal is exactly independent `re()` terms. Against
  `nlme::lme` with `pdCompSymm`: covariance to 1e-4, log-likelihood to
  1e-7 (#178).
- **`GAM.fixed_effects_`, with GLS standard errors for large data.** It
  gives the intercept and the coefficients of linear terms and blocks of
  columns, with standard errors from the posterior covariance (for the
  Gaussian, lme4's `vcov`), and `fixed_effects_covariance_` their
  covariance. With random effects as the only penalized terms they come
  from backfitting the random effects on the linear terms' columns,
  `X_F' W (I - S) X_F`, without forming the design: 200,000 observations
  of 2000 x 500 crossed levels take seconds. They agree with the dense
  posterior covariance to 1e-8 and with lmer's `vcov` to 1e-4.
  `summary()` no longer forms the dense covariance for a method that does
  without it (`covariance_edf = False`) (#170).
- **`glmnet.gam`: pluggable methods for the penalty parameters.**
  `GAM(method=...)` takes a `SmoothingMethod` (`glmnet.gam.selection`)
  as well as `'fixed'`, `'REML'` or `'ML'` (now the `Fixed` and `Laplace`
  methods). Its `select(gam, dispersion)` sets the estimated terms'
  parameters (`estimated_terms(gam)`), using fits at given parameters
  (`GAM.inner_fit()`), and says whether `GAM` needs the joint solve and
  mgcv's edf. Those form the dense covariance; a method for large data
  can do without them (#168).
- **`glmnet.gam.moments.moment_equations`: the method of moments for the
  variances of crossed (or nested) random intercepts**, in time linear in
  the number of observations. These are Gao and Owen's U statistics, for
  any number of factors, and their expectations' coefficients, from counts
  of the levels and their pairs. The page "Crossed random effects at
  scale" writes the `SmoothingMethod` around it, as in Ghosh, Hastie and
  Owen: it takes the residuals of the fit without the random effects,
  solves the equations, and then `GAM` backfits with the variances fixed,
  giving the GLS fit and the effects' predictions. On a balanced design
  they are the ANOVA estimators. 200,000 observations of 2000 × 500
  crossed levels take about 2 seconds (#168). The method was
  `glmnet.gam.CrossedMoments`; it is now the page's example of writing a
  method, and the library keeps the equations (#175).
- **Cox REML/ML for every kind of penalty:** tensor products, several
  penalties per term and correlated random effects, on the penalty objects
  and `DenseHessian`, as the other families. As in theirs, one Newton step
  polishes the coefficients with the criterion's factorization, and the
  inner fit stops at 1e-10. Against mgcv's `cox.ph` with
  `te(x, z, bs="cr")`: our criterion is as low (mgcv stops sooner where
  it is flat) (#166).
- **Correlated random effects, `correlated_random_effect(group,
  slopes=[...])`** (`CorrelatedRandomEffectTerm`), lme4's `(1 + v | g)`: an
  intercept and slopes per group, their vectors `N(0, Sigma)` with
  `Sigma` unstructured. Its precision is given (`precision=`) or estimated
  by REML/ML (`glmnet.gam.penalties.CholeskyPenalty`). That uses the
  log-Cholesky factor `L` of the relative covariance, as lme4's relative
  covariance factor, with the criterion evaluated in the coordinates of
  `b = L u` per group, where the penalty is the identity. A singular
  covariance (lme4's boundary fit) is then a diagonal of `L` going to 0,
  with the other parameters finite. `variance_components_` has the
  variances and `covariance_` the covariance (#165).

  Against `lmer`, REML and ML: the standard deviations and correlation to
  1e-5, fitted values and BLUPs, and ML's log-likelihood to 1e-8. Against
  `glmer` (binomial, singular): to 1e-4. Backfitting (per group `q x q`
  solves) agrees with the joint solve.
- REML/ML: an evaluation where the deviance or its derivatives are not
  finite, or the inner fit cannot decrease its objective (far from the
  optimum), counts as an unconverged fit. Newton's method runs up to 100
  steps, for an optimum at infinity such as a singular covariance. The
  search fits a correlated random effect's covariance diagonal first, with
  the Cholesky factor's off-diagonals at 0, then all of it. From the start
  the off-diagonals' gradient can be large, and a first step to the edge
  of the box led a binomial REML fit into a singular covariance's basin, a
  local minimum. At an unconverged inner fit the search turns towards
  more penalty, which for a correlated random effect means a smaller
  covariance, where it had turned towards a larger one (#165).
- **`glmnet.gam` REML/ML: penalties as objects** (`glmnet.gam.penalties`).
  A term's penalty is a function `S(theta)` of its parameters. Each
  evaluation gives its matrix, its derivatives in the parameters,
  `log|S|_+` and its derivatives, and starting values, so the criterion
  doesn't depend on the kind of penalty. `ScaledPenalty` (one penalty,
  `lam S`) and `SumPenalty` (several, in Wood's stable basis) are today's.
  A new kind of term, such as a correlated random effect parametrized by
  the Cholesky factor of its covariance, adds a class. There is no change
  in the fits (#164).
- **Tensor product smooths, `tensor_spline(x, z, ...)`** (`TensorTerm`,
  mgcv's `te(..., bs="cr")`), in two or more columns. Each margin is a cubic
  regression spline: a natural cubic spline parametrized by its values at
  `k` knots (5 by default), placed as mgcv's `place.knots`, with its own
  penalty, as `S_1 (x) I` and `I (x) S_2`. So the term has one smoothing
  parameter per margin and is invariant to the columns' scales. Its
  smoothing parameters are fixed (`lam`) or estimated by REML/ML. It works
  with backfitting (modified backfitting fits the null space `x, z, x z`
  with the other linear parts), the joint solve, every family, weights and
  standard errors. Against mgcv `te(x, z, bs="cr")` the fits agree to 1e-4
  (to 5e-11 alone) and the ML criteria to 1e-7. Where the criterion is
  flat, mgcv stops sooner and ours is as low, except along a smoothing
  parameter going to infinity, which our bounds stop. `plot` draws a
  two-column term as filled contours, as `gaussian_process(u, v)` (#163).
- **`glmnet.gam` REML/ML: the criterion on four operations of the Hessian.**
  The criterion and its gradient now need only `log|H|`, solves, the traces
  `tr(H^{-1} S_k)` and the leverages `x_i' H^{-1} x_i`, of
  `glmnet.gam.linalg.DenseHessian`. Another implementation of these four
  operations (a sparse Cholesky with selected inversion, or matrix-free
  solves with stochastic estimates) can replace the dense one without
  changing the criterion. JAX is now used only for the elementwise
  derivatives of the family's unit deviance, not for dense matrices. The
  tests run in about half the time (#162).
- **Wood's (2011) stable reparametrization** for terms with several
  penalties. At each evaluation the dominant penalty's range is split off,
  with its null space set exactly to 0, then the next penalty in that null
  space, and so on. One Newton step on the coefficients with the
  criterion's own factorization makes the criterion independent of the
  inner fit's warm start. A smoothing parameter going to infinity (mgcv
  `paraPen`'s difference penalty, at about 1e13) no longer leaves the
  criterion's value noisy at 1e-9, and the gradient at the optimum is
  about 1e-11 (#162).
- **`glmnet.gam`: terms with several penalties.** A term can now have
  penalties `sum_m lam_m c' S_m c / 2` (`Term.penalty_matrices()`,
  `penalty_lams()` and `set_penalty_lams()`; a term with one penalty is
  unchanged). This is the foundation for tensor products and correlated
  random effects. All three fitting paths take them: backfitting, the joint
  solve, and REML/ML. REML and ML estimate one smoothing parameter per
  penalty. `log|S|_+` is that of the term's weighted sum of penalties on the
  range of their sum. Each evaluation rotates the term to the eigenvectors
  of that sum and equilibrates H (as Wood's, 2011, stable
  reparametrization), so a dominant penalty (a smoothing parameter going to
  infinity) does not cost the gradient its precision (#160; Cox REML with
  several penalties followed, below).
- **`penalized_columns(columns, penalties, lam=None)`**
  (`PenalizedColumnsTerm`), a block of columns with penalties on their
  coefficients, as mgcv's `paraPen`. Its
  smoothing parameters are fixed or estimated, and the fit agrees with mgcv's
  `paraPen` under REML and ML (#160).

- **Gaussian process smooths in `glmnet.gam`:** `gaussian_process(x)`,
  `gaussian_process(lon, lat)` (any number of columns), the low rank Gaussian process (kriging) smoother of
  Kammann & Wand (2003), as mgcv's `s(..., bs="gp")`: its knots, linear
  trend (or none, `trend=False`), Matern, power exponential and spherical
  correlation functions, default range and basis dimension, and mgcv's `m`
  codes. It takes `lam`, `df`, or neither under REML/ML, with both solvers,
  every family, weights and offsets; its basis spans mgcv's, so fits at
  fixed `sp`, edf, standard errors and REML/ML estimates agree with mgcv's;
  at full rank without a trend it is scikit-learn's
  `GaussianProcessRegressor` at fixed hyperparameters. `plot` draws a
  two-column term as filled contours. A docs page, "Gaussian processes as
  GAM terms" (#159).

- **`glmnet.gam`: random effects of combinations of columns.**
  `re(['batch', 'cask'])` gives one effect per combination, labelled
  `'batch:cask'`, so nested effects such as lme4's `(1 | batch/cask)` are
  `re('batch') + re(['batch', 'cask'])`. `by=` slopes, new data (unseen
  combinations predict 0) and positional columns work as for one column.
  Checked against `lmer` under REML and ML (#161).

### Documentation

- **`glmnet/gam/reml.py` explains its math.** The module docstring maps
  the criterion's symbols to the code's names, lists the four steps of an
  evaluation, and writes out the gradient (the implicit derivative through
  the coefficients, `log|H|` through the weights). Comments in `_value`,
  `value_and_grad`, `_factor`, `_evaluation`, `_set_structure` and the Cox
  criterion tie each line to its term of the formula (#184).
- The "Generalized additive models" chapter comes after "GLMNet paths" in
  the table of contents, before "Inference" (#180).
- **The crossed-effects page has a binary response**, as in Ghosh, Hastie
  and Owen's scalable logistic regression with crossed random effects. It
  writes Schall's (1991) PQL as a second `SmoothingMethod`, one that
  alternates warm-started inner fits with updates of the variances. It
  uses their O(N) trace approximation, which ignores the crossing in the
  random effects' penalized information. The section compares it with
  Laplace approximate ML: PQL's variances are smaller, as Breslow and
  Clayton found, the fixed effects close, and it is much faster. It also
  shows a naive logistic regression's standard errors, and run times up
  to 100,000 ratings, about linear. The tests copy the method and check
  the trace approximation against the exact traces (#176).
- **"Crossed random effects at scale"**, a page following Ghosh, Hastie and
  Owen on simulated ratings (customers × items, very unbalanced). It
  writes the method-of-moments variances as a `SmoothingMethod`, with a
  discussion of how and why `select()` is designed as it is. It then
  covers GLS by backfitting, and `fixed_effects_`. It shows OLS's naive standard errors,
  the BLUPs' shrinkage, and run times about linear in N up to 400,000
  ratings (48,000 random effects; the fit about 2 s), against REML on a
  small problem (#171).
- **A Poisson GAM: bike sharing** (`docs/gam_bikeshare.md`), on ISLP's
  `Bikeshare` data, after the ISLR lab's Poisson regression: smooths of the
  hour and the temperature with REML, a tensor product of the two, the
  model split by the working day (there are no factor-by smooths), compared
  by ML criteria and AIC; then the overdispersion (Pearson's statistic, a
  fixed dispersion, a random effect of the day) and its effect on the
  smoothing and the standard errors, and predictions with standard errors
  (#169).
- **A page translating R models**, "GAMs and mixed models from R": tables
  from `gam::gam`, `mgcv` and `lme4`/`nlme` to `glmnet.gam` terms and
  attributes, with what each translation is checked against in the tests
  and how closely it agrees. Also the differences in parametrization (the
  smoothing parameters' scale, the REML criterion's constant, the standard
  errors of random effects) and what has no translation yet (#167).
- The REML design page describes the four operations, the penalty
  objects (with the reason for the covariance-side Cholesky factor) and
  the updated list of what is not supported yet. It adds ML and REML
  versus ML: `V_REML - V_ML = log|C_F|/2 - M log(2 pi phi)/2`, `C_F` the
  unpenalized coefficients' (GLS) information, checked on the by-hand
  example. It also says what ML needs of the linear algebra: the four
  operations plus `M` solves, a rank-`M` correction (#166).
- The GAM page has a section on tensor products (`tensor_spline('age',
  'year')` on the Wage data, its smoothing parameters and surface). The
  mixed-models page has correlated intercepts and slopes
  (`correlated_random_effect`, lme4's `(days|subject)`), and its list of
  what is not supported is updated (#166).
- **Mixed models as GAMs:** a page, "Mixed models as GAMs", on fitting
  classical random-effects models with `glmnet.gam`'s `re()` terms: random
  intercepts (shrinkage, REML and ML), crossed and nested random effects,
  independent random intercepts and slopes, a binomial GLMM (Laplace ML and
  REML), an observation-level random effect for overdispersed counts, and a
  smooth with a random effect, with what is not supported yet (correlated
  random effects). Its fits are checked against lme4's `lmer` and `glmer`,
  nlme's `lme` and mgcv's `gam(bs="re")` in tests. The docs have a
  "Generalized additive models" chapter, with the GAM page (#157).
- **How REML and ML work in `glmnet.gam`** (`docs/gam_reml.md`): the
  criterion, derived for statisticians, then written out in a few lines of
  JAX for a Gaussian and a Poisson additive model and minimized by hand. It
  matches `GAM(method='REML')`. The page checks the optimum's identities in
  the residuals (REML's dispersion, and each term's prior size against its
  penalized degrees of freedom) and the implicit derivative of the fitted
  coefficients. It ends with the design of `glmnet.gam.reml`, and the
  lessons from lme4, MixedModels.jl, AI-REML, TMB, mgcv's `bam` and
  Fellner–Schall, matrix-free GP and genetics codes, and nlme that shape its
  path to large problems. The same
  checks are tests (`tests/gam/test_gam_reml_byhand.py`) (#158).

### Fixes

- **`glmnet.gam.factor()` of a pandas categorical keeps its categories'
  order** (those seen in fitting) for its effects, plots and baseline,
  instead of sorting the levels as strings: the months of ISLP's
  `Bikeshare` came out April, Aug, Dec. New data's levels are looked up
  by value. The Bikeshare page now uses its categorical month (#173).
- **`glmnet.gam` REML/ML ignored a given dispersion for the binomial and
  Poisson:** `fit(X, y, dispersion=phi)` set `dispersion_` (and so the
  standard errors) to `phi`, but the criterion that chose the smoothing
  parameters used 1. It now uses `phi`, a quasi-likelihood's dispersion:
  the fit is that with prior weights `1 / phi` (a test) (#169).
- **`glmnet.gam` REML/ML for the Cox family converged short on Windows**
  (two tests failed on its CI, Efron's ties with start times and strata).
  Near the optimum, the Cox criterion's value depends a little (about 3e-10,
  relative) on the search's path, through the warm-started inner fit, which
  is more than the 1e-10 within which a step that halves the gradient was
  taken: the step was refused, and the search stopped with a gradient of
  1e-4. The Cox criterion's precision is now 1e-9 (the gradient, accurate,
  must still halve). Rounding-level perturbations of the data reproduced it
  on macOS (2 of 16 REML fits); all converge now (#156).
- **ISLP's empty design without a warning:** a `DataFrame` with a single
  column of zeros named `zero` (ISLP's `ModelSpec` with no terms, the null
  model of a stepwise search such as `sklearn_selected(GLM(), ...)`) is the
  intercept-only model (Cox: the null model). It is excluded silently, as a
  lone `intercept` column. It warned "excluding the constant column(s)
  ['zero']" at every null model of the search (the ISLP Ch06 lab). Other
  constant columns still warn (#155).

### Changed

- `glmnet.gam`'s modules import at their tops, not inside functions. The
  optional dependencies (JAX, matplotlib, scatter_smooth) are still
  imported when used. `estimated_terms` is defined in `glmnet.gam.reml`,
  and shared by `glmnet.gam.selection` (#177).
- **`glmnet.gam`'s REML and ML are marked experimental:**
  `GAM(method='REML')` and `'ML'` warn (`glmnet.gam.ExperimentalWarning`, a
  `UserWarning`), and the docstrings and the GAM page say so. Their fits
  agree with mgcv, lme4 and nlme in the tests, but the interface may change
  and the optimization is less tested than the rest of `glmnet.gam` (#154).

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
