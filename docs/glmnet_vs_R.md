# Comparison with R's glmnet

`glmnet` follows R's glmnet (version 5.1). This page lists what R has
that `glmnet` does not yet, and how the shared features are tested.
Most tests compare with R itself through `rpy2`; they are skipped if R
or `rpy2` is not installed.

## Not yet in `glmnet`

| R | Status in `glmnet` |
|---|---|
| `print.glmnet`, `print.cv.glmnet` | The path summary (df, fraction of deviance explained, lambda) is in `summary_`. CV objects have no printed table. |
| `deviance(fit)` | Not provided. Compute it from `summary_` and the null deviance. |
| `predict(..., gamma = c(...))` with several gammas | `predict` takes a single `gamma`. Call it once for each value. |
| `relax.glmnet(fit, x, ...)`, `path = TRUE` | The relaxed fits are computed by `fit(relax=True)`; they can't be added to an existing fit. `relax_maxp` is R's `maxp`. |
| `plot(cvfit)` for relaxed CV, plotted against gamma | `RelaxedScorePath.plot` plots against lambda only, with optional SE bands. |
| `cv.glmnet(grouped = FALSE)` | Only the ungrouped MSE and MAE scorers. There is no switch and no ungrouped deviance. |
| `cv.glmnet(keep = TRUE)` `foldid` | The held-out predictions are returned. The folds are whatever was passed as `cv`. |
| `cv.glmnet(trace.it = 1)` | Fold progress through sklearn's `verbose` only. |
| `glmnet.measures()` | No public listing. The Cox C-index is a scorer (`CoxCIndexScorer`) but is not among the defaults. |
| `survfit(cvfit)` | Use `CoxNet.survfit(lambda_val=...)` with `cv_choice()`. |
| `coxnet.deviance`, `coxgrad` | Not public. |
| `na.replace`, `rmult` | `na.replace` exists only as `make_x(na_impute=True)`. There is no `rmult`. |
| `glmnet.control()`, `control = list(...)` | Each estimator takes a `control` dataclass (`FastNetControl`, `GLMNetControl`). There are no global settings. |
| `type.logistic` for a family object | `modified_newton` exists for `LogNet`, and `type_logistic` for `MultiClassNet`, but not for the IRLS `GLMNet`. |

## Shared features and their tests

| Feature | Tests |
|---|---|
| Gaussian, binomial, Poisson, multinomial (including grouped) and multi-response Gaussian paths | `tests/paths/test_{gaussnet,lognet,fishnet,multiclassnet,multigaussnet}.py`, `tests/compare_R/`. Coefficients and lambda values are compared with R. The tests vary weights, offsets, `standardize`, `fit_intercept`, penalty factors, limits and `alpha`. |
| Cox paths: strata, start/stop times, Breslow and Efron ties | `tests/paths/test_coxnet.py`, `tests/compare_R/test_coxnet_r_comparison.py`, `tests/paths/test_cox_predict_ties.py` (prediction types and the ties default). |
| Any GLM family (`family = <family object>`, IRLS) | `tests/test_irls_r_parity.py` compares with R's `glmnet.path` at tight tolerances: gaussian, logit, probit, Poisson and Gamma(log); penalty factors (0 and ∞ included), `alpha`, limits, offset, no intercept, `standardize=False`, `exclude`; R's path stopping rules. See also `tests/flex/` and `tests/compare_R/test_probit_r_comparison.py`. |
| `exclude`, `penalty.factor` (fixed or functions) | `get_penalty_factor`, where an infinite factor excludes a variable: `tests/test_penalty_factor_hook.py` (against R 5.1 for an adaptive lasso; reruns on each CV fold), `tests/paths/test_gaussnet.py`. |
| `lower.limits`, `upper.limits` | The path tests, `tests/test_glm_problem.py` (limits are respected, and the user's limits are not overwritten). |
| `dfmax`, `pmax` | `tests/paths/test_pmax.py`, `test_multiclassnet_df_max`. Compared with R, including the "exceeds pmax" warning. |
| `cv.glmnet`: deviance, MSE, MAE, class, AUC, C-index; `lambda.min`, `lambda.1se` | The CV tests in `tests/paths/` compare `cvm`/`cvsd` with R given the same `foldid`. Also `tests/test_scorer.py`. |
| `coef(cvfit, s = "lambda.1se")`, `predict(cvfit, ...)` | `cv_coefs`, `cv_predict`: `tests/paths/test_cv_choice.py`, compared with R. |
| `relax = TRUE`, `gamma` | `tests/paths/test_relax.py`. Compared with R: relaxed predictions (including when some active sets are too large to refit) and relaxed CV curves and `lambda`/`gamma` choices. |
| `predict(..., exact = TRUE)`, `coef(..., s =)` | `exact_coefs` / `refit_path`: `tests/paths/test_exact.py`; `get_fixed_lambda`: `tests/paths/test_fixed_lambda.py`. |
| `predict(..., newoffset =)` | `predict(offset=)`: `tests/paths/test_predict_offset.py`, compared with R. |
| `predict(type = "nonzero")` | `nonzero()`: `tests/test_nonzero.py`, compared with R. |
| `assess.glmnet`, `confusion.glmnet`, `roc.glmnet` | `tests/test_assess.py`, compared with R, with and without weights and offsets. |
| `makeX`, `bigGlm` | `tests/test_make_x.py`, `tests/test_big_glm.py`, compared with R. |
| `survfit.coxnet`, `Cindex` | `tests/paths/test_coxnet.py`, compared with R's `survfit` and `Cindex`. |
| Sparse `x` | `tests/sparse/`: sparse fits match dense ones, for the C++ paths and the IRLS `GLMNet`. |
| `plot(fit, xvar =, label = TRUE)`, `plot(cvfit)` | `tests/test_coef_path_plot.py` and the plot tests in `tests/paths/test_relax.py` and `tests/paths/test_cv_choice.py`. |
| `trace.it` for fits | `FastNetControl(itrace=1)`: `tests/paths/test_progress.py`. |
