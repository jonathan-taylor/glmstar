# Changelog

## Unreleased

- **Renamed, for the lasso / elastic net:** the modules
  `glmnet.inference.glm_problem` and `glm_inference` are now `glmnet_problem`
  and `glmnet_inference`, and the function `glmnet_problem` is now
  `problem_from_solution`. The old names still work, with a
  `DeprecationWarning`, and will be removed in a later release.
- **Requirements:** coxdev >= 0.3.0. `c_index` now uses coxdev's C-index,
  and glmstar's own concordance extension is removed.

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
