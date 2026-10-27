# Data Input Rationale: X and y

When using the `glmnet` estimator API, the `fit(X, y)` method follows the familiar `scikit-learn` signature. However, you will notice a divergence in how we prefer you to structure these two arguments.

Specifically, while `X` is expected to be a standard array-like matrix, we highly recommend passing a **column-focused data structure** (such as a `pandas.DataFrame`) for `y` or the broader dataset used to define the response variables.

## Why `X` is Array-Like

The feature matrix `X` has a single, well-defined mathematical purpose: to be multiplied by the coefficient vector $\beta$.

*   **Shape:** `(n_samples, n_features)`
*   **Format:** `numpy.ndarray` or `scipy.sparse` matrices.
*   **Rationale:** Treating `X` purely as a numerical matrix allows our computationally intensive C++ backends (powered by Eigen) to perform linear algebra operations with maximum efficiency and minimal overhead.

## Why `y` Prefers Columnar Access (e.g., `pandas.DataFrame`)

In standard OLS regression, the response `y` is just a single vector of length `n_samples`. However, Generalized Linear Models (GLMs) are frequently more complex. Constructing the appropriate loss function often requires **multiple `(n_samples,)` vectors** beyond just the primary target variable. 

By providing a `pandas.DataFrame` whose columns are named by their role (below), the `fit` method can find perfectly aligned vectors by their names, without a cluttered function signature. 

Here are the primary cases where columnar access is essential:

### 1. Offsets
In models like Poisson regression (used for modeling rates), you often have an exposure or `offset` term. This is an *a priori* known component of the linear predictor. A DataFrame allows you to keep the `offset` perfectly aligned with your `response` and easily extract it by name.

### 2. Cox Proportional Hazards (Survival Data)
Survival models require more than a single scalar per observation. To properly construct the risk sets, the model needs to retrieve:
*   **`start`**: Start time (for counting processes or left-truncated data).
*   **`stop`**: The observed time of the event or censoring.
*   **`status`**: A boolean/integer indicator of whether the event occurred.

A DataFrame naturally groups these multiple columns into a single object passed to `y`, avoiding the need for multiple array arguments.

### 3. Binomial / Multinomial Trials
When dealing with aggregated binomial or multinomial data, each observation isn't a single trial but a collection of trials. The model must ingest both the **number of successes** and the **number of failures** (or the counts of each category), the columns of the response.

### 4. Weights, and Cross-Validation
The weights could be an argument of `fit`, as scikit-learn's
`sample_weight`, but cross-validation is why glmstar takes them, and the
offset and survival data, as columns of `y`. scikit-learn's
cross-validation (`cross_val_score`, `cross_validate`, `GridSearchCV`)
and glmstar's own `cross_validation_path` split `X` and `y` into folds by
their rows, so every column of `y` goes with its rows, to the fits on the
training folds *and* to the score on the held-out fold.
An array beside `y` is split only when each call passes it on
(`cross_validate(..., params={'sample_weight': w})`, or through a
`Pipeline` with metadata routing,
`sklearn.set_config(enable_metadata_routing=True)`), and it reaches the
scorer of the held-out fold only with metadata routing and a scorer that
asks for it (`set_score_request(sample_weight=True)`): otherwise the
held-out deviance silently drops the weights or the offset. A column
of `y` needs none of this, and the same goes for any other column a
model needs, e.g. a Cox model's `strata`.

`fit(X, y, sample_weight=w)` and `score(X, y, sample_weight=w)` still
work, for scikit-learn's sake: `w` becomes the column `weight` of `y`.
The weights are given once, so `sample_weight` with a column `weight`
raises.

## The Columns of `y` by Role

A data frame `y` names each column by its role:

| Column | Role |
|---|---|
| `offset` | the offset, added to the linear predictor; for a multi-response model (`MultiGaussNet`, `MultiClassNet`) the same for every response |
| `offset[r]` | the offset of the response `r` (a column, or a class of a multinomial) of a multi-response model, one for each |
| `weight` | the prior weights of the observations |
| `stop`, `status` | a Cox model's times (of the events, or of censoring) and status (1 = event, 0 = censored) |
| `start` | a Cox model's start times, of (start, stop] data |
| `strata` | a Cox model's strata (labels of any type) |
| every other column | the response |

The response depends on the family:

- **Binomial**: one column, of the labels of two classes (they are the
  classes of `predict`), of 0/1, or of proportions (with the trials as
  `weight`); or **two columns, read as R's `cbind(successes,
  failures)`**: the response is the proportion `s / (s + f)`, and the
  number of trials `s + f` multiplies the weights. (glmstar 0.6.0 read
  two columns as (trials, successes): give them in R's order.)
- **Multinomial**: a column of the classes (a categorical's levels in
  their order are the categories), or a column of counts (or one-hot
  indicators) for each class.
- **Multi-response Gaussian**: a column for each response.
- **Cox**: the response is `stop` and `status` (and `start`); a Cox `y`
  with a column other than the Cox roles, `offset` and `weight` raises.

A response alone (a `Series` or an array) has no offset and weights.

modelmatrix's `Outcome` builds such a frame from the data, whatever the
names of its columns, as `ModelSpec` builds `X`:

```python
from modelmatrix import ModelSpec as MS, Outcome, Surv, Binomial

X = MS(['x1', 'x2']).fit_transform(D)
Y = Outcome('claims', offset='O', weight='w').fit_transform(D)   # claims, offset, weight
GLM(family=sm.families.Poisson()).fit(X, Y)

Y = Outcome(Surv('time', 'status'), strata='inst').fit_transform(D)   # stop, status, strata
CoxNet().fit(X, Y)

Y = Outcome(Binomial('cases', trials='size')).fit_transform(D)   # cases, size-cases
LogNet().fit(X, Y)
```

An offset can be a feature of the data, e.g. R's `offset(log(exposure))`,
and `outcome.transform(new, response=False)` gives the offset (and
strata) of new data, for `predict(X, offset=...)`.

The former `response_id`, `offset_id` and `weight_id`, and the
`CoxFamily`'s `event_id`, `status_id`, `start_id` and `strata_id`, which
named the columns of a frame of one's own, are deprecated: they still
work, with a FutureWarning.

## Polars and Other Data Frames

`X` and `y` may also be a polars `DataFrame` (or `LazyFrame`, which is
collected), a pyarrow `Table`, or any data frame
[narwhals](https://narwhals-dev.github.io/narwhals/) supports. Each is
converted to pandas once, when it is passed in. A frame of numeric
columns becomes a single array: one copy, as a pandas `X` already gets.
A polars `Enum` keeps its levels in their order (the levels of a GAM's
`categorical()` term), and a `Categorical`'s levels are sorted.

```python
import polars as pl
from modelmatrix import Outcome
from glmnet import GLM

df = pl.read_csv('data.csv')
fit = GLM().fit(df.select('x1', 'x2'), Outcome('y', weight='w').fit_transform(df))
fit.summary_     # a pandas DataFrame
```

The outputs (coefficients, summaries, `anova` tables) are pandas, and
predictions are numpy arrays.

## Conclusion

Keeping `X` as a numeric array optimizes the core math, while promoting `y` (or a related dataset object) to a `pandas.DataFrame` provides the robust, flexible data routing necessary to construct advanced GLM loss functions cleanly.
