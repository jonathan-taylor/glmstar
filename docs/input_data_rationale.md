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

By providing a `pandas.DataFrame` and specifying column names (or IDs) for these components, the `fit` method can seamlessly extract perfectly aligned vectors without requiring a massive, cluttered function signature. 

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
*(Note: Planned for future releases)*
When dealing with aggregated binomial or multinomial data, each observation isn't a single trial but a collection of trials. The model must ingest both the **number of successes** and the **total trial count** (or totals per category). Column-focused structures allow the estimator to retrieve both vectors explicitly by name.

## Conclusion

Keeping `X` as a numeric array optimizes the core math, while promoting `y` (or a related dataset object) to a `pandas.DataFrame` provides the robust, flexible data routing necessary to construct advanced GLM loss functions cleanly.
