import numpy as np
from glmnet.paths import LogNet, MultiClassNet
from glmnet.regularized_glm import BinomialRegGLM
from glmnet.glm import BinomialGLM

def test_lognet_trials_successes():
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    trials = rng.integers(1, 10, size=n)
    successes = rng.binomial(trials, 0.5)
    
    Y_counts = np.column_stack([trials, successes])
    
    # We provide a fixed lambda sequence so path generation is identical
    lambda_grid = np.geomspace(0.5, 0.001, 10)
    
    # Fit with LogNet
    lognet = LogNet(lambda_values=lambda_grid)
    lognet.fit(X, Y_counts)
    
    # Expand the data manually to compare
    X_expanded = np.repeat(X, trials, axis=0)
    Y_expanded = np.concatenate([
        np.concatenate([np.ones(s), np.zeros(t - s)])
        for t, s in zip(trials, successes)
    ])
    
    lognet_expanded = LogNet(lambda_values=lambda_grid)
    lognet_expanded.fit(X_expanded, Y_expanded)
    
    # Ensure coefs and intercepts are very close
    np.testing.assert_allclose(lognet.coefs_, lognet_expanded.coefs_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lognet.intercepts_, lognet_expanded.intercepts_, rtol=1e-5, atol=1e-5)

def test_multiclassnet_counts():
    rng = np.random.default_rng(43)
    n, p, k = 100, 5, 3
    X = rng.standard_normal((n, p))
    
    # Generate random counts for 3 classes
    Y_counts = rng.integers(0, 5, size=(n, k))
    # Ensure no all-zero rows
    Y_counts[Y_counts.sum(axis=1) == 0, 0] = 1
    
    lambda_grid = np.geomspace(0.5, 0.001, 10)
    
    mnet = MultiClassNet(lambda_values=lambda_grid)
    mnet.fit(X, Y_counts)
    
    # Expand data manually
    trials = Y_counts.sum(axis=1)
    X_expanded = np.repeat(X, trials, axis=0)
    Y_expanded = []
    for row in Y_counts:
        expanded_row = []
        for class_idx, count in enumerate(row):
            expanded_row.extend([class_idx] * count)
        Y_expanded.extend(expanded_row)
    Y_expanded = np.array(Y_expanded)
    
    mnet_expanded = MultiClassNet(lambda_values=lambda_grid)
    mnet_expanded.fit(X_expanded, Y_expanded)
    
    # Ensure coefs and intercepts are very close (tol loosened slightly due to grouped vs expanded numerical differnces in coordinate descent)
    np.testing.assert_allclose(mnet.coefs_, mnet_expanded.coefs_, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(mnet.intercepts_, mnet_expanded.intercepts_, rtol=1e-3, atol=1e-3)

def test_binomial_reg_glm_trials_successes():
    rng = np.random.default_rng(44)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    trials = rng.integers(1, 10, size=n)
    successes = rng.binomial(trials, 0.5)
    
    Y_counts = np.column_stack([trials, successes])
    
    # Fit with BinomialRegGLM
    reg_glm = BinomialRegGLM(lambda_val=0.1)
    # Provide a very small regularizer manually or just fit null
    reg_glm.fit(X, Y_counts)
    
    # Expand the data manually to compare
    X_expanded = np.repeat(X, trials, axis=0)
    Y_expanded = np.concatenate([
        np.concatenate([np.ones(s), np.zeros(t - s)])
        for t, s in zip(trials, successes)
    ])
    
    reg_glm_expanded = BinomialRegGLM(lambda_val=0.1)
    reg_glm_expanded.fit(X_expanded, Y_expanded)
    
    # Ensure coefs and intercepts are very close
    np.testing.assert_allclose(reg_glm.coef_, reg_glm_expanded.coef_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(reg_glm.intercept_, reg_glm_expanded.intercept_, rtol=1e-5, atol=1e-5)

def test_binomial_glm_trials_successes():
    rng = np.random.default_rng(45)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    trials = rng.integers(1, 10, size=n)
    successes = rng.binomial(trials, 0.5)
    
    Y_counts = np.column_stack([trials, successes])
    
    glm = BinomialGLM()
    glm.fit(X, Y_counts)
    
    X_expanded = np.repeat(X, trials, axis=0)
    Y_expanded = np.concatenate([
        np.concatenate([np.ones(s), np.zeros(t - s)])
        for t, s in zip(trials, successes)
    ])
    
    glm_expanded = BinomialGLM()
    glm_expanded.fit(X_expanded, Y_expanded)
    
    np.testing.assert_allclose(glm.coef_, glm_expanded.coef_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(glm.intercept_, glm_expanded.intercept_, rtol=1e-5, atol=1e-5)
