import pytest
import numpy as np

def pytest_addoption(parser):
    parser.addoption(
        "--test-size", action="store", default="large", help="run subset of tests: small, medium, large"
    )

collect_ignore = ["glmnet/blah", "sandbox/compare_R"]

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set parameter values if the
    # parameter is not marked (via get_closest_marker) and is parametrized
    # (via 'parametrize' in funcargnames).
    rng = np.random.default_rng(0)
    if 'sample_weight' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("sample_weight", [np.ones])
        else:
            metafunc.parametrize("sample_weight", [np.ones, lambda n: rng.uniform(0, 1, size=n)])
    if 'df_max' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("df_max", [None])
        else:
            metafunc.parametrize("df_max", [None, 5])
    if 'exclude' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("exclude", [[], [1,2,3]])
        else:
            metafunc.parametrize("exclude", [[], [1,2,3]])
    if 'lower_limits' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("lower_limits", [None])
        else:
            metafunc.parametrize("lower_limits", [None, np.zeros])
    if 'covariance' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("covariance", [None])
        else:
            metafunc.parametrize("covariance", [None, np.eye])

    if 'standardize' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("standardize", [True])
        else:
            metafunc.parametrize("standardize", [True, False])

    if 'fit_intercept' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("fit_intercept", [True])
        else:
            metafunc.parametrize("fit_intercept", [True, False])

    if 'nlambda' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("nlambda", [None])
        else:
            metafunc.parametrize("nlambda", [None, 20])
        
    if 'lambda_min_ratio' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("lambda_min_ratio", [None])
        else:
            metafunc.parametrize("lambda_min_ratio", [None, 0.02])

    if 'n' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("n", [100])
        else:
            metafunc.parametrize("n", [100, 500])
        
    if 'p' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("p", [10])
        else:
            metafunc.parametrize("p", [10, 200])

    if 'limits' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("limits", [(-1, 1), None])
        else:
            metafunc.parametrize("limits", [(-1, 1), None, (-np.inf, 1), (-1, np.inf), (0, np.inf)])
    if 'penalty_factor' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("penalty_factor", [None])
        else:
            metafunc.parametrize("penalty_factor", [None, lambda p:rng.uniform(0, 1, size=p) + 0.1])
    if 'alignment' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("alignment", ['lambda'])
        else:
            metafunc.parametrize("alignment", ['lambda', 'fraction'])
    if 'offset' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("offset", [None])
        else:
            metafunc.parametrize('offset', [None, np.zeros, lambda n: rng.uniform(0, 1, size=n)])
    if 'alpha' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("alpha", [0.5])
        else:
            metafunc.parametrize("alpha", [0.1, 0.5, 0.9])
    if 'path' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("path", [True])
        else:
            metafunc.parametrize("path", [True, False])
    if 'q' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("q", [3])
        else:
            metafunc.parametrize("q", [2, 3, 4])
    if 'use_offset' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("use_offset", [True])
        else:
            metafunc.parametrize("use_offset", [True, False])
    if 'use_weights' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("use_weights", [True])
        else:
            metafunc.parametrize("use_weights", [True, False])
    if 'glmnet' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("glmnet", [True])
        else:
            metafunc.parametrize("glmnet", [True, False])
    if 'scaled_input' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("scaled_input", [True])
        else:
            metafunc.parametrize("scaled_input", [True, False])
    if 'scaled_output' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("scaled_output", [True])
        else:
            metafunc.parametrize("scaled_output", [True, False])
    if 'ridge_coef' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("ridge_coef", [0])
        else:
            metafunc.parametrize("ridge_coef", [0, 0.5, 1.0])
    if 'X' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            rng = np.random.default_rng(0)
            metafunc.parametrize("X", [rng.standard_normal((10, 5))])
        else:
            rng = np.random.default_rng(0)
            metafunc.parametrize("X", [rng.standard_normal((100, 50)), rng.standard_normal((100, 200))])
    if 'weights' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("weights", [np.ones(10)])
        else:
            metafunc.parametrize("weights", [np.ones(100), rng.uniform(0, 1, size=100)])
    if 'intercept' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("intercept", [True])
        else:
            metafunc.parametrize("intercept", [True, False])
    if 'gls' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("gls", [None])
        else:
            metafunc.parametrize("gls", [None, np.linalg.pinv])
    if 'lambda_val' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("lambda_val", [1e-4])
        else:
            metafunc.parametrize("lambda_val", [1e-4, 1e-2])
    if 'modified_newton' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("modified_newton", [True])
        else:
            metafunc.parametrize("modified_newton", [True, False])

