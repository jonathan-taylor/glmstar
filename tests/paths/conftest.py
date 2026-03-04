import pytest
import numpy as np

def pytest_addoption(parser):
    parser.addoption("--paths-standardize", action="store_true", help="Run tests with standardize=True")
    parser.addoption("--no-paths-standardize", action="store_true", help="Run tests with standardize=False")
    
    parser.addoption("--paths-intercept", action="store_true", help="Run tests with intercept=True")
    parser.addoption("--no-paths-intercept", action="store_true", help="Run tests with intercept=False")
    
    parser.addoption("--paths-offset", action="store_true", help="Run tests with offset")
    parser.addoption("--no-paths-offset", action="store_true", help="Run tests without offset")
    
    parser.addoption("--paths-weights", action="store_true", help="Run tests with weights")
    parser.addoption("--no-paths-weights", action="store_true", help="Run tests without weights")

def pytest_generate_tests(metafunc):
    rng = np.random.default_rng(0)
    
    # standardize
    if 'standardize' in metafunc.fixturenames:
        if metafunc.config.getoption("--paths-standardize"):
            metafunc.parametrize("standardize", [True])
        elif metafunc.config.getoption("--no-paths-standardize"):
            metafunc.parametrize("standardize", [False])
        else:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("standardize", [True])
            else:
                metafunc.parametrize("standardize", [True, False])

    # intercept / fit_intercept
    for name in ['intercept', 'fit_intercept']:
        if name in metafunc.fixturenames:
            if metafunc.config.getoption("--paths-intercept"):
                metafunc.parametrize(name, [True])
            elif metafunc.config.getoption("--no-paths-intercept"):
                metafunc.parametrize(name, [False])
            else:
                if metafunc.config.getoption("test_size") == "small":
                    metafunc.parametrize(name, [True])
                else:
                    metafunc.parametrize(name, [True, False])

    # offset
    if 'offset' in metafunc.fixturenames:
        if metafunc.config.getoption("--paths-offset"):
            metafunc.parametrize('offset', [np.zeros, lambda n: rng.uniform(0, 1, size=n)])
        elif metafunc.config.getoption("--no-paths-offset"):
            metafunc.parametrize("offset", [None])
        else:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("offset", [None])
            else:
                metafunc.parametrize('offset', [None, np.zeros, lambda n: rng.uniform(0, 1, size=n)])

    if 'use_offset' in metafunc.fixturenames:
        if metafunc.config.getoption("--paths-offset"):
            metafunc.parametrize("use_offset", [True])
        elif metafunc.config.getoption("--no-paths-offset"):
            metafunc.parametrize("use_offset", [False])
        else:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("use_offset", [True])
            else:
                metafunc.parametrize("use_offset", [True, False])

    # weights
    if 'weights' in metafunc.fixturenames:
        if metafunc.config.getoption("--paths-weights"):
            metafunc.parametrize("weights", [rng.uniform(0, 1, size=100)])
        elif metafunc.config.getoption("--no-paths-weights"):
            metafunc.parametrize("weights", [np.ones(100)])
        else:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("weights", [np.ones(10)])
            else:
                metafunc.parametrize("weights", [np.ones(100), rng.uniform(0, 1, size=100)])

    if 'use_weights' in metafunc.fixturenames:
        if metafunc.config.getoption("--paths-weights"):
            metafunc.parametrize("use_weights", [True])
        elif metafunc.config.getoption("--no-paths-weights"):
            metafunc.parametrize("use_weights", [False])
        else:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("use_weights", [True])
            else:
                metafunc.parametrize("use_weights", [True, False])
