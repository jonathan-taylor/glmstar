import pytest
import numpy as np

def pytest_generate_tests(metafunc):
    rng = np.random.default_rng(0)
    
    # standardize
    if 'standardize' in metafunc.fixturenames:
        if metafunc.config.getoption("--standardize"):
            metafunc.parametrize("standardize", [True])
        elif metafunc.config.getoption("--no-standardize"):
            metafunc.parametrize("standardize", [False])
        else:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("standardize", [True])
            else:
                metafunc.parametrize("standardize", [True, False])

    # intercept / fit_intercept
    for name in ['intercept', 'fit_intercept']:
        if name in metafunc.fixturenames:
            if metafunc.config.getoption("--intercept"):
                metafunc.parametrize(name, [True])
            elif metafunc.config.getoption("--no-intercept"):
                metafunc.parametrize(name, [False])
            else:
                if metafunc.config.getoption("test_size") == "small":
                    metafunc.parametrize(name, [True])
                else:
                    metafunc.parametrize(name, [True, False])

    # offset
    if 'offset' in metafunc.fixturenames:
        if metafunc.config.getoption("--offset"):
            metafunc.parametrize('offset', [np.zeros, lambda n: rng.uniform(0, 1, size=n)])
        elif metafunc.config.getoption("--no-offset"):
            metafunc.parametrize("offset", [None])
        else:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("offset", [None])
            else:
                metafunc.parametrize('offset', [None, np.zeros, lambda n: rng.uniform(0, 1, size=n)])

    if 'use_offset' in metafunc.fixturenames:
        if metafunc.config.getoption("--offset"):
            metafunc.parametrize("use_offset", [True])
        elif metafunc.config.getoption("--no-offset"):
            metafunc.parametrize("use_offset", [False])
        else:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("use_offset", [True])
            else:
                metafunc.parametrize("use_offset", [True, False])

    # weights

    if 'use_weights' in metafunc.fixturenames:
        if metafunc.config.getoption("--weights"):
            metafunc.parametrize("use_weights", [True])
        elif metafunc.config.getoption("--no-weights"):
            metafunc.parametrize("use_weights", [False])
        else:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("use_weights", [True])
            else:
                metafunc.parametrize("use_weights", [True, False])
