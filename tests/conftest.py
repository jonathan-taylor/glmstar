from dataclasses import dataclass, field
import pytest
import numpy as np
import os

def pytest_addoption(parser):
    parser.addoption(
        "--test-size", action="store", default="large", help="run subset of tests: small, medium, large"
    )
    parser.addoption("--standardize", action="store_true", help="Run tests with standardize=True")
    parser.addoption("--no-standardize", action="store_true", help="Run tests with standardize=False")
    
    parser.addoption("--intercept", action="store_true", help="Run tests with intercept=True")
    parser.addoption("--no-intercept", action="store_true", help="Run tests with intercept=False")
    
    parser.addoption("--offset", action="store_true", help="Run tests with offset")
    parser.addoption("--no-offset", action="store_true", help="Run tests without offset")
    
    parser.addoption("--weights", action="store_true", help="Run tests with weights")
    parser.addoption("--no-weights", action="store_true", help="Run tests without weights")

collect_ignore = ["sandbox/compare_R"]

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set parameter values if the
    # parameter is not marked (via get_closest_marker) and is parametrized
    # (via 'parametrize' in funcargnames).
    rng = np.random.default_rng(0)
    
    # Check if we are in a subdirectory that has its own handling
    # We use string matching on the path.
    # Note: metafunc.definition.fspath is a LocalPath object in older pytest or Path in newer.
    fspath = str(metafunc.definition.fspath)
    is_subdir = any(d in fspath for d in ['/glm/', '/compare_R/', '/flex/', '/paths/'])

    if 'offset' in metafunc.fixturenames:
        metafunc.parametrize("offset", [None, lambda n: rng.standard_normal(n)])

    if 'standardize' in metafunc.fixturenames:
        metafunc.parametrize("standardize", [True, False])

    if 'penalty_facs' in metafunc.fixturenames:
        metafunc.parametrize("penalty_facs", [True, False])

    if 'use_offset' in metafunc.fixturenames:
        metafunc.parametrize("use_offset", [True, False])

    if 'use_weights' in metafunc.fixturenames:
        metafunc.parametrize("use_weights", [True, False])

    if 'fit_intercept' in metafunc.fixturenames:
        metafunc.parametrize("fit_intercept", [True, False])

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
            metafunc.parametrize("lower_limits", [-np.inf])
        else:
            metafunc.parametrize("lower_limits", [-np.inf, 0])

    if 'upper_limits' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("upper_limits", [np.inf])
        else:
            metafunc.parametrize("upper_limits", [np.inf, 0])

    if 'covariance' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("covariance", [False])
        else:
            metafunc.parametrize("covariance", [True, False])

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
            metafunc.parametrize("limits", [(-1, 1), (-np.inf, np.inf)])
        else:
            metafunc.parametrize("limits", [(-1, 1), (-np.inf, np.inf), (-np.inf, 1), (-1, np.inf), (0, np.inf)])

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
        if not is_subdir:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("offset", [None])
            else:
                metafunc.parametrize('offset', [None, np.zeros, lambda n: rng.uniform(0, 1, size=n)])

    if 'alpha' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("alpha", [1.])
        else:
            metafunc.parametrize("alpha", [0.1, 0.5, 1.])

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
        if not is_subdir:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("weights", [np.ones(10)])
            else:
                metafunc.parametrize("weights", [np.ones(100), rng.uniform(0, 1, size=100)])

    if 'intercept' in metafunc.fixturenames:
        if not is_subdir:
            if metafunc.config.getoption("test_size") == "small":
                metafunc.parametrize("intercept", [True])
            else:
                metafunc.parametrize("intercept", [True, False])

    if 'gls' in metafunc.fixturenames:
        if metafunc.config.getoption("test_size") == "small":
            metafunc.parametrize("gls", [None])
        else:
            def wish(n,p):
                X = rng.standard_normal((n, p))
                return X @ X.T
            def diag(n, p):
                return np.linspace(1, 2, n)
            metafunc.parametrize("gls", [None, wish, diag])
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


# R-specific fixture

@pytest.fixture(scope='session')
def Rdata():
    val = {}
    try:
        import rpy2.robjects as rpy

    except ImportError:
        pytest.skip('rpy2 is not importable')
        return {}
    
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter

    val['np_cv_rules'] = default_converter + numpy2ri.converter
    np_cv_ruls = val['np_cv_rules']
    
    val['glmnetR'] = importr('glmnet')
    val['baseR'] = importr('base')
    val['statR'] = importr('stats')
    val['survivalR'] = importr('survival')

    @dataclass
    class RGLMNet(object):

        family: str='"gaussian"'
        covariance: bool=False
        standardize: bool=True
        fit_intercept: bool=True
        exclude: list = field(default_factory=list)
        df_max: int=None
        nlambda: int=None
        lambda_min_ratio: float=None
        lower_limits: float=None
        upper_limits: float=None
        penalty_factor: float=None
        offset: float=None
        weights: float=None
        foldid: int=None
        grouped: bool=True
        alignment: str='lambda'

        def __post_init__(self):

            with np_cv_rules.context():

                args = {}
                args['family'] = self.family

                if self.df_max is not None:
                    rpy.r.assign('dfmax', self.df_max)
                    args['dfmax'] = 'dfmax'

                if self.weights is not None:
                    rpy.r.assign('weights', self.weights)
                    rpy.r('weights=as.numeric(weights)')
                    args['weights'] = 'weights'

                if self.offset is not None:
                    rpy.r.assign('offset', self.offset)
                    args['offset'] = 'offset'

                if self.lambda_min_ratio is not None:
                    rpy.r.assign('lambda.min.ratio', self.lambda_min_ratio)
                    args['lambda.min.ratio'] = 'lambda.min.ratio'

                if self.lower_limits is not None:
                    rpy.r.assign('lower.limits', self.lower_limits)
                    args['lower.limits'] = 'lower.limits'
                if self.upper_limits is not None:
                    rpy.r.assign('upper.limits', self.upper_limits)
                    args['upper.limits'] = 'upper.limits'

                if self.penalty_factor is not None:
                    rpy.r.assign('penalty.factor', self.penalty_factor)
                    args['penalty.factor'] = 'penalty.factor'

                if self.nlambda is not None:
                    rpy.r.assign('nlambda', self.nlambda)
                    args['nlambda'] = 'nlambda'

                if self.standardize:
                    rpy.r.assign('standardize', True)
                else:
                    rpy.r.assign('standardize', False)
                args['standardize'] = 'standardize'

                if self.fit_intercept:
                    rpy.r.assign('intercept', True)
                else:
                    rpy.r.assign('intercept', False)
                args['intercept'] = 'intercept'

                if len(self.exclude) > 1:
                    rpy.r.assign('exclude', np.array(self.exclude) + 1)
                else:
                    rpy.r.assign('exclude', np.array(self.exclude))
                args['exclude'] = 'exclude'

                self.args = args
                self.cvargs = copy(self.args)

                rpy.r.assign('doCV', self.foldid is not None)

                if self.foldid is not None:
                    rpy.r.assign('foldid', self.foldid)
                    rpy.r('foldid = as.integer(foldid)')
                    self.cvargs['foldid'] = 'foldid'

                rpy.r.assign('grouped', self.grouped)
                self.cvargs['grouped'] = 'grouped'

                self.cvargs['alignment'] = f'"{self.alignment}"'

        def parse(self):
            args = ','.join([f'{k}={v}' for k, v in self.args.items()])
            cvargs = ','.join([f'{k}={v}' for k, v in self.cvargs.items()])
            return args, cvargs



    val['RGLMNet'] = RGLMNet

    return val



