from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from copy import copy

import pytest
rng = np.random.default_rng(0)

from sklearn.model_selection import KFold

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

np_cv_rules = default_converter + numpy2ri.converter

from glmnet import GaussNet

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

@dataclass
class RGaussNet(RGLMNet):

    covariance: bool = False

    def __post_init__(self):

        super().__post_init__()
        if self.covariance:
            self.args['type.gaussian'] = '"covariance"'



def get_glmnet_soln(parser_cls,
                    X,
                    Y,
                    **args):

    parser = parser_cls(**args)
    args, cvargs = parser.parse()
    
    with np_cv_rules.context():
        rpy.r.assign('X', X)
        rpy.r.assign('Y', Y)
        cmd = f'''
library(glmnet)
Y = as.numeric(Y)
X = as.matrix(X)
G = glmnet(X, Y, {args})
C = as.matrix(coef(G))
if (doCV) {{
    foldid = as.integer(foldid)
    CVG = cv.glmnet(X, Y, {cvargs})
    CVM = CVG$cvm
    CVSD = CVG$cvsd
}}
            '''
        print(cmd)

        rpy.r(cmd)
        C = rpy.r('C')
        if parser.foldid is not None:
            CVM = rpy.r('CVM')
            CVSD = rpy.r('CVSD')

    if parser.foldid is None:
        return C.T
    else:
        return C.T, CVM, CVSD

# weight functions

def sample1(n):
    return rng.uniform(0, 1, size=n)

def sample2(n):
    V = sample1(n)
    V[:n//5] = 0
    return V

def get_data(n, p, sample_weight, offset):

    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    D = pd.DataFrame({'Y':Y})
    col_args = {'response_id':'Y'}
    
    if offset is not None:
        offset = offset(n)
        offset_id = 'offset'
        D['offset'] = offset
        offsetR = offset
    else:
        offset_id = None
        offsetR = None
    if sample_weight is not None:
        sample_weight = sample_weight(n)
        weight_id = 'weight'
        D['weight'] = sample_weight
        weightsR = sample_weight
    else:
        weight_id = None
        weightsR = None
        
    col_args = {'response_id':'Y',
                'weight_id':weight_id,
                'offset_id':offset_id}
    return X, Y, D, col_args, weightsR, offsetR


sample_weight_pyt = pytest.mark.parametrize('sample_weight', [None, np.ones, sample1, sample2])
df_max_pyt = pytest.mark.parametrize('df_max', [None, 5])
exclude_pyt = pytest.mark.parametrize('exclude', [[], [1,2,3]])
lower_limits_pyt = pytest.mark.parametrize('lower_limits', [-1, None])
# covariance changes type.gaussian, behaves unpredictably even in R
covariance_pyt = pytest.mark.parametrize('covariance', [None]) 
standardize_pyt = pytest.mark.parametrize('standardize', [True, False])
fit_intercept_pyt = pytest.mark.parametrize('fit_intercept', [True, False])
nlambda_pyt = pytest.mark.parametrize('nlambda', [None, 20])
lambda_min_ratio_pyt = pytest.mark.parametrize('lambda_min_ratio', [None,0.02])
nsample_pyt = pytest.mark.parametrize('n', [1000,50,500])
nfeature_pyt = pytest.mark.parametrize('p', [10,100])
limits_pyt = pytest.mark.parametrize('limits', [(-1, np.inf), (-np.inf, 1),
                                                (-np.inf, 0), (0, np.inf),
                                                (-np.inf, np.inf),
                                                (-1, 1),
                                                (0, 1)])
penalty_factor_pyt = pytest.mark.parametrize('penalty_factor', [None,
                                                                sample1,
                                                                sample2])
alignment_pyt = pytest.mark.parametrize('alignment', ['lambda', 'fraction'])
offset_pyt = pytest.mark.parametrize('offset', [None, np.zeros, lambda n: 20*sample1(n)]) # should match n=100 below

@sample_weight_pyt
@df_max_pyt
@exclude_pyt
@lower_limits_pyt
# covariance changes type.gaussian, behaves unpredictably even in R
@covariance_pyt
@standardize_pyt
@fit_intercept_pyt 
@nlambda_pyt
@lambda_min_ratio_pyt
@nsample_pyt
@nfeature_pyt
def test_gaussnet(covariance,
                  standardize,
                  fit_intercept,
                  exclude,
                  lower_limits,
                  nlambda,
                  lambda_min_ratio,
                  sample_weight,
                  df_max,
                  n,
                  p):

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, None)

    if lower_limits is not None:
        lower_limits = np.ones(p) * lower_limits
    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 exclude=exclude,
                 df_max=df_max, **col_args)

    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)

    C = get_glmnet_soln(RGaussNet,
                        X,
                        Y,
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        lower_limits=lower_limits,
                        exclude=exclude,
                        weights=weightsR,
                        offset=offsetR,
                        nlambda=nlambda,
                        lambda_min_ratio=lambda_min_ratio,
                        df_max=df_max)

    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < 1e-10
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < 1e-10


@limits_pyt
@penalty_factor_pyt
@sample_weight_pyt
def test_limits(limits,
                penalty_factor,
                sample_weight,
                df_max=None,
                covariance=None,
                standardize=True,
                fit_intercept=True,
                exclude=[],
                nlambda=None,
                lambda_min_ratio=None,
                n=100,
                p=50):

    lower_limits, upper_limits = limits
    if lower_limits is not None:
        lower_limits = np.ones(p) * lower_limits
    if upper_limits is not None:
        upper_limits = np.ones(p) * upper_limits
    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, None)

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 upper_limits=upper_limits,
                 lower_limits=lower_limits,
                 penalty_factor=penalty_factor,
                 exclude=exclude,
                 df_max=df_max, **col_args)
    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)

    C = get_glmnet_soln(RGaussNet,
                        X,
                        Y,
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                        penalty_factor=penalty_factor,
                        exclude=exclude,
                        weights=weightsR,
                        nlambda=nlambda,
                        df_max=df_max,
                        lambda_min_ratio=lambda_min_ratio)

    tol = 1e-10
    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < tol
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < tol

@offset_pyt
@penalty_factor_pyt
@sample_weight_pyt
def test_offset(offset,
                penalty_factor,
                sample_weight,
                df_max=None,
                covariance=None,
                standardize=True,
                fit_intercept=True,
                exclude=[],
                nlambda=None,
                lambda_min_ratio=None,
                n=100,
                p=50):

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset)

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 penalty_factor=penalty_factor,
                 exclude=exclude,
                 df_max=df_max, **col_args)
    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)

    C = get_glmnet_soln(RGaussNet,
                        X,
                        Y.copy(),
                        covariance=covariance,
                        standardize=standardize,
                        fit_intercept=fit_intercept,
                        penalty_factor=penalty_factor,
                        exclude=exclude,
                        weights=weightsR,
                        nlambda=nlambda,
                        offset=offsetR,
                        df_max=df_max,
                        lambda_min_ratio=lambda_min_ratio)

    tol = 1e-10
    assert np.linalg.norm(C[:,1:] - L.coefs_) / np.linalg.norm(L.coefs_) < tol
    if fit_intercept:
        assert np.linalg.norm(C[:,0] - L.intercepts_) / np.linalg.norm(L.intercepts_) < tol

@offset_pyt
@penalty_factor_pyt
@sample_weight_pyt
@alignment_pyt
def test_CV(offset,
            penalty_factor,
            sample_weight,
            alignment,
            df_max=None,
            covariance=None,
            standardize=True,
            fit_intercept=True,
            exclude=[],
            nlambda=None,
            lambda_min_ratio=None,
            n=103,
            p=50):

    if penalty_factor is not None:
        penalty_factor = penalty_factor(p)

    X, Y, D, col_args, weightsR, offsetR = get_data(n, p, sample_weight, offset)

    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(n)
    for i, (train, test) in enumerate(cv.split(np.arange(n))):
        foldid[test] = i+1

    L = GaussNet(covariance=covariance,
                 standardize=standardize,
                 fit_intercept=fit_intercept,
                 lambda_min_ratio=lambda_min_ratio,
                 penalty_factor=penalty_factor,
                 exclude=exclude,
                 df_max=df_max, **col_args)
    if nlambda is not None:
        L.nlambda = nlambda

    L.fit(X,
          D)
    L.cross_validation_path(X,
                            D,
                            alignment=alignment,
                            cv=cv)
    CVM_ = L.score_path_.scores['Mean Squared Error']
    CVSD_ = L.score_path_.scores['SD(Mean Squared Error)']
    C, CVM, CVSD = get_glmnet_soln(RGaussNet,
                                   X,
                                   Y.copy(),
                                   covariance=covariance,
                                   standardize=standardize,
                                   fit_intercept=fit_intercept,
                                   penalty_factor=penalty_factor,
                                   exclude=exclude,
                                   weights=weightsR,
                                   nlambda=nlambda,
                                   offset=offsetR,
                                   df_max=df_max,
                                   lambda_min_ratio=lambda_min_ratio,
                                   foldid=foldid,
                                   alignment=alignment)

    assert np.allclose(CVM, CVM_)
    assert np.allclose(CVSD, CVSD_)


def test_prefilter_excludes_features():
    from glmnet.paths.gaussnet import GaussNet

    class PrefilterGaussNet(GaussNet):
        def prefilter(self, X, y):
            # Exclude features where the sum of X[:, j] * y is negative
            return np.nonzero(X.T @ y > 0)[0]

    X = rng.standard_normal((100, 10))
    y = rng.standard_normal(100)
    X[:,:2] *= np.sign(X.T @ y)[:2][None,:]
    model = PrefilterGaussNet()
    model.fit(X, y)
    # All excluded features should have all-zero coefficients for all lambdas
    excluded = model.exclude
    assert excluded, "No features were excluded by prefilter"
    coefs = model.coefs_
    assert np.allclose(coefs[:, excluded], 0)


def test_prefilter_and_explicit_exclude():
    from glmnet.paths.gaussnet import GaussNet

    class PrefilterGaussNet(GaussNet):
        def prefilter(self, X, y):
            # Exclude features where the sum of X[:, j] * y is negative
            return np.nonzero(X.T @ y > 0)[0]

    X = rng.standard_normal((100, 10))
    y = rng.standard_normal(100)
    X[:,:2] *= -np.sign(X.T @ y)[:2][None,:]

    # Explicitly exclude feature 0, and let prefilter exclude others
    model = PrefilterGaussNet(exclude=[0])
    model.fit(X, y)
    excluded = model.exclude
    assert 0 in excluded, "Explicitly excluded feature 0 not in exclude list"
    coefs = model.coefs_
    assert np.allclose(coefs[:, excluded], 0)

