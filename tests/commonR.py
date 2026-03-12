from dataclasses import dataclass, field

import pytest

try:
    import rpy2.robjects as rpy
    has_rpy2 = True

except ImportError:
    has_rpy2 = False

if has_rpy2:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter

    np_cv_rules = default_converter + numpy2ri.converter

    glmnetR = importr('glmnet')
    baseR = importr('base')
    statR = importr('stats')
    importr('survival')

else:
    np_cv_rules = glmnetR = baseR = statR = rpy = None

ifrpy = pytest.mark.skipif(not has_rpy2, reason='requires rpy2')




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

