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

