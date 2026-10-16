"""
Deprecated: renamed to :mod:`glmnet.inference.glmnet_problem`, as it is specific to
the lasso / elastic net. Its public names are also in :mod:`glmnet.inference`.
"""
import importlib
import warnings

_renamed = importlib.import_module('glmnet.inference.glmnet_problem')

warnings.warn('glmnet.inference.glm_problem is deprecated and will be removed: use '
              'glmnet.inference.glmnet_problem (or import from glmnet.inference)',
              DeprecationWarning,
              stacklevel=2)


def __getattr__(name):
    if name == 'glmnet_problem':     # the function, renamed problem_from_solution
        return _renamed._glmnet_problem
    return getattr(_renamed, name)
