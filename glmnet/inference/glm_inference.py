"""
Deprecated: renamed to :mod:`glmnet.inference.glmnet_inference`, as it is specific to
the lasso / elastic net. Its public names are also in :mod:`glmnet.inference`.
"""
import importlib
import warnings

_renamed = importlib.import_module('glmnet.inference.glmnet_inference')

warnings.warn('glmnet.inference.glm_inference is deprecated and will be removed: use '
              'glmnet.inference.glmnet_inference (or import from glmnet.inference)',
              DeprecationWarning,
              stacklevel=2)


def __getattr__(name):
    return getattr(_renamed, name)
