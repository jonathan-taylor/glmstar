from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("glmnet")
except PackageNotFoundError:
    # package is not installed, perhaps we are in a git repo
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "unknown"

from .glmnet import GLMNet
from .glm import GLM
from .cox import CoxNet
from .paths.gaussnet import GaussNet
from .paths.lognet import LogNet
from .paths.fishnet import FishNet
from .paths.multigaussnet import MultiGaussNet
from .paths.multiclassnet import MultiClassNet
