"""
The problem solved by a glmnet / glmstar fit, and the inputs for selective
inference after it. The inference itself is done by ``lassoinf``.

Adapted from github.com/jonathan-taylor/lassoinf.
"""

from .glmnet_problem import (GLMProblem,
                          GLMFamily,
                          resolve_family,
                          glm_moments,
                          problem_from_solution,
                          glmstar_problem,
                          glmnet_scaling,
                          glmnet_response_scale,
                          glmnet_penalty_factor,
                          kkt_violation,
                          XTVXOperator)
from .glmnet_inference import (GLMInferenceProblem,
                            glm_inference_problem,
                            glmstar_inference_problem,
                            score_perturbation)
