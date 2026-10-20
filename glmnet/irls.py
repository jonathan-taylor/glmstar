"""
Iteratively reweighted least squares (IRLS), with step halving.

`IRLS` is generic in its pieces, which only need:

- `family.get_response_and_weights(state, y, offset, weights)`: the working
  response and weights at `state`; optionally `family.valid(state)`, whether
  the mean is in the family's domain; and `family.is_gaussian`, whether one
  step solves the problem;
- `regularizer.newton_step(design, pseudo_response, newton_weights, state)`:
  the next state; `regularizer.half_step(state, oldstate)`, the state halfway
  back to `oldstate`; `regularizer.check_state(state)`, which raises if
  `state` is not usable; optionally `regularizer.has_objective`, False if
  the steps do not decrease an objective (see `quasi_newton_step`);
- `state.update(design, family, offset, objective)`, which sets
  `state.linear_predictor`, `state.mu` and `state.obj_val`.
"""
import logging

import numpy as np
from copy import deepcopy


def quasi_newton_step(regularizer,
                      family,
                      design,
                      y,
                      offset,
                      weights,
                      state,
                      objective,
                      control):
    """
    Performs a single quasi-Newton step for the IRLS algorithm.

    This function computes the pseudo-response and weights, and then
    calls the regularizer's Newton step to update the model state.
    The step is halved, towards the current state, until the objective is
    finite (and below `control.big`), the mean is in the family's domain,
    and the objective has not increased. The last check is skipped if
    `regularizer.has_objective` is False, as for smoothers that do not
    minimize an objective.

    Parameters
    ----------
    regularizer : object
        The regularizer object, which must have a `newton_step` method.
    family : object
        The GLM family object.
    design : object
        The design matrix object.
    y : array-like
        The response variable.
    offset : array-like
        The offset term.
    weights : array-like
        The observation weights.
    state : object
        The current state of the model (e.g., coefficients).
    objective : callable
        The objective function to be minimized.
    control : object
        A control object with parameters for the optimization.

    Returns
    -------
    state : object
        The updated model state.
    boundary : bool
        True if the step was halved because the objective was not finite or
        the mean left the family's domain.
    halved : bool
        True if the step size was halved.
    newton_weights : array-like
        The weights used in the Newton step.
    """

    oldstate = deepcopy(state)

    pseudo_response, newton_weights = family.get_response_and_weights(state,
                                                                      y,
                                                                      offset,
                                                                      weights)

    state = regularizer.newton_step(design,
                                    pseudo_response,
                                    newton_weights,
                                    state)

    state.update(design,
                 family,
                 offset,
                 objective)

    # check to make sure it is a feasible descent step

    # a mean in the family's domain (CoxFamilySpec has no constraint)
    _valid = getattr(family, 'valid', lambda state: True)

    def finite_objective(state):
        return np.isfinite(state.obj_val) and state.obj_val < control.big

    def decreased_obj(state):
        return state.obj_val <= oldstate.obj_val + 1e-7

    # (name, test, whether failing it is a boundary, message)
    tests = [('finite objective', finite_objective, True,
              "Non finite objective function! Step size truncated due to divergence."),
             ('valid mean', _valid, True,
              "Invalid eta/mu! Step size truncated: out of bounds.")]
    if getattr(regularizer, 'has_objective', True):
        tests.append(('decreased objective', decreased_obj, False,
                      "Objective did not decrease!"))

    boundary = False
    halved = False  # did we have to halve the step size?

    for name, test, at_boundary, msg in tests:

        if test(state):
            continue

        if control.logging: logging.debug(msg)
        regularizer.check_state(oldstate)
        boundary = boundary or at_boundary
        halved = True

        ii = 1
        while not test(state):
            if ii > control.mxitnr:
                raise ValueError(f"IRLS step halving: no {name} after "
                                 f"{control.mxitnr} halvings; cannot correct step size")
            ii += 1

            state = regularizer.half_step(state,
                                          oldstate)
            state.update(design,
                         family,
                         offset,
                         objective)

    if control.logging: logging.debug(f'old value: {oldstate.obj_val}, new value: {state.obj_val}')

    return state, boundary, halved, newton_weights


def _relative_change(new, old):
    """`|new - old| / (0.1 + |new|)`, in norm for vectors."""
    new, old = np.asarray(new, float), np.asarray(old, float)
    return np.linalg.norm(new - old) / (0.1 + np.linalg.norm(new))


def IRLS(regularizer,
         family,
         design,
         y,
         offset,
         weights,
         state,
         objective,
         control,
         info=None):
    """
    Iteratively Reweighted Least Squares (IRLS) algorithm.

    This function implements the IRLS algorithm for fitting Generalized
    Linear Models. It iteratively calls `quasi_newton_step` until
    convergence or the maximum number of iterations is reached.

    The iterations stop when the change in the objective, relative to
    `0.1 + |objective|`, is below `control.epsnr` (as R's `glm.fit`, with the
    deviance). With `control.convergence='eta'`, or a regularizer whose
    `has_objective` is False, the change in the linear predictor is used
    instead, in norm. For the Gaussian with identity link
    (`family.is_gaussian`) one step solves the problem: the iterations stop
    after it, so `regularizer.newton_step` must then solve its problem
    exactly.

    Parameters
    ----------
    regularizer : object
        The regularizer object.
    family : object
        The GLM family object.
    design : object
        The design matrix object.
    y : array-like
        The response variable.
    offset : array-like
        The offset term.
    weights : array-like
        The observation weights.
    state : object
        The initial state of the model.
    objective : callable
        The objective function to be minimized.
    control : object
        A control object with parameters for the optimization: `mxitnr`,
        `epsnr`, `big`, `logging` and, optionally, `convergence`
        (`'objective'`, the default, or `'eta'`).
    info : dict, optional
        If given, filled with `'n_iter'`, the number of iterations, and
        `'n_halved'`, the number of them whose step was halved.

    Returns
    -------
    converged : bool
        True if the algorithm converged.
    boundary : bool
        True if a step was halved at a boundary (see `quasi_newton_step`).
    state : object
        The final model state.
    newton_weights : array-like
        The weights from the final Newton step.
    """

    converged = False
    boundary = False
    n_halved = 0

    criterion = getattr(control, 'convergence', 'objective')
    if not getattr(regularizer, 'has_objective', True):
        criterion = 'eta'
    if criterion not in ('objective', 'eta'):
        raise ValueError(f"convergence should be 'objective' or 'eta', not {criterion!r}")

    if control.logging:
        logging.info('Starting IRLS')
        logging.debug(f'{regularizer._debug_msg(state)}')

    for i in range(control.mxitnr):

        if criterion == 'objective':
            old_value = state.obj_val
        else:
            old_value = np.array(state.linear_predictor, float)

        (state,
         boundary_,
         halved,
         newton_weights) = quasi_newton_step(regularizer,
                                             family,
                                             design,
                                             y,
                                             offset,
                                             weights,
                                             state,
                                             objective,
                                             control)
        boundary = boundary or boundary_
        n_halved += halved

        if control.logging:
            logging.debug(f'Iteration {i}, {regularizer._debug_msg(state)}')
            logging.info(f'Objective: {state.obj_val}')

        # test for convergence
        new_value = state.obj_val if criterion == 'objective' else state.linear_predictor
        if (family.is_gaussian or
            _relative_change(new_value, old_value) < control.epsnr):
            converged = True
            break

    if info is not None:
        info.update(n_iter=i + 1, n_halved=n_halved)

    if control.logging:
        logging.info(f'Terminating IRLS after {i+1} iterations.')
        logging.debug(f'{regularizer._debug_msg(state)}')
    return converged, boundary, state, newton_weights
