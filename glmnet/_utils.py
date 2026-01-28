import numpy as np
import pandas as pd

from sklearn.utils import check_X_y

from dataclasses import fields

def _get_data(estimator,
              X,
              y,
              offset_id=None,
              weight_id=None,
              response_id=None,
              check=True,
              multi_output=True # always true below
              ):
    """
    Extracts response, offset, and weights from the input `y`.

    This function handles various input formats for `y` (DataFrame or array)
    and separates the response variable from any specified offset or weight columns.

    Parameters
    ----------
    estimator : object
        The estimator calling this function (for `check_X_y`).
    X : array-like
        The input data matrix.
    y : array-like or pd.DataFrame
        The target variable, which may contain response, offset, and weights.
    offset_id : str or int, optional
        Column identifier for the offset in `y`.
    weight_id : str or int, optional
        Column identifier for the weights in `y`.
    response_id : str or int, optional
        Column identifier for the response in `y`.
    check : bool, default=True
        Whether to perform `sklearn.utils.check_X_y` validation.
    multi_output : bool, default=True
        Whether to allow multi-output for `check_X_y`.

    Returns
    -------
    X : array-like
        The input data matrix (possibly validated).
    y : array-like or pd.DataFrame
        The original target variable.
    response : np.ndarray
        The extracted response variable.
    offset : np.ndarray or None
        The extracted offset term, or None if not provided.
    weight : np.ndarray
        The extracted weights, or an array of ones if not provided.
    """

    weight = None
    if offset_id is None and weight_id is None:
        # col could be 0 so check for None
        if response_id is not None:
            if isinstance(y, pd.DataFrame):
                response = y.loc[:,response_id]
            else:
                response = y[:,response_id]
        else:
            response = y
        if check:
            X, _ = check_X_y(X, response,
                             accept_sparse=['csc'],
                             multi_output=True,
                             estimator=estimator)
        offset, weight = None, None
    else:
        if isinstance(y, pd.DataFrame):
            response = y
            if offset_id is not None:
                offset = np.asarray(y.loc[:,offset_id])
                if type(offset_id) in [str, int]:
                    response = response.drop(columns=[offset_id])
                else:
                    response = response.drop(columns=offset_id)
            else:
                offset = None
            if weight_id is not None:
                weight = np.asarray(y.loc[:,weight_id])
                if type(weight_id) in [str, int]:
                    response = response.drop(columns=[weight_id])
                else:
                    response = response.drop(columns=weight_id)
            else:
                weight = None
        else:
            keep = np.ones(y.shape[1], bool)
            if offset_id is not None:
                offset = y[:,offset_id]
                keep[offset_id] = 0
            else:
                offset = None
            if weight_id is not None:
                weight = y[:,weight_id]
                keep[weight_id] = 0
            else:
                weight = None

        if check:
            X, _ = check_X_y(X, y,
                             accept_sparse=['csc'],
                             multi_output=True,
                             estimator=estimator)

    # col could be 0 so check for None
    if response_id is not None:
        if isinstance(y, pd.DataFrame):
            response = y.loc[:,response_id]
        else:
            response = y[:,response_id]
    else:
        # we already removed columns of the data frame
        if not isinstance(y, pd.DataFrame):
            if offset_id is not None or weight_id is not None: 
                response = y[:,keep]
            else:
                response = np.asarray(y)
    if weight is None:
        weight = np.ones(y.shape[0])
    return X, y, np.squeeze(np.asarray(response)), offset, weight

def _jerr_elnetfit(n, maxit, k=None):
    """
    Interprets error codes from `elnet` (C++ or Fortran) routines.

    Parameters
    ----------
    n : int
        The error code returned by the `elnet` routine.
    maxit : int
        Maximum number of iterations set for the `elnet` routine.
    k : int, optional
        The index of the lambda value for which convergence failed.

    Returns
    -------
    dict
        A dictionary containing the error code, a boolean indicating if it's a
        fatal error, and a descriptive message.
    """
    if n == 0:
        fatal = False
        msg = ''
    elif n > 0:
        # fatal error
        fatal = True
        msg =(f"Memory allocation error; contact package maintainer" if n < 7777 else
              "Unknown error")
    else:
        fatal = False
        msg = (f"Convergence for {k}-th lambda value not reached after maxit={maxit}" +
               " iterations; solutions for larger lambdas returned")
    return {'n':n,
            'fatal':fatal,
            'msg':f"Error code {n}:" + msg}

def _parent_dataclass_from_child(cls,
                                 parent_dict,
                                 **modified_args):
    """
    Creates an instance of a dataclass from a dictionary, filtering keys.

    This utility function is useful when initializing a dataclass with arguments
    that might come from a larger dictionary, ensuring only relevant keys are used.

    Parameters
    ----------
    cls : type
        The dataclass type to instantiate.
    parent_dict : dict
        A dictionary containing potential arguments for the dataclass.
    **modified_args
        Additional arguments to update or override values from `parent_dict`.

    Returns
    -------
    object
        An instance of the specified dataclass.
    """
    _fields = [f.name for f in fields(cls)]
    _cls_args = {k:parent_dict[k] for k in parent_dict.keys() if k in _fields}
    _cls_args.update(**modified_args)
    return cls(**_cls_args)
