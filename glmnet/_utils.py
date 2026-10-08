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
    offset_id : str, int, or list, optional
        Column identifier for the offset in `y`.
    weight_id : str, int, or list, optional
        Column identifier for the weights in `y`.
    response_id : str, int, or list, optional
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
    is_df = isinstance(y, pd.DataFrame)

    def _as_list(val):
        """Helper to ensure column identifiers are lists for dropping/indexing."""
        if isinstance(val, (list, tuple, np.ndarray)):
            return list(val)
        return [val]

    # Extract Offset
    offset = None
    if offset_id is not None:
        offset = np.asarray(y.loc[:, offset_id] if is_df else y[:, offset_id])

    # Extract Weight
    weight = None
    if weight_id is not None:
        weight = np.asarray(y.loc[:, weight_id] if is_df else y[:, weight_id])

    # Extract Response
    if response_id is not None:
        response = np.asarray(y.loc[:, response_id] if is_df else y[:, response_id])
    else:
        if is_df:
            drop_cols = []
            if offset_id is not None:
                drop_cols.extend(_as_list(offset_id))
            if weight_id is not None:
                drop_cols.extend(_as_list(weight_id))
            response = np.asarray(y.drop(columns=drop_cols) if drop_cols else y)
        else:
            if y.ndim == 1:
                response = np.asarray(y)
            else:
                keep = np.ones(y.shape[1], dtype=bool)
                if offset_id is not None:
                    keep[_as_list(offset_id)] = False
                if weight_id is not None:
                    keep[_as_list(weight_id)] = False
                response = np.asarray(y[:, keep])

    # Default Weights
    if weight is None:
        weight = np.ones(y.shape[0])

    # Validate against Response
    if check:
        X, _ = check_X_y(X, response,
                         accept_sparse=['csc'],
                         multi_output=multi_output,
                         estimator=estimator)

    return X, y, np.squeeze(response), offset, weight

def _jerr_elnetfit(n, maxit, k=None, pmax=None):
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
    pmax : int, optional
        Maximum number of variables ever nonzero (`nx`), used in the
        message when it is exceeded.

    Returns
    -------
    dict
        A dictionary containing the error code, a boolean indicating if it's a
        fatal error, and a descriptive message.
    """
    if k is None and -10000 < n < 0:
        k = -n # index of the lambda value that did not converge, as in R
    if n == 0:
        fatal = False
        msg = ''
    elif n > 0:
        # fatal error
        fatal = True
        msg =(f"Memory allocation error; contact package maintainer" if n < 7777 else
              "Unknown error")
    elif n < -10000:
        fatal = False
        msg = (f"Number of nonzero coefficients along the path exceeds pmax={pmax}" +
               f" at {-n-10000}-th lambda value; solutions for larger lambdas returned")
    else:
        fatal = False
        msg = (f"Convergence for {k}-th lambda value not reached after maxit={maxit}" +
               " iterations; solutions for larger lambdas returned")
    return {'n':n,
            'fatal':fatal,
            'msg':f"Error code {n}:" + msg}

def _jerr_lognet(n, maxit, k=None, pmax=None):
    """
    Interprets error codes from the logistic and multinomial path routines
    (as in R's `jerr.lognet`).

    Parameters
    ----------
    n : int
        The error code returned by the `lognet` routine.
    maxit : int
        Maximum number of iterations set for the `lognet` routine.
    k : int, optional
        The index of the lambda value for which convergence failed.
    pmax : int, optional
        Maximum number of variables ever nonzero (`nx`).

    Returns
    -------
    dict
        A dictionary containing the error code, a boolean indicating if it's a
        fatal error, and a descriptive message.
    """
    if n < -20000:
        msg = (f"Max(p(1-p),1.0e-6 at {-n-20000}-th value of lambda;" +
               " solutions for larger values of lambda returned")
        return {'n':n,
                'fatal':False,
                'msg':f"Error code {n}:" + msg}
    if 8000 < n < 9000:
        msg = f"Null probability for class {n-8000} < 1.0e-5"
    elif 9000 < n < 10000:
        msg = f"Null probability for class {n-9000} > 1.0 - 1.0e-5"
    else:
        return _jerr_elnetfit(n, maxit, k, pmax=pmax)
    return {'n':n,
            'fatal':True,
            'msg':f"Error code {n}:" + msg}

def _jerr_coxnet(n, maxit, k=None, pmax=None):
    """
    Interprets error codes from the Cox path routine (as in R's `jerr.coxnet`).

    Parameters
    ----------
    n : int
        The error code returned by the `coxnet` routine.
    maxit : int
        Maximum number of iterations set for the `coxnet` routine.
    k : int, optional
        The index of the lambda value for which convergence failed.
    pmax : int, optional
        Maximum number of variables ever nonzero (`nx`).

    Returns
    -------
    dict
        A dictionary containing the error code, a boolean indicating if it's a
        fatal error, and a descriptive message.
    """
    if n > 0:
        if n == 8888:
            msg = "All observations censored - cannot proceed"
        elif n == 9999:
            msg = "No positive observation weights"
        elif n in [20000, 30000]:
            msg = "Initialization numerical error; probably too many censored observations"
        else:
            return _jerr_elnetfit(n, maxit, k, pmax=pmax)
        return {'n':n,
                'fatal':True,
                'msg':f"Error code {n}:" + msg}
    elif n <= -30000:
        msg = (f"Numerical error at {-n-30000}-th lambda value;" +
               " solutions for larger values of lambda returned")
        return {'n':n,
                'fatal':False,
                'msg':f"Error code {n}:" + msg}
    if k is None and -10000 < n < 0:
        k = -n # index of the lambda value that did not converge, as in R
    return _jerr_elnetfit(n, maxit, k, pmax=pmax)

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

# ===================================================================
# ARGUMENT DICTIONARY EXTRACTED FROM C++ WRAPPER FILES
# ===================================================================

# Using the dictionary structure you suggested for better organization
ALL_CPP_ARGS = {
    # From: elnet_point.cpp
    "elnet_point": [
        'alm0', 'almc', 'alpha', 'm', 'no', 'ni', 'x', 'r', 'xv', 'v',
        'intr', 'ju', 'vp', 'cl', 'nx', 'thr', 'maxit', 'a', 'aint', 'g',
        'ia', 'iy', 'iz', 'mm', 'nino', 'rsqc', 'nlp', 'jerr'
    ],
    "spelnet_point": [
        'alm0', 'almc', 'alpha', 'm', 'no', 'ni', 'x_data_array',
        'x_indices_array', 'x_indptr_array', 'xm', 'xs', 'r', 'xv', 'v',
        'intr', 'ju', 'vp', 'cl', 'nx', 'thr', 'maxit', 'a', 'aint', 'g',
        'ia', 'iy', 'iz', 'mm', 'nino', 'rsqc', 'nlp', 'jerr'
    ],

    # From: lognet.cpp
    "lognet": [
        'parm', 'ni', 'no', 'x', 'y', 'g', 'jd', 'vp', 'cl', 'ne', 'nx',
        'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'kopt', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "splognet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'g', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam',
        'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'kopt', 'pb', 'lmu',
        'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],

    # From: gaussnet.cpp
    "gaussnet": [
        'ka', 'parm', 'ni', 'no', 'x', 'y', 'w', 'jd', 'vp', 'cl', 'ne',
        'nx', 'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'rsq', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spgaussnet": [
        'ka', 'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam',
        'flmin', 'ulam', 'thr', 'isd', 'intr', 'pb', 'lmu', 'a0', 'ca',
        'ia', 'nin', 'rsq', 'maxit', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    
    # From: multigaussnet.cpp
    "multigaussnet": [
        'parm', 'ni', 'no', 'x', 'y', 'w', 'jd', 'vp', 'cl', 'ne', 'nx',
        'nlam', 'flmin', 'ulam', 'thr', 'isd', 'jsd', 'intr', 'maxit', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'rsq', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spmultigaussnet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam',
        'flmin', 'ulam', 'thr', 'isd', 'jsd', 'intr', 'maxit', 'pb', 'lmu',
        'a0', 'ca', 'ia', 'nin', 'rsq', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],

    # From: fishnet.cpp
    "fishnet": [
        'parm', 'ni', 'no', 'x', 'y', 'w', 'g', 'jd', 'vp', 'cl', 'ne',
        'nx', 'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spfishnet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'w', 'g', 'jd', 'vp', 'cl', 'ne', 'nx',
        'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'pb', 'lmu',
        'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],

    # From: coxnet.cpp
    "coxnet": [
        'parm', 'ni', 'no', 'x', 'start', 'stop', 'status', 'strata', 'efron',
        'g', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam', 'flmin', 'ulam', 'thr',
        'isd', 'maxit', 'pb', 'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev',
        'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spcoxnet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'start', 'stop', 'status', 'strata', 'efron',
        'g', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam', 'flmin', 'ulam', 'thr',
        'isd', 'maxit', 'pb', 'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev',
        'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ]
}


# ===================================================================
# VALIDATION FUNCTION (Unchanged, it works with the new structure)
# ===================================================================

def _validate_cpp_args(
    args_dict: dict,
    function_name: str
) -> bool:
    """
    Compares a dictionary of arguments against a list of expected names.
    """
    expected_set = set(ALL_CPP_ARGS[function_name])
    actual_set = set(args_dict.keys())

    missing_args = expected_set - actual_set
    extra_args = actual_set - expected_set

    msg = ''
    if missing_args:
        msg = f"ERROR: The following arguments for {function_name} are MISSING: {sorted(list(missing_args))}, "
    if extra_args:
        msg += f"ERROR: The following EXTRA arguments for {function_name} were found: {sorted(list(extra_args))}"
    if msg:
        return msg

