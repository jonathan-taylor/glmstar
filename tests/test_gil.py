"""Tests that the compiled path solvers release the GIL.

The solvers wrap their call into the C++ driver in a
``py::gil_scoped_release`` block (``src/*.cpp``), and the progress-bar
callback reacquires it (``src/update_pb.cpp``).  These tests check both
halves: that other Python threads make progress while a solve is running,
and that concurrent fits still produce the serial answer.
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from glmnet import GaussNet, CoxNet
from glmnet.cox import CoxFamily



def _problem(nrow, ncol, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((nrow, ncol))
    beta = np.zeros(ncol)
    beta[:20] = 2 * rng.standard_normal(20)
    resp = D @ beta + rng.standard_normal(nrow)
    return D, resp


def _cox_problem(nrow, ncol, seed=0, sparse=False):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((nrow, ncol))
    if sparse:
        D[rng.uniform(size=D.shape) < 0.7] = 0
    beta = np.zeros(ncol)
    beta[:20] = 0.3 * rng.standard_normal(20)
    T = rng.exponential(np.exp(-D @ beta))
    C = rng.exponential(np.median(T) * 2, size=nrow)
    resp = pd.DataFrame({'stop': np.minimum(T, C),
                         'status': (T <= C).astype(int)})
    if sparse:
        D = scipy.sparse.csc_array(D)
    return D, resp


def _coxnet(**args):
    return CoxNet(family=CoxFamily(event_id='stop', status_id='status'), **args)


def _gaussnet_solve():
    L = GaussNet(nlambda=200, lambda_min_ratio=1e-4)
    L.fit(*_problem(1000, 2000))
    return L._dense, L._args


def _coxnet_solve(sparse):
    def build():
        L = _coxnet(nlambda=100, lambda_min_ratio=1e-3)
        L.fit(*_cox_problem(800, 400, sparse=sparse))
        return (L._sparse if sparse else L._dense), L._args
    return build


# each builds a fitted model and returns its compiled solver and arguments
SOLVES = {'gaussnet': _gaussnet_solve,
          'coxnet': _coxnet_solve(False),
          'coxnet_sparse': _coxnet_solve(True)}


def _spin_rate(counter_ref, duration):
    """Counts/second the spinner achieves while the main thread is idle."""
    start = counter_ref()
    tic = time.time()
    time.sleep(duration)
    return (counter_ref() - start) / (time.time() - tic)


@pytest.mark.parametrize('model', SOLVES)
def test_gil_released_during_solve(model):
    """A plain Python thread must keep running while a solve is in flight.

    The timed region has to be the extension call on its own: a whole
    ``fit()`` spends most of its wall clock in Python (building the design,
    extracting coefficients), and another thread trivially runs during
    that.  So fit once to build the argument dict, then re-enter the
    compiled solver directly with no Python in the region being measured.

    The comparison is a rate, not "did it move at all".  A counter thread
    picks up a burst of increments at the edges of the measured window
    while the main thread is still in Python, which is enough to move an
    absolute count even when the GIL is held for the entire solve.  What
    separates the two cases is what fraction of the spinner's unobstructed
    rate it sustains: ~all of it when the GIL is released for the solve,
    versus a switch interval or two out of the whole solve when it is not.
    """
    solve, args = SOLVES[model]()

    counter = 0
    stop = threading.Event()

    def spin():
        nonlocal counter
        while not stop.is_set():
            counter += 1

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-3)   # keep the edge bursts small
    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    try:
        free_rate = _spin_rate(lambda: counter, 0.2)

        start = counter
        tic = time.time()
        solve(**args)
        elapsed = time.time() - tic
        solve_rate = (counter - start) / elapsed
    finally:
        stop.set()
        spinner.join()
        sys.setswitchinterval(old_interval)

    if elapsed < 0.05:
        pytest.skip(f'solve too fast ({elapsed:.3f}s) to say anything about the GIL')

    fraction = solve_rate / free_rate
    assert fraction > 0.3, (
        f'other thread ran at {fraction:.1%} of its unobstructed rate during a '
        f'{elapsed:.3f}s solve; the GIL looks held'
    )


@pytest.mark.parametrize('model', ['gaussnet', 'coxnet'])
def test_concurrent_fits_match_serial(model):
    if model == 'gaussnet':
        problems = [_problem(500, 300, seed=seed) for seed in range(4)]
        estimator = lambda: GaussNet(nlambda=100)
    else:
        problems = [_cox_problem(500, 300, seed=seed) for seed in range(4)]
        estimator = lambda: _coxnet(nlambda=100)

    def fit(problem):
        return estimator().fit(*problem)

    serial = [fit(p) for p in problems]

    with ThreadPoolExecutor(max_workers=len(problems)) as pool:
        threaded = list(pool.map(fit, problems))

    for expected, got in zip(serial, threaded):
        assert np.allclose(expected.coefs_, got.coefs_)
        assert np.allclose(expected.intercepts_, got.intercepts_)
        assert np.allclose(expected.lambda_values_, got.lambda_values_)
