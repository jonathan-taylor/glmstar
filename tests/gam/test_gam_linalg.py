"""
The operations on the Hessian of the REML and ML criteria (`glmnet.gam.linalg`):
log determinant, solves, traces and leverages, against dense numpy, with
rotations of some coordinates and restricted to some coordinates.
"""
import numpy as np
import pytest

from glmnet.gam.linalg import DenseHessian


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    n, P = 50, 8
    X = rng.standard_normal((n, P))
    h = rng.uniform(0.5, 2, n)
    A = rng.standard_normal((P, P))
    S = A.T @ A * 1e3                     # (badly scaled against X' h X)
    V = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    return X, h, S, [(np.array([2, 4, 5]), V)]


@pytest.mark.parametrize('rotate', [False, True])
@pytest.mark.parametrize('restrict', [False, True])
def test_operations(problem, rotate, restrict):
    X, h, S, rotations = problem
    H = X.T @ (h[:, None] * X) + S
    F = DenseHessian(X, h, S, rotations if rotate else ())
    coords = np.arange(X.shape[1])
    if restrict:
        coords = np.array([0, 2, 3, 4, 5, 7])
        F = F.restrict(coords)
    Hc = H[np.ix_(coords, coords)]
    np.testing.assert_allclose(F.logdet(), np.linalg.slogdet(Hc)[1], rtol=1e-12)
    B = np.random.default_rng(1).standard_normal((X.shape[1], 3))
    expected = np.zeros_like(B)
    expected[coords] = np.linalg.solve(Hc, B[coords])
    np.testing.assert_allclose(F.solve(B), expected, rtol=1e-10, atol=1e-12)
    Hinv = np.zeros_like(H)
    Hinv[np.ix_(coords, coords)] = np.linalg.inv(Hc)
    idx = np.array([2, 4])
    U = np.array([[2., 0.5], [0.5, 1.]])
    np.testing.assert_allclose(F.trace(idx, U), np.sum(Hinv[np.ix_(idx, idx)] * U.T), rtol=1e-10)
    np.testing.assert_allclose(F.trace(idx, np.array([2., 1.])),
                               2 * Hinv[2, 2] + Hinv[4, 4], rtol=1e-10)
    np.testing.assert_allclose(F.leverages(), np.einsum('ij,jk,ik->i', X, Hinv, X), rtol=1e-10)


def test_indefinite():
    # (the observed information of a non-canonical link can be negative)
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 4))
    h = np.r_[-np.ones(18) * 3, np.ones(2)]
    F = DenseHessian(X, h, np.identity(4))
    H = X.T @ (h[:, None] * X) + np.identity(4)
    np.testing.assert_allclose(F.logdet(), np.linalg.slogdet(H)[1], rtol=1e-12)
    b = np.ones(4)
    np.testing.assert_allclose(F.solve(b), np.linalg.solve(H, b), rtol=1e-10)
