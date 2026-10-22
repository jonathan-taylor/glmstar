"""
The penalties of the REML and ML criteria as objects (`glmnet.gam.penalties`):
their matrices, log determinants and derivatives in their parameters, and
Wood's stable basis.
"""
import numpy as np
import pytest

from glmnet.gam.penalties import ScaledPenalty, SumPenalty, Dense, Diagonal


def _dense(piece, P):
    S = np.zeros((P, P))
    piece.add_to(S)
    return S


def test_scaled():
    ev = np.array([3., 1., 0.5])
    p = ScaledPenalty(np.array([1, 3, 4]), ev, ['s(x)'])
    theta = np.array([0.7])
    e = p.evaluate(theta)
    S = _dense(e.matrix, 6)
    np.testing.assert_allclose(np.diag(S)[[1, 3, 4]], np.exp(0.7) * ev)
    np.testing.assert_allclose(e.logdet, np.log(np.exp(0.7) * ev).sum())
    np.testing.assert_allclose(e.dlogdet, [3])
    beta = np.arange(6.)
    np.testing.assert_allclose(e.derivatives[0].apply(beta), S @ beta)


def test_sum():
    rng = np.random.default_rng(0)
    A1, A2 = rng.standard_normal((4, 4)), rng.standard_normal((2, 4))
    U1, U2 = A1.T @ A1, A2.T @ A2
    p = SumPenalty(np.array([0, 2, 3, 5]), [U1, U2], ['a[0]', 'a[1]'])
    theta = np.array([np.log(2.), 20.])       # the second dominant
    e = p.evaluate(theta)
    lam = np.exp(theta)
    V = e.V
    np.testing.assert_allclose(V.T @ V, np.identity(4), atol=1e-12)
    # in the stable basis: the weighted sum, rotated
    A = V.T @ (lam[0] * U1 + lam[1] * U2) @ V
    np.testing.assert_allclose(e.matrix.A, A, rtol=1e-8, atol=1e-6)
    # (the direct log determinant, at lam = e^20, is the less precise)
    np.testing.assert_allclose(e.logdet, np.linalg.slogdet(lam[0] * U1 + lam[1] * U2)[1],
                               rtol=1e-8)
    # the dominant penalty is exactly 0 beyond its range (rank 2)
    T2 = e.derivatives[1].A / lam[1]
    np.testing.assert_array_equal(T2[2:, :], 0)
    np.testing.assert_array_equal(T2[:, 2:], 0)
    # derivatives of log|S| in theta, against differences
    eps = 1e-6
    for k in range(2):
        d = np.zeros(2)
        d[k] = eps
        fd = (p.evaluate(theta + d).logdet - p.evaluate(theta - d).logdet) / (2 * eps)
        np.testing.assert_allclose(e.dlogdet[k], fd, rtol=1e-5)


def test_pieces():
    beta = np.arange(5.)
    D = Diagonal([0, 2], [2., 3.])
    np.testing.assert_allclose(D.apply(beta), [0, 0, 6, 0, 0])
    A = np.array([[1., 2.], [2., 5.]])
    M = Dense([1, 3], A)
    np.testing.assert_allclose(M.apply(beta), [0, 1 * 1 + 2 * 3, 0, 2 * 1 + 5 * 3, 0])
