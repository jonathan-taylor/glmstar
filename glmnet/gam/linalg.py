"""
The linear algebra of the REML and ML criteria (`glmnet.gam.reml`): the
operations on the Hessian `H = X' diag(h) X + S` of the penalized deviance
over 2 that the criterion and its gradient need, and nothing else:

- `logdet()`, `log|H|`;
- `solve(B)`, `H^{-1} B`;
- `trace(idx, U)`, `tr(H^{-1} A)` for a penalty `A` that is `U` on the
  coordinates `idx` (a matrix, or the diagonal as a vector) and 0 elsewhere;
- `leverages()`, the diagonal of `X H^{-1} X'`.

A factorization of `H` restricted to some coordinates (ML integrates only
the penalized coefficients) is `restrict(coords)`.

`DenseHessian` forms `H` (a dense `P x P` matrix) and factors it. Others,
with the same operations, need not form it: a sparse Cholesky, whose
selected inverse gives the traces and leverages (Takahashi, Fagan & Chen,
1973), or matrix-free solves (conjugate gradients, backfitting) with
stochastic estimates of the log determinant and traces (Hutchinson, 1989;
Ubaru, Chen & Saad, 2017). See the documentation's page "REML and ML for
GAMs: the design".
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve, LinAlgError


class DenseHessian(object):
    """
    `H = X' diag(h) X + S` as a dense matrix, factored: in coordinates in
    which the penalties of each term with several are diagonal (`rotations`,
    pairs `(coordinates, eigenvectors)`, as Wood's (2011) stable
    reparametrization), equilibrated (its diagonal scaled to 1), by Cholesky,
    or by LU when it is not positive definite (the observed information of a
    non-canonical link can be negative away from the optimum).

    Parameters
    ----------
    X : np.ndarray
        The design (n, P), in the criterion's coordinates.
    h : np.ndarray
        The weights of the observations in `H` (n,).
    S : np.ndarray
        The penalty matrix (P, P).
    rotations : list
        `(idx, V)`: the coordinates `idx` are rotated by the orthogonal `V`.
    """

    def __init__(self, X, h, S, rotations=()):
        self.X, self.h = X, h
        self.rotations = list(rotations)
        self._H = X.T @ (h[:, None] * X) + S
        self._factor(np.arange(X.shape[1]))

    def _factor(self, coords):
        self.coords = coords
        H = self._rotate_matrix(self._H)[np.ix_(coords, coords)]
        d = 1 / np.sqrt(np.maximum(np.abs(np.diag(H)), np.finfo(float).tiny))
        self._d = d
        Hs = d[:, None] * H * d[None, :]
        try:
            self._chol = cho_factor(Hs, lower=False)
            self._lu = None
        except LinAlgError:
            self._chol = None
            self._lu = lu_factor(Hs)
        # the positions of the coordinates (P) among those factored
        self._position = np.full(self._H.shape[0], -1)
        self._position[coords] = np.arange(len(coords))

    def restrict(self, coords):
        """The factorization of `H` restricted to the coordinates `coords`."""
        new = object.__new__(type(self))
        new.X, new.h, new.rotations, new._H = self.X, self.h, self.rotations, self._H
        new._factor(np.asarray(coords))
        return new

    # the rotations: of the rows and columns of a matrix, of vectors

    def _rotate_matrix(self, H):
        H = H.copy()
        for idx, V in self.rotations:
            H[:, idx] = H[:, idx] @ V
            H[idx, :] = V.T @ H[idx, :]
        return H

    def _rotate(self, B, back=False):
        B = np.array(B, float)
        for idx, V in self.rotations:
            B[idx] = (V if back else V.T) @ B[idx]
        return B

    def logdet(self):
        """`log|H|` (of the absolute value of its determinant)."""
        if self._chol is not None:
            logdet = 2 * np.sum(np.log(np.diag(self._chol[0])))
        else:
            logdet = np.sum(np.log(np.abs(np.diag(self._lu[0]))))
        return float(logdet - 2 * np.sum(np.log(self._d)))

    def _solve_factored(self, B):
        # B on the factored coordinates (rotated)
        B = self._d.reshape((-1,) + (1,) * (B.ndim - 1)) * B
        x = cho_solve(self._chol, B) if self._chol is not None else lu_solve(self._lu, B)
        return self._d.reshape((-1,) + (1,) * (B.ndim - 1)) * x

    def solve(self, B):
        """`H^{-1} B` (on the factored coordinates; `B` with a row per coordinate of `H`)."""
        Br = self._rotate(B)[self.coords]
        x = np.zeros((self._H.shape[0],) + np.shape(B)[1:])
        x[self.coords] = self._solve_factored(Br)
        return self._rotate(x, back=True)

    def _inverse_block(self, idx):
        # H^{-1} restricted to the coordinates idx, in the original
        # coordinates (idx among the factored ones)
        E = np.zeros((self._H.shape[0], len(idx)))
        E[idx, np.arange(len(idx))] = 1.
        return self.solve(E)[idx]

    def trace(self, idx, U):
        """`tr(H^{-1} A)`, `A` equal to `U` on the coordinates `idx` (a vector: its diagonal)."""
        Hinv = self._inverse_block(np.asarray(idx))
        U = np.asarray(U, float)
        if U.ndim == 1:
            return float(np.sum(np.diag(Hinv) * U))
        return float(np.sum(Hinv * U.T))

    def leverages(self):
        """The diagonal of `X H^{-1} X'` (`X` on the factored coordinates)."""
        Z = self.solve(self.X.T)                    # (P, n)
        return np.einsum('ij,ji->i', self.X, Z)
