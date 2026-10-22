"""
The penalties of the REML and ML criteria (`glmnet.gam.reml`) as objects:
a term's penalty `S(theta)` on its coordinates, a function of its
parameters `theta`, gives at each evaluation its matrix, the derivatives in
the parameters, `log|S(theta)|_+` and its derivatives. The criterion needs
nothing else of a penalty, so a new kind (an unstructured covariance of
correlated random effects, by its Cholesky factor) is a new class.

- `ScaledPenalty`: one penalty, `lam S` (`theta = log lam`), diagonal in
  the coordinates of the eigenvectors of `S`.
- `SumPenalty`: several, `sum_k lam_k S_k` (`theta_k = log lam_k`), in
  Wood's (2011) stable reparametrization at each evaluation.

A penalty's pieces are `Diagonal` or `Dense` on coordinates: they apply to
coefficients, add to a matrix and give traces with a factorization of `H`
(`glmnet.gam.linalg`).

References
----------
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Pinheiro, J. C. and Bates, D. M. (1996). Unconstrained parametrizations
  for variance-covariance matrices. *Statistics and Computing* 6(3),
  289–296.
"""
import numpy as np


class Diagonal(object):
    """The diagonal matrix `diag(d)` on the coordinates `idx` (0 elsewhere)."""

    def __init__(self, idx, d):
        self.idx, self.d = np.asarray(idx), np.asarray(d, float)

    def apply(self, beta):
        """The product with `beta` (all the coordinates)."""
        out = np.zeros_like(beta)
        out[self.idx] = self.d * beta[self.idx]
        return out

    def add_to(self, S):
        S[self.idx, self.idx] += self.d

    def trace(self, H):
        """`tr(H^{-1} A)`, `H` a factorization (`DenseHessian`)."""
        return H.trace(self.idx, self.d)


class Dense(object):
    """The matrix `A` on the coordinates `idx` (0 elsewhere)."""

    def __init__(self, idx, A):
        self.idx, self.A = np.asarray(idx), np.asarray(A, float)

    def apply(self, beta):
        out = np.zeros_like(beta)
        out[self.idx] = self.A @ beta[self.idx]
        return out

    def add_to(self, S):
        S[np.ix_(self.idx, self.idx)] += self.A

    def trace(self, H):
        return H.trace(self.idx, self.A)


class Evaluation(object):
    """
    A penalty at its parameters: `matrix` (a piece), `derivatives` (one
    piece per parameter), `logdet` and `dlogdet` (its derivatives), and the
    rotation `V` of its coordinates (None: none) in which they are given.
    """

    def __init__(self, matrix, derivatives, logdet, dlogdet, V=None):
        self.matrix, self.derivatives = matrix, derivatives
        self.logdet, self.dlogdet = float(logdet), np.asarray(dlogdet, float)
        self.V = V


class ScaledPenalty(object):
    """
    One penalty, `lam S`, on the coordinates `idx` in which `S` is diagonal
    (its positive eigenvalues `ev`): `theta = (log lam,)`.
    """

    def __init__(self, idx, ev, names):
        self.idx, self.ev = np.asarray(idx), np.asarray(ev, float)
        self.names = list(names)
        self.n_params = 1

    @staticmethod
    def theta(lams):
        return np.log(np.asarray(lams, float))

    @staticmethod
    def lams(theta):
        return np.exp(np.asarray(theta, float))

    def evaluate(self, theta):
        lam = float(np.exp(theta[0]))
        piece = Diagonal(self.idx, lam * self.ev)
        return Evaluation(piece, [piece], len(self.ev) * theta[0] + np.log(self.ev).sum(),
                          [len(self.ev)])

    def start(self, xwx):
        """mgcv's starting value: the mean of the diagonal of `X' W X` (`xwx`) on its coordinates."""
        return np.array([np.log(xwx[self.idx].mean() / self.ev.mean())])


class SumPenalty(object):
    """
    Several penalties, `sum_k lam_k U_k`, on the coordinates `idx` (the
    range of their sum), `U_k` dense there: `theta_k = log lam_k`. Each
    evaluation is in Wood's (2011) stable reparametrization (`stable_basis`).
    """

    def __init__(self, idx, Us, names):
        self.idx = np.asarray(idx)
        self.Us = [np.asarray(U, float) for U in Us]
        self.names = list(names)
        self.n_params = len(self.Us)

    theta = staticmethod(ScaledPenalty.theta)
    lams = staticmethod(ScaledPenalty.lams)

    def stable_basis(self, lam):
        """
        The range of the dominant penalty (the largest `lam_k |U_k|`) first,
        then, in its null space, that of the next, and so on. Each penalty
        is exactly 0 beyond the block where it dominates (its null space
        there), so a large smoothing parameter multiplies only its own block:
        its rounding does not reach the directions it does not penalize.
        Returns the basis `V` and the penalties `T_k = V' U_k V` in it.
        """
        Us = self.Us
        r = Us[0].shape[0]
        Q, cols, ends, active = np.identity(r), [], {}, list(range(len(Us)))
        while active and Q.shape[1] > 0:
            sizes = [lam[i] * np.linalg.norm(Q.T @ Us[i] @ Q) for i in active]
            i = active[int(np.argmax(sizes))]
            ev, W = np.linalg.eigh(Q.T @ Us[i] @ Q)
            if len(active) == 1:
                cols.append(Q @ W)
                break
            on = ev > max(ev.max(), 0) * 1e-10
            cols.append(Q @ W[:, on])
            ends[i] = sum(c.shape[1] for c in cols)
            Q = Q @ W[:, ~on]
            active.remove(i)
        V = np.hstack(cols)
        T = []
        for i, U in enumerate(Us):
            Ti = V.T @ U @ V
            Ti = (Ti + Ti.T) / 2
            if i in ends:
                Ti[ends[i]:, :] = 0.
                Ti[:, ends[i]:] = 0.
            T.append(Ti)
        return V, T

    def evaluate(self, theta):
        lam = np.exp(np.asarray(theta, float))
        V, T = self.stable_basis(lam)
        A = sum(l * Tk for l, Tk in zip(lam, T))
        dlogdet = [l * np.trace(np.linalg.solve(A, Tk)) for l, Tk in zip(lam, T)]
        return Evaluation(Dense(self.idx, A), [Dense(self.idx, l * Tk) for l, Tk in zip(lam, T)],
                          np.linalg.slogdet(A)[1], dlogdet, V=V)

    def start(self, xwx):
        mean = xwx[self.idx].mean()
        return np.array([np.log(mean / max(np.diag(U).mean(), np.finfo(float).tiny))
                         for U in self.Us])
