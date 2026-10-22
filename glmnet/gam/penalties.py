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
    piece per parameter), `logdet` and `dlogdet` (its derivatives, of the
    penalty in the original coordinates), and the change of its coordinates
    in which the pieces are given: `beta = V beta~` on its coordinates (for
    each group's, if they are a `(G, q)` array; None: none), `V_inverse`,
    and `log|det V|` over all of them (`logdet_V`; 0 for a rotation). In
    the new coordinates the design is `X V`, the penalty `V' S V`, and
    `log|H|` is that of the original plus `2 logdet_V`.
    """

    def __init__(self, matrix, derivatives, logdet, dlogdet, V=None, V_inverse=None,
                 logdet_V=0.):
        self.matrix, self.derivatives = matrix, derivatives
        self.logdet, self.dlogdet = float(logdet), np.asarray(dlogdet, float)
        self.V = V
        self.V_inverse = V.T if V is not None and V_inverse is None else V_inverse
        self.logdet_V = float(logdet_V)


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


class Repeated(object):
    """
    The block diagonal matrix with the block `B` (`q x q`) on each group's
    coordinates `idx[g]` (`idx` of shape `(G, q)`): `I_G (x) B`, never
    formed.
    """

    def __init__(self, idx, B):
        self.idx, self.B = np.asarray(idx), np.asarray(B, float)

    def apply(self, beta):
        out = np.zeros_like(beta)
        out[self.idx] = beta[self.idx] @ self.B.T
        return out

    def add_to(self, S):
        S[self.idx[:, :, None], self.idx[:, None, :]] += self.B

    def trace(self, H):
        return H.trace_blocks(self.idx, self.B)


class CholeskyPenalty(object):
    """
    The precision `I_G (x) Sigma^{-1}` of `G` groups' vectors of `q`
    correlated random effects (coordinates `idx`, `(G, q)`), the relative
    covariance `Sigma = L L'` (lme4's relative covariance factor, the
    covariance over the dispersion on the prior weights' scale), `L` lower
    triangular, parametrized by the logs of its diagonal and its entries
    below (the log-Cholesky parametrization, Pinheiro & Bates, 1996).

    A singular covariance (a correlation of 1, or a variance of 0, lme4's
    boundary fit) is a diagonal entry of `L` at 0, its log at its lower
    bound, the others finite. So the evaluation is in the coordinates `u`
    of `b = L u` in each group (as lme4's): there the penalty is the
    identity, and its derivatives `-(L^{-1} E + E' L^{-T})` (`E` the
    derivative of `L`), without the precision's large entries.
    """

    def __init__(self, idx, names):
        self.idx = np.asarray(idx)
        self.G, self.q = self.idx.shape
        self.rows, self.cols = np.tril_indices(self.q)
        self.diag = self.rows == self.cols
        self.off_diagonal_params = ~self.diag
        self.names = list(names)
        self.n_params = len(self.rows)

    def factor(self, theta):
        L = np.zeros((self.q, self.q))
        L[self.rows, self.cols] = np.where(self.diag, np.exp(theta), theta)
        return L

    def theta(self, B):
        """The parameters of the precision `B` (on the prior weights' scale)."""
        L = np.linalg.cholesky(np.linalg.inv(np.asarray(B, float)))
        theta = L[self.rows, self.cols].copy()
        theta[self.diag] = np.log(theta[self.diag])
        return theta

    def precision(self, theta):
        Linv = np.linalg.inv(self.factor(theta))
        return Linv.T @ Linv

    def evaluate(self, theta):
        from scipy.linalg import solve_triangular
        theta = np.asarray(theta, float)
        L = self.factor(theta)
        Linv = solve_triangular(L, np.identity(self.q), lower=True)
        derivatives = []
        for i, j, d, t in zip(self.rows, self.cols, self.diag, theta):
            E = np.zeros((self.q, self.q))
            E[i, j] = np.exp(t) if d else 1.
            A = Linv @ E
            derivatives.append(Repeated(self.idx, -(A + A.T)))
        G = self.G
        return Evaluation(Repeated(self.idx, np.identity(self.q)), derivatives,
                          -2 * G * theta[self.diag].sum(), -2 * G * self.diag,
                          V=L, V_inverse=Linv, logdet_V=G * theta[self.diag].sum())

    def more_penalty(self, theta):
        """
        The direction of more penalty (a smaller covariance): the diagonal's
        logs down, the entries below it towards 0.
        """
        theta = np.asarray(theta, float)
        return np.where(self.diag, -1., -np.sign(theta))

    def start(self, xwx):
        """As mgcv's: the precision diagonal, matching the mean diagonal of each effect's `X' W X`."""
        d = xwx[self.idx].mean(0)
        return self.theta(np.diag(np.maximum(d, np.finfo(float).tiny)))


class CovariancePenalty(object):
    """
    The precision `I_G (x) Sigma(theta)^{-1}` of `G` groups' vectors of `q`
    random effects (coordinates `idx`, `(G, q)`), for a structured relative
    covariance `Sigma(theta)` given by a subclass (`covariance`,
    `derivatives`, `parameters`). As `CholeskyPenalty`, the evaluation is in
    the coordinates of `b = L u` per group, `L` the Cholesky factor of
    `Sigma` at `theta`: the penalty is the identity there, and its
    derivatives `-L^{-1} (dSigma / dtheta_k) L^{-T}`.

    Subclasses set `off_diagonal_params` (a boolean per parameter: the
    search fits the covariance diagonal first, with those held at 0).
    """

    def __init__(self, idx, names):
        self.idx = np.asarray(idx)
        self.G, self.q = self.idx.shape
        self.names = list(names)
        self.n_params = len(self.names)

    def precision(self, theta):
        return np.linalg.inv(self.covariance(theta))

    def theta(self, B):
        """The parameters of the precision `B` (on the prior weights' scale)."""
        return self.parameters(np.linalg.inv(np.asarray(B, float)))

    def evaluate(self, theta):
        from scipy.linalg import solve_triangular
        theta = np.asarray(theta, float)
        Sigma = self.covariance(theta)
        L = np.linalg.cholesky(Sigma)
        Linv = solve_triangular(L, np.identity(self.q), lower=True)
        Sinv = Linv.T @ Linv
        derivatives, dlogdet = [], []
        for dSigma in self.derivatives(theta):
            derivatives.append(Repeated(self.idx, -(Linv @ dSigma @ Linv.T)))
            dlogdet.append(-self.G * np.trace(Sinv @ dSigma))
        logdet_Sigma = 2 * np.sum(np.log(np.diag(L)))
        return Evaluation(Repeated(self.idx, np.identity(self.q)), derivatives,
                          -self.G * logdet_Sigma, dlogdet,
                          V=L, V_inverse=Linv, logdet_V=self.G * logdet_Sigma / 2)

    def start(self, xwx):
        """As mgcv's: the precision diagonal, matching the mean diagonal of each effect's `X' W X`."""
        d = np.maximum(xwx[self.idx].mean(0), np.finfo(float).tiny)
        return self.parameters(np.diag(1 / d))


class CompoundSymmetryPenalty(CovariancePenalty):
    """
    Compound symmetry (nlme's `pdCompSymm`): `Sigma = s ((1 - rho) I + rho
    J)`, a common variance `s` and correlation `rho` of the `q` effects,
    `theta = (log s, t)` with `rho = (e^t - 1) / (e^t + q - 1)` (nlme's: any
    `t` is a positive definite `Sigma`, `rho` in `(-1/(q - 1), 1)`).
    """
    off_diagonal_params = np.array([False, True])

    def _rho(self, t):
        e = np.exp(t)
        return (e - 1) / (e + self.q - 1), self.q * e / (e + self.q - 1)**2   # rho, drho/dt

    def covariance(self, theta):
        s, (rho, _) = np.exp(theta[0]), self._rho(theta[1])
        q = self.q
        return s * ((1 - rho) * np.identity(q) + rho * np.ones((q, q)))

    def derivatives(self, theta):
        s, (rho, drho) = np.exp(theta[0]), self._rho(theta[1])
        q = self.q
        C = (1 - rho) * np.identity(q) + rho * np.ones((q, q))
        return [s * C, s * drho * (np.ones((q, q)) - np.identity(q))]

    def parameters(self, Sigma):
        q = self.q
        s = np.mean(np.diag(Sigma))
        rho = (np.sum(Sigma) - np.trace(Sigma)) / (q * (q - 1)) / s if q > 1 else 0.
        # rho = (e^t - 1) / (e^t + q - 1)  =>  e^t = (1 + (q - 1) rho) / (1 - rho)
        return np.array([np.log(s), np.log((1 + (q - 1) * rho) / (1 - rho))])

    def more_penalty(self, theta):
        """A smaller covariance: the variance down, the correlation towards 0."""
        return np.array([-1., -np.sign(theta[1])])


class DiagonalPenalty(CovariancePenalty):
    """A diagonal covariance (nlme's `pdDiag`): independent effects, `theta` their log variances."""

    @property
    def off_diagonal_params(self):
        return np.zeros(self.q, bool)

    def covariance(self, theta):
        return np.diag(np.exp(np.asarray(theta, float)))

    def derivatives(self, theta):
        v = np.exp(np.asarray(theta, float))
        out = []
        for k in range(self.q):
            D = np.zeros((self.q, self.q))
            D[k, k] = v[k]
            out.append(D)
        return out

    def parameters(self, Sigma):
        return np.log(np.diag(Sigma))

    def more_penalty(self, theta):
        return -np.ones(self.q)


def covariance_penalty(kind, idx, name, effect_names):
    """The penalty of a correlated random effect term's covariance structure `kind`."""
    q = np.asarray(idx).shape[1]
    if kind == 'unstructured':
        names = [f'{name}[{a},{b}]' for a, b in zip(*np.tril_indices(q))]
        return CholeskyPenalty(idx, names)
    if kind == 'compound_symmetry':
        return CompoundSymmetryPenalty(idx, [f'{name}[variance]', f'{name}[correlation]'])
    if kind == 'diagonal':
        return DiagonalPenalty(idx, [f'{name}[{e}]' for e in effect_names])
    raise ValueError("covariance should be one of 'unstructured', 'compound_symmetry' or 'diagonal'")
