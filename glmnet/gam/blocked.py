"""
REML and ML with one large random effect, exactly and without dense `P x P`
matrices: the blocked factorization of the Hessian (as lme4's and
MixedModels.jl's, for one grouping factor first).

A random effect `random_effect(g)` (or `random_effect(g, by=v)`) of `R` levels has a column per
level, each row in one of them: its block of `X' W X` is diagonal, as is its
penalty, the identity. With its coefficients first (`a`) and the others
(`e`, the intercept, the linear terms, the smooths: `m` of them),

    H = [ A   B' ]      A diagonal (R), B (m x R), E (m x m),
        [ B   E  ]

and with the Schur complement `C = E - B A^{-1} B'` (`m x m`):

- `log|H| = sum log A + log|C|`;
- solves by block elimination: `v = C^{-1} (r_e - B A^{-1} r_a)`,
  `u = A^{-1} (r_a - B' v)`;
- the diagonal of `H^{-1}`: `1/A + diag(G C^{-1} G')` on `a`, `G = A^{-1}
  B'`, and `diag(C^{-1})` on `e` (the traces of the penalties);
- the leverages `x_i' H^{-1} x_i = v_i^2 / A_l + u_i' C^{-1} u_i`, row `i`
  of level `l` with value `v_i`, `u_i = x_{e,i} - v_i G_l`.

These are the four operations of `glmnet.gam.linalg`: the criterion
(`glmnet.gam.reml`) is unchanged. The cost is `O(N m^2 + R m^2 + m^3)`;
nothing is `R x R` or `N x R` (the random effect's design is its rows'
levels and values, `SplitDesign`).

The inner fit is the GAM's backfitting (the random effect's smoother is a
division per level), polished by a Newton step with this factorization.
"""
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve, LinAlgError

from .joint import JointDesign


class SplitDesign(object):
    """
    The design `X` (`n x P`) with the large random effect's columns implicit:
    the coordinates `a` (contiguous, `R` of them) are its levels, row `i`
    having `values[i]` in column `a[codes[i]]`; the others, `e_idx`, are the
    dense `E` (`n x m`).
    """
    __array_ufunc__ = None          # (`v @ X` and `X @ v` are ours)

    def __init__(self, P, a_start, R, codes, values, e_idx, E):
        self.P, self.R = P, R
        self.a = np.arange(a_start, a_start + R)
        self.codes, self.values = np.asarray(codes), np.asarray(values, float)
        self.e_idx, self.E = np.asarray(e_idx), np.asarray(E, float)
        self._epos = np.full(P, -1)
        self._epos[self.e_idx] = np.arange(self.e_idx.shape[0])

    @property
    def shape(self):
        return (self.E.shape[0], self.P)

    def copy(self):
        new = object.__new__(SplitDesign)
        new.__dict__.update(self.__dict__)
        new.E = self.E.copy()
        return new

    def _e_positions(self, idx):
        pos = self._epos[np.asarray(idx)]
        if np.any(pos < 0):
            raise ValueError("a change of coordinates of the large random effect's")
        return pos

    def __getitem__(self, key):
        rows, idx = key
        return self.E[rows][:, self._e_positions(idx)]

    def __setitem__(self, key, value):
        rows, idx = key
        if rows != slice(None):
            raise ValueError('only whole columns')
        self.E[:, self._e_positions(idx)] = value

    def __matmul__(self, beta):
        beta = np.asarray(beta, float)
        out = self.E @ beta[self.e_idx]
        ba = beta[self.a][self.codes]
        return out + (self.values[:, None] * ba if beta.ndim > 1 else self.values * ba)

    def rmatvec(self, v):
        """`X' v`, `v` (n,) or (n, K)."""
        v = np.asarray(v, float)
        out = np.zeros((self.P,) + v.shape[1:])
        if v.ndim == 1:
            out[self.a] = np.bincount(self.codes, weights=self.values * v, minlength=self.R)
        else:
            out[self.a] = self.indicators().T @ v
        out[self.e_idx] = self.E.T @ v
        return out

    @property
    def T(self):
        return _Transposed(self)

    def indicators(self):
        """The random effect's columns, sparse (`n x R`)."""
        n = self.codes.shape[0]
        return sparse.csr_matrix((self.values, (np.arange(n), self.codes)), shape=(n, self.R))

    def weighted_colsq(self, h):
        """`diag(X' diag(h) X)`."""
        out = np.zeros(self.P)
        out[self.a] = np.bincount(self.codes, weights=h * self.values**2, minlength=self.R)
        out[self.e_idx] = np.einsum('ij,i,ij->j', self.E, h, self.E)
        return out


class _Transposed(object):
    __array_ufunc__ = None

    def __init__(self, X, scale=1.):
        self.X, self.scale = X, scale

    def __matmul__(self, v):
        return self.scale * self.X.rmatvec(v)

    def __rmul__(self, c):
        # (`c * X.T @ v`, as `(c * X.T) @ v`)
        return _Transposed(self.X, c * self.scale)

    __mul__ = __rmul__


class SplitPenalty(object):
    """
    The penalty matrix `S` (`P x P`) of the criterion with a large diagonal
    block: its diagonal `diag` (all the coordinates) and its off-diagonal
    part, dense on the coordinates `e_idx` only.
    """
    __array_ufunc__ = None

    def __init__(self, P, e_idx):
        self.diag = np.zeros(P)
        self.e_idx = np.asarray(e_idx)
        self._epos = np.full(P, -1)
        self._epos[self.e_idx] = np.arange(self.e_idx.shape[0])
        self.EE = np.zeros((self.e_idx.shape[0],) * 2)       # (off the diagonal)

    def add_diagonal(self, idx, d):
        np.add.at(self.diag, np.asarray(idx), d)

    def add_block(self, idx, A):
        idx = np.asarray(idx)
        A = np.asarray(A, float)
        pos = self._epos[idx]
        if np.any(pos < 0):
            raise ValueError('a dense penalty on the large random effect')
        self.diag[idx] += np.diag(A)
        self.EE[np.ix_(pos, pos)] += A - np.diag(np.diag(A))

    def __matmul__(self, beta):
        beta = np.asarray(beta, float)
        out = (self.diag[:, None] if beta.ndim > 1 else self.diag) * beta
        out[self.e_idx] += self.EE @ beta[self.e_idx]
        return out

    __rmatmul__ = __matmul__            # (symmetric)

    def __rmul__(self, c):
        # (`c * S @ beta`, as `(c * S) @ beta`)
        new = object.__new__(SplitPenalty)
        new.__dict__.update(self.__dict__)
        new.diag, new.EE = c * self.diag, c * self.EE
        return new

    __mul__ = __rmul__

    def dense_e(self):
        """`S` on the coordinates `e_idx`."""
        return self.EE + np.diag(self.diag[self.e_idx])


def _factor_dense(C):
    """Equilibrated Cholesky (LU when not positive definite) of the small `C`."""
    d = 1 / np.sqrt(np.maximum(np.abs(np.diag(C)), np.finfo(float).tiny))
    Cs = d[:, None] * C * d[None, :]
    try:
        return ('chol', cho_factor(Cs, lower=False), d)
    except LinAlgError:
        return ('lu', lu_factor(Cs), d)


def _solve_dense(factor, B):
    kind, f, d = factor
    shape = (-1,) + (1,) * (np.ndim(B) - 1)
    B = d.reshape(shape) * B
    x = cho_solve(f, B) if kind == 'chol' else lu_solve(f, B)
    return d.reshape(shape) * x


def _logdet_dense(factor):
    kind, f, d = factor
    diag = np.diag(f[0])
    logdet = 2 * np.sum(np.log(diag)) if kind == 'chol' else np.sum(np.log(np.abs(diag)))
    return float(logdet - 2 * np.sum(np.log(d)))


class BlockedHessian(object):
    """
    `H = X' diag(h) X + S` with `X` a `SplitDesign` and `S` a `SplitPenalty`,
    factored by blocks (see `glmnet.gam.blocked`), with the operations of
    `glmnet.gam.linalg.DenseHessian`: `logdet`, `solve`, `trace`,
    `trace_blocks`, `leverages`, `restrict`.
    """

    def __init__(self, X, h, S):
        self.X = X
        a, e = X.a, X.e_idx
        h = np.asarray(h, float)
        self._P = X.P
        self.A = np.bincount(X.codes, weights=h * X.values**2, minlength=X.R) + S.diag[a]
        hE = h[:, None] * X.E
        self.Bt = np.asarray(X.indicators().T @ hE)              # (R, m): B'
        self.Ee = X.E.T @ hE + S.dense_e()
        self._S_ee = S.dense_e()
        self._S_aa = S.diag[a]
        self._setup(np.ones(e.shape[0], bool))

    def _setup(self, keep_e):
        X = self.X
        self._keep_e = keep_e
        self.e = X.e_idx[keep_e]
        self.coords = np.sort(np.concatenate([X.a, self.e]))
        Bt = self.Bt[:, keep_e]
        self.G = Bt / self.A[:, None]                              # A^{-1} B'
        C = self.Ee[np.ix_(keep_e, keep_e)] - Bt.T @ self.G
        self._C = (C + C.T) / 2
        self._factor = _factor_dense(self._C)
        self._Cinv = None
        self._diag = None

    def restrict(self, coords):
        """The factorization of `H` restricted to the coordinates `coords` (all of `a`)."""
        coords = np.asarray(coords)
        if not np.all(np.isin(self.X.a, coords)):
            raise ValueError("a restriction of the large random effect's coordinates")
        new = object.__new__(BlockedHessian)
        new.__dict__.update(self.__dict__)
        new._setup(np.isin(self.X.e_idx, coords))
        return new

    def _inverse_e(self):
        if self._Cinv is None:
            m = self._C.shape[0]
            Cinv = _solve_dense(self._factor, np.identity(m))
            self._Cinv = (Cinv + Cinv.T) / 2
        return self._Cinv

    def logdet(self):
        """`log|H|` (of the absolute value of its determinant)."""
        return float(np.sum(np.log(np.abs(self.A))) + _logdet_dense(self._factor))

    def solve(self, B):
        """`H^{-1} B` on the coordinates (0 on the others), `B` with a row per coordinate."""
        B = np.asarray(B, float)
        ra, re = B[self.X.a], B[self.e]
        A = self.A.reshape((-1,) + (1,) * (B.ndim - 1))
        v = _solve_dense(self._factor, re - self.G.T @ ra)
        u = ra / A - self.G @ v
        x = np.zeros_like(B)
        x[self.X.a], x[self.e] = u, v
        return x

    def inverse_diagonal(self):
        """The diagonal of `H^{-1}` (P; 0 off the coordinates)."""
        if self._diag is None:
            Cinv = self._inverse_e()
            d = np.zeros(self._P)
            d[self.X.a] = 1 / self.A + np.einsum('ij,jk,ik->i', self.G, Cinv, self.G)
            d[self.e] = np.diag(Cinv)
            self._diag = d
        return self._diag

    def _e_positions(self, idx):
        pos = np.full(self._P, -1)
        pos[self.e] = np.arange(self.e.shape[0])
        p = pos[np.asarray(idx)]
        if np.any(p < 0):
            raise ValueError("a dense block on the large random effect's coordinates")
        return p

    def trace(self, idx, U):
        """`tr(H^{-1} A)`, `A` equal to `U` on the coordinates `idx` (a vector: its diagonal)."""
        U = np.asarray(U, float)
        idx = np.asarray(idx)
        if U.ndim == 1:
            return float(np.sum(self.inverse_diagonal()[idx] * U))
        p = self._e_positions(idx)
        return float(np.sum(self._inverse_e()[np.ix_(p, p)] * U.T))

    def trace_blocks(self, idx, B):
        """`tr(H^{-1} (I_G (x) B))`, the block `B` on each row of coordinates `idx` (on `e`)."""
        idx = np.asarray(idx)
        p = self._e_positions(idx.reshape(-1)).reshape(idx.shape)
        Cinv = self._inverse_e()
        blocks = Cinv[p[:, :, None], p[:, None, :]]
        return float(np.einsum('gij,ji->', blocks, B))

    def leverages(self):
        """The diagonal of `X H^{-1} X'` (`X` on the coordinates)."""
        X = self.X
        E = X.E[:, self._keep_e]
        U = E - X.values[:, None] * self.G[X.codes]
        return X.values**2 / self.A[X.codes] + np.einsum('ij,jk,ik->i', U, self._inverse_e(), U)

    def edf(self, S):
        """`diag(H^{-1} X' W X) = 1 - diag(H^{-1} S)`, `S` the `SplitPenalty` (P)."""
        Cinv = self._inverse_e()
        out = np.zeros(self._P)
        out[self.X.a] = 1 - self.inverse_diagonal()[self.X.a] * self._S_aa
        S_ee = self._S_ee[np.ix_(self._keep_e, self._keep_e)]
        out[self.e] = 1 - np.einsum('ij,ji->i', Cinv, S_ee)
        return out


class SplitJointDesign(JointDesign):
    """
    `JointDesign` with the large random effect (the term `big`) implicit: its
    rows' levels and values, no `N x R` block, and no `R x R` penalty (its
    penalty set is `None`; it is the identity). The other blocks are dense,
    `X_E` on the coordinates `e_idx` (with the intercept).
    """

    def __init__(self, terms, big):
        others = [t for j, t in enumerate(terms) if j != big]
        # (the others as JointDesign builds them, then the big one inserted)
        n = terms[big].codes_.shape[0]
        inner = JointDesign(others) if others else _InterceptOnly(n)
        self.terms = terms
        self.big = big
        R = terms[big].n_coef()
        self.Z, self.means, self.penalties, self.slices, self.penalty_sets = [], [], [], [], []
        self._roots = []
        start, k = 1, 0
        e_cols, e_idx = [inner.X[:, :1]], [0]
        for j, term in enumerate(terms):
            if j == big:
                self.Z.append(None)
                self.means.append(None)
                self.penalties.append(None)
                self.penalty_sets.append(None)
                self._roots.append(None)
                self.slices.append(slice(start, start + R))
                self.a_start = start
                start += R
                continue
            sl = inner.slices[k]
            width = sl.stop - sl.start
            self.Z.append(inner.Z[k])
            self.means.append(inner.means[k])
            self.penalties.append(inner.penalties[k])
            self.penalty_sets.append(inner.penalty_sets[k])
            self._roots.append(inner._roots[k])
            self.slices.append(slice(start, start + width))
            e_cols.append(inner.X[:, sl])
            e_idx.extend(range(start, start + width))
            start += width
            k += 1
        self._n_coef = start
        self.R = R
        self.X_E = np.hstack(e_cols)
        self.e_idx = np.array(e_idx)
        term = terms[big]
        self.codes = term.codes_
        self.values = np.asarray(term.v_, float)

    @property
    def n_coef(self):
        return self._n_coef

    @property
    def X(self):
        raise ValueError('the design with a large random effect is not formed (SplitDesign)')

    def gamma(self, state):
        gamma = np.zeros(self.n_coef)
        gamma[0] = state.intercept
        coefs = state.coefs if state.coefs is not None else [None] * len(self.terms)
        pos = np.full(self.n_coef, -1)
        pos[self.e_idx] = np.arange(self.e_idx.shape[0])
        for j, (term, Z, sl) in enumerate(zip(self.terms, self.Z, self.slices)):
            coef = coefs[j]
            if term.penalized and coef is not None:
                gamma[sl] = Z.T @ coef if Z is not None else coef
            elif not term.penalized:
                cols = self.X_E[:, pos[sl]]
                gamma[sl] = np.linalg.lstsq(cols, state.fits[:, j], rcond=None)[0]
        return gamma


class _InterceptOnly(object):
    """(The design of no terms: the intercept's column.)"""

    def __init__(self, n):
        self.X, self.Z, self.means, self.penalties, self.slices = np.ones((n, 1)), [], [], [], []
        self.penalty_sets, self._roots = [], []


def large_random_effect(terms, max_dense_coef):
    """
    The index of the random effect to keep implicit (the one with the most
    levels, `random_effect()` without a precision), when the coefficients are more than
    `max_dense_coef` and the others are at most `max_dense_coef`; else None.
    """
    from .terms import RandomEffectSmoother
    sizes = [term.n_coef() for term in terms]
    if 1 + sum(sizes) <= max_dense_coef:
        return None
    candidates = [j for j, term in enumerate(terms)
                  if isinstance(term, RandomEffectSmoother)]
    if not candidates:
        return None
    big = max(candidates, key=lambda j: sizes[j])
    if 1 + sum(sizes) - sizes[big] > max_dense_coef:
        return None
    return big
