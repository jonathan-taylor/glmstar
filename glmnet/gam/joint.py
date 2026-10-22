"""
The joint (mgcv-style) solve of the working problems of local scoring: the
terms' bases side by side in one design, and the penalized weighted least
squares fit of the working response on it. It has the same solution as
backfitting the terms (for terms with a basis and a quadratic penalty), so
`GAMControl(solver='joint')` fits the same GAM, `df` terms included (their
smoothing parameters are matched at each working problem's weights, as R's
`gam`). The design also gives the Bayesian covariance of the fit.

References
----------
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
- Wahba, G. (1983). Bayesian "confidence intervals" for the cross-validated
  smoothing spline. *Journal of the Royal Statistical Society, Series B*
  45(1), 133–150.
- Nychka, D. (1988). Bayesian confidence intervals for smoothing splines.
  *Journal of the American Statistical Association* 83(404), 1134–1143.
- Marra, G. and Wood, S. N. (2012). Coverage properties of confidence
  intervals for generalized additive model components. *Scandinavian
  Journal of Statistics* 39(1), 53–74.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .backfit import GAMRegularizer, GAMState


class JointDesign(object):
    """
    The design of the terms (`glmnet.gam.terms`, set up on the training data):
    the column of ones, then a block of columns per term. A term's block is
    its basis `B`:

    - with the sum-to-zero constraint over the training rows absorbed (as
      mgcv: `B Z`, `Z` a basis of the coefficients `c` with `1' B c = 0`)
      when the constant is among the functions it fits (a spline, a factor's
      indicators);
    - centered (`B - mean(B)`) when it is not (a linear term, a block of
      columns);
    - as is for a term that is not centered (a random effect).

    The fits of the centered terms have mean 0 over the training rows, as
    those of backfitting. Their penalties are `Z' S Z` (or `S`).
    """

    def __init__(self, terms):
        self.terms = terms
        self.Z, self.means, self.penalties, self.slices = [], [], [], []
        # the penalties of each term (several for some), and their sum
        self.penalty_sets = []
        blocks, start = [], 1
        for term in terms:
            B = term.basis()
            if B is None:
                raise ValueError(f'{term.name} has no basis: it can only be backfitted '
                                 '(GAMControl(solver="backfit"))')
            B = np.asarray(B, float)
            n = B.shape[0]
            Ss = term.penalty_matrices()
            Z = mean = None
            if term.centered:
                ones = np.ones(n)
                c = np.linalg.lstsq(B, ones, rcond=None)[0]
                if np.linalg.norm(B @ c - ones) < 1e-8 * np.sqrt(n):
                    # the constant is in the span: the constraint 1' B c = 0
                    Q = np.linalg.qr(B.sum(0)[:, None], mode='complete')[0]
                    Z = Q[:, 1:]
                else:
                    mean = B.mean(0)
            block = B @ Z if Z is not None else (B - mean if mean is not None else B)
            if Z is not None:
                Ss = [Z.T @ S @ Z for S in Ss]
            self.Z.append(Z)
            self.means.append(mean)
            self.penalty_sets.append(Ss)
            self.penalties.append(sum(Ss) if Ss else None)
            self.slices.append(slice(start, start + block.shape[1]))
            start += block.shape[1]
            blocks.append(block)
        self.X = np.hstack([np.ones((blocks[0].shape[0], 1))] + blocks)
        # the square roots of the single penalties: S = R' R
        self._roots = [_root(Ss[0]) if len(Ss) == 1 else None for Ss in self.penalty_sets]

    @property
    def n_coef(self):
        return self.X.shape[1]

    def block(self, j, x):
        """The block of the term `j` at its values `x` (new data)."""
        B = np.asarray(self.terms[j].basis(x), float)
        if self.Z[j] is not None:
            return B @ self.Z[j]
        if self.means[j] is not None:
            return B - self.means[j]
        return B

    def rows(self, xs):
        """The design at new data: `xs` the terms' columns."""
        blocks = [self.block(j, x) for j, x in enumerate(xs)]
        return np.hstack([np.ones((blocks[0].shape[0], 1))] + blocks)

    def _penalty_sets(self):
        # a term whose penalty matrix changes (`penalty_varies`: a precision
        # matrix estimated by REML) gives it at each use (it is not centered)
        return [term.penalty_matrices() if getattr(term, 'penalty_varies', False) else Ss
                for term, Ss in zip(self.terms, self.penalty_sets)]

    def penalty(self, lams):
        """
        The penalty matrix, `sum_j sum_m lams[j][m] S_jm` (on the coefficients
        of the design): `lams[j]` the smoothing parameters of the term `j`'s
        penalties (a number for a term with one).
        """
        P = np.zeros((self.n_coef, self.n_coef))
        for sl, Ss, lam in zip(self.slices, self._penalty_sets(), lams):
            for S, l in zip(Ss, np.atleast_1d(lam)):
                P[sl, sl] += l * S
        return P

    def root(self, lams):
        """A square root `E` of the penalty matrix, `E' E = penalty(lams)`."""
        rows = []
        for term, sl, Ss, R, lam in zip(self.terms, self.slices, self._penalty_sets(),
                                        self._roots, lams):
            lam = np.atleast_1d(lam)
            if not Ss or not np.any(lam > 0):
                continue
            if R is not None and not getattr(term, 'penalty_varies', False):
                R = np.sqrt(lam[0]) * R
            else:
                R = _root(sum(l * S for l, S in zip(lam, Ss)))
            E = np.zeros((R.shape[0], self.n_coef))
            E[:, sl] = R
            rows.append(E)
        return np.vstack(rows) if rows else np.zeros((0, self.n_coef))

    def solve(self, z, w, lams):
        """
        The coefficients minimizing `sum(w (z - X gamma)^2) / 2 +
        gamma' penalty(lams) gamma / 2`. By the Cholesky factorization of
        `X' W X + penalty(lams)` when it is well conditioned (the normal
        equations square the condition number), else by the QR decomposition
        of the augmented design `[sqrt(W) X; E]` (`E' E = penalty(lams)`),
        and if that is rank deficient, by its least squares solution of
        minimum norm.
        """
        from scipy.linalg import cho_factor, cho_solve, solve_triangular, LinAlgError
        from scipy.linalg.lapack import dpocon
        Xw = w[:, None] * self.X
        A = self.X.T @ Xw + self.penalty(lams)
        d = np.sqrt(np.diag(A))
        if np.all(d > 0):
            A_scaled = A / d[:, None] / d[None, :]
            try:
                factor = cho_factor(A_scaled, lower=False)
            except LinAlgError:
                pass
            else:
                rcond, _ = dpocon(factor[0], np.abs(A_scaled).sum(0).max())
                if rcond > 1e-10:
                    return cho_solve(factor, Xw.T @ z / d) / d
        sq = np.sqrt(w)
        E = self.root(lams)
        A = np.vstack([sq[:, None] * self.X, E])
        b = np.concatenate([sq * z, np.zeros(E.shape[0])])
        Q, R = np.linalg.qr(A)
        r = np.abs(np.diag(R))
        if r.min() > 1e-10 * r.max():
            return solve_triangular(R, Q.T @ b)
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def gamma(self, state):
        """
        The coefficients of the design of a fitted state: its intercept, the
        penalized terms' coefficients (`state.coefs`, on their bases) and,
        for the others, the least squares fit of their block to their fit.
        The design may be rank deficient (a random effect's indicators span
        the constant), so the coefficients are not recovered from the linear
        predictor.
        """
        gamma = np.zeros(self.n_coef)
        gamma[0] = state.intercept
        coefs = state.coefs if state.coefs is not None else [None] * len(self.terms)
        for j, (term, Z, sl) in enumerate(zip(self.terms, self.Z, self.slices)):
            coef = coefs[j]
            if term.penalty_matrix() is not None and coef is not None:
                gamma[sl] = Z.T @ coef if Z is not None else coef
            elif term.penalty_matrix() is None:
                gamma[sl] = np.linalg.lstsq(self.X[:, sl], state.fits[:, j], rcond=None)[0]
        return gamma

    def split(self, gamma):
        """
        The fits of the terms (at the training rows), the intercept and the
        terms' coefficients (on their bases, as `Term.coef()`; None for terms
        whose fit is enough to finalize them) from the coefficients `gamma`.
        """
        fits = np.column_stack([self.X[:, sl] @ gamma[sl] for sl in self.slices])
        coefs = []
        for term, Z, sl in zip(self.terms, self.Z, self.slices):
            if term.penalty_matrix() is None:
                coefs.append(None)
            else:
                coefs.append(Z @ gamma[sl] if Z is not None else gamma[sl].copy())
        return fits, float(gamma[0]), coefs


def _root(S):
    """A square root `R` of the positive semi-definite `S`, `R' R = S`."""
    ev, V = np.linalg.eigh((S + S.T) / 2)
    return np.sqrt(np.maximum(ev, 0))[:, None] * V.T


def _solve_psd(A, b):
    """
    The solution of `A x = b`, `A` positive semi-definite: equilibrated (its
    diagonal scaled to 1), by Cholesky when it is well conditioned, else on
    its eigenvectors with eigenvalues above 1e-13 of the largest (minimum
    norm).
    """
    from scipy.linalg import cho_factor, cho_solve, LinAlgError
    from scipy.linalg.lapack import dpocon
    d = np.sqrt(np.maximum(np.diag(A), 0))
    d = np.where(d > 0, d, 1.)
    As = A / d[:, None] / d[None, :]
    bs = b / d
    try:
        factor = cho_factor(As, lower=False)
    except LinAlgError:
        pass
    else:
        rcond, _ = dpocon(factor[0], np.abs(As).sum(0).max())
        if rcond > 1e-12:
            return cho_solve(factor, bs) / d
    ev, V = np.linalg.eigh((As + As.T) / 2)
    keep = ev > ev.max() * 1e-13
    return (V[:, keep] @ ((V[:, keep].T @ bs) / ev[keep])) / d


@dataclass
class JointRegularizer(GAMRegularizer):
    """
    The regularizer of local scoring whose Newton step solves the working
    problem jointly: the penalized weighted least squares fit of the working
    response on the terms' bases (`JointDesign`), instead of backfitting.
    Its objective, step halving and state are those of `GAMRegularizer`.

    For a family whose Hessian in the linear predictor is not diagonal (Cox),
    `information(state)` gives the score and the Hessian (an operator) of the
    log-likelihood in the linear predictor at `state`, on the scale of the
    objective, and the step is Newton's for the penalized objective, with
    the full Hessian and no intercept (the smoothing parameters of the `df`
    terms are those at the diagonal working weights).
    """
    design_: Optional[JointDesign] = None
    information: Optional[object] = None

    def lams(self):
        """The terms' smoothing parameters at the current weights."""
        return [term.penalty_lams() for term in self.terms]

    def newton_step(self,
                    design,                 # ignored
                    pseudo_response,
                    sample_weight,
                    cur_state):
        z, w = self._working(pseudo_response, sample_weight)
        if self.design_ is None:
            self.design_ = JointDesign(self.terms)
        if self.information is None:
            gamma = self.design_.solve(z, w, self.lams())
        else:
            gamma = self._full_newton(cur_state)
        fits, intercept, coefs = self.design_.split(gamma)
        self.n_sweeps_ = 1
        self.warm_state = GAMState(fits, intercept, coefs=coefs)
        self.warm_state.gamma = gamma
        return self.warm_state

    def _full_newton(self, state):
        """
        The Newton step from `state` with the full Hessian `H`: `gamma_0 +
        delta`, `(X' H X + S) delta = X' score - S gamma_0` (without the
        column of ones), `gamma_0` the coefficients of `state`. (Solved for
        the step, whose error vanishes with it, not for `gamma`.)
        """
        D = self.design_
        score, H = self.information(state)
        if self.linear_term is not None:
            score = score - np.asarray(self.linear_term, float)
        X = D.X[:, 1:]
        gamma0 = getattr(state, 'gamma', None)
        if gamma0 is None:
            gamma0 = D.gamma(state)
        S = D.penalty(self.lams())[1:, 1:]
        A = X.T @ (H @ X) + S
        b = X.T @ score - S @ gamma0[1:]
        return np.concatenate([[0.], gamma0[1:] + _solve_psd(A, b)])
