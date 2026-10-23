"""
Plots of the fitted terms of a GAM, as R's `plot.Gam` (and mgcv's
`plot.gam`): each term's centered fit against its column, with pointwise
standard error bands from the Bayesian covariance when it is available (all
terms but loess), and a rug of the training data. A term of two columns (a
surface: `gaussian_process(u, v)`, `tensor_spline(u, v)`) is drawn as filled
contours over the data's bounding box, as mgcv's `plot.gam` (`too.far`).
"""
import numpy as np
from scipy.spatial import cKDTree

from .terms import FactorTerm, ColumnsTerm, RandomEffectTerm, PenalizedColumnsTerm
from .tensor import TensorTerm
from .gp import GPTerm
from .varying import VaryingTerm


def _term_index(gam, term):
    if isinstance(term, (int, np.integer)):
        return int(term)
    names = [t.name for t in gam.terms_]
    if term not in names:
        raise ValueError(f'{term!r} is not a term of the GAM: {names}')
    return names.index(term)


def _has_covariance(gam):
    return gam.posterior_covariance_ is not None


def plottable(term):
    """
    Can `plot_term` draw the term (all but blocks of several columns, and
    functions of more than two: Gaussian processes, tensor products)?
    """
    if isinstance(term, (ColumnsTerm, PenalizedColumnsTerm)):
        return term.X_.shape[1] == 1
    if isinstance(term, GPTerm):
        return term.dim_ <= 2
    if isinstance(term, TensorTerm):
        return term.X_.shape[1] == 2
    if isinstance(term, VaryingTerm):
        # (a curve per level, or the varying coefficient, of one column)
        return len(term._inner_columns) == 1
    return True


def _surface(term):
    """Is the term a function of two columns (drawn as a surface)?"""
    return ((isinstance(term, GPTerm) and term.dim_ == 2) or
            (isinstance(term, TensorTerm) and term.X_.shape[1] == 2))


def _surface_data(term):
    """The training rows of the columns of a surface's term."""
    if isinstance(term, GPTerm):
        return term._X + term.shift_
    return term.X_


def _plot_surface(gam, j, ax, n_grid, too_far, rug, cmap):
    """Filled contours of a term of two columns over the data's bounding box."""
    term = gam.terms_[j]
    X = _surface_data(term)
    lo, hi = X.min(0), X.max(0)
    g1, g2 = np.linspace(lo[0], hi[0], n_grid), np.linspace(lo[1], hi[1], n_grid)
    U, V = np.meshgrid(g1, g2)
    grid = np.column_stack([U.ravel(), V.ravel()])
    fit = gam.partial_effect(j, grid)
    if too_far is not None and too_far > 0:
        # (as mgcv's exclude.too.far: distances in the unit square)
        scale = np.where(hi > lo, hi - lo, 1.)
        dist = cKDTree((X - lo) / scale).query((grid - lo) / scale)[0]
        fit = np.where(dist > too_far, np.nan, fit)
    Z = fit.reshape(U.shape)
    cs = ax.contourf(U, V, Z, levels=15, cmap=cmap)
    ax.contour(U, V, Z, levels=cs.levels, colors='k', linewidths=0.4, alpha=0.5)
    ax.figure.colorbar(cs, ax=ax, label=term.name)
    if rug:
        ax.plot(X[:, 0], X[:, 1], '.', color='k', markersize=2, alpha=0.4)
    names = term.columns
    ax.set_xlabel(str(names[0]) if isinstance(names, list) else 'x0')
    ax.set_ylabel(str(names[1]) if isinstance(names, list) else 'x1')
    ax.set_title(term.name)
    return ax


def plot_term(gam,
              term,
              ax=None,
              se=True,
              rug=True,
              n_grid=100,
              multiplier=2.,
              color='C0',
              se_color='C3',
              too_far=0.1,
              cmap='viridis'):
    """
    Plot the fit of a term of a fitted GAM against its column, as R's
    `plot.Gam(fit, se=TRUE)`.

    A term of a numeric column (a spline, loess, a linear term) is drawn as
    its curve over the range of the column (`n_grid` points); a factor or a
    random effect as one point per level (per group; for `re(g, by=x)` the
    groups' slopes). The fits are centered, as the GAM's terms. A term of
    two columns (`gaussian_process(u, v)`, `tensor_spline(u, v)`) is drawn
    as filled contours on an `n_grid` by `n_grid` grid over the data's
    bounding box, without bands, the data marked as points (`rug`).

    Parameters
    ----------
    gam: GAM
        A fitted GAM.
    term: int or str
        The term: its index or its name (e.g. `'s(age, df=5)'`).
    ax: matplotlib Axes, optional
        Where to draw (by default a new figure's).
    se: bool
        Draw the pointwise bands `fit +/- multiplier * std` (dashed, as R's)
        from the posterior covariance, when it is available (every term has a
        basis: not with loess terms).
    rug: bool
        Mark the training values of the column along the x axis.
    n_grid: int
        Number of points of a curve.
    multiplier: float
        Of the standard errors (2, as R's `plot.Gam`).
    color, se_color:
        Colors of the fit and of the bands.
    too_far: float
        For a surface: grid points farther than this from the data (in the
        bounding box scaled to the unit square) are not drawn, as mgcv's
        `plot.gam(too.far=0.1)`; None draws all.
    cmap:
        The colormap of a surface.

    Returns
    -------
    ax: matplotlib Axes
    """
    import matplotlib.pyplot as plt     # (optional: only for plots)

    j = _term_index(gam, term)
    term = gam.terms_[j]
    if not plottable(term):
        if isinstance(term, (GPTerm, TensorTerm)):
            raise ValueError(f'{term.name} is a function of more than two columns: '
                             'it has no curve or surface to plot')
        if isinstance(term, VaryingTerm):
            raise ValueError(f'{term.name} varies a function of several columns: '
                             'plot it with partial_effect')
        raise ValueError(f'{term.name} is a block of several columns: it has no curve to plot')
    if ax is None:
        _, ax = plt.subplots()
    if _surface(term):
        return _plot_surface(gam, j, ax, n_grid, too_far, rug, cmap)
    se = se and _has_covariance(gam)
    column = term.columns if not isinstance(term, ColumnsTerm) else term.column[0]

    if isinstance(term, VaryingTerm) and term.is_factor_:
        # a curve per level of the factor, over that level's range
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        values_all = np.asarray(term.xi_, float)
        for l, level in enumerate(term.levels_):
            values = values_all[term.codes_ == l]
            grid = np.linspace(values.min(), values.max(), n_grid)
            x = np.empty((n_grid, 2), dtype=object)
            x[:, 0], x[:, 1] = grid, level
            c = colors[l % len(colors)]
            if se:
                fit, std = gam.partial_effect(j, x, return_std=True)
                ax.fill_between(grid, fit - multiplier * std, fit + multiplier * std,
                                color=c, alpha=0.2, lw=0)
            else:
                fit = gam.partial_effect(j, x)
            ax.plot(grid, fit, color=c, lw=2, label=str(level))
        ax.legend(title=str(term.by), fontsize='small')
        if rug:
            ax.plot(np.unique(values_all), np.zeros(len(np.unique(values_all))), '|', color='k',
                    alpha=0.4, markersize=8, transform=ax.get_xaxis_transform())
        xlabel = term._inner_columns[0]
    elif isinstance(term, VaryingTerm):
        # a numeric `by`: the varying coefficient f(x), the term at z = 1
        values = np.asarray(term.xi_, float)
        grid = np.linspace(values.min(), values.max(), n_grid)
        x = np.column_stack([grid, np.ones(n_grid)])
        if se:
            fit, std = gam.partial_effect(j, x, return_std=True)
        else:
            fit, std = gam.partial_effect(j, x), None
        ax.plot(grid, fit, color=color, lw=2)
        if std is not None:
            for sign in (-1, 1):
                ax.plot(grid, fit + sign * multiplier * std, color=se_color, ls='--', lw=1.5)
        if rug:
            ax.plot(np.unique(values), np.zeros(len(np.unique(values))), '|', color='k',
                    alpha=0.4, markersize=8, transform=ax.get_xaxis_transform())
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        xlabel = term._inner_columns[0]
        ax.set_xlabel(str(xlabel))
        ax.set_ylabel(f'coefficient of {term.by}: {term.name}')
        return ax
    elif isinstance(term, (FactorTerm, RandomEffectTerm)):
        levels = np.asarray(term.levels_)
        if isinstance(term, RandomEffectTerm) and term.by is not None:
            x = np.column_stack([levels, np.ones(len(levels))])
        else:
            x = levels
        if se:
            fit, std = gam.partial_effect(j, x, return_std=True)
        else:
            fit, std = gam.partial_effect(j, x), None
        labels = levels
        if isinstance(term, RandomEffectTerm) and term.level_names is not None:
            labels = np.asarray(term.level_names)[levels.astype(int)]
        pos = np.arange(len(levels))
        if std is not None:
            ax.errorbar(pos, fit, yerr=multiplier * std, fmt='o', color=color,
                        ecolor=se_color, capsize=3)
        else:
            ax.plot(pos, fit, 'o', color=color)
        ax.set_xticks(pos)
        labels = [str(l) for l in labels]
        # (slanted, when they would overlap)
        crowded = len(pos) > 6 or sum(len(l) for l in labels) > 30
        ax.set_xticklabels(labels, rotation=30 if crowded else 0,
                           ha='right' if crowded else 'center')
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        xlabel = term.column
    else:
        if isinstance(term, ColumnsTerm):
            values = term.X_[:, 0]
        else:
            values = term.x_
        grid = np.linspace(values.min(), values.max(), n_grid)
        x = grid[:, None] if isinstance(term, ColumnsTerm) else grid
        if se:
            fit, std = gam.partial_effect(j, x, return_std=True)
        else:
            fit, std = gam.partial_effect(j, x), None
        ax.plot(grid, fit, color=color, lw=2)
        if std is not None:
            for sign in (-1, 1):
                ax.plot(grid, fit + sign * multiplier * std, color=se_color, ls='--', lw=1.5)
        if rug:
            ax.plot(np.unique(values), np.zeros(len(np.unique(values))), '|', color='k',
                    alpha=0.4, markersize=8, transform=ax.get_xaxis_transform())
        xlabel = column
    ax.set_xlabel(str(xlabel))
    ax.set_ylabel(term.name)
    return ax


def plot_terms(gam,
               terms=None,
               axes=None,
               ncols=3,
               figsize=None,
               **kwargs):
    """
    Plot the fits of the terms of a fitted GAM, one panel each (by default
    every term that has a curve, `plottable`), as R's `plot.Gam`. The other
    arguments are those of `plot_term`.

    Returns
    -------
    axes: np.ndarray of matplotlib Axes, one per plotted term
    """
    import matplotlib.pyplot as plt     # (optional: only for plots)

    if terms is None:
        terms = [j for j, t in enumerate(gam.terms_) if plottable(t)]
    else:
        terms = [_term_index(gam, t) for t in terms]
    if axes is None:
        ncols = min(ncols, len(terms))
        nrows = int(np.ceil(len(terms) / ncols))
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=figsize or (4.5 * ncols, 3.5 * nrows))
        axes = axes.ravel()
        for ax in axes[len(terms):]:
            ax.set_visible(False)
        axes = axes[:len(terms)]
    else:
        axes = np.asarray(axes).ravel()
        if axes.shape[0] < len(terms):
            raise ValueError(f'{len(terms)} terms to plot, {axes.shape[0]} axes given')
    for ax, j in zip(axes, terms):
        plot_term(gam, j, ax=ax, **kwargs)
    return axes
