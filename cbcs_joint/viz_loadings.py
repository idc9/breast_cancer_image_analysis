import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_loading(v, abs_sorted=True, show_var_names=True,
                 significant_vars=None, show_top=None,
                 colors=None, vert=True):
    """
    Plots a single loadings component.

    Parameters
    ----------
    v: array-like
        The loadings component.

    abs_sorted: bool
        Whether or not to sort components by their absolute values.


    significant_vars: {array-like, None}
        Indicated which features are significant in this component.

    show_top: {None, array-like}
        Will only display this number of top loadings components when
        sorting by absolute value.

    colors: None, array-like
        Colors for each loading. If None, will use sign.

    vert: bool
        Make plot vertical or horizontal
    """
    if hasattr(v, 'name'):
        xlab = v.name
    else:
        xlab = ''

    if type(v) != pd.Series:
        v = pd.Series(v, index=['feature {}'.format(i) for i in range(len(v))])
        if significant_vars is not None:
            significant_vars = v.index.iloc[significant_vars]
    else:
        if colors is not None:
            colors = colors.loc[v.index]

    if abs_sorted:
        v_abs_sorted = np.abs(v).sort_values()
        v = v[v_abs_sorted.index]

        if show_top is not None:
            v = v[-show_top:]

            if significant_vars is not None:
                significant_vars = significant_vars[-show_top:]

    inds = np.arange(len(v))

    signs = v.copy()
    signs[v > 0] = 'pos'
    signs[v < 0] = 'neg'
    if significant_vars is not None:
        signs[v.index.difference(significant_vars)] = 'zero'
    else:
        signs[v == 0] = 'zero'
    s2c = {'pos': 'red', 'neg': 'blue', 'zero': 'grey'}

    if colors is None:
        colors = signs.apply(lambda x: s2c[x])

    if vert:
        plt.scatter(v, inds, color=colors)
        plt.axvline(x=0, alpha=.5, color='black')
        plt.xlabel(xlab)
        if show_var_names:
            plt.yticks(inds, v.index)
    else:
        v = v[::-1]
        colors = colors[::-1]
        plt.scatter(inds, v, color=colors)
        plt.axhline(y=0, alpha=.5, color='black')
        plt.ylabel(xlab)
        if show_var_names:
            plt.xticks(inds, v.index)

    max_abs = np.abs(v).max()
    xmin = -1.2 * max_abs
    xmax = 1.2 * max_abs
    if np.mean(signs == 'pos') == 1:
        xmin = 0
    elif np.mean(signs == 'neg') == 1:
        xmax = 0
    elif np.mean(signs == 'zero') == 1:
        xmin = 1
        xmax = 1

    if vert:
        plt.xlim(xmin, xmax)
    else:
        plt.ylim(xmin, xmax)

    if vert:
        ticklabs = plt.gca().get_yticklabels()
    else:
        ticklabs = plt.gca().get_xticklabels()

    for t, c in zip(ticklabs, colors):
        t.set_color(c)
        if c != 'grey':
            t.set_fontweight('bold')

    if not vert:
        plt.gca().xaxis.set_tick_params(rotation=55)
