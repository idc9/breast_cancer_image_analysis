from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import matplotlib.pyplot as plt


def jitter_hist(x, jit_how='order',
                hist_kws={'color': 'black',
                          'alpha': .5},
                scatter_kws={'color': 'black', 's': 1}):
    """
    Jitter plot histogram

    Parameters
    ----------

    jit_how: str, None
        How to jitter (see iain_bcia.viz.jitter.jitter_yvals).
        Must be one of ['cdf', jitter_cdf, 'order', 'random'].

    hist_kws: dict
        Key word arguments for plt.hist

    scatter_kws: dict
        Key word arguments for plt.scatter

    scatter_kws
    """

    # update default parameters
    hist_kws_ = {'color': 'black',
                 'alpha': .5}
    for k in hist_kws.keys():
        hist_kws_[k] = hist_kws[k]

    scatter_kws_ = {'color': 'black', 's': 1}
    for k in scatter_kws.keys():
        scatter_kws_[k] = scatter_kws[k]

    # make histogram
    n, bins, patches = plt.hist(x, zorder=0, **hist_kws_)

    # maybe add jittering
    if jit_how is not None:
        # y = np.random.uniform(low=.05 * max(n), high=.1 * max(n), size=len(x))
        y = jitter_yvals(x, how=jit_how, yprops=(.2, .5), ylim=None,
                         jitter_cdf_width=0.05)

        plt.scatter(x, y, zorder=1, **scatter_kws_)


def jitter_yvals(values, how='order', yprops=(.2, .5), ylim=None,
                 jitter_cdf_width=0.05):
    """
    Computes the y-values for a jitter plot.

    Parameters
    ----------
    values: array-like
        The values to be jittered

    how: str ['cdf', jitter_cdf, 'order', 'random']
        How to jitter.
        If cdf, will return cdf values.
        If jitter_cdf will return jittered cdf values (prevents overplotting).
        If random, will do random jittering.
        If order, will plot height proportional to the order in the dataset.


    yprops: (float, float)
        How to pick the lower and upper y limits for the jittering points.


    ylim: (float, flaot)
        Explicit values for lower/upper limits.

    """
    values = np.array(values).reshape(-1)

    if ylim is None:
        _, ymax = plt.gca().get_ylim()

        ymin = ymax * yprops[0]
        ymax = ymax * yprops[1]

    else:
        ymin, ymax = ylim

    if how == 'cdf':
        # compute empirical CDF
        yvals = ECDF(values)(values)
        yvals = yvals * (ymax - ymin) + ymin  # rescale

        return yvals

    elif how == 'jitter_cdf':

        yvals = ECDF(values)(values)

        yvals += np.random.uniform(low=-jitter_cdf_width, high=jitter_cdf_width, size=len(yvals))

        yvals += np.min(yvals)
        yvals /= np.max(yvals)

        yvals = yvals * (ymax - ymin) + ymin

        return yvals

    elif how == 'order':
        yvals = np.linspace(start=0, stop=1, num=len(values))
        yvals = yvals * (ymax - ymin) + ymin
        return yvals

    elif how == 'random':
        return np.random.uniform(low=ymin, high=ymax, size=len(values))

    else:
        raise ValueError("how must be one of ['cdf', 'random'], not {}".format(how))
