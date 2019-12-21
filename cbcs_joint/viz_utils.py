import matplotlib.pyplot as plt
import matplotlib as mpl


def savefig(fpath, dpi=100):
    """
    Save and close a figure.
    """
    plt.savefig(fpath, bbox_inches='tight', frameon=False, dpi=dpi)
    plt.close()


def mpl_noaxis(labels=False):
    """
    Do not display any axes for any figure.
    """

    mpl.rcParams['axes.linewidth'] = 0

    if not labels:
        mpl.rcParams['xtick.bottom'] = False
        mpl.rcParams['xtick.labelbottom'] = 0

        mpl.rcParams['ytick.left'] = False
        mpl.rcParams['ytick.labelleft'] = 0
