import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def PlotBayOps(D,
               color_col=None,
               verbose=False):
    D["vp"] = ' Vehicle: ' +\
        D.v.astype(str) +\
        ' Procedure: ' +\
        D.p.astype(str) +\
        ' Operation: ' +\
        D.o.astype(str)

    # Which column to use as the color gradient (g).
    g = 'o' if color_col is None else color_col

    _D = D.copy()

    uniques = _D[g].unique()
    M = dict(zip(uniques, range(len(uniques))))
    _D['i_g'] = _D[g].astype(int).map(M)

    n = len(_D[g].unique()) + 1
    cmap = plt.get_cmap("gist_rainbow", n)
    palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    _D['color'] = _D.i_g.astype(int).map(pd.Series(palette))

    n_bays = len(_D.b.unique())
    fig, axs = plt.subplots(n_bays,
                            figsize=(15, 15))

    x_lim = _D.t_e.max()

    B = _D.groupby('b',
                   as_index=False,
                   group_keys=False)

    for (i, b), ax in zip(B, axs):
        ax.barh(y=b.vp,
                width=b.d,
                left=b.t_s,
                color=b.color)

        ax.set_title(f"Bay {i}")
        ax.set_xlim([0, x_lim])

    plt.show()

    if verbose is True:
        print(_D)
