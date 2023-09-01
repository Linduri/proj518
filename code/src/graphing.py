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


def PlotVehicleLocations(D, F, title=None):
    # Plot locations to graph.
    fig, ax = plt.subplots()

    if title is not None:
        fig.suptitle(title,
                     fontsize=16)

    # Plot vehicles.
    ax.scatter(x=D['latitude'],
               y=D['longitude'])

    # Group vehicle IDs
    L = D.groupby('loc',
                  as_index=False,
                  group_keys=False)

    for _, l in L:
        vp_tags = []
        # Group by vehicle
        V = l.groupby('vehicle',
                      as_index=False,
                      group_keys=False)

        for _, v in V:
            i_v = v.iloc[0].vehicle
            P_v = ", ".join(v.procedure.astype(str))
            vp_tags.append(f"{int(i_v)} ({P_v})")

        # names = l['vehicle'].unique()
        # names = [str(name) for name in names]
        txt = ', '.join(vp_tags)
        ax.annotate(txt,
                    (l['latitude'].iloc[0],
                     l['longitude'].iloc[0]),
                    xytext=(5, -10),
                    textcoords='offset points')

    # Plot facilities.
    ax.scatter(x=F['latitude'],
               y=F['longitude'])

    for _, name, lat, lon in F.itertuples():
        ax.annotate(str(name).capitalize(),
                    (lat,
                    lon),
                    xytext=(5, 5),
                    textcoords='offset points')

    plt.show()
