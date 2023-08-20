import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def PlotBayOps(D):
    D["vp"] = 'Vehicle: ' +\
        D.v.astype(str) +\
        ' Procedure: ' +\
        D.p.astype(str) +\
        ' Operation: ' +\
        D.o.astype(str)

    # Group adjacent vehicle procedures (b).
    B = D.groupby('b',
                  as_index=False,
                  group_keys=False)

    _D = D.copy()
    
    n_oc = len(_D.oc.unique()) + 1
    cmap = plt.get_cmap("gist_rainbow", n_oc)
    palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    
    _D['color'] = _D.oc.astype(int).map(pd.Series(palette))
    
    # for _, b in B:
    #     # Find the number of operation clusters per bay.
    #     n_oc = len(b.oc.unique()) + 1
    #     cmap = plt.get_cmap("gist_rainbow", n_oc)
    #     custom_palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    #     C = pd.Series(custom_palette)
    #     print(C)
    #     _D = D.copy()
    #     _D['color'] = _.oc.astype(int).map(C)

    #     print(_D)

    plt.barh(y=_D.vp,
             width=_D.d,
             left=_D.t_s,
             color=_D.color)
    plt.show()
