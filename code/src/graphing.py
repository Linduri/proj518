import matplotlib.pyplot as plt


def PlotBayOps(D): 
    D["vp"] = D.v.astype(str) + ' ' + D.p.astype(str) + ' ' + D.o.astype(str)
    print(D.vp)
    plt.barh(y=D.vp, width=D.d, left=D.t_s)
    plt.show()
