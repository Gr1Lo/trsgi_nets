import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from pyEOF import *

def visualization(da, pcs, eofs_da, evf):
    fig = plt.figure(figsize = (12,24))

    ax = fig.add_subplot(n+1,2,1)
    da.mean(dim=["latitude","longitude"]).plot(ax=ax)
    ax.set_title("average scpdsi")

    ax = fig.add_subplot(n+1,2,2)
    da.mean(dim="time").plot(ax=ax)
    ax.set_title("average scpdsi")

    for i in range(1,n+1):
        pc_i = pcs["PC"+str(i)].to_xarray()
        eof_i = eofs_da.sel(EOF=i)["scpdsi"]
        frac = str(np.array(evf[i-1]*100).round(2))

        ax = fig.add_subplot(n+1,2,i*2+1)
        pc_i.plot(ax=ax)
        ax.set_title("PC"+str(i)+" ("+frac+"%)")

        ax = fig.add_subplot(n+1,2,i*2+2)
        eof_i.plot(ax=ax,
                   vmin=-0.75, vmax=0.75, cmap="RdBu_r",
                   cbar_kwargs={'label': ""})
        ax.set_title("EOF"+str(i)+" ("+frac+"%)")

    plt.tight_layout()
    plt.show()

def find_n_for_kmeans(d_tab):

    '''
    Поиск оптимального значения классов для k-mean
    1) Методом "локтя"
    2) Методом "силуэтов"
    '''

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42}

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 15):
      kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
      kmeans.fit(d_tab)
      sse.append(kmeans.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 15), sse)
    plt.xticks(range(1, 15))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(1, 15), sse, curve="convex", direction="decreasing")
    print('Elbow: ' + str(kl.elbow))

    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(d_tab)
        score = silhouette_score(d_tab, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 15), silhouette_coefficients)
    plt.xticks(range(2, 15))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

def save_pickle(f_path, vari):
    with open(f_path, 'wb') as f:
        pickle.dump(vari, f)

def read_pickle(f_path):
    with open(f_path, 'rb') as f:
        df_test = pickle.load(f)
        return df_test

def eof_an(df_clim_index, n = 10):

    '''
    EOF-анализ
    df_clim_index - переменная со значениями климатических индексов из r_netCDF()
    n - количество EOF
    '''

    pca = df_eof(df_clim_index,pca_type="varimax",n_components=n)
    eofs = pca.eofs(s=2, n=n)
    eofs_da = eofs.stack(["latitude","longitude"]).to_xarray()
    pcs = pca.pcs(s=2, n=n)
    evfs = pca.evf(n=n)

    # plot
    visualization(ds_n, pcs, eofs_da, evfs)

    return (pca, eofs, pcs, evfs)