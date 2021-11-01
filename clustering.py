import utils
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import pandas as pd

def k_means(pcs, n_clust, text_c, use_norm=False):

    '''
    Кластеризация методом к-среднего
    pcs - главные компоненты, полученные через eof_an,
    n_clust - количество кластеров,
    text_c - текст подписи на выходе,
    use_norm - параметр, отвечающий за использование нормализвоанных значений
    '''

    if use_norm == True:
        data_scaled = normalize(pcs)
        data_s = pd.DataFrame(data_scaled, columns=pcs.columns)

    else:
        data_s = pcs

    utils.find_n_for_kmeans(data_s)
    kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42}

    kmeans = KMeans(n_clusters=n_clust, **kmeans_kwargs)
    kmeans.fit(data_s)
    print(text_c)
    print(kmeans.labels_)
    return kmeans


def aggl_clust(pcs, n_clust, text_c, use_norm=False):

    '''
    Кластеризация агломеративным методом
    pcs - главные компоненты, полученные через eof_an,
    n_clust - количество кластеров,
    text_c - текст подписи на выходе,
    use_norm - параметр, отвечающий за использование нормализвоанных значений
    '''

    if use_norm == True:
        data_scaled = normalize(pcs)
        data_s = pd.DataFrame(data_scaled, columns=pcs.columns)

    else:
        data_s = pcs

    cluster = AgglomerativeClustering(n_clusters=n_clust, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data_s)
    print(text_c)
    print(cluster.labels_)
    return cluster
