import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from pyEOF import *

def visualization(da, pcs, eofs_da, evf, n):
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

def eof_an(df_clim_index, ds_n, n = 10, scale_type = 0, pca_type = "varimax"):

    '''
    EOF-анализ
    df_clim_index - переменная со значениями климатических индексов из r_netCDF()
    n - количество EOF
    scale_type - параметр, отвечающий за масштабирование при расчете EOF
    '''

    pca = df_eof(df_clim_index,pca_type=pca_type,n_components=n)
    eofs = pca.eofs(s=scale_type, n=n)
    eofs_da = eofs.stack(["latitude","longitude"]).to_xarray()
    pcs = pca.pcs(s=scale_type, n=n)
    evfs = pca.evf(n=n)
    eigvals = pca.eigvals(n=n)

    # plot
    visualization(ds_n, pcs, eofs_da, evfs, n)

    return (pca, eofs, pcs, evfs, eigvals)

def regr_graph_check(year, 
                     base_year = 1901, 
                     df_data = None, 
                     ds_n = None, 
                     pca = None,
                     pcs = None, 
                     eofs = None, 
                     eigvals = None, 
                     type_f = 'Both',
                     scale_type = 2):
  '''
  Функция сравнения реальный и предсказанных значений PDSI для регрессии:
  year - индекс года в наборе данных, 
  base_year - год отсчета (нужен для перевод индекса в реальный год), 
  df_data - исходные данные по PDSI, 
  ds_n - трехмерный массив, используется для извлечения параметров исходных данных, 
  pca - объект, полученный в результате eof_an,
  pcs - предсказанные моделью значения компонент, 
  eofs - набор значений функций EOF, 
  eigvals - собственные числа EOF,
  type_f - формат вывода:
    'Real' - выводить рисунок только по исходным данным,
    'Predicted' - выводить рисунок только по предсказанным данным,
    'Both' - выводить оба варианта
  scale_type - параметр отвечающий за масштабирование главных компонент и 
                EOF через умножение/деление значений на собственные числа
  '''

  if type_f in ('Both', 'Real'):
    if type(df_data).__module__ != np.__name__:
      u = df_data.to_numpy()[year]
    else:
      u = df_data[year]
     
    
    new = np.reshape(u, (-1, ds_n.shape[2]))
    im = plt.imshow(new[::-1], interpolation='none')

    cbar = plt.colorbar(im, ticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4], 
                        orientation='vertical',
                        fraction=0.045, pad=0.05)
    plt.title('Реальные PDSI, год: ' + str(year + base_year))
    plt.axis('off')
    plt.tight_layout()
    plt.show()



  if type_f in ('Both', 'Predicted'):

    if scale_type == 2:
      eofs = eofs[0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])[:, np.newaxis]
      if type(pcs).__module__ != np.__name__:
        pcs = pcs.to_numpy()[:, 0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])
      else:
        pcs = pcs[:, 0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])

    if scale_type == 1:
      eofs = eofs[0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])[:, np.newaxis]
      if type(pcs).__module__ != np.__name__:
        pcs = pcs.to_numpy()[:, 0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])
      else:
        pcs = pcs[:, 0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])


    Yhat = np.dot(pcs, eofs.to_numpy())
    Yhat = pca._scaler.inverse_transform(Yhat)
    u = Yhat[year]
    

    new = np.reshape(u, (-1, ds_n.shape[2]))
    im = plt.imshow(new[::-1], interpolation='none')

    cbar = plt.colorbar(im, ticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4], 
                        orientation='vertical',
                        fraction=0.045, pad=0.05)
    plt.title('Предсказанные PDSI, год: ' + str(year + base_year))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


  if type_f in ('Both'):

    Yhat = np.dot(pcs, eofs.to_numpy())

    if type(df_data).__module__ != np.__name__:
      u = df_data.to_numpy()[year]
    else:
      u = df_data[year]

    u = u - Yhat[year]

    new = np.reshape(u, (-1, ds_n.shape[2]))
    im = plt.imshow(new[::-1], interpolation='none')

    cbar = plt.colorbar(im, ticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4], 
                        orientation='vertical',
                        fraction=0.045, pad=0.05)
    plt.title('Разница PDSI, год: ' + str(year + base_year))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    
def class_graph_check(year, base_year = 1901, df_data = None, ds_n = None, pcs = None, eofs = None, type_f = 'Both'):
  '''
  Функция сравнения реальный и предсказанных значений PDSI для классификации:
  year - индекс года в наборе данных, 
  base_year - год отсчета (нужен для перевод индекса в реальный год), 
  df_data - исходные данные по PDSI, 
  ds_n = трехмерный массив, используется для извлечения параметров исходных данных, 
  pcs - массив с вероятностями принадлежности года к тому или иному классу, 
  eofs - набор значений функций EOF, 
  type_f - формат вывода:
    'Real' - выводить рисунок только по исходным данным,
    'Predicted' - выводить рисунок только по предсказанным данным,
    'Both' - выводить оба варианта
  '''

  if type_f in ('Both', 'Real'):
    u = df_data.to_numpy()[year]
    new = np.reshape(u, (-1, ds_n.shape[2]))
    im = plt.imshow(new[::-1], interpolation='none')

    cbar = plt.colorbar(im, ticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4], 
                        orientation='vertical',
                        fraction=0.045, pad=0.05)
    plt.title('Реальные PDSI, год: ' + str(year + base_year))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

  if type_f in ('Both', 'Predicted'):
    eof_in = pcs[year].argmax()
    eofs_da = eofs.stack(["latitude","longitude"]).to_xarray()
    eof_i = eofs_da.sel(EOF=1)["scpdsi"]
    fig = plt.figure()
    ax = fig.add_subplot()
    eof_i.plot(ax=ax,vmin=-0.75, vmax=0.75, cmap="RdBu_r",cbar_kwargs={'label': ""})
    ax.set_title('Класс (EOF=' + str(eof_in) + '), год: ' + str(base_year+year))
    
def diff(row, name_c):
  #Расчет полей с разницей метрик модели
  if name_c == 'test_loss-loss':
    res = row['test_loss'] - row['loss']
    return res

  if name_c == 'test_loss-val_loss':
    res = row['test_loss'] - row['val_loss']
    return res

  if name_c == 'test_accuracy-accuracy':
    res = row['test_accuracy'] - row['accuracy']
    return res

  if name_c == 'test_accuracy-val_accuracy':
    res = row['test_accuracy'] - row['val_accuracy']
    return res
