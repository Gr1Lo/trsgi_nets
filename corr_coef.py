import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import kstest, normaltest, pearsonr, spearmanr

def corr_coef_year(year,
                     base_year = 1901, 
                     df_data = None, 
                     ds_n = None, 
                     pca = None,
                     pcs = None, 
                     eofs = None, 
                     eigvals = None, 
                     scale_type = 2,
                     type_ret = None):

    '''
    Функция подсчета коэффициента корреляции для реальных и предсказанных значений scPDSI для регрессии в рамках одного года:
    year - индекс года в наборе данных, 
    base_year - год отсчета (нужен для перевод индекса в реальный год), 
    df_data - исходные данные по PDSI, 
    ds_n - трехмерный массив, используется для извлечения параметров исходных данных, 
    pca - объект, полученный в результате eof_an,
    pcs - предсказанные моделью значения компонент, 
    eofs - набор значений функций EOF, 
    eigvals - собственные числа EOF,
    scale_type - параметр отвечающий за масштабирование главных компонент и 
                  EOF через умножение/деление значений на собственные числа,
    type_ret = 'Mute' - не выводить картинки и лишний текст
    '''

    if type(df_data).__module__ != np.__name__:
      u0 = df_data.to_numpy()[year]
    else:
      u0 = df_data[year]
      
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

    nas = np.logical_or(np.isnan(u), np.isnan(u0))
    normed_data0 = (u0[~nas] - u0[~nas].mean()) / u0[~nas].std() 
    normed_data = (u[~nas] - u[~nas].mean()) / u[~nas].std() 

    corr = pearsonr(u0[~nas],u[~nas])
    corr_s = spearmanr(u0[~nas],u[~nas])

    if type_ret != 'Mute':
      _ = plt.hist(u0, bins='auto')
      plt.title('Распределение значений scPDSI в исходных данных для (' + str(base_year+year)+ ') года')
      plt.show()
      D, pval = kstest(normed_data0, 'norm')
      print(normaltest(u0[~nas]))
      print('p-value теста Колмогорова-Симрнова для исходных данных = ' + str(pval) + '\n')
      

      _ = plt.hist(u, bins='auto')
      plt.title('Распределение значений scPDSI в предсказанных данных для (' + str(base_year+year) + ') года')
      plt.show()
      D, pval = kstest(normed_data, 'norm')
      print(normaltest(u[~nas]))
      print('p-value теста Колмогорова-Симрнова для предсказанных данных = ' + str(pval) + '\n')
      

      print('Коэффициент корреляции Пирсона = ' + str(corr[0]) + ', p-value = ' + str(corr[1]))
      print('Коэффициент корреляции Спирмана (для ненормального распределения) = ' + str(corr_s[0]) + ', p-value = ' + str(corr_s[1]))

    return corr, corr_s




def corr_coef_pixel(str_lat,
                    str_lon,
                     base_year = 1901, 
                     df_data = None, 
                     ds_n = None, 
                     pca = None,
                     pcs = None, 
                     eofs = None, 
                     eigvals = None,
                     scale_type = 2,
                     type_ret = None):
  
    '''
    Функция подсчета коэффициента корреляции для реальных и предсказанных значений scPDSI для регрессии по отдельно взятому пикселю за все годы:
    str_lat - географическая широта пикселя, 
    str_lon - географическая долгота пикселя, 
    base_year - год отсчета (нужен для перевод индекса в реальный год), 
    df_data - исходные данные по PDSI, 
    ds_n - трехмерный массив, используется для извлечения параметров исходных данных, 
    pca - объект, полученный в результате eof_an,
    pcs - предсказанные моделью значения компонент, 
    eofs - набор значений функций EOF, 
    eigvals - собственные числа EOF,
    scale_type - параметр отвечающий за масштабирование главных компонент и 
                  EOF через умножение/деление значений на собственные числа,
    type_ret = 'Mute' - не выводить картинки и лишний текст
    '''

    irr_lat = np.array(list(df_data))[:,1]
    irr_lon = np.array(list(df_data))[:,2]

    k_lat = [i for i in range(len(irr_lat)) if irr_lat[i] == str_lat]
    k_lon = [i for i in range(len(irr_lon)) if irr_lon[i] == str_lon]

    m_ind = list(set(k_lat).intersection(k_lon))[0]
    
    if type(df_data).__module__ != np.__name__:
      u0 = df_data.to_numpy()[:,m_ind]
    else:
      u0 = df_data[:,m_ind]
      
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
    u = Yhat[:,m_ind]
    u0 = u0[:len(u)]

    nas = np.logical_or(np.isnan(u), np.isnan(u0))
    normed_data0 = (u0[~nas] - u0[~nas].mean()) / u0[~nas].std() 
    normed_data = (u[~nas] - u[~nas].mean()) / u[~nas].std() 

    
    

    corr = pearsonr(u0[~nas],u[~nas])
    corr_s = spearmanr(u0[~nas],u[~nas])

    if type_ret != 'Mute':
      _ = plt.hist(u0, bins='auto')
      plt.title('Распределение значений scPDSI в исходных данных для (' + str_lat + ', ' + str_lon + ') пикселя')
      plt.show()
      D, pval = kstest(normed_data0, 'norm')
      print(normaltest(u0[~nas]))
      print('p-value теста Колмогорова-Симрнова для исходных данных = ' + str(pval) + '\n')
      

      _ = plt.hist(u, bins='auto')
      plt.title('Распределение значений scPDSI в предсказанных данных для (' + str_lat + ', ' + str_lon + ') пикселя')
      plt.show()
      D, pval = kstest(normed_data, 'norm')
      print(normaltest(u[~nas]))
      print('p-value теста Колмогорова-Симрнова для предсказанных данных = ' + str(pval) + '\n')
      

      print('Коэффициент корреляции Пирсона = ' + str(corr[0]) + ', p-value = ' + str(corr[1]))
      print('Коэффициент корреляции Спирмана (для ненормального распределения) = ' + str(corr_s[0]) + ', p-value = ' + str(corr_s[1]))

    return corr, corr_s

  
  
  def corr_coef_pixel_all_years(
                     df_data = None, 
                     ds_n = None, 
                     pca = None,
                     pcs = None, 
                     eofs = None, 
                     eigvals = None,
                     scale_type = 2):
    '''
    Вывод карты с коэффициентами корреляции по всем пикселям за все годы:
    df_data - исходные данные по PDSI, 
    ds_n - трехмерный массив, используется для извлечения параметров исходных данных, 
    pca - объект, полученный в результате eof_an,
    pcs - предсказанные моделью значения компонент, 
    eofs - набор значений функций EOF, 
    eigvals - собственные числа EOF,
    scale_type - параметр отвечающий за масштабирование главных компонент и 
                  EOF через умножение/деление значений на собственные числа,
    '''
  
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

    if type(df_data).__module__ != np.__name__:
      l_ar = len(df_data.to_numpy()[0])
    else:
      l_ar = len(df_data[:,m_ind][0])
    

    corr_s_l = []
    for m_ind in range(l_ar):
      if type(df_data).__module__ != np.__name__:
        u0 = df_data.to_numpy()[:,m_ind]
      else:
        u0 = df_data[:,m_ind]
      
      if np.isnan(u0[0]):
        corr_s = np.nan
      else:
        u = Yhat[:,m_ind]
        u0 = u0[:len(u)]
        nas = np.logical_or(np.isnan(u), np.isnan(u0))
        corr_s = spearmanr(u0[~nas],u[~nas])[0]
      
      corr_s_l.append(corr_s)

    new = np.reshape(corr_s_l, (-1, ds_n.shape[2]))
    im = plt.imshow(new[::-1], interpolation='none',cmap='coolwarm')
    cbar = plt.colorbar(im, ticks=[-1, -0.5, -0.25, 0, 0.25, 0.5, 1], 
                        orientation='vertical',
                        fraction=0.045, pad=0.05)
    plt.title('Коэффициент корреляции по всем годам')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return new
