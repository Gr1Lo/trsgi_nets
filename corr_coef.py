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
                     scale_type = 2):

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
                  EOF через умножение/деление значений на собственные числа
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

    _ = plt.hist(u0, bins='auto')
    plt.title('Распределение значений scPDSI в исходных данных для ' + str(year + base_year) + ' года')
    plt.show()
    D, pval = kstest(normed_data0, 'norm')
    print(normaltest(normed_data0))
    print('p-value теста Колмогорова-Симрнова для исходных данных = ' + str(pval) + '\n')

    _ = plt.hist(u, bins='auto')
    plt.title('Распределение значений scPDSI в предсказанных данных для ' + str(year + base_year) + ' года')
    plt.show()
    D, pval = kstest(normed_data, 'norm')
    print(normaltest(normed_data))
    print('p-value теста Колмогорова-Симрнова для предсказанных данных = ' + str(pval) + '\n')
    
    corr = pearsonr(u0[~nas],u[~nas])
    print('Коэффициент корреляции Пирсона = ' + str(corr[0]) + ', p-value = ' + str(corr[1]))
    
    corr_s = spearmanr(u0[~nas],u[~nas])
    print('Коэффициент корреляции Спирмана (для ненормального распределения) = ' + str(corr_s[0]) + ', p-value = ' + str(corr_s[1]))




def corr_coef_pixel(str_lat,
                    str_lon,
                     base_year = 1901, 
                     df_data = None, 
                     ds_n = None, 
                     pca = None,
                     pcs = None, 
                     eofs = None, 
                     eigvals = None,
                     scale_type = 2):
  
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
                  EOF через умножение/деление значений на собственные числа
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
    

    corr = pearsonr(u0[~nas],u[~nas])
    print('Коэффициент корреляции Пирсона = ' + str(corr[0]) + ', p-value = ' + str(corr[1]))
    
    corr_s = spearmanr(u0[~nas],u[~nas])
    print('Коэффициент корреляции Спирмана (для ненормального распределения) = ' + str(corr_s[0]) + ', p-value = ' + str(corr_s[1]))
