import pandas as pd
import numpy as np
import xarray as xr
from pyEOF import *
from sklearn import linear_model

def r_execel(f_path, drop_val = 2):
    '''
    Формирование двумерного массива по годам из данных,
    полученных из ДКХ
    f_path - путь до xlsx файла
    drop_val - число строчек, которые будут удалены с конца табилцы
    '''
    df = pd.read_excel(f_path, index_col=None)
    fn_list = df['file_name'].unique()

    trsgi_values = []
    for i in (range(1901, np.max(df['age'])+1)):
      one_year_arr = []
      print(i)
      df0 = df[df['age'] == i]
      for j in fn_list:
        re = df0[df0['file_name'] == j]['trsgi'].values
        if len(re)>0:
          one_year_arr.append(re[0])
          #print(len(one_year_arr))
        else:
          one_year_arr.append(None)
          #print(len(one_year_arr))

      trsgi_values.append(one_year_arr)

    df_trsgi_values = pd.DataFrame(data=trsgi_values)
    ind_list = []
    for index, val in df_trsgi_values.isna().sum().iteritems():
      if val < drop_val+1:
        ind_list.append(index)

    df_trsgi_values.drop(df_trsgi_values.tail(drop_val).index,inplace=True)
    arrX = df_trsgi_values[ind_list].to_numpy()

    m_list = []
    for i in range(len(df_trsgi_values.columns)):
      arrY = df_trsgi_values[i].to_numpy()
      ind_NONE = np.where(np.isnan(arrY))
      ind_not_NONE = np.where(~np.isnan(arrY))

      regr = linear_model.LinearRegression()
      regr.fit(arrX[ind_not_NONE], arrY[ind_not_NONE])
      if len(ind_NONE[0])>0:
        arrY[ind_NONE] = np.around(regr.predict(arrX[ind_NONE]),3)
      m_list.append(arrY)

    mat = np.array(m_list)
    res = np.transpose(mat)

    return res

def r_netCDF(f_path, min_lon = -145, min_lat = 14, max_lon = -52, max_lat = 71):
    '''
    Формирование таблицы по годам из netCDF с индексами scpdsi
    '''

    ds = xr.open_dataset(f_path)["scpdsi"]

    coor = [] 
    for key in ds.coords.keys():
      coor.append(key)

    #Выбор территории анализа
    if coor[1] == 'latitude':
      mask_lon = (ds.longitude >= min_lon) & (ds.longitude <= max_lon)
      mask_lat = (ds.latitude >= min_lat) & (ds.latitude <= max_lat)
    else:
      mask_lon = (ds.lon >= min_lon) & (ds.lon <= max_lon)
      mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)

    ds_n = ds.where(mask_lon & mask_lat, drop=True)
    df_nn = ds_n.to_dataframe().reset_index()

    #Используется информация только по летним месяцам
    df_nn0 = df_nn[(df_nn['time'].dt.month < 9)&(df_nn['time'].dt.month > 5)]
    grouped_df = df_nn0.groupby([coor[1], coor[0] ,df_nn0['time'].dt.year])
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()

    mean_df['time_n'] = pd.to_datetime(mean_df.time.astype(str), format='%Y')
    del mean_df['time']
    mean_df = mean_df.rename(columns={'time_n': 'time'})
    mean_df = mean_df[['time', coor[1], coor[0], 'scpdsi']]
    df_data = get_time_space(mean_df, time_dim = "time", lumped_space_dims = [coor[1],coor[0]])

    return df_data, ds_n
