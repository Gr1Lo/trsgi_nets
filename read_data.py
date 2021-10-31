import pandas as pd
import numpy as np
import xarray as xr
from pyEOF import *

def r_execel(f_path):
    '''
    Формирование двумерного массива по годам из данных,
    полученных из ДКХ
    '''
    df = pd.read_excel(f_path, index_col=None)
    fn_list = df['file_name'].unique()

    trsgi_values = []
    for i in (range(1900, np.max(df['age'])+1)):
      one_year_arr = []
      print(i)
      df0 = df[df['age'] == i]
      for j in fn_list:
        re = df0[df0['file_name'] == j]['trsgi'].values
        if len(re)>0:
          one_year_arr.append(re[0])
        else:
          one_year_arr.append(0.)

      trsgi_values.append(one_year_arr)

    return trsgi_values

def r_netCDF(f_path, min_lon = -145, min_lat = 14, max_lon = -52, max_lat = 71):
    '''
    Формирование таблицы по годам из netCDF с индексами scpdsi
    '''

    ds = xr.open_dataset(f_path)["scpdsi"]

    #Выбор территории анализа
    mask_lon = (ds.longitude >= min_lon) & (ds.longitude <= max_lon)
    mask_lat = (ds.latitude >= min_lat) & (ds.latitude <= max_lat)
    ds_n = ds.where(mask_lon & mask_lat, drop=True)
    df_nn = ds_n.to_dataframe().reset_index()

    #Используется информация только по летним месяцам
    df_nn0 = df_nn[(df_nn['time'].dt.month < 9)&(df_nn['time'].dt.month > 5)]
    grouped_df = df_nn0.groupby(["latitude", "longitude" ,df_nn0['time'].dt.year])
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()

    mean_df['time_n'] = pd.to_datetime(mean_df.time.astype(str), format='%Y')
    del mean_df['time']
    mean_df = mean_df.rename(columns={'time_n': 'time'})
    mean_df = mean_df[['time', 'latitude', 'longitude', 'scpdsi']]
    df_data = get_time_space(mean_df, time_dim = "time", lumped_space_dims = ["latitude","longitude"])

    return df_data
