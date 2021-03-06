
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd

def train_and_test(trsgi, labels, p_v):

  '''
  Разделение выборки на тренировочную и тестовую (соотношение 4 к 1)
  trsgi - значения предикторов,
  labels - предсказываемые значения,
  p_v - доля тестовой выборки
  '''

  nums = np.ones(len(trsgi))
  nums[:int(len(trsgi)*p_v)] = 0
  np.random.shuffle(nums)

  mask = 1 == nums
  mask = np.array(mask)

  train_trsgi = np.array(trsgi)[mask]
  train_labels = labels[mask]
  test_trsgi = np.array(trsgi)[~mask]
  test_labels = labels[~mask]

  return train_trsgi, train_labels, test_trsgi, test_labels

def train_and_test5(trsgi, labels, start_p,type_parts):

  '''
  Разделение выборки на тренировочную и тестовую (по пятилеткам)
  trsgi - значения предикторов,
  labels - предсказываемые значения
  start_p - индекс года, с которого начинается тестовая пятилетка
  type_parts - единица разделения (пятилетки, шестилетки и т.д.)
  Схема пятилеток:
  1) Тестовая пятилетка (tes) --------------- 2) Первая тренировочная пятилетка (tr0)
            |                                              |
  4) Вторая тренировочная пятилетка (tr1)-----3) Валидационная пятилетка (val)
  '''

  prev = 'tr1'
  prev0  = 'tr1'
  train_nums = np.ones(len(trsgi))
  val_nums = np.ones(len(trsgi))
  test_nums = np.ones(len(trsgi))
  loc_count = 0
  for i in range(len(trsgi)):
    if i >= start_p:
      loc_count += 1
      if prev == 'tr1':
        if loc_count > type_parts-1:
          prev0 = 'tes'
          loc_count = 0

        train_nums[i] = 0
        val_nums[i] = 0
        test_nums[i] = 1

      if prev == 'tes':
        if loc_count > type_parts-1:
          prev0 = 'tr0'
          loc_count = 0

        train_nums[i] = 1
        val_nums[i] = 0
        test_nums[i] = 0

      if prev == 'tr0':
        if loc_count > type_parts-1:
          prev0 = 'val'
          loc_count = 0

        train_nums[i] = 0
        val_nums[i] = 1
        test_nums[i] = 0

      if prev == 'val':
        if loc_count > type_parts-1:
          prev0 = 'tr1'
          loc_count = 0

        train_nums[i] = 1
        val_nums[i] = 0
        test_nums[i] = 0

      prev = prev0

    else:
      train_nums[i] = 1
      val_nums[i] = 0
      test_nums[i] = 0

  train_mask = 1 == train_nums
  val_mask = 1 == val_nums
  test_mask = 1 == test_nums

  train_mask = np.array(train_mask)
  val_mask = np.array(val_mask)
  test_mask = np.array(test_mask)

  train_trsgi = np.array(trsgi)[train_mask]
  train_labels = labels[train_mask]
  val_trsgi = np.array(trsgi)[val_mask]
  val_labels = labels[val_mask]
  test_trsgi = np.array(trsgi)[test_mask]
  test_labels = labels[test_mask]

  train_trsgi1 = np.concatenate((train_trsgi, val_trsgi))
  train_labels1 = np.concatenate((train_labels, val_labels))

  #Доля валидационных лет от суммы лет валидационной и тренировочных пятилеток
  val_rate = len(val_labels)/len(train_labels1)

  return train_trsgi, train_labels, val_labels, val_trsgi, test_trsgi, test_labels, val_rate

def augment(pcs_copy1):
  pcs_copy1 = pcs_copy1.replace(0, 100000000000)
  new_df = pd.DataFrame(np.where(pcs_copy1.T.pow(2) == pcs_copy1.T.pow(2).min(), 0, 1),index=pcs_copy1.columns).T
  pcs_copy1 = pcs_copy1.replace(100000000000, 0)
  new_df1 = pd.DataFrame(new_df.values*pcs_copy1.values, columns=new_df.columns, index=new_df.index)
  return new_df1

def sta_augment(trsgi_values, pcs, n_eof):
  '''
  Искусственное увеличение тренировочной выборки
  '''
  pcs_copy = pcs.copy()
  n_tab = augment(pcs_copy)
  pcs_copy = pcs_copy.append(n_tab)

  for t in range(len(n_tab.columns)-2):
    n_tab = augment(n_tab)
    pcs_copy = pcs_copy.append(n_tab)

  trsgi_copy= trsgi_values.tolist() * n_eof

  return trsgi_copy, pcs_copy

def sta_split(trsgi_values, pcs_or_kmeans, use_norm = True, type_op = 'regr', use5 = None, type_parts = 5, use_aug = False, p_v=0.2):
    '''
    Запуск разделения выборки на тренировочную и тестовую
    trsgi_values - набор предикторов, 
    pcs_or_kmeans - набор предсказываемых значений, 
    use_norm - использование нормализации, 
    type_op - тип модели (классификационная или регрессионная), 
    use5 - параметр, отвечающий за разделение набора данных по пятилеткам, 
    use_aug - параметр, отвечающий за использования синтетического расширения,
    p_v - доля тестовой выборки
    '''

    trsgi_values = np.asarray(trsgi_values)
    
    if use_norm:
      trsgi_values = normalize(trsgi_values, axis = 0)

    if use5 == None:
      if type_op == 'regr':
        # разбивка для регрессионной задачи
        train_trsgi, train_labels, test_trsgi, test_labels = train_and_test(trsgi_values, pcs_or_kmeans[:106], p_v)
        val_rate = 1
        if use_aug:
          train_trsgi, train_labels = sta_augment(train_trsgi, train_labels, len(pcs_or_kmeans.columns))
          train_trsgi = np.array(train_trsgi)
          train_labels = np.array(train_labels)

      if type_op == 'class':
        # разбивка для классификационной задачи
        train_trsgi, train_labels, test_trsgi, test_labels = train_and_test(trsgi_values, pcs_or_kmeans.labels_[:106], p_v)
        val_rate = 1

    else:
      if type_op == 'regr':
        # разбивка для регрессионной задачи
        train_trsgi, train_labels, val_labels, val_trsgi, test_trsgi, test_labels, val_rate = train_and_test5(trsgi_values, pcs_or_kmeans[:106], use5, type_parts)
        if use_aug:
          train_trsgi, train_labels = sta_augment(train_trsgi, train_labels, len(pcs_or_kmeans.columns))
          val_trsgi, val_labels = sta_augment(val_trsgi, val_labels, len(pcs_or_kmeans.columns))
          train_trsgi = np.concatenate((train_trsgi, val_trsgi))
          train_labels = pd.concat([train_labels, val_labels])
        else:
          train_trsgi = np.concatenate((train_trsgi, val_trsgi))
          train_labels = np.concatenate((train_labels, val_labels))

      if type_op == 'class':
        # разбивка для классификационной задачи
        train_trsgi, train_labels, val_labels, val_trsgi, test_trsgi, test_labels, val_rate = train_and_test5(trsgi_values, pcs_or_kmeans.labels_[:106], use5, type_parts)
        train_trsgi = np.concatenate((train_trsgi, val_trsgi))
        train_labels = np.concatenate((train_labels, val_labels))

    return train_trsgi, train_labels, test_trsgi, test_labels, val_rate
  
def filt_arr(type_m, norm, aug, use_01, use_02, eof):
  all_arr = []
  for norm_i in norm:
    for aug_i in aug:
      for use_01_i in use_01:
        for use_02_i in use_02:
          for eof_i in eof:
            m_str = type_m + '_norm' + str(norm_i) + '_aug' + str(aug_i) + '_use' + str(use_01_i) + str(use_02_i) + '_useEOF' + str(eof_i)
            all_arr.append(m_str)

  return all_arr

