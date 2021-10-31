import numpy as np

def train_and_test(trsgi, labels):

  '''
  Разделение выборки на тренировочную и тестовую (соотношение 4 к 1)
  trsgi - значения предикторов,
  labels - предсказываемые значения
  '''

  nums = np.ones(len(trsgi))
  nums[:int(len(trsgi)/5)] = 0
  np.random.shuffle(nums)

  mask = 1 == nums
  mask = np.array(mask)

  train_trsgi = np.array(trsgi)[mask]
  train_labels = labels[mask]
  test_trsgi = np.array(trsgi)[~mask]
  test_labels = labels[~mask]

  return train_trsgi, train_labels, test_trsgi, test_labels

def train_and_test5(trsgi, labels, start_p):

  '''
  Разделение выборки на тренировочную и тестовую (по пятилеткам)
  trsgi - значения предикторов,
  labels - предсказываемые значения
  start_p - индекс года, с которого начинается тестовая пятилетка

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
        if loc_count > 4:
          prev0 = 'tes'
          loc_count = 0

        train_nums[i] = 0
        val_nums[i] = 0
        test_nums[i] = 1

      if prev == 'tes':
        if loc_count > 4:
          prev0 = 'tr0'
          loc_count = 0

        train_nums[i] = 1
        val_nums[i] = 0
        test_nums[i] = 0

      if prev == 'tr0':
        if loc_count > 4:
          prev0 = 'val'
          loc_count = 0

        train_nums[i] = 0
        val_nums[i] = 1
        test_nums[i] = 0

      if prev == 'val':
        if loc_count > 4:
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

def sta_augment(trsgi_values, pcs):
  '''
  Искусственное увеличение тренировочной выборки
  '''
  pcs_copy = pcs.copy()
  n_tab = augment(pcs_copy)
  pcs_copy = pcs_copy.append(n_tab)

  for t in range(len(n_tab.columns)-2):
    n_tab = augment(n_tab)
    pcs_copy = pcs_copy.append(n_tab)

  trsgi_copy= trsgi_values.tolist() * 10

  return trsgi_copy, pcs_copy

def split_regr(trsgi_values, pcs):
    trsgi_values = np.asarray(trsgi_values)
    trsgi_scaled = normalize(trsgi_values, axis = 0)

    # разбивка для регрессионной задачи с ненормализованными trsgi
    train_trsgi, train_labels, test_trsgi, test_labels = train_and_test(trsgi_values, pcs[:109])
    # разбивка для регрессионной задачи с нормализованными trsgi
    train_trsgi_norm, train_labels_norm, test_trsgi_norm, test_labels_norm = train_and_test(trsgi_scaled, pcs[:109])

def split_class(trsgi_values, kmeans10):
    trsgi_values = np.asarray(trsgi_values)
    trsgi_scaled = normalize(trsgi_values, axis = 0)

    # разбивка для классификационной задачи с ненормализованными trsgi (10 классов)
    train_trsgi10, train_labels10, test_trsgi10, test_labels10 = train_and_test(trsgi_values, kmeans10.labels_[:109])
    # разбивка для классификационной задачи с нормализованными trsgi (10 классов)
    train_trsgi_norm10, train_labels_norm10, test_trsgi_norm10, test_labels_norm10 = train_and_test(trsgi_scaled, kmeans10.labels_[:109])
