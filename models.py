from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import skimage.measure
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

def get_model_nofrozen_classification(n_inputs, n_classes, use_drop = False):
  '''
  Описание сети для задачи классификации без использования нейронов с EOF
  n_inputs - число предикторов,
  n_classes - число классов на выходе,
  use_drop - параметр, отвечающий за рандомное отключение доли нейронов (30%)

  Модель состоит из трех fully-connected (fc) слоев
  и выходного слоя с softmax-функцией из n_classes нейронов:
  n_inputs -> 30 -> 30 -> 1000 -> n_classes
  '''
  model = Sequential()
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))

  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))

  layer_1 = Dense(1000, kernel_initializer='he_uniform', activation='relu')
  model.add(layer_1)
  if use_drop == True:
    model.add(Dropout(0.30))

  layer_last = Dense(n_classes, activation='softmax')
  model.add(layer_last)
  return model

def get_model_nofrozen_regression(n_inputs, n_outputs, use_drop = False):

  '''
  Описание сети для задачи регрессии без использования нейронов с EOF
  n_inputs - число предикторов,
  n_outputs - количество предсказываемых значений,
  use_drop - параметр, отвечающий за рандомное отключение доли нейронов (30%)

  Модель состоит из четырех fully-connected (fc) слоев:
  n_inputs -> 30 -> 30 -> 1000 -> 10
  '''

  model = Sequential()
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))
  model.add(Dense(1000, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.3))
  model.add(Dense(10, kernel_initializer='he_uniform'))
  return model

def get_model_frozen_regression(n_inputs, n_outputs, eofs, use_drop = False, primEOF = False):

  '''
  Описание сети для задачи регрессии с применением нейронов с EOF
  n_inputs - число предикторов,
  n_outputs - количество предсказываемых значений,
  eofs - набор двумерных EOF,
  use_drop - параметр, отвечающий за рандомное отключение доли нейронов (30%)
  primEOF - параметр, отвечающий за использование примитивов значений EOF

  Модель состоит из четырех fully-connected (fc) слоев:
  n_inputs -> 30 -> 30 -> 30 -> число EOF
  '''

  n_last_layer = len(eofs)
  fr_weights = []
  fr_bais = []
  ar2 = []
  for i in range(n_last_layer):
      fr_bais.append(np.float32(0.))

  if primEOF == False:
    #Формирование двумерного массива из значений EOF, который можно применить
    #в качестве весов для выходного слоя
    t_tab0 = eofs['scpdsi'].iloc()[0].values
    for j in range(len(t_tab0[~np.isnan(t_tab0)])):
      tem_lis = []
      for i in range(n_last_layer):
          t_tab = eofs['scpdsi'].iloc()[i].values
          tem_lis.append(np.float32(t_tab[~np.isnan(t_tab)][j]))

      fr_weights.append(tem_lis)

  else:
    #Формирование двумерного массива из примитивов значений EOF,
    #который можно применить в качестве весов для выходного слоя
    for i in range(len(eofs)):
      te_ar = eofs.iloc()[i].to_numpy()
      te_ar.shape = (te_ar.size//186, 186)
      te_ar = np.flip(te_ar, 0)

      #Уменьшение размеров изображений (EOF) в пять раз по каждой из осей
      te_ar = skimage.measure.block_reduce(te_ar, (5,5), np.mean)
      #Повторное уменьшение размеров изображений в два раза по каждой из осей
      #В обоих случаях применялся пуллинг по среднему значению
      te_ar = skimage.measure.block_reduce(te_ar, (2,2), np.mean)

      x = te_ar.ravel()
      x = x[~np.isnan(x)]
      print(x)
      fr_weights.append(x)
      plt.imshow(te_ar, interpolation='none')
      plt.show()

    fr_weights = np.transpose(fr_weights)

  ar2.append(np.array(fr_weights))
  ar2.append(np.array(fr_bais))

  model = Sequential()
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))

  layer_1 = Dense(len(fr_weights), kernel_initializer='he_uniform', activation='relu')
  model.add(layer_1)
  if use_drop == True:
    model.add(Dropout(0.30))
  layer_last = Dense(len(fr_bais), activation='relu')
  model.add(layer_last)

  #Присвоение последнему слою весов fr_weights и заморозка его параметров
  layer_last.set_weights(ar2)
  layer_last.trainable=False
  return model

def get_model_frozen_classification(n_inputs, eofs, use_drop = False, primEOF = False):

  '''
  Описание сети для задачи классификации с применением нейронов с EOF
  n_inputs - число предикторов,
  eofs - набор двумерных EOF,
  use_drop - параметр, отвечающий за рандомное отключение доли нейронов (30%)
  primEOF - параметр, отвечающий за использование примитивов значений EOF

  Модель состоит из четырех fully-connected (fc) слоев
  и выходного слоя с softmax-функцией из len(eofs) нейронов:
  n_inputs -> 30 -> 30 -> 30 -> число EOF
  '''

  n_last_layer = len(eofs)
  fr_weights = []
  fr_bais = []
  ar2 = []
  for i in range(n_last_layer):
      fr_bais.append(np.float32(0.))

  if primEOF == False:
    #Формирование двумерного массива из значений EOF, который можно применить
    #в качестве весов для выходного слоя
    t_tab0 = eofs['scpdsi'].iloc()[0].values

    for j in range(len(t_tab0[~np.isnan(t_tab0)])):
      tem_lis = []
      for i in range(n_last_layer):
          t_tab = eofs['scpdsi'].iloc()[i].values
          tem_lis.append(np.float32(t_tab[~np.isnan(t_tab)][j]))

      fr_weights.append(tem_lis)

  else:
    #Формирование двумерного массива из примитивов значений EOF,
    #который можно применить в качестве весов для выходного слоя
    for i in range(len(eofs)):
      te_ar = eofs.iloc()[i].to_numpy()
      te_ar.shape = (te_ar.size//186, 186)
      te_ar = np.flip(te_ar, 0)

      #Уменьшение размеров изображений (EOF) в пять раз по каждой из осей
      te_ar = skimage.measure.block_reduce(te_ar, (5,5), np.mean)
      #Повторное уменьшение размеров изображений в два раза по каждой из осей
      #В обоих случаях применялся пуллинг по среднему значению
      te_ar = skimage.measure.block_reduce(te_ar, (2,2), np.mean)

      x = te_ar.ravel()
      x = x[~np.isnan(x)]
      print(x)
      fr_weights.append(x)
      plt.imshow(te_ar, interpolation='none')
      plt.show()

    fr_weights = np.transpose(fr_weights)

  ar2.append(np.array(fr_weights))
  ar2.append(np.array(fr_bais))

  model = Sequential()
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))
  model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_drop == True:
    model.add(Dropout(0.30))
  layer_1 = Dense(len(fr_weights), kernel_initializer='he_uniform', activation='relu')
  model.add(layer_1)
  layer_last = Dense(len(fr_bais), activation='softmax')
  model.add(layer_last)

  #Присвоение последнему слою весов fr_weights и заморозка его параметров
  layer_last.set_weights(ar2)
  layer_last.trainable=False
  return model




def simp_net_classification(trsgi_values, clust, ttl, model, use5 = None):
    '''
    Запуск обучения модели классификации
    trsgi_values - набор значений, полученных по ДКХ
    clust - набор предсказываемых значений, в данном случае, классов
    ttl - название графика, описывающего ход обучения
    model - модель, сформированная через get_model_*
    use5 - параметр, отвечающий за разделение выборки по пятилеткам
    '''

    trsgi = trsgi_values[:]
    all_arr = clust[:]

    trsgi_values = np.asarray(trsgi_values)
    all_arr = np.asarray(all_arr)

    #Прописан параметр ранней остановки:
    #В случае, когда значения val_accuracy в течении 30 итераций не улучшаются
    #больше, чем на 0.1, обучение прекращается
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, min_delta=0.1)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])



    if use5 == None or use5 == 1:
      #shuffle=True
      trsgi_values, all_arr = shuffle(trsgi_values, all_arr)
      history = model.fit(trsgi_values,
                          all_arr,
                          verbose=0,
                          epochs=200,
                          batch_size = 10,
                          shuffle=True,
                          validation_split = 0.2,
                          callbacks=[callback]
                          )
    else:
      #shuffle=False
      history = model.fit(trsgi_values,
                          all_arr,
                          verbose=0,
                          epochs=200,
                          batch_size = 10,
                          shuffle=False,
                          validation_split = use5,
                          callbacks=[callback]
                          )

    #Формирование графиков с ходом обучения
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.title(ttl)
    plt.show()

    return model, history

def simp_net_regression(trsgi_values, clust, ttl, eofs, model, use5 = None):

    '''
    Запуск обучения модели регрессии
    trsgi_values - набор значений, полученных по ДКХ
    clust - набор предсказываемых значений, в данном случае, главных компонент
    ttl - название графика, описывающего ход обучения
    model - модель, сформированная через get_model_*
    use5 - параметр, отвечающий за разделение выборки по пятилеткам
    '''

    trsgi = trsgi_values[:]
    all_arr = clust[:]

    trsgi_values = np.asarray(trsgi_values)
    all_arr = np.asarray(all_arr)


    #Прописан параметр ранней остановки:
    #В случае, когда значения val_loss в течении 20 итераций не улучшаются
    #больше, чем на 10, обучение прекращается
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=10)

    model.compile(optimizer='adam',
                  loss='mean_squared_error')


    if use5 == None or use5 == 1:
      #shuffle=True
      trsgi_values, all_arr = shuffle(trsgi_values, all_arr)
      history = model.fit(trsgi_values,
                          all_arr,
                          verbose=0,
                          epochs=200,
                          batch_size = 10,
                          shuffle=True,
                          validation_split = 0.2,
                          callbacks=[callback]
                          )
    else:
      #shuffle=False
      history = model.fit(trsgi_values,
                          all_arr,
                          verbose=0,
                          epochs=200,
                          batch_size = 10,
                          shuffle=False,
                          validation_split = use5,
                          callbacks=[callback]
                          )

    #Формирование графиков с ходом обучения
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.title(ttl)
    plt.show()

    return model, history

  
  
  
def train_model(df, list_t, li_m, type_m = 'regr', useEOF = 0):
    '''
    Запуск тренировки моделей
    df - таблица с результатами
    list_t - список переменных с данными для обучения
    li_m - тип сети (регрессия или классификация)
    useEOF - использование EOF на выходных слоях:
      0 - не использовать
      1 - использовать
      2 - использовать примитивы
    '''

    if type_m == 'class':
      tr_t, tr_l, te_t, te_l, v_r = list_class[li_m]
      n_inputs, n_outputs = tr_t.shape[1], 1
      stri = post_list_class[li_m]

      if useEOF == 0:
        #без использования EOF
        stri = stri + '_eof0'
        model = get_model_nofrozen_classification(n_inputs, 10, True)
        model, hystory = simp_net_classification(tr_t, tr_l, stri, model, v_r)
        
        
      if useEOF == 1:
        #с использованием EOF
        stri = stri + '_eof1'
        model = get_model_frozen_classification(n_inputs, eofs, True)
        model, hystory = simp_net_classification(tr_t, tr_l, stri, model, v_r)
        
        
      if useEOF == 2:
        #с использованием примитивов EOF
        stri = stri + '_eof1'
        model = get_model_frozen_classification(n_inputs, eofs, True, True)
        model, hystory = simp_net_classification(tr_t, tr_l, stri, model, v_r)
        
        
      score = model.evaluate(te_t, te_l, verbose=2)
      print(score)
      df_class = df_class.append({'Name': stri, 
                                    'params': hystory.params, 
                                    'loss': hystory.history['loss'][-1], 
                                    'accuracy': hystory.history['accuracy'][-1], 
                                    'val_loss': hystory.history['val_loss'][-1], 
                                    'val_accuracy': hystory.history['val_accuracy'][-1], 
                                    'test_loss': score[0], 
                                    'test_accuracy': score[1]}, ignore_index=True)

      return df_class, model


    else:
      tr_t, tr_l, te_t, te_l, v_r = list_regr[li_m]
      n_inputs, n_outputs = tr_t.shape[1], 1
      stri = post_list_regr[li_m]

      if useEOF == 0:
        #без использования EOF
        stri = stri + '_eof0'
        model = get_model_nofrozen_regression(n_inputs, n_outputs, True)
        model, hystory = simp_net_regression(tr_t, tr_l, stri, eofs, model, v_r)

        
      if useEOF == 1:
        #с использованием EOF
        stri = stri + '_eof1'
        model = get_model_frozen_regression(n_inputs, n_outputs, eofs, True)
        model, hystory = simp_net_regression(tr_t, tr_l, stri, eofs, model, v_r)
        
        
      if useEOF == 2:
        #с использованием примитивов EOF
        stri = stri + '_eof1'
        model = get_model_frozen_regression(n_inputs, n_outputs, eofs, True, True)
        model, hystory = simp_net_regression(tr_t, tr_l, stri, eofs, model, v_r)
        
        
      score = model.evaluate(te_t, te_l, verbose=2)
      print(score)
      df_regr = df_regr.append({'Name': stri, 
                              'params': hystory.params, 
                              'loss': hystory.history['loss'][-1],
                              'val_loss': hystory.history['val_loss'][-1], 
                              'test_loss': score}, ignore_index=True)

      return df_regr, model
