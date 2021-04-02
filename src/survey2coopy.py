# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:12:53 202

@author: dimi
"""


#%%
# =============================================================================
# import tensorflow as tf
# tf.config.experimental.list_physical_devices()
# =============================================================================

#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LinDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error

from datetime import datetime
import time 
import timeit
import csv

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

## Read data
data_1 = pd.read_csv(file_path_in + file_name_in,  header = 0)
data_1.shape

df1 = data_1.loc[data_1["EventId"] == "left"]
df1.shape
df2 = data_1.loc[data_1["EventId"] == "right"]
df2.shape
df_event = data_1["EventId"]
df_event.shape



data_1.EventId = data_1.EventId.replace({"left": 0 ,"right": 1})
#%%

print(data_1.columns)
data_1.columns

#%%
      
# ##  Feature selection - отделяне на полетата с характеристиките и колоната със събитията

s1 = 'All '
electrodes_to_keep =  ["AF3", "F7", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# s1 = 'No F7'
# electrodes_to_keep =  ["AF3", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# s1 = 'No F7, F5'
# electrodes_to_keep =  ["AF3", "F3", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# s1 = 'No F7, F5, F3'
# electrodes_to_keep =  ["AF3", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# s1 = 'No F7, F5, F3, F8'
# electrodes_to_keep =  ["AF3", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "AF4"]

# s1 = 'No F7, F5, F3, F8, FC6'
# electrodes_to_keep =  ["AF3", "T7", "P7", "O1", "O2", "P8", "T8", "F4", "AF4"]

# s1 = 'No F7, F5, F3, F8, FC6, F4'
# electrodes_to_keep =  ["AF3", "T7", "P7", "O1", "O2", "P8", "T8", "AF4"]

# s1 = 'Only F7, F5, F3, F8, FC6, F4'
# electrodes_to_keep =  [ "F7", "F3", "F5","FC6", "F4", "F8"]

# # High Corelation
# s1 = 'High Corelation  - No P7'
# electrodes_to_keep =  ["AF3", "F7", "F3", "F5", "T7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# s1 = 'High Corelation  -  P7, O2'
# electrodes_to_keep =  ["AF3", "F7", "F3", "F5", "T7", "O1", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# s1 = 'High Corelation  -  P7, O2, O1'
# electrodes_to_keep =  ["AF3", "F7", "F3", "F5", "T7", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# s1 = 'High Corelation  -  P7, O2, O1, P8'
# electrodes_to_keep =  ["AF3", "F7", "F3", "F5", "T7",  "T8", "FC6", "F4", "F8", "AF4"]

#electrodes_to_keep = ["AF3", "F7", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
events_to_keep = ["EventId"]
features_to_keep = electrodes_to_keep + events_to_keep
data_2 = data_1[features_to_keep]

# Определяме размера на тестовата извадка
test_size = 0.4

# Описание на изследването
servey_name  = s1 + ' RFR = RandomForestRegressor(max_dept = '

#%%
print(data_2.head())
data_2.shape
print(data_2.columns)

# # Separating attributes and target columns
# Разделяне на датафрейма на атрибутии (AF3....) колони със събития (EventId)

features, event_ids = data_2.drop("EventId", axis = 1), data_2.EventId
print(features.head())
print(event_ids.sample(10))



print(data_2.head(10))
x = data_2.values[:, [2,7, 9]] #trim values to 10 entries and only columns 2 and 5 (indices 1, 4)

# x.columns
print(x)
x.shape

#%%
# Филтриране на dataframe поотделно за събитията
df1 = data_2.loc[data_2["EventId"] == "left"]
df2 = data_2.loc[data_2["EventId"] == "right"]
df1.shape
df2.shape

#df_event = data_2["ID_all_events"]

df_1_1 = df1['F4']
df_2_1 = df2['F4']
df_1_1.shape

print(df_1_1.head())
print(df_2_1.head())
df_event.head(10)
print(df_event.sample(10))


df_event = data_2.loc[data_2["EventId"] == "left"]


#%%
# ### Train / test split
# Initially, it's simplest to consider each  point in time independently. We could add time dependence (i.e. history of measurements) later. To do this, we perform a standard shuffle and split. Let's fix the test size at 20%. We need to split evenly (stratify) by event ID to ensure the same distribution of classes in the train and test data.
# 
# Първоначално е най-просто да се разглежда всеки момент във времето независимо. По-късно бихме могли да добавим времева зависимост (т.е. история на измерванията). За целта извършваме стандартно разбъркване и разделяне. Нека определим размера на даннте за тестване на 20%. Трябва да разделим равномерно (стратифициране) по идентификатор на събитието, за да осигурим еднакво разпределение на класовете с данни за трениране и тестване 

# Разделяне на dataset на 
# обучителна - features_train
# и 
# тестова част - features_test
# и на 
# обучителни - event_ids_train
# и 
# тестови събития (events) - event_ids_test

# Размет на тестовата част - test_size 
# По този начин се разделя dataset на части за обучение и тест. Обучението ще се извърши на тренировъчна извадка, а на тестова  ще се проверят придобитите „знания“. 
# test_size се използва за разделяне на пробата (във нашия случай 20% ще бъдат използвани за тестове

features_train, features_test, event_ids_train, event_ids_test = train_test_split(
    features,
    event_ids,
    test_size = test_size,
    stratify = event_ids)

# Връща размера на извадките
print(features_train.shape, features_test.shape, event_ids_train.shape, event_ids_test.shape)

#%%
def score_model(estimator, features_train, features_test, event_ids_train, event_ids_test):
    """
    Returns the model accuracy scores on the training and testing data
    Връща оценката на точността на модела на базата данните за тестване и обучение
    и записва във файл времето за изпълнение
    """
    
    # Оценява тренировъчната и тестовата извадка 
    f_train_score = str(estimator.score(features_train, event_ids_train))
    f_test_score = str(estimator.score(features_test, event_ids_test))
    
    print("Train score:    ", f_train_score)
    print("Test score:    ", f_test_score)
    

#%%


# Python program to convert a list to string 
    
# Function to convert   
def listToString(list_to_str):  
    
    # initialize an empty string 
    str1 = " "  
    
    # traverse in the string   
    for ele in list_to_str:  
        str1 += ele 
        str1 += ' , ' 
    
    # return string   
    return str1  
        

#%%
def score_model_file(  estimator, features_train, features_test, event_ids_train, event_ids_test, file_name_out, servey_name, electrode, events):
    """
    Returns the model accuracy scores on the training and testing data
    Връща оценката на точността на модела на базата данните за тестване и обучение
    и записва във файл времето за изпълнение
    """
    start = time.time()

    # Оценява тренировъчната и тестовата извадка 
    f_train_score = str(estimator.score(features_train, event_ids_train))
    f_test_score = str(estimator.score(features_test, event_ids_test))

    end = time.time()
    time_exec = end - start

    print('\n Servey name : '+servey_name +'\n')
    print("Train score:    ", f_train_score)
    print("Test score:    ", f_test_score)
    
    print('\n Time to execute code - '+str(time_exec) + '\n\n')   

    now = datetime.now()
    now = now.strftime("%m.%d.%Y %H:%M:%S")
    
    f = open(file_name_out,"a")
    f.write('\n ==================== \n')
    f.write('\n Servey time: '+ str(now))
    f.write('\n Servey name : '+servey_name +'\n')
    f.write('\n Electrodes : '+ listToString(electrode))
    f.write('\n Events : '+ listToString(events) +'\n')
    
    f.write('\n'+"    Train score:    " + f_train_score+'\n')
    f.write("    Test score:    "+f_test_score+'\n')
    f.write('\n Time to execute code - '+str(time_exec))
    f.write("\n")
    f.close()
   
  
   
#%%
# f_train_score = ''
# f_test_score = ''
# t_exec = ''
# t_now = ''

def score_model_small_result( estimator, features_train, features_test, event_ids_train, event_ids_test, file_name_out, servey_name, electrode, events):
    """
    Returns the model accuracy scores on the training and testing data
    Връща оценката на точността на модела на базата данните за тестване и обучение
    и записва във файл времето за изпълнение
    """
    now = datetime.now()
    now = now.strftime("%m.%d.%Y %H:%M:%S")
    start = time.time()

    # Оценка на резултата - estimator
    
    
    # Оценка на съответият алгоритъм - в случая за тренировъчната и тестовата извадка 
    f_train_score = str(estimator.score(features_train, event_ids_train))
    f_test_score = str(estimator.score(features_test, event_ids_test))

    end = time.time()
    time_exec = end - start
    
    
    t_exec = str(time_exec)    
    t_now = str(now)
    
    
    return t_exec, t_now, f_train_score, f_test_score




#%%
## features_train, features_test, event_ids_train, event_ids_test
#%%
#  RandomForestRegressor classifier.
# A random forest regressor.

# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

#%%
#  RandomForestRegressor classifier.
# Случайно горски регресор.

# Случайната гора е мета-оценител, който се вписва в редица класифициращи дървета за вземане на решения в различни подпроби от набора от данни и използва усредняване за подобряване на точността на предсказване и контрол на пренастройването. Размерът на подпробата се контролира с параметъра max_samples, ако bootstrap = True (по подразбиране), в противен случай целият набор от данни се използва за изграждане на всяко дърво.


# n_estimators - Число деревьев

# Чем больше деревьев, тем лучше качество, но время настройки и работы RF также пропорционально увеличиваются. Обратите внимание, что часто при увеличении n_estimators качество на обучающей выборке повышается (может даже доходить до 100%), а качество на тесте выходит на асимптоту (можно прикинуть, скольких деревьев Вам достаточно).

# max_features - Число признаков для выбора расщепления 

# График качества на тесте от значения этого праметра унимодальный, на обучении он строго возрастает. При увеличении max_features увеличивается время построения леса, а деревья становятся «более однообразными». По умолчанию он равен sqrt(n) в задачах классификации и n/3 в задачах регрессии. Это самый важный параметр! Его настраивают в первую очередь (при достаточном числе деревьев в лесе).

# min_samples_leaf - Ограничение на число объектов в листьях
# Добре е да е 5

# max_depth - Максимальная глубина деревьев

# Ясно, что чем меньше глубина, тем быстрее строится и работает RF. При увеличении глубины резко возрастает качество на обучении, но и на контроле оно, как правило, увеличивается. Рекомендуется использовать максимальную глубину (кроме случаев, когда объектов слишком много и получаются очень глубокие деревья, построение которых занимает значительное время). При использовании неглубоких деревьев изменение параметров, связанных с ограничением числа объектов в листе и для деления, не приводит к значимому эффекту (листья и так получаются «большими»). Неглубокие деревья рекомендуют использовать в задачах с большим числом шумовых объектов (выбросов).
#%%
forest_regressor = RandomForestRegressor(n_estimators = 10, max_depth = 20)
forest_regressor.fit(features_train, event_ids_train)

#%%
fr_train = forest_regressor.score(features_train, event_ids_train)
print(fr_train)
fr_test = forest_regressor.score(features_test, event_ids_test)
print(fr_test)

#%%

list = [10, 20, 30, 40, 50]

# Using for loop
for i in list:
    
    # Задаваме параметрите 
    # Брой дървета - n_estimators = 9
    # Участват всички процесорни ядра - n_jobs=-1
    
    forest_regressor = RandomForestRegressor(n_estimators = 10, max_depth = i, n_jobs=-1)
    
    # Обучаваме калсификатора
    forest_regressor.fit(features_train, event_ids_train)

    servey_name_new  = servey_name +str(i)

    # Връща резултата в променливи t_exec, t_now, f_train_score, f_test_score
    t_exec, t_now, f_train_score, f_test_score = score_model_small_result(forest_regressor, features_train, features_test, event_ids_train, event_ids_test, File_Out_Full, servey_name_new,electrodes_to_keep, events_to_keep)

    
    #Изчисляваме predict - прогнозния резултат
    predictions = forest_regressor.predict(features_test).astype(int)
    print(predictions)
    
    # Получаваме данните от съответната метрика за оценка 
    accuracy_score(y_true = event_ids_test, y_pred = predictions, normalize=False)
   
    # Печат на резултатат от оценката на класификатора
    print(classification_report(y_true = event_ids_test, y_pred = predictions))
    
    report = classification_report(y_true=event_ids_test, y_pred=predictions, output_dict=True)
    # Транспонираме резултата
    df = pd.DataFrame(report).transpose()
    
    # Заместваме индекса с нов ( добавяме индексна колона)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Type', 'precision', 'recall', 'f1-score','support']
    
    # Добавяме данните от оценката като нови колони в dataframe
    df['Servey_name'] = servey_name_new
    df['Servey_time'] = t_now
    df['Train_score'] = f_train_score
    df['Test_score'] = f_test_score
    df['Time_Exec'] = t_exec
    df['Electrodes'] = listToString(electrodes_to_keep)
    df['Events'] = listToString(events_to_keep)
    df['test_size'] = str(test_size)
    df['F_Name'] = file_name_in
    
    # Печат на всики колони
    pd.set_option('max_columns',None)
    
    # Колони в dataframe - необходимо е за да бъдат записани правилно в CSV файла
    fieldnames = ['Servey_name', 'Time_Exec', 'Train_score', 'Test_score','Type', 'precision', 'recall', 'f1-score', 'support',  'Electrodes', 'Events','Servey_time','test_size','F_Name']
    
    # Преподреждаме колоните
    df = df.reindex(columns=fieldnames)
    
    print(df.head())
    # Добавяне на dataframe в CSV
    df.to_csv(File_Out_Full_csv, mode='a', header=False)
    
print(classification_report(y_true = event_ids_test, y_pred = predictions))

