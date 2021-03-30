# #%%
# ## features_train, features_test, event_ids_train, event_ids_test
# #%%
# #  RandomForestRegressor classifier.
# # A random forest regressor.

# # A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

# #%%
# #  RandomForestRegressor classifier.
# # Случайно горски регресор.

# # Случайната гора е мета-оценител, който се вписва в редица класифициращи дървета за вземане на решения в различни подпроби от набора от данни и използва усредняване за подобряване на точността на предсказване и контрол на пренастройването. Размерът на подпробата се контролира с параметъра max_samples, ако bootstrap = True (по подразбиране), в противен случай целият набор от данни се използва за изграждане на всяко дърво.


# # n_estimators - Число деревьев

# # Чем больше деревьев, тем лучше качество, но время настройки и работы RF также пропорционально увеличиваются. Обратите внимание, что часто при увеличении n_estimators качество на обучающей выборке повышается (может даже доходить до 100%), а качество на тесте выходит на асимптоту (можно прикинуть, скольких деревьев Вам достаточно).

# # max_features - Число признаков для выбора расщепления 

# # График качества на тесте от значения этого праметра унимодальный, на обучении он строго возрастает. При увеличении max_features увеличивается время построения леса, а деревья становятся «более однообразными». По умолчанию он равен sqrt(n) в задачах классификации и n/3 в задачах регрессии. Это самый важный параметр! Его настраивают в первую очередь (при достаточном числе деревьев в лесе).

# # min_samples_leaf - Ограничение на число объектов в листьях
# # Добре е да е 5

# # max_depth - Максимальная глубина деревьев

# # Ясно, что чем меньше глубина, тем быстрее строится и работает RF. При увеличении глубины резко возрастает качество на обучении, но и на контроле оно, как правило, увеличивается. Рекомендуется использовать максимальную глубину (кроме случаев, когда объектов слишком много и получаются очень глубокие деревья, построение которых занимает значительное время). При использовании неглубоких деревьев изменение параметров, связанных с ограничением числа объектов в листе и для деления, не приводит к значимому эффекту (листья и так получаются «большими»). Неглубокие деревья рекомендуют использовать в задачах с большим числом шумовых объектов (выбросов).
# #%%
# forest_regressor = RandomForestRegressor(n_estimators = 10, max_depth = 20)
# forest_regressor.fit(features_train, event_ids_train)

# #%%
# fr_train = forest_regressor.score(features_train, event_ids_train)
# print(fr_train)
# fr_test = forest_regressor.score(features_test, event_ids_test)
# print(fr_test)

# #%%

# list = [10, 20, 30, 40, 50]

# # Using for loop
# for i in list:
    
#     # Задаваме параметрите 
#     # Брой дървета - n_estimators = 9
#     # Участват всички процесорни ядра - n_jobs=-1
    
#     forest_regressor = RandomForestRegressor(n_estimators = 10, max_depth = i, n_jobs=-1)
    
#     # Обучаваме калсификатора
#     forest_regressor.fit(features_train, event_ids_train)

#     servey_name_new  = servey_name +str(i)

#     # Връща резултата в променливи t_exec, t_now, f_train_score, f_test_score
#     t_exec, t_now, f_train_score, f_test_score = score_model_small_result(forest_regressor, features_train, features_test, event_ids_train, event_ids_test, File_Out_Full, servey_name_new,electrodes_to_keep, events_to_keep)

    
#     #Изчисляваме predict - прогнозния резултат
#     predictions = forest_regressor.predict(features_test).astype(int)
#     print(predictions)
    
#     # Получаваме данните от съответната метрика за оценка 
#     accuracy_score(y_true = event_ids_test, y_pred = predictions, normalize=False)
   
#     # Печат на резултатат от оценката на класификатора
#     print(classification_report(y_true = event_ids_test, y_pred = predictions))
    
#     report = classification_report(y_true=event_ids_test, y_pred=predictions, output_dict=True)
#     # Транспонираме резултата
#     df = pd.DataFrame(report).transpose()
    
#     # Заместваме индекса с нов ( добавяме индексна колона)
#     df.reset_index(level=0, inplace=True)
#     df.columns = ['Type', 'precision', 'recall', 'f1-score','support']
    
#     # Добавяме данните от оценката като нови колони в dataframe
#     df['Servey_name'] = servey_name_new
#     df['Servey_time'] = t_now
#     df['Train_score'] = f_train_score
#     df['Test_score'] = f_test_score
#     df['Time_Exec'] = t_exec
#     df['Electrodes'] = listToString(electrodes_to_keep)
#     df['Events'] = listToString(events_to_keep)
#     df['test_size'] = str(test_size)
#     df['F_Name'] = file_name_in
    
#     # Печат на всики колони
#     pd.set_option('max_columns',None)
    
#     # Колони в dataframe - необходимо е за да бъдат записани правилно в CSV файла
#     fieldnames = ['Servey_name', 'Time_Exec', 'Train_score', 'Test_score','Type', 'precision', 'recall', 'f1-score', 'support',  'Electrodes', 'Events','Servey_time','test_size','F_Name']
    
#     # Преподреждаме колоните
#     df = df.reindex(columns=fieldnames)
    
#     print(df.head())
#     # Добавяне на dataframe в CSV
#     df.to_csv(File_Out_Full_csv, mode='a', header=False)
    
# print(classification_report(y_true = event_ids_test, y_pred = predictions))

import time
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

st.set_page_config(layout='wide')

# CONFIG
# Глобални променливи
file_path_in = 'data/'
file_path_out = 'results/'
file_name_in_1 = '4_GD_14ch_LR_24_01_2020_VStim_20Repeat_3sec_epoch_3_off_0.25_chunk_0.025_8_Butter_Mean'
file_name_in = '2_KA_14ch_LR_23_01_2020_VStim_20Repeat_3sec_epoch_3_off_0.25_chunk_0.25_4_with_shunk_and_notnull'+'.csv' #"4_GD_14ch_LR_24_01_2020_VStim_20Repeat_3sec_epoch_3_off_0.25_chunk_0.025_8_Butter_Mean"+".csv"
# datetime object containing current date and time
now = datetime.now()  # dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
str_filename_Out = 'BCI_Classification'
File_Out_Full = file_path_out + str_filename_Out + dt_string + '.txt'
# Data_Result
f = open( File_Out_Full,"a")
f.write("\n")
f.close()

st.write('''
# PyBrain
This is our attempt at using python programming language to solve EEG motor imagery classification problem. (:

----
''')
@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
st.write('''
## read data
''')
fileslist = get_static_store()
st.text(os.getcwd())
folderPath = st.text_input('Enter folder path:', 'data/')
if folderPath:
    filename = file_selector(folderPath)
    # if not filename in fileslist.values():
        # fileslist[filename] = filename
else:
    fileslist.clear()  # Hack to clear list if the user clears the cache and reloads the page
    st.info("Select one or more files.")

if st.button("Clear file list"):
    fileslist.clear()
if st.checkbox("Show file list?", False):
    finalNames = list(fileslist.keys())
    st.write(list(fileslist.keys()))

@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

# DATASET PREPROCESS RELATED
def load_data():
    data = pd.read_csv(file_path_in + file_name_in,  header = 0)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# ### Процедура за генериране на последователносстите от 1 до n за целия файл
def update_event_ids_all(data):
    # Генерипа последователно ID за целия файл - може да се използва за TIME и др.
    chk = 0
    for index, row in data.iterrows(): #итерираме ред по ред с вградения метод iterrows() от pandas
            chk += 1
            data.at[index, 'ID_all'] = chk
    return data

def update_event_ids_all_event(data):
    # Генерипа последователно ID за всеки event - може да се използва за TIME серии и др.
    chk_33025 = 0
    chk_33026 = 0
    chk_33027 = 0
    chk_33028 = 0
    for index, row in data.iterrows(): #итерираме ред по ред с вградения метод iterrows() от pandas
        if row['EventId'] == 33025:  
            chk_33025 += 1
            data.at[index, 'ID_all_events'] = chk_33025
        if row['EventId'] == 33026:  
            chk_33026 += 1
            data.at[index, 'ID_all_events'] = chk_33026
        if row['EventId'] == 33027:  
            chk_33027 += 1
            data.at[index, 'ID_all_events'] = chk_33027
        if row['EventId'] == 33028:  
            chk_33028 += 1
            data.at[index, 'ID_all_events'] = chk_33028            
    return data
### forest
# def score_model(estimator, features_train, features_test, event_ids_train, event_ids_test):
#     """
#     Returns the model accuracy scores on the training and testing data
#     Връща оценката на точността на модела на базата данните за тестване и обучение
#     и записва във файл времето за изпълнение
#     """
    
#     # Оценява тренировъчната и тестовата извадка 
#     f_train_score = str(estimator.score(features_train, event_ids_train))
#     f_test_score = str(estimator.score(features_test, event_ids_test))
    
#     print("Train score:    ", f_train_score)
#     print("Test score:    ", f_test_score)
# # Python program to convert a list to string 
    
# # Function to convert   
# def listToString(list_to_str):  
    
#     # initialize an empty string 
#     str1 = " "  
    
#     # traverse in the string   
#     for ele in list_to_str:  
#         str1 += ele 
#         str1 += ' , ' 
    
#     # return string   
#     return str1  

# def score_model_file(  estimator, features_train, features_test, event_ids_train, event_ids_test, file_name_out, servey_name, electrode, events):
#     """
#     Returns the model accuracy scores on the training and testing data
#     Връща оценката на точността на модела на базата данните за тестване и обучение
#     и записва във файл времето за изпълнение
#     """
#     start = time.time()

#     # Оценява тренировъчната и тестовата извадка 
#     f_train_score = str(estimator.score(features_train, event_ids_train))
#     f_test_score = str(estimator.score(features_test, event_ids_test))

#     end = time.time()
#     time_exec = end - start

#     print('\n Servey name : '+servey_name +'\n')
#     print("Train score:    ", f_train_score)
#     print("Test score:    ", f_test_score)
    
#     print('\n Time to execute code - '+str(time_exec) + '\n\n')   

#     now = datetime.now()
#     now = now.strftime("%m.%d.%Y %H:%M:%S")
    
#     f = open(file_name_out,"a")
#     f.write('\n ==================== \n')
#     f.write('\n Servey time: '+ str(now))
#     f.write('\n Servey name : '+servey_name +'\n')
#     f.write('\n Electrodes : '+ listToString(electrode))
#     f.write('\n Events : '+ listToString(events) +'\n')
    
#     f.write('\n'+"    Train score:    " + f_train_score+'\n')
#     f.write("    Test score:    "+f_test_score+'\n')
#     f.write('\n Time to execute code - '+str(time_exec))
#     f.write("\n")
#     f.close()
# def score_model_small_result( estimator, features_train, features_test, event_ids_train, event_ids_test, file_name_out, servey_name, electrode, events):
#     """
#     Returns the model accuracy scores on the training and testing data
#     Връща оценката на точността на модела на базата данните за тестване и обучение
#     и записва във файл времето за изпълнение
#     """
#     now = datetime.now()
#     now = now.strftime("%m.%d.%Y %H:%M:%S")
#     start = time.time()

#     # Оценка на резултата - estimator
    
    
#     # Оценка на съответият алгоритъм - в случая за тренировъчната и тестовата извадка 
#     f_train_score = str(estimator.score(features_train, event_ids_train))
#     f_test_score = str(estimator.score(features_test, event_ids_test))

#     end = time.time()
#     time_exec = end - start
    
    
#     t_exec = str(time_exec)    
#     t_now = str(now)
    
    
#     return t_exec, t_now, f_train_score, f_test_score


### forest

# DATASET PREPROCESS RELATED
st.write('''
## Dataset
Motor imagery left/right visual stimulation dataset
''')

if not filename:
    pass
else:
    loading = st.info('loading . . .')
    data = pd.read_csv(filename, header = 0)
    if st.checkbox('Show raw data', True):
        st.text('Raw data')
        st.write(data.head(3))
    loading.info('loading . . . step 1 done')
    data_2 = update_event_ids_all(data)
    loading.info('loading . . . step 1 done, step 2 done')
    # st.write(data_2.head())
    data_3 = update_event_ids_all_event(data_2)
    # loading.info('loading . . . step 1 done, step 2 done, step 3 done')
    st.write('Result df after preprocessing')
    st.write(data_3.head())
    # time.sleep(0.2)
    loading.empty()

#FEATURE SELECTION

st.write('''
## Features
Feature selection; Separating attributes and target columns

Oтделяне на полетата с характеристиките и колоната със събитията
''')
with st.echo():
    electrodes_to_keep = ["AF3", "F7", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "ID_all_events"]
    features_to_keep = electrodes_to_keep + ["EventId"]
brain_wave_data = data_3[features_to_keep]
features, event_ids = brain_wave_data.drop("EventId", axis = 1), brain_wave_data.EventId
col1, col2 = st.beta_columns([5, 1])
with col1:
    col1.write('Разделяне на датафрейма на атрибутии (AF3....)')
    st.write(features.head())
with col2:
    col2.write('и колони със събития')
    st.write(event_ids.head())

def prepare(df):
    # something like LabelEncoder
    df.replace({"left": 0 ,"right": 1}, inplace=True)
    # Feature selection - отделяне на полетата с характеристиките и колоната със събитията
    electrodes_to_keep =  ["AF3", "F7", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    predict = "EventId"
    features_to_keep = electrodes_to_keep # + predict
    features_to_keep.append(predict)
    df = df[features_to_keep]
    # X = df.loc[:, df.columns != 'EventId'].values
    # y = df.loc[:, df.columns == 'EventId'].values
    X = np.array(df.drop([predict], 1))
    Y = np.array(df[predict])
    return X,Y
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    # return X_train, X_test, y_train, y_test

X, Y = prepare(brain_wave_data)
n_train = st.text_input('N times to train')
train_placeholder = st.empty()
if n_train:
    train_placeholder.info('training . .')
    for _ in range(int(n_train)):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        mlp_model = MLPClassifier()
        mlp_model.fit(x_train,y_train)
        accu = mlp_model.score(x_test,y_test)
        st.write('ACCURACY')
        st.write(accu)
        #save model
    # train_placeholder.info('saving model')
    # with open("trained_model.pickle", "wb") as f:
    #     pickle.dump(mlp_model, f)
    train_placeholder.info('done')
else:
    st.info('Enter how much times to iterate')
# loaded_mlp_model = pickle.load(open("trained_model.pickle", "rb"))
# acc = loaded_mlp_model.score(X,Y)
# st.write(acc)
# predictions = loaded_mlp_model.predict(X[1,:].reshape(1,-1))
# # st.write(X[1,:])

# st.write(predictions)
# st.write(Y[1])
# # displaying the predictions
# # st.write("\nExperience\tActual Salary\tPredicted Salary")
# # for i in range(len(predictions)):
#     # st.write(X[i], "\t\t", Y[i], "\t\t", predictions[i])
