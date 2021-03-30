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
