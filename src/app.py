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

#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Sidebar - Specify parameter settings
st.sidebar.header('Set Parameters')
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

st.sidebar.subheader('Learning Parameters')
parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10,50), 50)
parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
st.sidebar.write('---')
parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1,3), 1)
st.sidebar.number_input('Step size for max_features', 1)
st.sidebar.write('---')
parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

st.sidebar.subheader('General Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+1, 1)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
# -----------------------

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
    df.EventId.replace({"left": 0 ,"right": 1}, inplace=True)
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
st.sidebar()
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