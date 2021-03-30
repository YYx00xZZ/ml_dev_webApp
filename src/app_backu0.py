import time
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

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
# CONFIG

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

with header:
    '''
    # PyBrain
    This is our attempt at using python programming language to solve EEG motor imagery classification problem. (:
    
    ----
    '''
    # st.title('PyBrain')
    # st.text or st.write('This is our attempt at using python programming language to solve EEG motor-imagery classification problem. (:')


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

with dataset:
    '''
    ## Dataset
    Motor imagery left/right visual stimulation dataset
    '''
    loading = st.info('loading . . .')
    data = pd.read_csv(file_path_in + file_name_in,  header = 0)
    if st.checkbox('Show raw data'):
        st.text('Raw data')
        st.write(data)
    loading.info('loading . . . step 1 done')
    data_2 = update_event_ids_all(data)
    loading.info('loading . . . step 1 done, step 2 done')
    # st.write(data_2.head())
    data_3 = update_event_ids_all_event(data_2)
    # loading.info('loading . . . step 1 done, step 2 done, step 3 done')
    st.write(data_3.head())
    # time.sleep(0.2)
    loading.empty()


with features:
    st.header('Features')
    st.write('Feature selection; Separating attributes and target columns')
    'Oтделяне на полетата с характеристиките и колоната със събитията'
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

with model_training:
    st.header('Time to train the model')
    st.text('Soon here will be config widget')
