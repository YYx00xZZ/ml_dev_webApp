import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


st.set_page_config(layout='wide')

# @st.cache
def load_data():
    df = pd.read_csv("data/1_KA_14ch_LR_23_01_2020_VStim_20Repeat_3sec_epoch_3_off_0.25_chunk_0.25_4_with_shunk_and_notnull_scaler.csv", header=0)
    return df

def preprocessing(df):
    # something like LabelEncoder
    df.replace({"left": 0 ,"right": 1}, inplace=True)
    # Feature selection - отделяне на полетата с характеристиките и колоната със събитията
    electrodes_to_keep =  ["AF3", "F7", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    events_to_keep = ["EventId"]
    features_to_keep = electrodes_to_keep + events_to_keep
    df = df[features_to_keep]
    X = df.loc[:, df.columns != 'EventId'].values
    y = df.loc[:, df.columns == 'EventId'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test

def neuralNet(X_train, X_test, y_train, y_test):
    # Scalling the data before feeding it to the Neural Network.
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Instantiate the Classifier and fit the model.
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,2), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score1 = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score1, report, clf

# Accepting user data for predicting its Member Type
def accept_user_data():
    electrodes = ["AF3", "F7", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    user_data_to_predict = [] # np.array(empty=True)
    for x in electrodes:
        n_input =  st.text_input(f'Enter value for {x}')
        user_data_to_predict.append(n_input)
    return user_data_to_predict

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

def main():
    st.title('PyBrain')
    st.write('This is our attempt at using python programming language to solve EEG motor imagery classification problem. (:')
    st.write('---')
    
    # st.write('Import file')
    st.write('Import na faila')
    data = load_data()

    if st.checkbox('Show raw data'):
        st.write(data.head())

    X_train, X_test, y_train, y_test = preprocessing(data)
    score, report, clf = neuralNet(X_train,X_test,y_train,y_test)
    st.text("Accuracy of Neural Network model is: ")
    st.write(score,"%")
    st.text("Report of Neural Network model is: ")
    st.write(report)
    # accept_user_data()
    try:
        if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
            user_prediction_data = accept_user_data()
            return user_pridiction_data
    #         scaler = StandardScaler()  
    #         scaler.fit(X_train)  
    #         user_prediction_data = scaler.transform(user_prediction_data)	
    #         pred = clf.predict(user_prediction_data)
    #         st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
    except:
        pass

if __name__ == "__main__":
    main()