import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


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


if __name__ == '__main__':
    # main()
    data=load_data()
    X,Y=preprocessing(data)
    # for _ in range(1):
    #     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    #     mlp_model = MLPClassifier()
    #     mlp_model.fit(x_train,y_train)
    #     accu = mlp_model.score(x_test,y_test)
    #     st.write('ACCURACY')
    #     st.write(accu)
    #     #save model
    #     with open("trained_model.pickle", "wb") as file:
    #         pickle.dump(mlp_model, file)
    loaded_mlp_model = pickle.load(open("trained_model.pickle", "rb"))
    acc = loaded_mlp_model.score(X,Y)
    st.write(acc)
    predictions = loaded_mlp_model.predict(X[1,:].reshape(1,-1))
    # st.write(X[1,:])

    st.write(predictions)
    st.write(Y[1])
    # displaying the predictions
    # st.write("\nExperience\tActual Salary\tPredicted Salary")
    # for i in range(len(predictions)):
        # st.write(X[i], "\t\t", Y[i], "\t\t", predictions[i])
