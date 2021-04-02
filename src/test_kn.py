import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

st.title("Predicting Diabetes Web App")
st.sidebar.title("Model Selection Panel")
st.markdown("Affected by Diabetes or not ?")
st.sidebar.markdown("Choose your model and its parameters")

#@st.cache(allow_output_mutation=True) if data have to be changed after looaded and cached
@st.cache(persist=True)
def load_data():
    data = pd.read_csv("data/diabetes.csv")
    return data
df=load_data()
if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Diabetes Raw Dataset")
    st.write(df)

st.sidebar.subheader("Select your Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Decision Tree","Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))