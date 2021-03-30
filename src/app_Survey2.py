import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Hyperparameter Optimization App',
    layout='wide')

#---------------------------------#
st.write("""
# Machine Learning App
**(Regression Edition)**
In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

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

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('Dataset')