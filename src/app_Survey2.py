import pickle
import requests
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import plotly.graph_objects as go
#knn imports
from sklearn.neighbors import KNeighborsClassifier



#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')
default_columns = ["AF3", "F7", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "EventId"]

#---------------------------------#
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

#---------------------------------#
# Model building
def build_forest_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X.shape)
    st.write('Test set')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, stratify=Y)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        max_depth=parameter_max_depth,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)
    # knc KNeighborsClassifier
    knc = KNeighborsClassifier(n_neighbors=5,
        # *,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None)
    knc.fit(X_train, Y_train)
    #     #save model
    with open("randomforest_model.pickle", "wb") as file:
        pickle.dump(rf, file)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())
    #open model and use it
    loaded_forest_model = pickle.load(open("randomforest_model.pickle", "rb"))
    # acc = loaded_mlp_model.score(X,Y)
    # st.write(acc)
    # predictions = loaded_mlp_model.predict(X[1,:].reshape(1,-1))
    # st.write(X[1,:])

    # st.write(predictions)
    # st.write(Y[1])
    import re
    # collect_numbers = lambda x : [float(i) for i in re.split("[^0-9]", x) if i != ""]
    collect_numbers = lambda x : [float(i) for i in re.split(",", x) if i != ""]
    numbers = st.text_input("PLease enter 14 values (separated with ,). THis represent 1 row of BCI data for which we`ll predict if it's left or right.")
    st.text('for e.g. #4081.0256, 4091.6667, 4069.6155, 4088.4614, 4099.1025, 4092.3076, 4078.7180, 4078.2051, 4066.7949, 4066.4102, 4046.7949, 4050, 4092.9487, 4081.2820')
    if not numbers:
        st.info("Waiting for input")
    else:
        predictions = loaded_forest_model.predict(np.array(collect_numbers(numbers)).reshape(1,-1))
        st.write('prediction result', predictions)
def build_knn_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X.shape)
    st.write('Test set')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, stratify=Y)

    # rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
    #     max_depth=parameter_max_depth,
    #     random_state=parameter_random_state,
    #     max_features=parameter_max_features,
    #     criterion=parameter_criterion,
    #     min_samples_split=parameter_min_samples_split,
    #     min_samples_leaf=parameter_min_samples_leaf,
    #     bootstrap=parameter_bootstrap,
    #     oob_score=parameter_oob_score,
    #     n_jobs=parameter_n_jobs)
    # rf.fit(X_train, Y_train)
    # knc KNeighborsClassifier
    knc = KNeighborsClassifier(n_neighbors=parameter_n_neighbors,
        weights=parameter_weights,
        algorithm=parameter_algorithm,
        leaf_size=parameter_leaf_size,
        n_jobs=parameter_n_jobs)
    knc.fit(X_train, Y_train)
    #     #save model
    with open("knn_model.pickle", "wb") as file:
        pickle.dump(knc, file)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = knc.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = knc.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(knc.get_params())
    #open model and use it
    loaded_knc_model = pickle.load(open("knn_model.pickle", "rb"))
    # predict your data here with the loaded model
    import re
    # collect_numbers = lambda x : [float(i) for i in re.split("[^0-9]", x) if i != ""]
    collect_numbers = lambda x : [float(i) for i in re.split(",", x) if i != ""]
    numbers = st.text_input("PLease enter 14 values (separated with ,). THis represent 1 row of BCI data for which we`ll predict if it's left or right.")
    st.text('for e.g. #4081.0256, 4091.6667, 4069.6155, 4088.4614, 4099.1025, 4092.3076, 4078.7180, 4078.2051, 4066.7949, 4066.4102, 4046.7949, 4050, 4092.9487, 4081.2820')
    if not numbers:
        st.info("Waiting for input")
    else:
        predictions = loaded_knc_model.predict(np.array(collect_numbers(numbers)).reshape(1,-1))
        st.write('prediction result', predictions)
#---------------------------------#
st.write("""
# ml app
""")

#---------------------------------# DEMOS
st.write("""
# demos
""")

st.write("""
radio button group
""")
genre = st.radio(
    'What\'s your favorite movie genre',
    ('Comedy', 'Drama', 'Documentary')
)
if genre == 'Comedy':
    st.write('You selected comedy.')
else:
    st.write("You didn't select comedy.")
st.write("""
---
""")

st.write("""
checkbox
""")
agree = st.checkbox('I agree')

if agree:
    st.write('Great!')
st.write("""
---
""")

st.write("""
button
""")
if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
st.write("""
---
""")

st.write("""
example reading json api data
""")
URL = 'https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Coronavirus_2019_nCoV_Cases/FeatureServer/1/query?where=1%3D1&outFields=*&outSR=4326&f=json'
URL_STYLED = '[services1.arcgis.com](https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Coronavirus_2019_nCoV_Cases/FeatureServer/1/query?where=1%3D1&outFields=*&outSR=4326&f=json)'
st.markdown(f'current data source: {URL_STYLED}')
st.beta_container()
def load_api_data():
    raw= requests.get(URL)  #"https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Coronavirus_2019_nCoV_Cases/FeatureServer/1/query?where=1%3D1&outFields=*&outSR=4326&f=json")
    raw_json = raw.json()
    df = pd.DataFrame(raw_json["features"])
    return (raw_json, df)

demo_data = load_api_data()
show_raw_output = st.checkbox('Show raw API output', False)
if show_raw_output:
    raw_output = demo_data[0]
    st.write(raw_output)
demo_df = demo_data[1]
st.write(demo_df)

## radi group OK
## nqkolko checkboca OK
## button OK
## APIs function OK
## generate PDF
#---------------------------------# END DEMOS

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    # Select alg
    classifier = st.sidebar.selectbox("Select model", ("Random Forest", "KNN"))
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
if classifier == "Random Forest":
    # Sidebar - Specify parameter settings

    with st.sidebar.subheader('2.1. Learning Parameters'):
        parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_depth = st.sidebar.slider('Max depth', 0,50,20,5)
        parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
        parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
        parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    with st.sidebar.subheader('2.2. General Parameters'):
        parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
        parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
        parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
if classifier == "KNN":
    with st.sidebar.subheader('2.1. Parameters'):
        parameter_n_neighbors = st.sidebar.slider('n_neighbors', 0,20,5,1)
        parameter_weights = st.sidebar.select_slider('weights', options=['uniform', 'distance'])
        parameter_algorithm = st.sidebar.selectbox('Select algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        parameter_leaf_size = st.sidebar.slider('leaf_size', 0,120,30,5)
        # parameter_p ???
        parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = update_event_ids_all_event(df)
    df.EventId.replace({"left": 0 ,"right": 1}, inplace=True)
    df.EventId.replace({33025: 0 ,33026: 1}, inplace=True)

    st.markdown('**1.1. Glimpse of dataset**')
    chosen_columns = st.multiselect(
        'Exclude/Include column/s',
        df.columns.tolist(),
        default_columns)
    st.write(df[chosen_columns])
    if classifier == "Random Forest":
        build_forest_model(df[chosen_columns])
    if classifier == "KNN":
        build_knn_model(df[chosen_columns])
else:
    st.info('Awaiting for CSV file to be uploaded.')


