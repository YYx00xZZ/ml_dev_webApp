import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time

"""
# PyBrain
Here's our first attempt at using python programming language to solve EEG motor-imagery classification problem. (:
"""
# ------------------------------------
"""
# File Uploader
It's hard to test the ability to upload files in an automated way, so here you
should test it by hand. Please upload a CSV file and make sure a table shows up
below with its contents.
"""
w = st.file_uploader("Upload a CSV file", type="csv")
if w:
    import pandas as pd
    data = pd.read_csv(w)
    st.write(data)
# ------------------------------------

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw test data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')

hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

st.bar_chart(hist_values)


# df = pd.DataFrame({
#   'first column': [1, 2, 3, 4],
#   'second column': [10, 20, 30, 40]
# })

# df

chart_data = pd.DataFrame(
     np.random.randn(20, 3), 
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.map(map_data)

# option = st.sidebar.selectbox(
#     'Which number do you like best?',
#      df['first column'])

# 'You selected:', option

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("Woohoo!")

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")


'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'