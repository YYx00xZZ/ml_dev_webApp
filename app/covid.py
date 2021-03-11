import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly_express as px
import pandas as pd
import requests
from datetime import datetime

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import time
import base64
import panel as pn
import pathlib
import json

st.set_page_config(layout='wide')


URL = 'https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Coronavirus_2019_nCoV_Cases/FeatureServer/1/query?where=1%3D1&outFields=*&outSR=4326&f=json'
URL_STYLED = '[services1.arcgis.com](https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Coronavirus_2019_nCoV_Cases/FeatureServer/1/query?where=1%3D1&outFields=*&outSR=4326&f=json)'
st.markdown(f'current data source: {URL_STYLED}')
st.beta_container()

@st.cache
def load_data():
    raw= requests.get(URL)  #"https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Coronavirus_2019_nCoV_Cases/FeatureServer/1/query?where=1%3D1&outFields=*&outSR=4326&f=json")
    raw_json = raw.json()
    df = pd.DataFrame(raw_json["features"])
    return (raw_json, df)


data = load_data()
raw_output = data[0]
df = data[1]
csv = df.to_csv(r'app/covid_raw_output.csv', header=True, index=None, sep=',', mode='a')

with open('app/covid_raw_output.txt', 'w') as outfile:
    json.dump(raw_output, outfile)

# Create a text element and let the reader know the data is loading.
data_load_state = st.info('Loading data...')
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")


if st.checkbox('Show raw response data'):
    st.subheader('Raw data')
    st.write(raw_output['features'])


# csv = df.to_csv(r'app/covid_raw_output.csv', header=True, index=None, sep=',', mode='a')

download_raw_output = st.button('Download Raw Data')
if download_raw_output:
    'Download Started!'
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko= f'<a href="data:file/csv;base64,{b64}" download="download_raw_output.txt">Download csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)


st.subheader('transform')
# transform
data_list = df["attributes"].tolist()
df_final = pd.DataFrame(data_list)
df_final.set_index("OBJECTID")
df_final = df_final[["Country_Region", "Province_State", "Lat", "Long_", "Confirmed", "Deaths", "Recovered", "Last_Update"]]
st.write(df_final)
download_transformed_output = st.button('Download download_transformed_output')
if download_transformed_output:
    'Download started'
    csv = df_final.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko= f'<a href="data:file/csv;base64,{b64}" download="download_transformed_output.txt">Download csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)


st.subheader('clearing')
# clearing
df_clear = df_final.copy()
df_clear = df_clear.dropna(subset=["Last_Update"])
df_clear["Province_State"].fillna(value="", inplace=True)

def convertTime(t):
    """convert the timestamp into a date with format “yyyy-mm-dd-hh-mm-ss” 
    """
    t = int(t)
    return datetime.fromtimestamp(t)

df_clear["Last_Update"]= df_clear["Last_Update"]/1000
df_clear["Last_Update"] = df_clear["Last_Update"].apply(convertTime)
# clearing
st.dataframe(df_clear)

download_cleared_output = st.button('Download download_cleared_output')
if download_cleared_output:
    'Download started'
    csv = df_final.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko= f'<a href="data:file/csv;base64,{b64}" download="download_cleared_output.txt">Download csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)

st.subheader('Aggregate')
# Aggregate
df_total = df_final.groupby("Country_Region", as_index=False).agg(
    {
        "Confirmed" : "sum",
        "Deaths" : "sum",
        "Recovered" : "sum"
    }
)
df_mid = df_total.copy()
df_mid['Last_Update'] = df_clear['Last_Update']
st.line_chart(df_total)
st.dataframe(df_total)
download_aggregated_output = st.button('Download download_aggregated_output')
if download_aggregated_output:
    'Download started'
    csv = df_final.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko= f'<a href="data:file/csv;base64,{b64}" download="download_aggregated_output.txt">Download csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)


# if st.checkbox('Show raw test data'):
#     st.subheader('Raw data')
#     # st.write(raw.text)
#     st.json(raw_json)

st.header('Global statistics')
total_confirmed = df_final["Confirmed"].sum()
total_recovered = df_final["Recovered"].sum()
total_deaths = df_final["Deaths"].sum()
st.subheader(f'total_confirmed: {total_confirmed}')
st.subheader(f'total_recovered: {total_recovered}')
st.subheader(f'total_deaths: {total_deaths}')

# df_top10 = df_total.nlargest(10, "Confirmed")
# top10_countries_1 = df_top10["Country_Region"].tolist()
# top10_confirmed = df_top10["Confirmed"].tolist()

# df_top10 = df_total.nlargest(10, "Recovered")
# top10_countries_2 = df_top10["Country_Region"].tolist()
# top10_recovered = df_top10["Recovered"].tolist()

# df_top10 = df_total.nlargest(10, "Deaths")
# top10_countries_3 = df_top10["Country_Region"].tolist()
# top10_deaths = df_top10["Deaths"].tolist()

# fig = make_subplots(
#     rows = 4, cols = 6,

#     specs=[
#             [{"type": "scattergeo", "rowspan": 4, "colspan": 3}, None, None, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"} ],
#             [    None, None, None,               {"type": "bar", "colspan":3}, None, None],
#             [    None, None, None,              {"type": "bar", "colspan":3}, None, None],
#             [    None, None, None,               {"type": "bar", "colspan":3}, None, None],
#           ]
# )


# message = df_final["Country_Region"] + " " + df_final["Province_State"] + "<br>"
# message += "Confirmed: " + df_final["Confirmed"].astype(str) + "<br>"
# message += "Deaths: " + df_final["Deaths"].astype(str) + "<br>"
# message += "Recovered: " + df_final["Recovered"].astype(str) + "<br>"
# message += "Last updated: " + df_final["Last_Update"].astype(str)
# df_final["text"] = message

# fig.add_trace(
#     go.Scattergeo(
#         locationmode = "country names",
#         lon = df_final["Long_"],
#         lat = df_final["Lat"],
#         hovertext = df_final["text"],
#         showlegend=False,
#         marker = dict(
#             size = 10,
#             opacity = 0.8,
#             reversescale = True,
#             autocolorscale = True,
#             symbol = 'square',
#             line = dict(
#                 width=1,
#                 color='rgba(102, 102, 102)'
#             ),
#             cmin = 0,
#             color = df_final['Confirmed'],
#             cmax = df_final['Confirmed'].max(),
#             colorbar_title="Confirmed Cases<br>Latest Update",  
#             colorbar_x = -0.05
#         )

#     ),
    
#     row=1, col=1
# )

# fig.add_trace(
#     go.Indicator(
#         mode="number",
#         value=total_confirmed,
#         title="Confirmed Cases",
#     ),
#     row=1, col=4
# )

# fig.add_trace(
#     go.Indicator(
#         mode="number",
#         value=total_recovered,
#         title="Recovered Cases",
#     ),
#     row=1, col=5
# )

# fig.add_trace(
#     go.Indicator(
#         mode="number",
#         value=total_deaths,
#         title="Deaths Cases",
#     ),
#     row=1, col=6
# )

# fig.add_trace(
#     go.Bar(
#         x=top10_countries_1,
#         y=top10_confirmed, 
#         name= "Confirmed Cases",
#         marker=dict(color="Yellow"), 
#         showlegend=True,
#     ),
#     row=2, col=4
# )

# fig.add_trace(
#     go.Bar(
#         x=top10_countries_2,
#         y=top10_recovered, 
#         name= "Recovered Cases",
#         marker=dict(color="Green"), 
#         showlegend=True),
#     row=3, col=4
# )

# fig.add_trace(
#     go.Bar(
#         x=top10_countries_3,
#         y=top10_deaths, 
#         name= "Deaths Cases",
#         marker=dict(color="crimson"), 
#         showlegend=True),
#     row=4, col=4
# )


# fig.update_layout(
#     template="plotly_dark",
#     title = "Global COVID-19 Cases (Last Updated: " + str(df_final["Last_Update"][0]) + ")",
#     showlegend=True,
#     legend_orientation="h",
#     legend=dict(x=0.65, y=0.8),
#     geo = dict(
#             projection_type="orthographic",
#             showcoastlines=True,
#             landcolor="white", 
#             showland= True,
#             showocean = True,
#             lakecolor="LightBlue"
#     ),

#     annotations=[
#         dict(
#             text="Source: https://bit.ly/3aEzxjK",
#             showarrow=False,
#             xref="paper",
#             yref="paper",
#             x=0.35,
#             y=0)
#     ]
# )

# fig.write_html('first_figure.html', auto_open=True)