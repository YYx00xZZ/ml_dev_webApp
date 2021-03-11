# import yfinance as yf
# import streamlit as st

# st.write("""
# # Simple Stock Price App
# Shown are the stock closing price and volume of Google!
# """)

# # https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
# #define the ticker symbol
# tickerSymbol = 'GOOGL'
# #get data on this ticker
# tickerData = yf.Ticker(tickerSymbol)
# #get the historical prices for this ticker
# tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# # Open	High	Low	Close	Volume	Dividends	Stock Splits

# st.line_chart(tickerDf.Close)
# st.line_chart(tickerDf.Volume)

# import yfinance as yf
# import streamlit as st

# st.write("""
# # Simple Stock Price App
# Shown are the stock **closing price** and ***volume*** of Google!
# """)

# # https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
# #define the ticker symbol
# tickerSymbol = 'GOOGL'
# #get data on this ticker
# tickerData = yf.Ticker(tickerSymbol)
# #get the historical prices for this ticker
# tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# # Open	High	Low	Close	Volume	Dividends	Stock Splits

# st.write("""
# ## Closing Price
# """)
# st.line_chart(tickerDf.Close)
# st.write("""
# ## Volume Price
# """)
# st.line_chart(tickerDf.Volume)

# import streamlit as st
# import pandas as pd
# import base64
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# st.title('NBA Player Stats Explorer')

# st.markdown("""
# This app performs simple webscraping of NBA player stats data!
# * **Python libraries:** base64, pandas, streamlit
# * **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
# """)

# st.sidebar.header('User Input Features')
# selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2020))))

# # Web scraping of NBA player stats
# @st.cache
# def load_data(year):
#     url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
#     html = pd.read_html(url, header = 0)
#     df = html[0]
#     raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
#     raw = raw.fillna(0)
#     playerstats = raw.drop(['Rk'], axis=1)
#     return playerstats
# playerstats = load_data(selected_year)

# # Sidebar - Team selection
# sorted_unique_team = sorted(playerstats.Tm.unique())
# selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# # Sidebar - Position selection
# unique_pos = ['C','PF','SF','PG','SG']
# selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# # Filtering data
# df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

# st.header('Display Player Stats of Selected Team(s)')
# st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
# st.dataframe(df_selected_team)

# # Download NBA player stats data
# # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# def filedownload(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
#     return href

# st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# # Heatmap
# if st.button('Intercorrelation Heatmap'):
#     st.header('Intercorrelation Matrix Heatmap')
#     df_selected_team.to_csv('output.csv',index=False)
#     df = pd.read_csv('output.csv')

#     corr = df.corr()
#     mask = np.zeros_like(corr)
#     mask[np.triu_indices_from(mask)] = True
#     with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(7, 5))
#         ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
#     st.pyplot()

# import streamlit as st
# import pandas as pd
# import base64
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import yfinance as yf

# st.title('S&P 500 App')

# st.markdown("""
# This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
# * **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
# * **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
# """)

# st.sidebar.header('User Input Features')

# # Web scraping of S&P 500 data
# #
# @st.cache
# def load_data():
#     url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#     html = pd.read_html(url, header = 0)
#     df = html[0]
#     return df

# df = load_data()
# sector = df.groupby('GICS Sector')

# # Sidebar - Sector selection
# sorted_sector_unique = sorted( df['GICS Sector'].unique() )
# selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# # Filtering data
# df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]

# st.header('Display Companies in Selected Sector')
# st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
# st.dataframe(df_selected_sector)

# # Download S&P500 data
# # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# def filedownload(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
#     return href

# st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# # https://pypi.org/project/yfinance/

# data = yf.download(
#         tickers = list(df_selected_sector[:10].Symbol),
#         period = "ytd",
#         interval = "1d",
#         group_by = 'ticker',
#         auto_adjust = True,
#         prepost = True,
#         threads = True,
#         proxy = None
#     )

# # Plot Closing Price of Query Symbol
# def price_plot(symbol):
#   df = pd.DataFrame(data[symbol].Close)
#   df['Date'] = df.index
#   plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
#   plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
#   plt.xticks(rotation=90)
#   plt.title(symbol, fontweight='bold')
#   plt.xlabel('Date', fontweight='bold')
#   plt.ylabel('Closing Price', fontweight='bold')
#   return st.pyplot()

# num_company = st.sidebar.slider('Number of Companies', 1, 5)

# if st.button('Show Plots'):
#     st.header('Stock Closing Price')
#     for i in list(df_selected_sector.Symbol)[:num_company]:
#         price_plot(i)

# # This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
# import streamlit as st
# from PIL import Image
# import pandas as pd
# import base64
# import matplotlib.pyplot as plt
# from bs4 import BeautifulSoup
# import requests
# import json
# import time
# #---------------------------------#
# # New feature (make sure to upgrade your streamlit library)
# # pip install --upgrade streamlit

# #---------------------------------#
# # Page layout
# ## Page expands to full width
# st.set_page_config(layout="wide")
# #---------------------------------#
# # Title

# image = Image.open('./logo.jpg')

# st.image(image, width = 500)

# st.title('Crypto Price App')
# st.markdown("""
# This app retrieves cryptocurrency prices for the top 100 cryptocurrency from the **CoinMarketCap**!
# """)
# #---------------------------------#
# # About
# expander_bar = st.beta_expander("About")
# expander_bar.markdown("""
# * **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
# * **Data source:** [CoinMarketCap](http://coinmarketcap.com).
# * **Credit:** Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf).
# """)


# #---------------------------------#
# # Page layout (continued)
# ## Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
# col1 = st.sidebar
# col2, col3 = st.beta_columns((2,1))

# #---------------------------------#
# # Sidebar + Main panel
# col1.header('Input Options')

# ## Sidebar - Currency price unit
# currency_price_unit = col1.selectbox('Select currency for price', ('USD', 'BTC', 'ETH'))

# # Web scraping of CoinMarketCap data
# @st.cache
# def load_data():
#     cmc = requests.get('https://coinmarketcap.com')
#     soup = BeautifulSoup(cmc.content, 'html.parser')

#     data = soup.find('script', id='__NEXT_DATA__', type='application/json')
#     coins = {}
#     coin_data = json.loads(data.contents[0])
#     listings = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
#     for i in listings:
#       coins[str(i['id'])] = i['slug']

#     coin_name = []
#     coin_symbol = []
#     market_cap = []
#     percent_change_1h = []
#     percent_change_24h = []
#     percent_change_7d = []
#     price = []
#     volume_24h = []

#     for i in listings:
#       coin_name.append(i['slug'])
#       coin_symbol.append(i['symbol'])
#     #   price.append(i['quote'][currency_price_unit]['Price'])
#       percent_change_1h.append(i['quote'][currency_price_unit]['percent_change_1h'])
#       percent_change_24h.append(i['quote'][currency_price_unit]['percent_change_24h'])
#       percent_change_7d.append(i['quote'][currency_price_unit]['percent_change_7d'])
#       market_cap.append(i['quote'][currency_price_unit]['market_cap'])
#       volume_24h.append(i['quote'][currency_price_unit]['volume_24h'])

#     df = pd.DataFrame(columns=['coin_name', 'coin_symbol', 'market_cap', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'price', 'volume_24h'])
#     df['coin_name'] = coin_name
#     df['coin_symbol'] = coin_symbol
#     df['price'] = price
#     df['percent_change_1h'] = percent_change_1h
#     df['percent_change_24h'] = percent_change_24h
#     df['percent_change_7d'] = percent_change_7d
#     df['market_cap'] = market_cap
#     df['volume_24h'] = volume_24h
#     return df

# df = load_data()

# ## Sidebar - Cryptocurrency selections
# sorted_coin = sorted( df['coin_symbol'] )
# selected_coin = col1.multiselect('Cryptocurrency', sorted_coin, sorted_coin)

# df_selected_coin = df[ (df['coin_symbol'].isin(selected_coin)) ] # Filtering data

# ## Sidebar - Number of coins to display
# num_coin = col1.slider('Display Top N Coins', 1, 100, 100)
# df_coins = df_selected_coin[:num_coin]

# ## Sidebar - Percent change timeframe
# percent_timeframe = col1.selectbox('Percent change time frame',
#                                     ['7d','24h', '1h'])
# percent_dict = {"7d":'percent_change_7d',"24h":'percent_change_24h',"1h":'percent_change_1h'}
# selected_percent_timeframe = percent_dict[percent_timeframe]

# ## Sidebar - Sorting values
# sort_values = col1.selectbox('Sort values?', ['Yes', 'No'])

# col2.subheader('Price Data of Selected Cryptocurrency')
# col2.write('Data Dimension: ' + str(df_selected_coin.shape[0]) + ' rows and ' + str(df_selected_coin.shape[1]) + ' columns.')

# col2.dataframe(df_coins)

# # Download CSV data
# # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# def filedownload(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
#     return href

# col2.markdown(filedownload(df_selected_coin), unsafe_allow_html=True)

# #---------------------------------#
# # Preparing data for Bar plot of % Price change
# col2.subheader('Table of % Price Change')
# df_change = pd.concat([df_coins.coin_symbol, df_coins.percent_change_1h, df_coins.percent_change_24h, df_coins.percent_change_7d], axis=1)
# df_change = df_change.set_index('coin_symbol')
# df_change['positive_percent_change_1h'] = df_change['percent_change_1h'] > 0
# df_change['positive_percent_change_24h'] = df_change['percent_change_24h'] > 0
# df_change['positive_percent_change_7d'] = df_change['percent_change_7d'] > 0
# col2.dataframe(df_change)

# # Conditional creation of Bar plot (time frame)
# col3.subheader('Bar plot of % Price Change')

# if percent_timeframe == '7d':
#     if sort_values == 'Yes':
#         df_change = df_change.sort_values(by=['percent_change_7d'])
#     col3.write('*7 days period*')
#     plt.figure(figsize=(5,25))
#     plt.subplots_adjust(top = 1, bottom = 0)
#     df_change['percent_change_7d'].plot(kind='barh', color=df_change.positive_percent_change_7d.map({True: 'g', False: 'r'}))
#     col3.pyplot(plt)
# elif percent_timeframe == '24h':
#     if sort_values == 'Yes':
#         df_change = df_change.sort_values(by=['percent_change_24h'])
#     col3.write('*24 hour period*')
#     plt.figure(figsize=(5,25))
#     plt.subplots_adjust(top = 1, bottom = 0)
#     df_change['percent_change_24h'].plot(kind='barh', color=df_change.positive_percent_change_24h.map({True: 'g', False: 'r'}))
#     col3.pyplot(plt)
# else:
#     if sort_values == 'Yes':
#         df_change = df_change.sort_values(by=['percent_change_1h'])
#     col3.write('*1 hour period*')
#     plt.figure(figsize=(5,25))
#     plt.subplots_adjust(top = 1, bottom = 0)
#     df_change['percent_change_1h'].plot(kind='barh', color=df_change.positive_percent_change_1h.map({True: 'g', False: 'r'}))
#     col3.pyplot(plt)

# import streamlit as st
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.ensemble import RandomForestRegressor

# st.write("""
# # Boston House Price Prediction App
# This app predicts the **Boston House Price**!
# """)
# st.write('---')

# # Loads the Boston House Price Dataset
# boston = datasets.load_boston()
# X = pd.DataFrame(boston.data, columns=boston.feature_names)
# Y = pd.DataFrame(boston.target, columns=["MEDV"])

# # Sidebar
# # Header of Specify Input Parameters
# st.sidebar.header('Specify Input Parameters')

# def user_input_features():
#     CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
#     ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
#     INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
#     CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
#     NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
#     RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
#     AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
#     DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
#     RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
#     TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
#     PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
#     B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
#     LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
#     data = {'CRIM': CRIM,
#             'ZN': ZN,
#             'INDUS': INDUS,
#             'CHAS': CHAS,
#             'NOX': NOX,
#             'RM': RM,
#             'AGE': AGE,
#             'DIS': DIS,
#             'RAD': RAD,
#             'TAX': TAX,
#             'PTRATIO': PTRATIO,
#             'B': B,
#             'LSTAT': LSTAT}
#     features = pd.DataFrame(data, index=[0])
#     return features

# df = user_input_features()

# # Main Panel

# # Print specified input parameters
# st.header('Specified Input parameters')
# st.write(df)
# st.write('---')

# # Build Regression Model
# model = RandomForestRegressor()
# model.fit(X, Y)
# # Apply Model to Make Prediction
# prediction = model.predict(df)

# st.header('Prediction of MEDV')
# st.write(prediction)
# st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')

# Raw Package
import numpy as np
import pandas as pd

#Data Source
import yfinance as yf

#Data viz
import plotly.graph_objs as go

#Interval required 1 minute
data = yf.download(tickers='UBER', period='1d', interval='1m')

#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name = 'market data'))

# Add titles
fig.update_layout(
    title='Uber live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()