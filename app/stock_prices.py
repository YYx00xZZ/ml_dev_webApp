import streamlit as st
import yfinance as yf
import pandas as pd

st.write(""" # Online stock price ticker (yfinance data)""")
tickerSymbol = 'tsla'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d', start='2010-5-30', end='2021-03-08')

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)