# import all the libraries here
# basic libraries
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# statistics
from scipy.stats import chi2_contingency

# data preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# ml models


# ml metrics


# user defined functions
from functions import *

## Basic streamlit display
st.title("Predicting Financial Market Trend Using Economics Indicators")
st.header("An Investigation Based On State-of-the-art Machine Learning Models")

# First image
st.image("../image/image_1.jpg", use_container_width=True)

# Write about the project here, keep it updated
description = "The world of finance is fascinating. \
Tracking and understanding the financial market is one of the most \
complex tasks. I am passionate about the economics and financial market. \
One of my hobbies is to understand how economic indicators affect the \
financial market, especially, the stock market. In this project, I have \
delved deep into this problem and try to find an answer. This project is \
a complete end-to-end data analysis project needed for Ironhack bootcamp."

st.write(description)

# collect data here
indicators = {
    "gdp" : "GDP",
    "inflation" : "CPIAUCSL",
    "unemployment" : "UNRATE",
    "fed_int_rate" : "FEDFUNDS",
    "ten_y_tres_yield" : "DGS10",
    "m2_money_supp" : "M2SL",
    "cons_sent_idx" : "UMCSENT",
    "ind_pro_idx" : "INDPRO",
    "exports" : "EXPGS",
    "imports" : "IMPGS",
    "new_home_const" : "HOUST",
    "mortgage_rate" : "MORTGAGE30US",
    "volatility_index" : "VIXCLS",
    "crude_oil_wti" : "DCOILWTICO"
}

# saving into dictionary
data = {key: get_fred_data(value) for key, value in indicators.items()}

# converting to dataframes
df_dict = {}
for key, value in data.items():
    df_dict[key] = data[key].to_frame(name=key).reset_index().rename(columns={"index": "date"})


# get S&P 500 data
sp500 = get_ticker_data("^SPX", plot=False)
print(sp500.head(4))
print(sp500.shape)