# import the libraries here
# basic
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# ml preprocessing
from sklearn.model_selection import train_test_split

# ml models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# ml metrics
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_squared_error

# additional settings
import warnings
warnings.filterwarnings("ignore")

# functions here
def train_linear_models(data, model_list):
    # independent and dependant variables
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'100% of our data: {len(df)}.')
    print(f'70% for training data: {len(X_train)}.')
    print(f'30% for test data: {len(X_test)}.')

    # metric dataframe
    metric_dict = {
        "R2" : [],
        "RMSE" : [],
        "MSE" : [],
        "MAE" : []
    }

    # Train the model
    for key, value in model_list.items():
        model = value
        model.fit(X_train,y_train)

        # Make predictions on the test dataset
        predictions = model.predict(X_test)

        # calculating metrics
        r2 = r2_score(y_test, predictions)
        RMSE = np.sqrt(mean_squared_error(y_test, predictions))
        MSE = mean_squared_error(y_test, predictions)
        MAE = mean_absolute_error(y_test, predictions)

        # append to the metric dictionary
        metric_dict["R2"].append(r2)
        metric_dict["RMSE"].append(RMSE)
        metric_dict["MSE"].append(MSE)
        metric_dict["MAE"].append(MAE)

    # metric_dict to dataframe
    df_metric = pd.DataFrame(data=metric_dict, index=model_list.keys())
    
    return df_metric


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
a complete end-to-end data science project needed for Ironhack bootcamp."

st.write(description)

# load the financial data
def load_data(filename):
    data = pd.read_csv(filename, index_col=0)
    return data

data = load_data("../data/financial_data_ml.csv")
df = data.copy()


# Sidebar Navigation
st.sidebar.title("Menu")
menu = st.sidebar.radio("Select an option: ", ["Exploratory Data Analysis", "Model Testing"])

# eda
if menu == "Exploratory Data Analysis":
    st.subheader("Explore the Financial Dataset")

    # Dataset preview
    st.dataframe(df)

elif menu == "Model Testing":
    st.subheader("Test ML models on the Financial Dataset")

    # clean the data before ml
    df = df.drop(columns=["date", "month", "year", "sp_close_pct_change"])
    df = pd.get_dummies(df, drop_first=True)
    df = df * 1
    
    # move target to the end
    target = df.pop("sp_close")
    df["sp_close"] = target

    # Tabs for different models
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Linear Regression",
        "Lasso Regression",
        "Ridge Regression",
        "Decision Tree Regression",
        "K-Neighbours Regression",
        "Random Forest Regression",
        "XGBoost Regression"]
    )

    with tab1:
        model1 = {"LR" : LinearRegression()}

        metric1 = train_linear_models(df, model1)

        # metrics
        r2_score1 = round(metric1["R2"].values[0], 2)
        rmse1 = round(metric1["RMSE"].values[0], 2)

        st.metric(label=r"$R^2 \ \mathrm{score}: $", value=f"{r2_score1}")
        st.metric(label=r"RMSE: ", value=f"{rmse1}")

    with tab2:
        model2 = {"Lasso" : Lasso()}

        metric2 = train_linear_models(df, model2)

        # metrics
        r2_score2 = round(metric2["R2"].values[0], 2)
        rmse2 = round(metric2["RMSE"].values[0], 2)

        st.metric(label=r"$R^2 \ \mathrm{score}: $", value=f"{r2_score2}")
        st.metric(label=r"RMSE: ", value=f"{rmse2}")

    with tab3:
        model3 = {"Ridge" : Ridge()}

        metric3 = train_linear_models(df, model3)

        # metrics
        r2_score3 = round(metric3["R2"].values[0], 2)
        rmse3 = round(metric3["RMSE"].values[0], 2)

        st.metric(label=r"$R^2 \ \mathrm{score}: $", value=f"{r2_score3}")
        st.metric(label=r"RMSE: ", value=f"{rmse3}")

    with tab4:
        model4 = {"Decision Tree" : DecisionTreeRegressor()}

        metric4 = train_linear_models(df, model4)

        # metrics
        r2_score4 = round(metric4["R2"].values[0], 2)
        rmse4 = round(metric4["RMSE"].values[0], 2)

        st.metric(label=r"$R^2 \ \mathrm{score}: $", value=f"{r2_score4}")
        st.metric(label=r"RMSE: ", value=f"{rmse4}")

    with tab5:
        model5 = {"KNR" : KNeighborsRegressor()}

        metric5 = train_linear_models(df, model5)

        # metrics
        r2_score5 = round(metric5["R2"].values[0], 2)
        rmse5 = round(metric5["RMSE"].values[0], 2)

        st.metric(label=r"$R^2 \ \mathrm{score}: $", value=f"{r2_score5}")
        st.metric(label=r"RMSE: ", value=f"{rmse5}")

    with tab6:
        model6 = {"Random Forest" : RandomForestRegressor()}

        metric6 = train_linear_models(df, model6)

        # metrics
        r2_score6 = round(metric6["R2"].values[0], 2)
        rmse6 = round(metric6["RMSE"].values[0], 2)

        st.metric(label=r"$R^2 \ \mathrm{score}: $", value=f"{r2_score6}")
        st.metric(label=r"RMSE: ", value=f"{rmse6}")

    with tab7:
        model7 = {"XGBR" : xgb.XGBRegressor()}

        metric7 = train_linear_models(df, model7)

        # metrics
        r2_score7 = round(metric7["R2"].values[0], 2)
        rmse7 = round(metric7["RMSE"].values[0], 2)

        st.metric(label=r"$R^2 \ \mathrm{score}: $", value=f"{r2_score7}")
        st.metric(label=r"RMSE: ", value=f"{rmse7}")
