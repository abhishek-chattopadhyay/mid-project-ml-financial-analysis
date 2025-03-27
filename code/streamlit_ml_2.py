# import the libraries here
# basic
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    #print(f'100% of our data: {len(df)}.')
    #print(f'70% for training data: {len(X_train)}.')
    #print(f'30% for test data: {len(X_test)}.')

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


# load the financial data
def load_data(filename):
    data = pd.read_csv(filename, index_col=0)
    return data

data = load_data("../data/financial_data_ml.csv")
df = data.copy()

# numerical data
num = df._get_numeric_data()
num = num.drop(columns=["year"])

# categorical data
cat = df.drop(columns=num.columns)

# Sidebar Navigation
st.sidebar.title("Menu")
menu = st.sidebar.radio("Select an option: ", ["Business Challenge", 
                                               "Exploratory Data Analysis", 
                                               "Model Testing"])

# business challenge
if menu == "Business Challenge":
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

# eda
if menu == "Exploratory Data Analysis":
    st.title("Explore the Financial Dataset")
    st.write("This is the dataframe that is used for further analysis.")

    # Dataset preview
    st.dataframe(df)

    st.title("Some Descriptive Statistics of the Dataset")

    # Description of the data
    st.dataframe(df.describe().T)

    # Plot the distribution of numerical data

    # Define grid lay   out (3 columns per row)
    num_cols = 3
    num_rows = -(-len(num.columns) // num_cols)  # Ceiling division

    # Create subplots
    fig = sp.make_subplots(
        rows=num_rows, cols=num_cols,
        subplot_titles=num.columns,
        vertical_spacing=0.03, horizontal_spacing=0.1,
        shared_xaxes=False, shared_yaxes=False
    )

    # Plot histograms
    for i, col in enumerate(num.columns):
        row = i // num_cols + 1  # Calculate row index
        col_idx = i % num_cols + 1  # Calculate column index
        
        # Create histogram trace
        hist_trace = go.Histogram(
            x=num[col],
            nbinsx=30,
            name=col,
            marker=dict(color='rgb(246, 112, 137)', line=dict(color='black', width=1)),
            histnorm='',
            opacity=0.8
        )
        
        # Add the trace to the appropriate subplot
        fig.add_trace(hist_trace, row=row, col=col_idx)

    # Update layout
    fig.update_layout(
        title_text=" ",
        showlegend=False,
        height=num_rows * 350,
        width=num_cols * 250,
        template='plotly_white',  # Clean white background
        title_x=0.5  # Center the title
    )

    # Update axes labels and remove the axis ticks for aesthetic purposes
    for i in range(1, num_rows * num_cols + 1):
        fig.update_xaxes(showticklabels=True, row=(i - 1) // num_cols + 1, col=(i - 1) % num_cols + 1)
        fig.update_yaxes(showticklabels=True, row=(i - 1) // num_cols + 1, col=(i - 1) % num_cols + 1)

    # Streamlit title
    st.title('Histograms of the Numerical Data')

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    # SOME LINE PLOTS
    # Columns to plot
    line_columns = ['inflation', 'unemployment', 'gdp', 'm2_money_supp', 'exports',
                    'imports', 'new_home_const', 'wti', 'volume', 'sp_close']

    # Create a Plotly subplot grid
    fig = sp.make_subplots(
        rows=5, cols=2,
        subplot_titles=line_columns,
        vertical_spacing=0.1, horizontal_spacing=0.1,
        shared_xaxes=False, shared_yaxes=False
    )

    # Add a line plot for each column in the dataset
    for i, col in enumerate(line_columns):
        row = i // 2 + 1  # Row index (1-based)
        col_idx = i % 2 + 1  # Column index (1-based)
        
        # Create line trace
        trace = go.Scatter(
            x=df['date'],  # Assuming 'date' column exists in df
            y=num[col],  # Numeric column from df
            mode='lines',
            name=col.replace("_", " ").title(),
            line=dict(color='rgb(246, 112, 137)'),
            showlegend=False  # Disable legend for each trace; we set titles later
            )
        
        # Add the trace to the subplot grid
        fig.add_trace(trace, row=row, col=col_idx)

    # Update layout for dark background
    fig.update_layout(
        title_text=" ",
        title_x=0.5,
        title_font=dict(size=24, color='white'),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),  # White font color for labels and title
        height=1000, width=900,
        showlegend=False,
        template="plotly_dark",  # Use dark theme
    )

    # Update axes labels and make grid lines more visible
    fig.update_xaxes(
        title="", 
        showgrid=True, 
        gridcolor='white', 
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='white', 
        zeroline=False
    )

    # Streamlit title
    st.title('Time Series of Economic Data')

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)


elif menu == "Model Testing":
    st.subheader("Test ML models on the Financial Dataset")

    # clean the data before ml
    df = df.drop(columns=["date", "month", "year", "sp_close_pct_change"])
    df = pd.get_dummies(df, drop_first=True)
    df = df * 1
    
    # move target to the end
    target = df.pop("sp_close")
    df["sp_close"] = target

    # Dictionary of models
    model_dict = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(),
        "Ridge Regression": Ridge(),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "K-Neighbours Regression": KNeighborsRegressor(),
        "Random Forest Regression": RandomForestRegressor(),
        "XGBoost Regression": xgb.XGBRegressor()
    }

    # Select the model using a dropdown
    selected_model = st.selectbox("Choose a regression model", list(model_dict.keys()))

    # Train the selected model and display metrics
    model = {selected_model.split()[0]: model_dict[selected_model]}
    metric = train_linear_models(df, model)

    # Display metrics
    r2_score = round(metric["R2"].values[0], 2)
    rmse = round(metric["RMSE"].values[0], 2)
    st.metric(label=r"$R^2 \ \mathrm{score}: $", value=f"{r2_score}")
    st.metric(label=r"RMSE: ", value=f"{rmse}")
