# Predicting Financial Market Trend Using Economic Indicators: An Investigation Based On State-of-the-art Machine Learning Models
THe world of finance is fascinating. Tracking and understanding the financial market is one of the most complex tasks. I am passionate about the economics and financial market. One of my hobbies is to understand how economic indicators affect the financial market, especially, the stock market. In this project, I have delved deep into this problem and try to find an answer. This project is a complete end-to-end data analysis project needed for Ironhack bootcamp.

## Objective of this project
- To understand how economic indicators such as GDP, unemployment, inflation etc. affect S&P500.
- To collect relevant data, clean them and make a dataset for analysis
- To perform univariate and bivariate analysis between numerical and categorical data
- To apply state-of-the-art machine learning models
- To find the best model and the prediction
- To present the result to the stakeholders

## Data Sources
- Economic indicators: [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/)
- Stock market data: [Yahoo finance](https://finance.yahoo.com/)

## Methodology
- Data Collection & Data Wrangling:
  - Using FRED API and python yfinance module to collect data
  - Merging all the series data into a dataframe
  - Creating relevant categorical and numerical columns
  - Replacing NaN values
  - Creating a clean dataframe for further analysis
  - This can be found in the `data_cleaning.ipynb` notebook
- Exploratory Data Analysis(EDA):
  - Univariate analysis
  - Bivariate analysis
  - This is presented in the `eda.ipynb` notebook
- Training State-of-the-art Machine Learning Models
  - Applied linear and tree based machine learning models to predict S&P500
  - Calculated the accuracy of the models and the best model is chosen 
  - This is given in the `supervised_ml_classification.ipynb`
- Streamlit app
  - You can interact with the models using the Streamlit app
  - To run this, use: `streamlit run streamlit.py`
## Results
### Key insights
  1. Univariate analysis shows historical events such as 2008 financial crisis and 2019 Covid-19 pandemic when S&P500 crashed more than 10% in a month
  2. Multiple correlations can be found between important economic indicators and S&P500 from the correlation matrix which shows how economic indicators affect the stock market.
  3. Different other correlations can be found in the notebook (e.g. construction of new homes goes down when unemployment rate increases).
  4. Linear regression models work very well for this dataset
## Extra materials
- A detailed Jupyter notebook can be found in the [code](/code) folder.
- Relevant plots can be found in the [plot](/plot) folder.
- A ready-to-work-on data file (csv format) can be found in the [data](/data) folder.
- The final presentation is added to the [presentation](/presentation) folder.
