# 📊 Predicting Financial Market Trend Using Economic Indicators: An Investigation Based On State-of-the-art Machine Learning Models
![](./presentation/stock_market_2.png)
Understanding the financial market is one of the most challenging yet fascinating tasks in economics. This end-to-end data science project explores how key economic indicators—such as GDP, unemployment, and inflation—impact the performance of the stock market, specifically the S&P 500 index. As part of the Ironhack Data Science Bootcamp, this mid-term project blends my passion for financial analysis with cutting-edge machine learning techniques.

## 🎯 Project Goals

- Investigate the relationship between macroeconomic indicators and the stock market.
- Build a comprehensive dataset by collecting and cleaning real-world economic and financial data.
- Conduct detailed exploratory data analysis to uncover hidden patterns.
- Apply state-of-the-art machine learning models to predict S&P 500 trends.
- Evaluate model performance and select the best-performing algorithm.
- Communicate insights through interactive visualizations and a Streamlit app.

## 📚 Data Sources

- **Economic Data:** [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/)
- **Stock Market Data:** [Yahoo Finance](https://finance.yahoo.com/)

## 🛠 Methodology

### 🔹 Data Collection & Wrangling
- Collected macroeconomic and stock market data using FRED API and `yfinance` package.
- Merged time series into a single DataFrame, handled missing values, and engineered new features.
- Final cleaned dataset available in `data_cleaning.ipynb`.

### 🔹 Exploratory Data Analysis (EDA)
- Conducted univariate and bivariate analysis to understand variable relationships.
- Key plots and visual insights presented in `eda.ipynb`.

### 🔹 Machine Learning Modeling
- Applied linear models and tree-based models to predict S&P 500 direction.
- Evaluated performance and selected the best model.
- Implementation and results in `supervised_ml_classification.ipynb`.

### 🔹 Streamlit Web App
- Built an interactive app to visualize model predictions and explore the dataset.
- Run it locally with:
  ```bash
  streamlit run streamlit.py
  ```

## 📈 Key Insights
-  📉 The S&P 500 saw sharp declines during key historical events like the 2008 financial crisis and the COVID-19 pandemic.
-  🔗 Strong correlations exist between economic indicators and market performance — e.g., unemployment rate and housing data.
-  ✅ Linear regression models performed surprisingly well, offering high interpretability and solid accuracy.
-  📊 Visualizations such as heatmaps, time series trends, and scatter plots reveal economic dynamics influencing the stock market.

## Project Structure
```bash
  mid-project-ml-financial-analysis/
  ├── code/             # Jupyter notebooks and Streamlit app
  │   ├── data_cleaning.ipynb
  │   ├── eda.ipynb
  │   ├── supervised_ml_classification.ipynb
  │   └── streamlit.py      # Interactive Streamlit web app
  ├── data/             # Cleaned dataset in CSV format
  ├── plots/            # Visualizations and charts
  ├── presentation/     # Final presentation slides
  ├── requirements.txt  # Python dependencies
  └── README.md         # Project overview and documentation
```

## 📦 Getting Started
1. Clone the Repository
```bash
git clone https://github.com/abhishek-chattopadhyay/-mid-project-ml-financial-analysis.git
cd mid-project-ml-financial-analysis
```
2. Install Dependencies
  - It’s recommended to use a virtual environment: 
  ```bash
    pip install -r requirements.txt
  ```
3. Launch the App:
  ```bash
    streamlit run streamlit.py
  ```

## 💬 Extra Notes
- This project was developed as a mandatory mid-term project for the Ironhack Data Science Bootcamp..
- The code is modular, readable, and can be reused or extended for similar financial prediction problems.
- Includes comprehensive Jupyter notebooks, ready-to-use data, and a web app.

Feel free to ⭐️ the repo if you find it useful or open an issue if you’d like to collaborate or suggest improvements!
