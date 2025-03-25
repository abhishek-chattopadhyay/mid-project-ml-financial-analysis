# financial data collection
from fredapi import Fred
import yfinance as yf

# get stock price data
def get_ticker_data(ticker_symbol, plot=False):
    data = yf.Ticker(ticker_symbol)
    ticker = data.history(period='max')
    ticker.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
    ticker.reset_index(inplace=True)

    for i in range(len(ticker['Date'])):
        ticker.iloc[i, 0] = ticker.iloc[i, 0].date()

    if plot:
        plt.plot(np.arange(0, len(ticker['Date'])), ticker['Open'])
        plt.show()

    return ticker

def get_fred_data(symbol):
    f = open("../fred_api_key.dat", "r")
    fred_api = f.read()
    fred = Fred(api_key=fred_api)
    symbol_data = fred.get_series(symbol)
    return symbol_data