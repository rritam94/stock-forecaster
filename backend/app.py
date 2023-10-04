from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from flask_cors import cross_origin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime as dt
import yfinance as yf
from datetime import datetime
import numpy as np
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

symbol = 'TSLA'
y_close = None
next_day_op = 0
next_day_cl = 0
y_close_dates = []
y_close_prices = []
y_open_prices = []
t = 0

app = Flask(__name__)
CORS(app, origins='http://localhost:3000')

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

def calculate_sentiment(symbol: str) -> float:
    # find percent change from prediction to previous close
    mean = 0.0
    news = requests.get('https://www.benzinga.com/quote/' + symbol)
    headlines = parse_webpage(news.text)

    for headline in headlines:
        mean = mean + analyzer.polarity_scores(headline[1 : len(headline)])['compound']

    mean = mean/len(headlines)
    return mean

def parse_webpage(news: str) -> list[str]:
    headlines = []
    headline_indices = []
    headline_index = 0

    date_indices = []
    date_index = 0
    idx = 0

    while True:
        headline_index = news.find('noopener noreferrer', headline_index + 1)
        date_index = news.find('"text-gray-500">', date_index + 1)

        if headline_index == -1:
            break

        if date_index == -1:
            break

        headline_indices.append(headline_index)
        date_indices.append(date_index)

    for index in headline_indices:
        headline = ''
        curr_index = index

        # gets the date element only (3 days ago, 6/21/2023, etc)
        if news[date_indices[idx] + 16 : date_indices[idx] + 30].find('3 day') != -1:
            break

        while (news[curr_index] != '>'):
            curr_index = curr_index + 1

        while (news[curr_index] != '<'):
            headline = headline + news[curr_index]
            curr_index = curr_index + 1

        headlines.append(headline)
        idx = idx + 1

    return headlines

def get_fused_score(current_price: float, predicted_price: float, sentiment_score: float) -> float:
    max_delta = 0.0155 * predicted_price + 1.9 # max delta in price in one trading day
    percent_change = (predicted_price - current_price)/max_delta # percent change based on max delta
    print(((0.9 * percent_change) + (0.1 * sentiment_score)))
    print('delta: ', max_delta)

    return current_price + (((0.9 * percent_change) + (0.1 * sentiment_score)) * max_delta)

def prepredict():
    global next_day_price_op, next_day_price_cl, y_close_dates, y_close_prices, y_open_prices, volatility, dividend_yield, market_cap, volume, eps, price_earning_ratio, low_price, high_price 

    #df operations
    ticker = yf.Ticker(symbol)
    # df = yf.download(stock_symbol, dt.datetime(2010, 1, 1), dt.datetime.now())
    df = yf.download('TSLA', dt.datetime(2023, 1, 1), dt.datetime.now())
    df = df.drop('Adj Close', axis = 1)
    X_open = df.drop('Open', axis = 1)  
    y_open = df['Open']  
    X_close = df.drop('Close' , axis = 1)
    y_close = df['Close']
    
    #converting tolist & getting date array
    y_close_dates = [datetime.strftime(date, "%Y") for date in y_close.index.tolist()]
    y_close_prices = y_close.values.tolist()
    y_open_prices = y_open.values.tolist()

    #calculate volatility - ytd (standard deviation * sqrt of ytd days)
    stock_returns = df['Close'].pct_change()
    volatility = round(np.std(stock_returns) * np.sqrt(252) * 100, 3) #calculating standard deviation and taking into account the number of trading days per year (ytd)

    #calculate market cap with the formula of outstanding shares * current stock price
    market_cap = ticker.info['marketCap']

    #retrieve volume from df
    volume = ticker.info['volume']

    #retrieve eps from api
    eps = round(ticker.info['trailingEps'], 2)

    #retrieve dividend per yield from api
    if 'dividendYield' in ticker.info:
        dividend_yield = round(ticker.info['dividendYield'] * 100,2)
    else:
        dividend_yield = 'N/A'

    #calculate price to earning ratio
    price_earning_ratio = round(ticker.info['currentPrice']/eps, 2)

    # #get low and high for today
    low_price = round(ticker.history('1d')['Low'].iloc[-1], 2)
    high_price = round(ticker.history('1d')['High'].iloc[-1], 2)

    # split the data training and testing sets
    X_train_op, X_test_op, y_train_op, y_test_op = train_test_split(X_open, y_open, test_size = 0.2, random_state = 42)
    X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_close, y_close, test_size = 0.2, random_state = 42)

    # training model using lin reg
    model_op = LinearRegression()
    model_op.fit(X_train_op, y_train_op)

    model_cl = LinearRegression()
    model_cl.fit(X_train_cl, y_train_cl)

    # predicting next day price
    df2 = df
    last_data_op = df.tail(2).drop('Open', axis = 1)
    next_day_price_op = model_op.predict(last_data_op)

    last_data_cl = df2.tail(1).drop(['Close'], axis=1)
    last_data_cl['Open'] = next_day_price_op[0]
    next_day_price_cl = model_cl.predict(last_data_cl)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # post req 
    data = request.get_json()
    symbol = data['stock_symbol']

    #df operations
    ticker = yf.Ticker(symbol)
    # df = yf.download(stock_symbol, dt.datetime(2010, 1, 1), dt.datetime.now())
    df = yf.download(symbol, dt.datetime(2023, 7, 1), dt.datetime.now())
    df = df.drop('Adj Close', axis = 1)
    X_open = df.drop('Open', axis = 1)  
    y_open = df['Open']  
    X_close = df.drop('Close', axis = 1)
    y_close = df['Close']
    
    #converting tolist & getting date array
    y_close_dates = [datetime.strftime(date, "%Y") for date in y_close.index.tolist()]
    y_close_prices = y_close.values.tolist()
    y_open_prices = y_open.values.tolist()

    #calculate volatility - ytd (standard deviation * sqrt of ytd days)
    stock_returns = df['Close'].pct_change()
    volatility = round(np.std(stock_returns) * np.sqrt(252) * 100, 3) #calculating standard deviation and taking into account the number of trading days per year (ytd)

    #calculate market cap with the formula of outstanding shares * current stock price
    if 'marketCap' in ticker.info:
        market_cap = ticker.info['marketCap']
    else:
        market_cap = 'N/A'

    #retrieve volume from df
    volume = ticker.info['volume']

    #retrieve eps from api
    if 'trailingEps' in ticker.info:
        eps = round(ticker.info['trailingEps'], 2)
    else:
        eps = 'N/A'

    #retrieve dividend per yield from api
    if 'dividendYield' in ticker.info:
        dividend_yield = round(ticker.info['dividendYield'] * 100,2)
    else:
        dividend_yield = 'N/A'

    #calculate price to earning ratio
    if 'currentPrice' in ticker.info:
        price_earning_ratio = round(ticker.info['currentPrice']/eps, 2)
    else:
        price_earning_ratio = 'N/A'

    #get low and high for today
    low_price = round(ticker.history('1d')['Low'].iloc[-1], 2)
    high_price = round(ticker.history('1d')['High'].iloc[-1], 2)

    # split the data training and testing sets
    X_train_op, X_test_op, y_train_op, y_test_op = train_test_split(X_open, y_open, test_size = 0.2, random_state = 42)
    X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_close, y_close, test_size = 0.2, random_state = 42)

    # training model using lin reg
    model_op = LinearRegression()
    model_op.fit(X_train_op, y_train_op)

    model_cl = LinearRegression()
    model_cl.fit(X_train_cl, y_train_cl)

    test_predictions_op = model_op.predict(X_test_op)
    test_predictions_cl = model_cl.predict(X_test_cl)

    train_predictions_op = model_op.predict(X_train_op)
    train_rmse_op = mean_squared_error(y_train_op, train_predictions_op, squared=False)
    test_predictions_op = model_op.predict(X_test_op)
    test_rmse_op = mean_squared_error(y_test_op, test_predictions_op, squared=False)

    train_predictions_cl = model_cl.predict(X_train_cl)
    train_rmse_cl = mean_squared_error(y_train_cl, train_predictions_cl, squared=False)
    test_predictions_cl = model_cl.predict(X_test_cl)
    test_rmse_cl = mean_squared_error(y_test_cl, test_predictions_cl, squared=False)

    print('Open Test RMSE:', test_rmse_op)
    print('Open Train RMSE:', train_rmse_op)

    print('Close Test RMSE:', test_rmse_cl)
    print('Close Train RMSE:', train_rmse_cl)
    
    # predicting next day price
    df2 = df
    last_data_op = df.tail(2).drop('Open', axis = 1)
    next_day_price_op = model_op.predict(last_data_op)

    last_data_cl = df2.tail(1).drop(['Close'], axis=1)
    last_data_cl['Open'] = next_day_price_op[0]
    next_day_price_cl = model_cl.predict(last_data_cl)

    # fusing sentiment & linear regression
    sentiment_score = calculate_sentiment(symbol)
    fused_open = get_fused_score(ticker.info['currentPrice'], next_day_price_op[0], sentiment_score)
    fused_close = get_fused_score(ticker.info['currentPrice'], next_day_price_cl[0], sentiment_score)

    return jsonify({
        'next_day_open': round(fused_open,2),
        'next_day_close': round(fused_close,2),
        'dates': y_close_dates,
        'pricesop': y_open_prices,
        'pricescl': y_close_prices,
        'volatility': volatility,
        'dividend_yield': dividend_yield,
        'market_cap': market_cap,
        'volume': volume,
        'eps': eps,
        'pe_ratio': price_earning_ratio,
        'low': low_price,
        'high': high_price
    })

@app.route('/prepredict', methods=['POST'])
@cross_origin()
def default():
    global next_day_op, next_day_cl, y_close_dates, y_close_prices, y_open_prices
    return jsonify({'next_day_open': round(next_day_price_op[0].item(), 2),
                    'next_day_close': round(next_day_price_cl[0].item(), 2),
                    'dates': y_close_dates,
                    'pricesop': y_open_prices,
                    'pricescl': y_close_prices,
                    'volatility': volatility,
                    'dividend_yield': dividend_yield,
                    'market_cap': market_cap,
                    'volume': volume,
                    'eps': eps,
                    'pe_ratio': price_earning_ratio,
                    'low': low_price,
                    'high': high_price
                   })

if __name__ == '__main__':
    prepredict()
    app.run()
