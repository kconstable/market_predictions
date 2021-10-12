import pandas as pd
import numpy as np
import requests
import time
import datetime
from tabulate import tabulate
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sys import exit
from fbprophet import Prophet


def get_economic_indicators(funct, key, interval=None, maturity=None, throttle=0):
    """
    Returns Economic Indicator Data with missing values interpolated between dates
    Monthly Data:
      NONFARM_PAYROLL, INFLATION_EXPECTATION,CONSUMER_SENTIMENT,UNEMPLOYMENT
    Daily, Weekly, Monthly Data:
      FEDERAL_FUNDS_RATE = interval (daily,weekly,monthly)
      TREASURY_YIELD = interval (daily, weekly, monthly),
                       maturity (3month, 5year, 10year, and 30year)
    """

    # query strings
    # Monthly Data:
    if funct in ['NONFARM_PAYROLL', 'INFLATION_EXPECTATION', 'CONSUMER_SENTIMENT', 'UNEMPLOYMENT']:
        url = f'https://www.alphavantage.co/query?function={funct}&apikey={key}'

    # Daily, Weekly or Monthly Data:
    # Interest Rates
    if funct == 'FEDERAL_FUNDS_RATE':
        url = f'https://www.alphavantage.co/query?function={funct}&interval={interval}&apikey={key}'

    # Treasury Yield
    if funct == 'TREASURY_YIELD':
        url = f'https://www.alphavantage.co/query?function={funct}&interval={interval}&maturity={maturity}&apikey={key}'

    # pull data
    r = requests.get(url)
    time.sleep(throttle)
    d = r.json()

    # convert to df
    df = pd.DataFrame(d['data'])

    # move date to a datetime index
    df.date = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)

    # add the ticker name and frequency
    df['name'] = d['name']
    df['interval'] = d['interval']

    # clean data & interpolate missing values
    # missing data encoded with '.'
    # change datatype to float
    df.replace('.', np.nan, inplace=True)
    df.value = df.value.astype('float')

    # missing data stats
    missing = sum(df.value.isna())
    total = df.shape[0]
    missing_pct = round(missing / total * 100, 2)

    # interpolate using the time index
    if missing > 0:
        df.value.interpolate(method='time', inplace=True)
        action = 'interpolate'
    else:
        action = 'none'

    # Print the results
    if maturity is not None:
        summary = ['Economic Indicator', funct + ':' + maturity, str(total), str(missing), str(missing_pct) + '%',
                   action]
    else:
        summary = ['Economic Indicator', funct, str(total), str(missing), str(missing_pct) + '%', action]

    return {'summary': summary, 'data': df}


def get_technical_indicators(symbol, funct, key, interval, time_period=None, throttle=0):
    """
  Returns Technical Indicators (only works for stocks, not cyrpto)
      MACD:   symbol,interval
      RSI:    symbol,interval,time_period
      BBANDS: symbol,interval,time_period

  Parameters:
          interval: (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
          :param symbol: stock
          :param funct: the technical indicator name
          :param key: the AV key
          :param interval:  day, week,
          :param throttle: number of seconds to wait between API requests
          :param time_period: day, week,month
  """
    # build the query string
    if funct == 'MACD':
        url = f'https://www.alphavantage.co/query?function={funct}&symbol={symbol}&interval={interval}&series_type' \
              f'=close&apikey={key} '
    if funct in ['RSI', 'BBANDS']:
        url = f'https://www.alphavantage.co/query?function={funct}&symbol={symbol}&interval={interval}&series_type' \
              f'=close&time_period={time_period}&apikey={key} '

    # request data as json, convert to dict, pause request to avoid the data throttle
    r = requests.get(url)
    time.sleep(throttle)
    d = r.json()

    # extract to a df, add the indicator name, convert the index to datetime
    df = pd.DataFrame(d[f'Technical Analysis: {funct}']).T
    df.index = pd.to_datetime(df.index)

    # convert the data to float
    for col in df.columns:
        df[col] = df[col].astype('float')

    # check for missing data
    missing = df.isnull().any().sum()
    total = len(df)
    missing_pct = round(missing / total * 100, 2)

    # Print the results
    summary = ['Technical Indicator', funct, str(total), str(missing), str(missing_pct) + '%', 'none']

    return {'summary': summary, 'data': df}


def get_crypto_data(symbol, key):
    """
  Pulls daily crypto prices from alpha advantage.
  Inputs:
    symbol: ETH, BTC, DOGE
    key:    The alpha advantage API key
  Output:
    a dataframe of crypto prices: open,high,low, close, volume
  """
    # build query string, get data as json and convert to a dict
    url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market=CAD&apikey={key}'
    r = requests.get(url)
    d = r.json()

    # extract data to df
    df = pd.DataFrame(d['Time Series (Digital Currency Daily)']).T

    # remove columns not required
    # returns the price in two currencies, just keep USD
    cols = [c for c in df.columns if '(CAD)' not in c]
    df = df.loc[:, cols]
    df.columns = ['open', 'high', 'low', 'close', 'volume', 'marketcap']
    df.drop(['marketcap'], axis=1, inplace=True)

    # change data types
    df.index = pd.to_datetime(df.index)

    # convert datatype to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype('float')

    # add the cyrpto name
    df['symbol'] = d['Meta Data']['3. Digital Currency Name']

    return df


def calc_bollinger(df, feature, window=20, st=2):
    """
  Calculates bollinger bands for a price time-series.  Used for crypto currencies
  Input:
    df     : A dataframe of time-series prices
    feature: The name of the feature in the df to calculate the bands for
    window : The size of the rolling window.  Defaults to 20 days with is standard
    st     : The number of standard deviations to use in the calculation. 2 is standard
  Output:
    Returns the df with the bollinger band columns added
  """

    # rolling mean and stdev
    rolling_m = df[feature].rolling(window).mean()
    rolling_st = df[feature].rolling(window).std()

    # add the upper/lower and middle bollinger bands
    df['b-upper'] = rolling_m + (rolling_st * st)
    df['b-middle'] = rolling_m
    df['b-lower'] = rolling_m - (rolling_st * st)


def calc_rsi(df, feature='close', window=14):
    """
    Calculates the RSI for the input feature
    Input:
      df      : A dataframe with a time-series of prices
      feature : The name of the feature in the df to calculate the bands for
      window  : The size of the rolling window.  Defaults to 14 days which is standard
    Output:
      Returns the df with the rsi band column added
    """
    # RSI
    # calc the diff in daily prices, exclude nan
    diff = df[feature].diff()
    diff.dropna(how='any', inplace=True)

    # separate positive and negitive changes
    pos_m, neg_m = diff.copy(), diff.copy()
    pos_m[pos_m < 0] = 0
    neg_m[neg_m > 0] = 0

    # positive/negative rolling means
    prm = pos_m.rolling(window).mean()
    nrm = neg_m.abs().rolling(window).mean()

    # calc the rsi and add to the df
    ratio = prm / nrm
    rsi = 100.0 - (100.0 / (1.0 + ratio))
    df['rsi'] = rsi


def calc_macd(df, feature='close'):
    """
  Calculates the MACD and signial for the input feature
  Input:
    df      : A dataframe with a time-series of prices
    feature : The name of the feature in the df to calculate the bands for
  Output:
    Returns the df with the macd columns added
  """
    ema12 = df[feature].ewm(span=12, adjust=False).mean()
    ema26 = df[feature].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()


def get_ticker_data(symbol, key, outputsize='compact', throttle=0):
    """
  Returns daily data for a stock (symbol)
    outputsize: compact(last 100) or full (20 years)
    key: apikey
    symbols: OILK (oil ETF),BAR(gold ETF),VXZ (volatility ETF)
  """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={key}'
    r = requests.get(url)
    time.sleep(throttle)
    d = r.json()

    # extract data to a df
    df = pd.DataFrame(d['Time Series (Daily)']).T
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df['symbol'] = d['Meta Data']['2. Symbol']

    # change data types
    df.index = pd.to_datetime(df.index)

    # convert datatype to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype('float')

    # Calculate missing data
    missing = sum(df.close.isna())
    total = df.shape[0]
    missing_pct = round(missing / total * 100, 2)

    # Print the results
    summary = ['Ticker', symbol, str(total), str(missing), str(missing_pct) + '%', 'none']

    return {'summary': summary, 'data': df}


def get_consolidated_stock_data(symbol, key, config, outputsize='compact', throttle=30, dropna=True):
    """
    Pulls data from alpha advantage and consolidates
    API Limitations: 5 API requests per minute and 500 requests per day
    Inputs:
      symbol: stock ticker
      key   : api key
      config: dictionary which lists the economic, technical and commodities to pull
      outputsize: compact(latest 100) or full (up to 20 years of daily data)
      throttle: number of seconds to wait between api requests
      dropna: True/False, drops any records with nan
    Output:
      A dataframe with consolidated price data for the symbol + economic/technical
      indicators and commodity prices
    """

    # Result header and accumulator
    header = ['Type', 'Data', 'Total', 'Missing', ' % ', 'Action']
    summary = []

    # Get stock prices
    try:
        results = get_ticker_data(symbol, key, outputsize, 0)
        dff = results['data']
        summary.append(results['summary'])
        print(f'Complete:===>Ticker:{symbol}')
    except:
        print(f'Error:===>Ticker:{symbol}')

    # Get Commodity prices
    # ****************************************************************************
    for commodity in config['Commodities']:
        try:
            # get prices
            results = get_ticker_data(commodity, key, outputsize, throttle)
            df = results['data']
            summary.append(results['summary'])
            print(f'Complete:===>Commodity:{commodity}')

            # rename close to commodity name, remove unneeded columns and join with
            # the stock prices by date
            df.rename(columns={'close': commodity}, inplace=True)
            df.drop(['open', 'high', 'low', 'volume', 'symbol'], axis=1, inplace=True)
            dff = dff.join(df, how='left')
        except:
            print(f"Error===>Commodity:{commodity}")

    # Economic Indicators
    # ****************************************************************************
    # loop through the config to pull the requested data
    for indicator, values in config['Economic'].items():
        if indicator == 'TREASURY_YIELD':
            for tr in values:
                try:
                    results = get_economic_indicators(indicator, key, interval=tr['interval'], maturity=tr['maturity'],
                                                      throttle=throttle)
                    summary.append(results['summary'])
                    print(f"Complete:===>{indicator}:{tr['maturity']}")

                    df = results['data']
                    dff = dff.join(df, how='left')
                    dff.rename(columns={"value": tr['name']}, inplace=True)
                    dff.drop(['name', 'interval'], axis=1, inplace=True)
                except:
                    print(f"Error===>{indicator}:{tr['maturity']}")

        else:
            # daily
            if values['interval'] == 'daily':
                try:
                    results = get_economic_indicators(indicator, key, interval=values['interval'], throttle=throttle)
                    df = results['data']
                    summary.append(results['summary'])
                    print(f"Complete:===>{indicator}")

                    dff = dff.join(df, how='left')
                    dff.rename(columns={"value": values['name']}, inplace=True)
                    dff.drop(['name', 'interval'], axis=1, inplace=True)
                except:
                    print(f"Error===>{indicator}")

            else:
                try:
                    # monthly or weekly
                    results = get_economic_indicators(indicator, key, throttle=throttle)
                    summary.append(results['summary'])
                    df = results['data']
                    print(f"Complete:===>{indicator}")

                    # reindex to daily, fill missing values forward
                    days = pd.date_range(start=min(df.index), end=max(df.index), freq='D')
                    df = df.reindex(days, method='ffill')

                    # join with the other data
                    dff = dff.join(df, how='left')
                    dff.rename(columns={"value": values['name']}, inplace=True)
                    dff.drop(['name', 'interval'], axis=1, inplace=True)
                except:
                    print(f"Error===>{indicator}")

    # # Technical Indicators
    # ****************************************************************************
    for indicator, values in config['Technical'].items():
        try:
            results = get_technical_indicators(symbol, indicator, key, values['interval'], values['time_period'],
                                               throttle)
            df = results['data']
            summary.append(results['summary'])

            dff = dff.join(df, how='left')
            print(f"Complete:===>{indicator}")
        except:
            print(f"Error===>{indicator}")

    # clean column names
    dff.rename(columns={"Real Upper Band": 'b-upper',
                        "Real Lower Band": 'b-lower',
                        "Real Middle Band": "b-middle",
                        "RSI": "rsi",
                        "MACD_Hist": "macd_hist",
                        "MACD_Signal": "macd_signal",
                        "MACD": "macd"
                        }, inplace=True)

    # Fill in any missing data after joining all datasets
    dff.fillna(method='bfill', inplace=True, axis=0)

    # drop rows with missing commodity prices
    if dropna:
        dff.dropna(how='any', inplace=True)

    # print the results table
    print("\n\n")
    print(tabulate(summary, header))

    return dff


def get_consolidated_crypto_data(symbol, key, config, boll_window=20, boll_std=2, rsi_window=14, throttle=30,
                                 dropna=True):
    """
    Pulls data from alpha advantage and consolidates
    API Limitations: 5 API requests per minute and 500 requests per day
    Inputs:
      symbol: crypto ticker
      key   : api key
      config: dictionary which lists the economic indicators and commodities to pull
      throttle: number of seconds to wait between api requests
      dropna: True/False, drops any records with nan
    Output:
      A dataframe with consolidated price data for the symbol + economic/technical
      indicators and commodity prices
    """

    # Result header and accumulator
    header = ['Type', 'Data', 'Total', 'Missing', ' % ', 'Action']
    summary = []

    # Get crypto prices
    try:
        dff = get_crypto_data(symbol, key)

        # add month feature
        dff['month'] = dff.index.month

        print(f'Complete:===>Crypto:{symbol}')

    except:
        print(f'Error:===>Crypto:{symbol}')

    # Get Commodity prices
    # ****************************************************************************
    for commodity in config['Commodities']:
        try:
            # get prices
            results = get_ticker_data(commodity, key, 'full', throttle)
            df = results['data']
            summary.append(results['summary'])
            print(f'Complete:===>Commodity:{commodity}')

            # rename close to commodity name, remove unneeded columns and join with
            # the stock prices by date
            df.rename(columns={'close': commodity}, inplace=True)
            df.drop(['open', 'high', 'low', 'volume', 'symbol'], axis=1, inplace=True)
            dff = dff.join(df, how='left')
        except:
            print(f"Error===>Commodity:{commodity}")

    # Economic Indicators
    # ****************************************************************************
    # loop through the config to pull the requested data
    for indicator, values in config['Economic'].items():
        if indicator == 'TREASURY_YIELD':
            for tr in values:
                try:
                    results = get_economic_indicators(indicator, key, interval=tr['interval'], maturity=tr['maturity'],
                                                      throttle=throttle)
                    summary.append(results['summary'])
                    print(f"Complete:===>{indicator}:{tr['maturity']}")

                    df = results['data']
                    dff = dff.join(df, how='left')
                    dff.rename(columns={"value": tr['name']}, inplace=True)
                    dff.drop(['name', 'interval'], axis=1, inplace=True)
                except:
                    print(f"Error===>{indicator}:{tr['maturity']}")

        else:
            # daily
            if values['interval'] == 'daily':
                try:

                    results = get_economic_indicators(indicator, key, interval=values['interval'], throttle=throttle)
                    df = results['data']
                    summary.append(results['summary'])
                    print(f"Complete:===>{indicator}")

                    dff = dff.join(df, how='left')
                    dff.rename(columns={"value": values['name']}, inplace=True)
                    dff.drop(['name', 'interval'], axis=1, inplace=True)
                except:
                    print(f"Error===>{indicator}")

            else:
                try:
                    # monthly or weekly

                    results = get_economic_indicators(indicator, key, throttle=throttle)
                    summary.append(results['summary'])
                    df = results['data']
                    print(f"Complete:===>{indicator}")

                    # reindex to daily, fill missing values forward
                    days = pd.date_range(start=min(df.index), end=max(df.index), freq='D')
                    df = df.reindex(days, method='ffill')

                    # join with the other data
                    dff = dff.join(df, how='left')
                    dff.rename(columns={"value": values['name']}, inplace=True)
                    dff.drop(['name', 'interval'], axis=1, inplace=True)
                except:
                    print(f"Error===>{indicator}")

    # # Technical Indicators
    # ****************************************************************************
    calc_rsi(dff, 'close', rsi_window)
    calc_bollinger(dff, 'close', boll_window, boll_std)
    calc_macd(dff, 'close')

    # Fill in any missing data after joining all datasets
    dff.fillna(method='bfill', inplace=True, axis=0)

    # drop rows with missing commodity prices
    if dropna:
        dff.dropna(how='any', inplace=True)

    # print the results table
    print("\n\n")
    print(tabulate(summary, header))

    return dff


def transform_stationary(df, features_to_transform, transform='log', verbose=False):
    """
  Transform time-series data using a log or boxcox transform.  Calculate the augmented
  dickey-fuller (ADF) test for stationarity after the transform
  Inputs:
    df: a dataframe of features
    features_to_transform: A list of features to apply the transform
    transform: The transform to apply (log, boxbox)
  Output
    Applies the transforms inplace in df
  """
    # transform each column in the features_to_transform list
    for feature in df.columns:
        if feature in features_to_transform:
            # log transform
            if transform == 'log':
                df[feature] = df[feature].apply(np.log)

            # boxcox transform
            elif transform == 'boxcox':
                bc, _ = stats.boxcox(df[feature])
                df[feature] = bc

            else:
                print("Transformation not recognized")
    if verbose:
        # check the closing price for stationarity using the augmented dicky fuller test
        t_stat, p_value, _, _, critical_values, _ = adfuller(df.close.values, autolag='AIC')
        print('\n\nTransforming Data:Augmented Dicky Fuller Test for Stationarity')
        print("=" * 60)
        print(f'ADF Statistic: {t_stat:.2f}')
        for key, value in critical_values.items():
            print('Critial Values:')
            if t_stat < value:
                print(f'   {key}, {value:.2f} => non-stationary')
            else:
                print(f'   {key}, {value:.2f} => stationary')


def shift_features(df, features_shift):
    """
  Shifts features by time periods to convert them to lagged indicators
  Input:
    df: dataframe of features
    features_shift: dictionary  of {feature:period shift}
  Output:
    df: original dataframe + the shifted features
  """
    dff = df.copy()
    for feature, shift in features_shift.items():
        t_shift = pd.DataFrame(dff[feature].shift(periods=shift))
        dff = dff.join(t_shift, how='left', rsuffix='_shift')

    # remove nan introducted with lag features
    dff.dropna(how='any', inplace=True)

    return dff


def prepare_data(df, n_steps, features=[], verbose=False):
    """
  Filter, scale and convert dataframe data to numpy arrays

  Inputs:
    df       => A dataframe of observations with features and y-labels
    y        => The name of the column that is the truth labels
    features => A list of features.  Used to subset columns

  Outputs:
    scaled_y => numpy array of the y-label data
    scaled_x => numpy array of the training features

  """

    # subset the latest n_steps rows to be used for prediction
    df = df.iloc[0:n_steps, :]

    # reverse the index such that dates are in chronological order
    df = df.iloc[::-1]

    # Subset features, get the y-label values
    df_y = df['close']
    df_X = df[features]

    # replace the date index with an integer index
    idx_dates = df.index
    df_X.reset_index(drop=True, inplace=True)

    # convert to numpay arrays
    array_X = np.array(df_X)
    array_y = np.array(df_y).reshape(-1, 1)

    if verbose:
        # print the output
        print("\nPreparing Data")
        print("=" * 60)
        print(f"=> {len(features)} Features")
        print(f"=> Input Dimensions :{array_X.shape}")
        print("\n")

    return idx_dates, array_y, array_X


def get_prophet_df(df):
    """
  Convert a dataframe into prophet format
  Input:
    Dataframe of prices
  Output:
    dataframe matching the prohet requirements of datestamp,
    target variable(ds,y)
  """

    df_prophet = df[['close']].copy()
    df_prophet.reset_index(inplace=True)
    df_prophet.rename(columns={'index': 'ds', 'close': 'y'}, inplace=True)

    return df_prophet


def create_prophet_model(df):
    """
  Creates and fits a FB Prophet model on the input time-series
  Input:
    df: a dataframe containing the closing price time-series
  Output:
    a trained prophet model
  """
    # convert the data into prophet format
    df_prophet = get_prophet_df(df)

    # init the prophet model and fit to the dataset
    # m = Prophet(daily_seasonality=False,growth='logistic')
    m = Prophet(daily_seasonality=False)
    m.fit(df_prophet)

    return m


def get_prophet_forecast(df, periods=3, visualize=False):
    """
  Predicts the closing price using FB Prophat algorithm for the input
  number of periods
  Input:
    df: a dataframe of closing prices
    peirods: the number of periods to predict
    visualize: boolen, determines if a plot should be displayed
  Output:
    a dataframe of predcited closing prices
  """

    # get lastest date
    latest_date = df.index.max()

    # convert the data into prophet format
    df_prophet = get_prophet_df(df)

    # init the prophet model and fit to the dataset
    model = Prophet(daily_seasonality=False)
    model.fit(df_prophet)

    # create a df to hold the predictions
    future = model.make_future_dataframe(periods=periods)

    # forecast the data
    forecast = model.predict(future[future['ds'] > latest_date])

    # show history + forcast
    if visualize:
        model.plot(forecast)

    # subset rows/cols
    # df_subset = forecast[forecast['ds']>latest_date]
    df_subset = forecast.loc[:, ('ds', 'yhat', 'yhat_lower', 'yhat_upper')]

    return df_subset


def make_ensemble_predictions(df, lstm_model, scaler, scaled_X, n_steps, n_features, n_pred, start_date, weights,
                              stock_type):
    """
  Predict the next n_pred days with n_steps of daily data
  Input:
    model: A trained LSTM model
    scaler: The scaler used
    scaled_X: scaled input features
    n_steps: the number of input days used in the model
    n_features: the number of features used in the model
    n_pred: the number of days predicted in the model
    start_date: the start date of the prediction window
  Output:
    a data frame of predicted prices
  """
    # LSTM Prediction
    # Predict the prices
    y_pred_scaled = lstm_model.predict(scaled_X.reshape(1, n_steps, n_features))

    # convert units back to the original scale
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)

    # convert from log transform back to original scale
    y_pred_np = np.exp(y_pred_unscaled)

    # set the date index
    # crypto trades 24/7
    if stock_type == 'stock':
        pred_dates = pd.date_range(start_date + datetime.timedelta(days=1), periods=n_pred, freq='B').tolist()
    else:
        pred_dates = pd.date_range(start_date + datetime.timedelta(days=1), periods=n_pred, freq='D').tolist()

    # convert to dataframe
    lstm = pd.DataFrame(y_pred_np.T, columns=['lstm'])
    lstm['actual'] = np.nan
    lstm['date'] = pred_dates
    lstm.set_index(['date'], inplace=True)
    lstm.index = pd.to_datetime(lstm.index)
    lstm = lstm[['actual', 'lstm']]
    print('lstm-ok')

    # FB Prophat Prediction
    fbp = get_prophet_forecast(df, n_pred, False)
    fbp['yhat'] = np.exp(fbp['yhat'])

    # Get last closing price
    last_close = np.exp(df.loc[df.index.max()]['close'])
    last_close_lst = [last_close for p in range(n_pred)]

    # create ensemble prediction
    df_ensemble = pd.DataFrame(list(zip(lstm.lstm, fbp.yhat, last_close_lst)), columns=['lstm', 'fbp', 'last_close'])
    df_ensemble['date'] = lstm.index
    df_ensemble.set_index('date', inplace=True)
    df_ensemble['ens'] = df_ensemble['last_close'] * weights['last_close'] + df_ensemble['lstm'] * weights['lstm'] + \
                         df_ensemble['fbp'] * weights['fbp']

    return df_ensemble[['lstm', 'fbp', 'last_close', 'ens']]


def roll_ensemble_predictions(df_new, df_pred=None, df_hist=None):
    """
  Updates previous predicted prices with actual prices, and adds
  the next n_pred prediction window
  Input:
    df_new: The new dataset of inputs
    df_pred: A dataframe of predicted prices
    df_hist: A dataframe that stores the actual/predicted prices
            (the output of this function)
  Output:
    A dataframe of up to date prices with the next prediction window.
    Incluces the daily and cumulative prediction error
  """
    # First time creating history file
    if df_pred is None and df_hist is None:
        # create the initial history of prices (without predictions)
        df_hist_new = pd.DataFrame(df_new['close'], columns=['close'])
        df_hist_new.columns = ['actual']
        df_hist_new['last_close'] = np.nan
        df_hist_new['last_close_diff'] = np.nan
        df_hist_new['last_close_cum'] = np.nan
        df_hist_new['lstm'] = np.nan
        df_hist_new['lstm_diff'] = np.nan
        df_hist_new['lstm_cum'] = np.nan
        df_hist_new['fbp'] = np.nan
        df_hist_new['fbp_diff'] = np.nan
        df_hist_new['fbp_cum'] = np.nan
        df_hist_new['ens'] = np.nan
        df_hist_new['ens_diff'] = np.nan
        df_hist_new['ens_cum'] = np.nan

    else:
        # append to existing history file
        # make a copy of df_hist
        df = df_hist.copy()

        # Get yesterdays closing price
        yesterday = df_new.index.max()
        yesterdays_close = df_new.loc[yesterday, 'close'].item()

        # update df_hist with yesterdays_close,
        update_price(df, yesterday, yesterdays_close, 'actual')

        # remove old predictions
        # yesterdays nan should have been replaced in the prevous step
        df = df[~df['actual'].isnull()]

        # add new predictions
        df_hist_new = pd.concat([df, df_pred])

        # calculate the difference between actual/predicted values
        # for current period and cumulative
        df_hist_new['last_close_diff'] = df_hist_new['last_close'] - df_hist_new['actual']
        df_hist_new['last_close_cum'] = df_hist_new['last_close_diff'].cumsum()

        df_hist_new['lstm_diff'] = df_hist_new['lstm'] - df_hist_new['actual']
        df_hist_new['lstm_cum'] = df_hist_new['lstm_diff'].cumsum()

        df_hist_new['fbp_diff'] = df_hist_new['fbp'] - df_hist_new['actual']
        df_hist_new['fbp_cum'] = df_hist_new['fbp_diff'].cumsum()

        df_hist_new['ens_diff'] = df_hist_new['ens'] - df_hist_new['actual']
        df_hist_new['ens_cum'] = df_hist_new['ens_diff'].cumsum()

        # sort by date
        df_hist_new.sort_index(inplace=True)

    return df_hist_new


def update_price(df_hist, date, value, type='actual'):
    """
    Updates a price in the df_hist table
    :param df_hist: A dataframe with historical back-tested data
    :param date:  Date of the price to update
    :param value:  Value of the price to update
    :param type:  Column name in the df to update
    :return: none
    """

    # update the price as of the date
    # should be yesterdays price
    df_hist.at[date, type] = value


def update_predictions(stock, stock_type, config_data, transform, shift, feature_path, n_steps, n_predict,
                       model_path, df_histpath, key, weights):
    """
    Pulls new market data, predicts the next n_predict days, updates the df_hist files
    :param key:
    :param weights:
    :param stock:
    :param stock_type:
    :param config_data:
    :param transform:
    :param shift:
    :param feature_path:
    :param n_steps:
    :param n_predict:
    :param model_path:
    :param df_histpath:
    :return:
    """

    # Get new data
    print("\n\nGetting Market Data")
    print('=' * 60)
    if stock_type == 'stock':
        # stock
        df_new = get_consolidated_stock_data(stock, key, config_data, 'full')
    else:
        # crypto
        df_new = get_consolidated_crypto_data(stock, key, config_data)

    # shift features to lagged
    if shift is not None:
        df_transposed = shift_features(df_new, shift)
        print('\nShifting Features')
        print('=' * 60)
        print('=>features shifted')
    else:
        df_transposed = df_new.copy()

    # Transform the data to be stationary
    features_to_transform = ['open', 'high', 'low', 'close']
    transform_stationary(df_transposed, features_to_transform, transform)

    # Get the features used in the final model
    print("\nGetting Features")
    print('=' * 60)
    try:
        df_features = pd.read_pickle(feature_path)
        features = [f for f in df_features.columns if f not in ['symbol']]
        print('=>got features')
    except:
        print(f'=>Error getting features!{feature_path}')
        exit(0)

    # Prepare data:
    # Subset rows to the latest n_steps for prediction input
    # Subset cols to keep the features required for the mode input
    # convert to numpy arrays
    idx_dates, array_y, array_X = prepare_data(df_transposed, n_steps, features)

    # scale the input and outputs
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_X = scaler_X.fit_transform(array_X)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_y = scaler_y.fit_transform(array_y)

    # get the trained model
    print("\nLoading the Model")
    print("=" * 60)
    try:
        model = keras.models.load_model(model_path)
        print('=>Model Loaded')
    except:
        print('=>Model Failed to load!')
        exit(0)

    # make predictions
    print("\nMaking Price Predictions")
    print("=" * 60)
    try:
        df_pred = make_ensemble_predictions(df_new, model, scaler_y, scaled_X, n_steps, len(features), n_predict,
                                            df_new.index.max(), weights, stock_type)
        print("=>Prices predicted\n")
    except:
        print('=>Error making price predictions!\n')
        exit(0)

    # get the previous df_hist data
    print('\nLoading Price History')
    print('=' * 60)
    try:
        df_hist = pd.read_pickle(df_histpath)
        print('=>history loaded')
    except:
        print('=>Error loading price history file')
        exit(0)

    # update the hist file with yesterdays close price, and add the new predictions
    print('\nRolling price model forward')
    print('=' * 60)
    try:
        # df_hist_new = roll_predictions(df_new, df_pred, df_hist)
        df_hist_new = roll_ensemble_predictions(df_new, df_pred, df_hist)
        df_hist_new.to_pickle(df_histpath)
        print('=>price model rolled forward\n')
    except:
        print('=>Error rolling price model!\n')
        exit(0)

    return df_hist_new
