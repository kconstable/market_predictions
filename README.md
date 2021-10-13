# An ensemble method to predict closing prices for stocks/crypto using LSTM & Facebook Prophet Models
There are several goals for this project. The first is **to determine if stock/crypto prices can be accurately predicted** based on price history, commodity prices, market indexes, and economic and technical indicators. Second, **to build a flexible & reusable framework** to evaluate any number of structured data features to predict prices for any stock/crypto for which data is available. Lastly, **to build a plolty dashboard** to view price history, predictions, and prediction error and automatically roll the model forward daily.  Models were developed for Bitcoin (BTC), VM Ware (VMW) and Boralex (BLX.TO). For brevity, only the results for Bitcoin are included in this summary.

## Overview
An ensemble method was developed by combining a long-short-term-memory (LSTM) Model, a Facebook Prophet time-series forecasting model, and a naive model that always predicts tomorrow's closing price is equal to today's closing price.  The naive model was introduced to add more weight to the most current closing price. Linear programming was used to find the optimal allocation to each model that resulted in a cumulative and average prediction error of zero in a historical back-test from January 2021 to September 2021 for Bitcoin.  The LSTM model tends to underestimate price trends while Prophet tends to overestimate prices.  As a result, the ensemble method performs better than any individual model.
+ LSTM: 44.8%
+ Prophet: 44.6%
+ Naive: 10.5%

![image](https://user-images.githubusercontent.com/1649676/137159566-a37994e3-e75b-4741-bb7b-3cb94a754736.png)


## Data Aquisition & Analysis
Data was sourced from the free API at [Alpha Vantage](https://www.alphavantage.co/) to pull stock/cryto/commodity/market index prices, economic indicators, and technical indicators.  Bitcoin prices underwent rapid growth, from a low of ~4k to a high of ~63k during the period, resulting in a skewed/bi-model distribution of prices. The price data was log-transformed to make it stationary and features were max-min-scaled to convert each to a fixed scale.


![image](https://user-images.githubusercontent.com/1649676/137164160-713777d0-516d-4432-af37-1f3de06aa9bb.png)


## Feature Selection
Several economic and technical indicators, commodity prices, and market indexes were considered for inclusion in the LTSM model.  Lagged versions of each indicator were evaluated to determine if values of up to 60 days prior to current had a stronger influence on current prices.  Recursive Feature Elimination (RFE) with a random forest regressor was used to select features by influence on closing prices. In total, 44 features were considered, and the RFE process selected the top 14 for inclusion in the model.
+ Economic Indicators (non-farm payroll, unemployment rate, consumer sentiment, expected inflation, Fed fund rate, bond-yields 3-month to 30-year duration)
+ Technical Indicators (Bollinger bands, RSI, MACD)
+ Commodity prices (Gold, Oil, Natural Gas)
+ Market Indexes (S&P500, VXX Volatility Index)
+ Seasonal Factors (month, day-of-week)

![image](https://user-images.githubusercontent.com/1649676/137167664-4a27ea47-e4ba-4d9f-ba0f-c00cd57d81f5.png)


## LSTM Model
The LSTM model was built using Keras/Tensorflow, and the hyper-parameters were optimized using Keras tuner. 10% of the training data was withheld as an out-of-sample validation set, and early stopping was utilized to determine the number of epochs used in training.  The model takes the latest 25 days of price and feature data as inputs and predicts the next 3-days of prices.

![image](https://user-images.githubusercontent.com/1649676/137169731-e4382e9a-1b31-437d-ada5-b44681d12fec.png)


## Plotly/Dash Application
A plotly/dash application was built to automatically pull daily price/features data, make the next three days of predictions, and calculate the daily and cumulative prediction errors.  The user can select the stock/crypto, and the application displays the predicted vs. actual prices, the daily and cumulative prediction errors, and presents the model details.

![image](https://user-images.githubusercontent.com/1649676/137170867-b5ea6dc2-a58f-4d45-9532-5dc805cf82bb.png)


## File Structure
The data files in the repository are organzed as follows:
+ **root**: Jupyter Notebooks
  + market_data: data collection, analysis, consolidation, feature selection
  + LSTM_model: data pre-processing, LSTM model training, optimiztaion
  + predict_market_data: construction of the ensemble method, historical back-test,visualizations for the plolty/dash app
+ **config**: folder to host json config files used to define model and data details
+ **data**: folder containing finalized data used in the models and historical back-tests
+ **models**: finalized LSTM models
+ **dash**: plotly/dash files1
  + main.py - controls daily updates flow
  + preprocess.py - functions to collect and preprocess data, make predictions
  + visuals.py - functions to plot visuals in the app
  + dash_app.py - code that defines the app layout and callbacks



