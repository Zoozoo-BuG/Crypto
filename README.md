# Cryptocurrency Price Prediction Using Machine Learning Model
# Introduction
In the realm of finance, predicting cryptocurrency prices has become a significant area of interest,
especially with the volatility associated with digital currencies. This document outlines a Pythonbased approach to predict the price of Bitcoin (BTC) against the Indian Rupee (INR) using a Long
Short-Term Memory (LSTM) neural network. The model leverages historical price data to forecast
future prices, providing insights for traders and investors.
# Key Concepts
1. LSTM Neural Networks: A type of recurrent neural network (RNN) that is particularly
effective for time series prediction due to its ability to remember long-term dependencies.
2. Data Normalization: The process of scaling data to a specific range, which helps in
improving the performance and training stability of the neural network.
3. Training and Testing Data: The dataset is split into training and testing sets to evaluate the
model's performance on unseen data.
Code Structure
The code is structured into several key sections:
Importing Modules: Essential libraries for data manipulation, visualization, and machine learning.
Data Acquisition: Fetching historical cryptocurrency price data from Yahoo Finance.
Data Preprocessing: Normalizing the data and preparing it for training the LSTM model.
 Model Creation: Building the LSTM model architecture.
 Model Training: Fitting the model to the training data.
Prediction and Visualization: Making predictions on the test data and visualizing the results.
Code Examples
# Importing The Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
from tensorflow.keras.models import Model, Sequential
# Define cryptocurrency and currency
crypto_currency = 'BTC'
normal_currency = 'INR'
# Set the date range for historical data
start = dt.datetime(2016,1,1) end =
dt.datetime.now()
# Fetch historical data
data = web.DataReader(f'{crypto_currency}-{normal_currency}', 'yahoo', start, en
######## Open Source Data #########
# Normalize the closing prices
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
# Define prediction days
prediction_days = 60
# Prepare training data
x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)):
x_train.append(scaled_data[x-prediction_days:x,0]) y_train.append(scaled_data[x, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
######## Creating Network Layers ########
# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True)) model.add(Dropout(0.2))
model.add(LSTM(units=50)) model.add(Dropout(0.2))
model.add(Dense(units=1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
######## Testing The DATA ########
# Define the test data range
test_start = dt.datetime(2020,1,1) test_end =
dt.datetime.now()
# Fetch test data
test_data = web.DataReader(f'{crypto_currency}-{normal_currency}','yahoo', test_
actual_prices = test_data['Close'].values
# Prepare the model inputs
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days: model_inputs =
model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)
# Prepare test data for prediction
x_test = []
for x in range(prediction_days, len(model_inputs)):
x_test.append(model_inputs[x-prediction_days:x, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# Make predictions
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)
######## Plotting The GRAPH ########
# Visualize the results
plt.plot(actual_prices, color='Black', label='Actual Prices')
plt.plot(prediction_prices, color='Green', label='Predicted Prices') plt.title(f'{crypto_currency}
price prediction')
plt.xlabel('Time') plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
# Conclusion
The provided code demonstrates a comprehensive approach to predicting cryptocurrency prices using
an LSTM neural network. By leveraging historical price data, the model is capable of making
informed predictions, which can be invaluable for traders and investors in the cryptocurrency market.
The visualization of actual versus predicted prices further aids in assessing the model's performance.
As the cryptocurrency market continues to evolve, such predictive models will play a crucial role in
decision-making processes.
