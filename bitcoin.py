######## Importing The Modules ########
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf  # Using yfinance instead of pandas_datareader
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Define cryptocurrency and currency
crypto_currency = 'BTC'
against_currency = 'USD'

# Set date range for historical data
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

# Fetch historical data using yfinance
data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)
print(data.head())

######## Scaling Data ########
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Define prediction days
prediction_days = 60

# Prepare training data
x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

######## Creating Network Layers ########
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model with corrected loss function
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

######## Testing the Model ########
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

# Fetch test data using yfinance
test_data = yf.download(f'{crypto_currency}-{against_currency}', start=test_start, end=test_end)
actual_prices = test_data['Close'].values

# Prepare model inputs
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Prepare test data for prediction
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

######## Plotting the Graph ########
plt.plot(actual_prices, color='Black', label='Actual Prices')
plt.plot(prediction_prices, color='Green', label='Predicted Prices')
plt.title(f'{crypto_currency} Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
