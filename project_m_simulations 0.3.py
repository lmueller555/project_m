import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import time
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("LSTMProject").getOrCreate()

# Use SparkSQL to query the data
modeling_df = spark.sql('SELECT * FROM lax.lstm_model_test_data').toPandas()

# Converting 'Date' from string to datetime and setting it as the index
modeling_df['Date'] = pd.to_datetime(modeling_df['Date'])
modeling_df.set_index('Date', inplace=True)

# Using 'Close/Last' for prediction and scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(modeling_df['Close/Last'].values.reshape(-1, 1))

# Creating a dataset suitable for time series prediction
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Modify time step
time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Revised LSTM model
model = Sequential()
model.add(LSTM(30, input_shape=(time_step, 1)))
model.add(Dropout(0.15))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

# Train the model with validation split and early stopping
epochs = 5
batch_size = 128
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Define a function for the trading simulation with LSTM model
def trading_simulation(dataset, buy_threshold, sell_threshold, take_profit_threshold, stop_loss_threshold, deviation_threshold, model, scaler):
    cash = 100000  # Initial cash
    shares_held = 0  # Initial shares
    buy_price = 0  # Price at which shares were bought
    investment_values = []  # Track investment value over time
    trade_count = 0  # Count the number of trades

    for index in range(len(dataset) - time_step):
        current_price = dataset.iloc[index + time_step]['Close/Last']
        previous_data = scaled_data[index:index + time_step]
        predicted_price = scaler.inverse_transform(model.predict(previous_data.reshape(1, time_step, 1)))[0][0]

        ma_200d = dataset.iloc[index + time_step]['200d MA']
        deviation = (current_price - ma_200d) / ma_200d

        # Check for take-profit or stop-loss conditions
        if shares_held > 0:
            if current_price >= buy_price * take_profit_threshold or current_price <= buy_price * stop_loss_threshold:
                cash += shares_held * current_price
                shares_held = 0
                trade_count += 1
                buy_price = 0  # Reset buy_price since we've sold our shares

        # Buy or sell based on the deviation and predicted price
        if deviation < -deviation_threshold and predicted_price > current_price * buy_threshold and cash >= current_price:
            shares_to_buy = cash // current_price
            cash -= shares_to_buy * current_price
            shares_held += shares_to_buy
            buy_price = current_price  # Set the buy_price to the current price
            trade_count += 1
        elif deviation > deviation_threshold and predicted_price < current_price * sell_threshold and shares_held > 0:
            cash += shares_held * current_price
            shares_held = 0
            trade_count += 1

        investment_values.append(cash + shares_held * current_price)

        # Debugging logs
        if index % 100 == 0:
            print(f"Processing index {index}")
        if shares_held > 0:
            print(f"Index {index}: Bought at {current_price}, Shares held: {shares_held}, Cash left: {cash}")
        else:
            print(f"Index {index}: Sold at {current_price}, Cash: {cash}")

    return investment_values, cash + shares_held * dataset.iloc[-1]['Close/Last'] - 100000, trade_count

# Running multiple generations with take-profit and stop-loss
generations = 10
best_deviation_threshold = 0.09  # Initial deviation threshold
best_profit = -float('inf')
best_investment_values = []
best_trade_count = 0
best_parameters = {}

for i in range(generations):
    # Randomly adjust thresholds and parameters
    buy_threshold = random.uniform(0.95, 1.05)
    sell_threshold = random.uniform(0.95, 1.05)
    take_profit_threshold = random.uniform(0.95, 1.05)
    stop_loss_threshold = random.uniform(0.85, 0.95)
    deviation_threshold = best_deviation_threshold * random.uniform(0.95, 1.05)

    # Run the simulation
    investment_values, profit, trade_count = trading_simulation(
        modeling_df, buy_threshold, sell_threshold, take_profit_threshold, stop_loss_threshold, deviation_threshold, model, scaler)

    # Update best performing generation
    if profit > best_profit:
        best_profit = profit
        best_parameters = {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'take_profit_threshold': take_profit_threshold,
            'stop_loss_threshold': stop_loss_threshold,
            'deviation_threshold': deviation_threshold
        }
        best_investment_values = investment_values
        best_trade_count = trade_count

# Output the results of the best generation
print(f"Best Generation - Profit: {best_profit}, Trades: {best_trade_count}, Parameters: {best_parameters}")

# Plotting the investment value over time for the best generation
plt.figure(figsize=(12, 6))
plt.plot(best_investment_values, label='Investment Value')
plt.title('Best Generation AI Investment Value Over Time')
plt.xlabel('Days')
plt.ylabel('Investment Value in $')
plt.legend()
plt.grid(True)
plt.show()

# C:\\Users\\lmueller\\Desktop\\Project M\\Project M Data Processed.xlsx
